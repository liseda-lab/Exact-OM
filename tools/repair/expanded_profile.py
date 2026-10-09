"""Development-only native profiles and conservative structural-parent selection.

This stage never labels or evaluates the proposed held-out corpus. Its schedule
is input to a separately registered corpus-finalization stage, not a gate pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import pyowl_core as owl

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, canonical_json, promote_input_v3
from tools.repair.batch import read, sha
from tools.repair.corpus import MECHANISMS, _case, coherent_control
from tools.repair.prepare import case_from_dict, case_to_dict

IDENTITY_VERSION = "owl-connected-core-wl/v1"
MECHANISMS_SELECTED = (
    "papers",
    "overlap",
    "disjointness",
    "conjunct",
    "participant",
    "domain",
    "range",
    "filler",
)


def core_fingerprint(axioms, editable_axioms):
    """Conservative name-free incidence hash of the largest editable component.

    Ignore declarations, annotations, labels, provenance, side names and detached
    nuisance/distractor components. Constructor fields retain their roles; sets
    are unordered. WL collisions merge parents (reducing coverage), never split
    isomorphic parents. This is structural grouping, not semantic equivalence.
    """
    graph, entities = nx.Graph(), {}

    def node(label):
        index = len(graph)
        graph.add_node(index, label=label)
        return index

    def add(value):
        if hasattr(value, "iri") and hasattr(value, "kind"):
            key = (type(value).__name__, value.iri.value)
            if key not in entities:
                builtin = (
                    value.iri.value
                    if value.iri.value.startswith("http://www.w3.org/2002/07/owl#")
                    else ""
                )
                entities[key] = node(type(value).__name__ + builtin)
            return entities[key]
        if dataclasses.is_dataclass(value):
            result = node(type(value).__name__)
            for field in dataclasses.fields(value):
                if field.name == "annotations":
                    continue
                slot = node("field:" + field.name)
                graph.add_edge(result, slot)
                graph.add_edge(slot, add(getattr(value, field.name)))
            return result
        if isinstance(value, (owl.CanonicalSet, tuple, list, set, frozenset)):
            result = node("set" if not isinstance(value, (tuple, list)) else "sequence")
            for index, child in enumerate(value):
                slot = node("member" if not isinstance(value, (tuple, list)) else str(index))
                graph.add_edge(result, slot)
                graph.add_edge(slot, add(child))
            return result
        return node(type(value).__name__ + ":" + str(value))

    for axiom in set(axioms):
        if not isinstance(axiom, owl.Declaration):
            add(axiom)
    roots = {
        entities[key]
        for axiom in editable_axioms
        for entity in owl.signature(axiom)
        for key in [(type(entity).__name__, entity.iri.value)]
        if key in entities
    }
    components = [part for part in nx.connected_components(graph) if roots & part]
    if not components:
        raise ValueError("No editable connected structural core")
    maximum = max(map(len, components))
    # Equal largest components are kept as a set: duplicating an independent
    # component is not new structural evidence.
    hashes = sorted(
        {
            nx.weisfeiler_lehman_graph_hash(graph.subgraph(part), node_attr="label", iterations=8)
            for part in components
            if len(part) == maximum
        }
    )
    return canonical_hash((IDENTITY_VERSION, maximum, hashes))


def parent_fingerprints(case):
    editable = tuple(a for obj in case.problem.objects for a in obj.original_axioms)
    observed = core_fingerprint((*case.problem.fixed_axioms, *editable), editable)
    intended = (
        core_fingerprint(case.intended_theory, editable) if case.intended_theory else observed
    )
    return tuple(sorted({observed, intended}))


def parent_case(family, depth, seed, split="development"):
    if family not in MECHANISMS_SELECTED and family not in MECHANISMS:
        raise ValueError("Undeclared mechanism")
    parent = f"{family}:path-{depth}"
    case = _case(parent, family, split, depth, 0, seed, misleading_label_fraction=0.1)
    return dataclasses.replace(case, problem=promote_input_v3(case.problem), schema_revision="v3")


def checked_checkpoint(path, identity):
    if not path.exists():
        return None
    value = read(path)
    if value.get("identity") != identity:
        raise ValueError("Profile checkpoint dependencies changed: " + str(path))
    payload = {key: val for key, val in value.items() if key != "content_hash"}
    if canonical_hash(payload) != value.get("content_hash"):
        raise ValueError("Profile checkpoint content changed: " + str(path))
    return value


def checkpoint(path, identity, **payload):
    value = dict(identity=identity, **payload)
    write_artifact(path, dict(value, content_hash=canonical_hash(value)))
    return checked_checkpoint(path, identity)


def select_parents(candidates, exposed, *, counts, seed):
    """Union alias groups before selecting; all exposure is excluded everywhere."""
    owners, links = {}, {}

    def root(key):
        links.setdefault(key, key)
        if links[key] != key:
            links[key] = root(links[key])
        return links[key]

    for row in [*exposed, *candidates]:
        key = row["key"]
        root(key)
        for fingerprint in row["fingerprints"]:
            if fingerprint in owners:
                links[root(key)] = root(owners[fingerprint])
            owners[fingerprint] = key
    forbidden = {root(row["key"]) for row in exposed}
    selected, missing, used = [], [], set(forbidden)
    # Selection is fixed by structure and seed, never a runtime/success outcome.
    for family in MECHANISMS_SELECTED:
        pool = sorted(
            (row for row in candidates if row["family"] == family),
            key=lambda row: (row["depth"], canonical_hash((seed, row["key"]))),
        )
        fresh, local = [], set()
        for row in pool:
            if root(row["key"]) not in used | local:
                local.add(root(row["key"]))
                fresh.append(row)
        # Development profiling uses the two smallest eligible structures. The
        # remaining splits are hashed to avoid a train/test size-order confound.
        profile_count = counts["profile"]
        profile, remainder = fresh[:profile_count], fresh[profile_count:]
        remainder.sort(key=lambda row: canonical_hash((seed, row["key"])))
        available = profile + remainder
        offset = 0
        for split in ("profile", "train", "development", "test", "fresh_evaluation"):
            count = counts[split]
            assigned = available[offset : offset + count]
            selected.extend(dict(row, split=split, group_id=root(row["key"])) for row in assigned)
            used.update(root(row["key"]) for row in assigned)
            missing.extend(
                dict(family=family, split=split, status="unavailable_structural_parent")
                for _ in range(count - len(assigned))
            )
            offset += count
    return selected, missing


def prepare_inventory(manifest, output, identity):
    if manifest.get("schema") != "exact-repair/expanded-profile-plan/v1":
        raise ValueError("Unknown expanded profile plan")
    counts = manifest["parent_counts"]
    if set(counts) != {"profile", "train", "development", "test", "fresh_evaluation"} or any(
        type(value) is not int or value < 1 for value in counts.values()
    ):
        raise ValueError("Declare positive counts for every parent cohort")
    if (
        type(manifest["candidate_max_path_depth"]) is not int
        or not 1 <= manifest["candidate_max_path_depth"] <= 128
        or not manifest["exposure_sources"]
    ):
        raise ValueError("A finite candidate grid and historical exposure bindings are required")
    target = output / "inventory.json"
    existing = checked_checkpoint(target, identity)
    if existing:
        return existing
    exposed = []
    seen_cases = set()
    sources = []
    for binding in manifest["exposure_sources"]:
        path = Path(binding["path"])
        if sha(path) != binding["sha256"]:
            raise ValueError("Historical exposure source changed: " + str(path))
        # Labels are never deserialized or used. Only immutable case structures
        # contribute to this conservative all-splits exposure exclusion.
        document = read(path)
        sources.append(dict(binding, cases=len(document["cases"])))
        for record in document["cases"]:
            if record["hash"] in seen_cases:
                continue
            seen_cases.add(record["hash"])
            case = case_from_dict(record)
            exposed.append(
                dict(
                    key="exposed:" + record["hash"],
                    family=case.family,
                    fingerprints=parent_fingerprints(case),
                )
            )
        del document
    candidates = []
    for family in MECHANISMS_SELECTED:
        for depth in range(1, manifest["candidate_max_path_depth"] + 1):
            case = parent_case(family, depth, manifest["seed"])
            candidates.append(
                dict(
                    key=f"{family}:path-{depth}",
                    family=family,
                    depth=depth,
                    fingerprints=parent_fingerprints(case),
                )
            )
        write_artifact(output / "progress.json", dict(stage="structural_inventory", family=family))
    selected, missing = select_parents(
        candidates, exposed, counts=manifest["parent_counts"], seed=manifest["seed"]
    )
    aliases = {}
    for family in MECHANISMS:
        case = parent_case(family, 1, manifest["seed"])
        editable = tuple(a for obj in case.problem.objects for a in obj.original_axioms)
        key = core_fingerprint((*case.problem.fixed_axioms, *editable), editable)
        aliases.setdefault(key, []).append(family)
    return checkpoint(
        target,
        identity,
        schema="exact-repair/expanded-inventory/v1",
        status="provisional_pending_development_profile",
        sources=sources,
        exposure_policy="all historical splits excluded; labels not used",
        exposed=exposed,
        candidates=candidates,
        selected=selected,
        missing=missing,
        generator_aliases=MECHANISMS,
        alias_fingerprint_groups=list(aliases.values()),
        identity_version=IDENTITY_VERSION,
        requested_counts=manifest["parent_counts"],
        selected_counts=dict(Counter(row["split"] for row in selected)),
        test_outcomes_opened=False,
        corpus_finalized=False,
    )


def native_profile(case_record, settings, cache_directory):
    """One independently killable native development probe; no model or labels."""
    from exact.repair.grammar import compile_families, mapping_grammar, with_immutable_context
    from exact.repair.retrieval import RetrievalConfig, retrieve_vocabulary

    case = case_from_dict(case_record)
    if case.split != "development":
        raise ValueError("Profiling cannot open a held-out case")
    start = time.monotonic()
    retrieved = retrieve_vocabulary(case.problem, config=RetrievalConfig(**settings["retrieval"]))
    retrieval_seconds = time.monotonic() - start
    objects = []
    for obj in case.problem.objects:
        menu = retrieved.for_object(obj.object_id)
        encoding = mapping_grammar(
            obj,
            menu.classes,
            menu.properties,
            max_depth=settings["max_depth"],
            max_constructors=settings["max_constructors"],
            fixed_axioms=case.problem.fixed_axioms,
            source_classes=menu.source_classes,
            target_classes=menu.target_classes,
            source_properties=menu.source_properties,
            target_properties=menu.target_properties,
        )
        encoding = with_immutable_context(encoding, case.problem.fixed_axioms, case.problem.policy)
        compiled = compile_families(
            encoding,
            seconds=settings["circuit"]["aggregate_seconds"],
            cache_directory=str(cache_directory),
            circuit_limits=settings["circuit"],
        )
        families = [
            dict(
                name=f.name,
                status=f.status,
                detail=f.detail,
                telemetry=dict(f.circuit.telemetry if f.circuit else f.failure_telemetry),
            )
            for f in compiled.families
        ]
        objects.append(
            dict(
                object_id=obj.object_id,
                complete=compiled.complete,
                variables=encoding.variable_count,
                families=families,
                class_count=len(menu.classes),
                property_count=len(menu.properties),
            )
        )
    return json.loads(
        canonical_json(
            dict(
                status="complete" if all(obj["complete"] for obj in objects) else "partial",
                objects=objects,
                retrieval_seconds=retrieval_seconds,
                retrieval_provenance=dict(retrieved.provenance),
                elapsed_seconds=time.monotonic() - start,
                scope="retrieval and native semantic family compilation only; no solve/learning claim",
            )
        )
    )


def run(manifest_path, output):
    from exact.repair.study import runtime_manifest
    from exact.repair.workers import bounded_call

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = read(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_manifest()
    dependencies = [
        (name, sha(Path(__file__).with_name(name)))
        for name in ("expanded_profile.py", "corpus.py", "prepare.py")
    ]
    identity = canonical_hash(
        (
            sha(manifest_path),
            dependencies,
            IDENTITY_VERSION,
            nx.__version__,
            runtime["dependencies"],
            runtime["code_hashes"],
            runtime["ontology_implementations"],
        )
    )
    inventory = prepare_inventory(manifest, output, identity)
    rows = []
    for parent in inventory["selected"]:
        if parent["split"] != "profile":
            continue
        case = parent_case(parent["family"], parent["depth"], manifest["seed"])
        for variant in (case, coherent_control(case)):
            record = case_to_dict(variant)
            case_key = canonical_hash((parent["key"], variant.control))
            case_path = output / "development_cases" / (case_key + ".json")
            if case_path.exists() and read(case_path) != json.loads(json.dumps(record)):
                raise ValueError("Frozen development case changed")
            write_artifact(case_path, record)
            for mode in ("cold", "warm"):
                receipt = output / "rows" / (case_key + "-" + mode + ".json")
                row_identity = canonical_hash((identity, record["hash"], mode))
                row = checked_checkpoint(receipt, row_identity)
                if row is None:
                    # A fresh cold attempt gets its own directory. A interrupted
                    # cold population can never silently turn the retry warm.
                    attempts = output / "cache" / case_key
                    attempts.mkdir(parents=True, exist_ok=True)
                    if mode == "cold":
                        cache = attempts / ("population-" + str(len(list(attempts.iterdir())) + 1))
                        cache.mkdir()
                    else:
                        cold = checked_checkpoint(
                            output / "rows" / (case_key + "-cold.json"),
                            canonical_hash((identity, record["hash"], "cold")),
                        )
                        cache = Path(cold["cache_directory"])
                    write_artifact(
                        output / "progress.json",
                        dict(
                            stage="native_profile",
                            case=variant.case_id,
                            cache_mode=mode,
                            recorded=len(rows),
                        ),
                    )
                    started = time.monotonic()
                    result = bounded_call(
                        native_profile,
                        record,
                        manifest["profile_settings"],
                        str(cache),
                        timeout=manifest["per_case_seconds"],
                        memory_mb=manifest["per_case_memory_mb"],
                    )
                    row = checkpoint(
                        receipt,
                        row_identity,
                        case_id=variant.case_id,
                        parent=parent["key"],
                        family=parent["family"],
                        control=variant.control,
                        cache_mode_requested=mode,
                        cache_directory=str(cache),
                        call_status=result.status,
                        detail=result.detail,
                        result=result.value,
                        elapsed_seconds=time.monotonic() - started,
                        resources=dict(result.resource_usage),
                        cleanup_complete=result.cleanup_complete,
                    )
                if not row["cleanup_complete"]:
                    raise RuntimeError(
                        "Bounded native worker cleanup incomplete; inspect ownership before continuation"
                    )
                rows.append(
                    dict(
                        path=str(receipt),
                        sha256=sha(receipt),
                        call_status=row["call_status"],
                        result_status=(row.get("result") or {}).get("status", "unknown"),
                    )
                )
    report = checkpoint(
        output / "report.json",
        identity,
        schema="exact-repair/development-profile/v1",
        status="complete",
        inventory=dict(path=str(output / "inventory.json"), sha256=sha(output / "inventory.json")),
        scheduled=manifest["parent_counts"]["profile"] * len(MECHANISMS_SELECTED) * 4,
        recorded=len(rows),
        rows=rows,
        missing_parents=inventory["missing"],
        test_outcomes_opened=False,
        corpus_finalized=False,
        next_stage="xr21-expanded-corpus-001",
        gates="G0-G2 and learning efficiency remain unestablished",
        interpretation="All profile outcomes retained; operational evidence only. Corpus finalization remains registered.",
    )
    write_artifact(output / "progress.json", dict(stage="complete", recorded=len(rows)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
