"""Pin publisher matcher outputs and prepare whole-source TRAIN Conference inputs.

Reference alignments are hash-bound metadata only. Neither preparation nor
qualification opens development/test alignment rows or invents semantic targets.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.workers import bounded_call

MATCHER_URL = "https://oaei.ontologymatching.org/2025/results/conference/conference_files/conference-track-2025.zip"
MATCHER_SHA = "e2266b2dedbf7636196e315fa9b8f12ebe327aa2226381a355d9beb5becceb7f"
ONTOLOGY_SHA = "78688cda05857b594be188db6831abc5d890d2e38d6602e0cd8d3c82ecb24546"
RESULTS_URL = "https://oaei.ontologymatching.org/2025/results/conference/index.html"
ALIGN = "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment#}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def bound_bytes(record):
    raw = Path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Conference source hash mismatch")
    return raw


def immutable(path, value):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return immutable_bytes(path, raw)


def immutable_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to replace frozen Conference artifact: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    return binding(path)


def prepare(previous, matcher_archive, output):
    """Copy immutable sources; extract TRAIN matcher files by frozen membership."""
    previous, output = Path(previous), Path(output)
    assets = json.loads((previous / "assets.json").read_text())
    splits = json.loads((previous / "pair-splits.json").read_text())
    files = {row["id"]: row for row in assets["files"]}
    receipt = json.loads((previous / "conference-receipt.json").read_text())["value"]
    ontology = {"path": str(previous / "conference.zip"), "sha256": ONTOLOGY_SHA}
    archive = {"path": str(matcher_archive), "sha256": MATCHER_SHA}
    pinned = immutable_bytes(output / "conference-track-2025.zip", bound_bytes(archive))
    ontologies = immutable_bytes(output / "conference.zip", bound_bytes(ontology))
    rows = []
    with zipfile.ZipFile(pinned["path"]) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for pair in splits["pairs"]:
            if pair["cohort"] != "conference_2025":
                continue
            if pair["split"] == "train" and "ekaw" in pair["ontology_names"]:
                raise ValueError("Whole-ontology holdout cannot enter TRAIN")
            member = "LogMap-" + "-".join(pair["ontology_names"]) + ".rdf"
            row = {
                **pair,
                "historical_metadata": dict(pair),
                "theory_scope": "complete_source_ontologies_with_resolved_imports",
                "matcher_member": member,
                "matcher_archive": pinned,
                "reference_bindings": [
                    {
                        "path": files[key]["path"],
                        "sha256": files[key]["sha256"],
                        "role": "evaluator_reference_only_not_matcher_or_repair_label",
                    }
                    for key in pair["reference_assets"]
                ],
                "reference_rows_opened": False,
                "outcomes_opened": False,
                "status": "deferred_split_closed",
            }
            row["ontology_bindings"] = [
                {"path": files[key]["path"], "sha256": files[key]["sha256"]}
                for key in pair["ontology_assets"]
            ]
            if member not in names:
                row["status"] = "unavailable_publisher_member"
            elif pair["split"] == "train":
                destination = output / "train" / "-".join(pair["ontology_names"])
                row["matcher"] = immutable_bytes(destination / "LogMap.rdf", source.read(member))
                row["ontology_bindings"] = [
                    immutable_bytes(
                        output / "ontologies" / Path(item["path"]).name, bound_bytes(item)
                    )
                    for item in row["ontology_bindings"]
                ]
                row["status"] = "ready_for_whole_source_qualification"
            rows.append(row)
    if len(rows) != 21:
        raise ValueError("Expected all 21 frozen Conference pairs")
    provenance = dict(
        schema="exact-repair/conference-source-provenance/v1",
        release="OAEI-2025-Conference",
        matcher="LogMap",
        execution_origin="publisher-MELT",
        matcher_source_url=MATCHER_URL,
        publisher_results_url=RESULTS_URL,
        ontology_source_url=receipt["url"],
        matcher_archive=pinned,
        ontology_archive=ontologies,
        inherited_ontology_archive=ontology,
        inherited_split_binding=binding(previous / "pair-splits.json"),
        inherited_asset_binding=binding(previous / "assets.json"),
        ontology_hashes=receipt["ontology_hashes"],
        archive_members=names,
        matcher_selection="LogMap publisher/MELT provenance; fixed before own outcome inspection",
        use_scope="local public-benchmark research preparation authorized by user",
        redistribution_license="explicit redistribution license not established; do not assert CC-BY",
        license_evidence="Publisher distributes these systems' outputs for the benchmark; archive contains no license file",
        production_matcher_used=False,
        reference_rows_opened=False,
        heldout_matcher_rows_opened=False,
        exposure_disclosure=[
            "Historical source-only cmt-ekaw discussion/ontology provenance was exposed; no claim that all ontology bytes were unseen",
            "Published aggregate result page exists; it is not this study's held-out outcome and was not used to choose favourable pairs",
        ],
        whole_ontology_holdout="ekaw",
        intended_semantic_targets="not supplied by reference equivalences",
    )
    manifest = dict(
        schema="exact-repair/conference-input-release/v1",
        rows=rows,
        provenance=immutable(output / "provenance.json", provenance),
        train_count=sum(row["split"] == "train" for row in rows),
        scheduled=len(rows),
    )
    return immutable(output / "manifest.json", manifest)


def train_row(manifest, pair_id):
    manifest = json.loads(bound_bytes(manifest))
    rows = [row for row in manifest["rows"] if row["id"] == pair_id]
    if len(rows) != 1 or rows[0]["split"] != "train" or "ekaw" in rows[0]["ontology_names"]:
        raise ValueError("Only frozen TRAIN pair inputs may be opened")
    row = rows[0]
    if row["status"] != "ready_for_whole_source_qualification":
        raise ValueError("Pair has no qualified publisher matcher source")
    if row.get("matcher", {}).get("path") in {r["path"] for r in row["reference_bindings"]}:
        raise ValueError("Reference alignment cannot be used as matcher output")
    return row


def _mapping_axioms(raw, left, right):
    """Type every correspondence against full native source signatures."""
    import pyowl_core as owl

    kinds = (owl.EntityKind.CLASS, owl.EntityKind.OBJECT_PROPERTY, owl.EntityKind.DATA_PROPERTY)
    signatures = []
    for view in (left, right):
        signature = {}
        for kind in kinds:
            for entity in view.signature(kind):
                iri = entity.iri.value
                if iri in signature and signature[iri] != entity:
                    raise ValueError(
                        "Punned matcher endpoint requires an explicit entity-kind policy"
                    )
                signature[iri] = entity
        signatures.append(signature)
    constructors = {
        owl.EntityKind.CLASS: (owl.EquivalentClasses, owl.SubClassOf),
        owl.EntityKind.OBJECT_PROPERTY: (owl.EquivalentObjectProperties, owl.SubObjectPropertyOf),
        owl.EntityKind.DATA_PROPERTY: (owl.EquivalentDataProperties, owl.SubDataPropertyOf),
    }
    root = ET.fromstring(raw)
    if root.tag != ALIGN + "Alignment" and root.find(ALIGN + "Alignment") is None:
        raise ValueError("Matcher file lacks an Alignment envelope")
    rows = []
    for index, cell in enumerate(root.iter(ALIGN + "Cell")):
        endpoints = []
        for tag, signature in zip(("entity1", "entity2"), signatures):
            element = cell.find(ALIGN + tag)
            iri = None if element is None else element.get(RDF + "resource")
            if iri not in signature:
                raise ValueError(f"Unsupported or absent matcher endpoint at row {index}: {iri}")
            endpoints.append(signature[iri])
        first, second = endpoints
        if first.kind != second.kind:
            raise ValueError(f"Matcher endpoint kinds differ at row {index}")
        equivalent, subclass = constructors[first.kind]
        relation = cell.findtext(ALIGN + "relation", "").strip()
        axioms = {
            "=": lambda: (equivalent(owl.CanonicalSet(endpoints)),),
            "<": lambda: (subclass(first, second),),
            ">": lambda: (subclass(second, first),),
        }
        if relation not in axioms:
            raise ValueError(f"Unsupported matcher relation at row {index}: {relation}")
        score = float(cell.findtext(ALIGN + "measure", "nan"))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Matcher confidence must be finite in [0,1]")
        rows.append((first, second, axioms[relation](), score))
    return rows


def construct_native(manifest, pair_id):
    import pyowl_core as owl
    from exact.repair.candidates import make_candidate, replacement_cost_features
    from exact.repair.records import PolicyV2, RepairInputV2, RevisionObjectV2, promote_input_v3

    row = train_row(manifest, pair_id)
    snapshots, route = [], []
    for item in row["ontology_bindings"]:
        bound_bytes(item)
        snapshot = owl.load_snapshot(
            item["path"],
            options=owl.LoadOptions(
                backend=owl.BackendPreference.NATIVE,
                imports=owl.ImportPolicy.RESOLVE_LOCAL,
                offline=True,
                allow_partial_rdf_mapping=False,
            ),
        )
        report = snapshot.report
        route.append(
            dict(
                document=item,
                backend=report.backend,
                complete_imports=snapshot.is_complete,
                document_count=report.document_count,
                resolution_attempts=report.resolution_attempts,
                effective_axiom_count=report.effective_axiom_count,
                logical_fingerprint=snapshot.logical_fingerprint.hex,
                diagnostics=[str(v) for v in report.diagnostics],
            )
        )
        if (
            not snapshot.is_complete
            or tuple(snapshot.iter_extensions())
            or report.document_count != 1
        ):
            return dict(
                status="unsupported_full_source",
                source_routes=route,
                detail="Incomplete import closure, unsupported source extensions, or imported document hashes not yet pinned",
            )
        snapshots.append(snapshot)
    mapping_rows = _mapping_axioms(bound_bytes(row["matcher"]), *snapshots)
    objects, evidence = [], []
    for index, (left, right, axioms, score) in enumerate(mapping_rows):
        oid = pair_id + f":LogMap:{index}"
        candidates = (
            make_candidate(oid, axioms, ("keep",)),
            make_candidate(
                oid, (), ("delete",), cost_features=replacement_cost_features(axioms, ())
            ),
        )
        objects.append(
            RevisionObjectV2(
                oid,
                "mapping",
                axioms,
                candidates,
                source="OAEI-2025-MELT-LogMap",
                source_entity=left,
                target_entity=right,
            )
        )
        evidence.append(
            (
                oid,
                {
                    "score": score,
                    "matcher_archive_hash": row["matcher_archive"]["sha256"],
                    "matcher_member": row["matcher_member"],
                    "matcher_row_index": index,
                },
            )
        )
    source_axioms, target_axioms = (tuple(snapshot.iter_axioms()) for snapshot in snapshots)
    problem = promote_input_v3(
        RepairInputV2(
            (*source_axioms, *target_axioms),
            tuple(objects),
            PolicyV2(),
            source_identity=row["ontology_bindings"][0]["sha256"],
            target_identity=row["ontology_bindings"][1]["sha256"],
            matcher_identity="OAEI-2025-MELT-LogMap:" + row["matcher"]["sha256"],
            evidence=tuple(evidence),
            source_axioms=source_axioms,
            target_axioms=target_axioms,
            source_documents=(
                (
                    row["ontology_names"][0],
                    row["ontology_bindings"][0]["path"],
                    row["ontology_bindings"][0]["sha256"],
                ),
            ),
            target_documents=(
                (
                    row["ontology_names"][1],
                    row["ontology_bindings"][1]["path"],
                    row["ontology_bindings"][1]["sha256"],
                ),
            ),
            candidate_coverage="elementary_keep_delete; rich grammar not yet generated",
        )
    )
    return dict(
        status="constructed",
        problem=problem.to_dict(),
        source_routes=route,
        mapping_count=len(objects),
        complete_source_axiom_counts=[len(source_axioms), len(target_axioms)],
        reference_rows_opened=False,
        known_intended_theory=False,
    )


def _check_native(problem_record, scope, reasoner="auto", backend="auto"):
    from exact.repair.records import read_record
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms

    problem = read_record(problem_record)
    axioms = {
        "source": problem.source_axioms,
        "target": problem.target_axioms,
        "union": problem.fixed_axioms,
        "alignment": (
            *problem.fixed_axioms,
            *(a for obj in problem.objects for a in obj.original_axioms),
        ),
    }[scope]
    return (
        OwlVerifier(reasoner, backend=backend).check_theory(snapshot_from_axioms(axioms)).to_dict()
    )


def inspect_profile(problem_record):
    """Explain the strict existing HermiT gate without disabling it or editing OWL."""
    import pyowl_core as owl
    from pyhermit.profile import validate_owl2_dl_view
    from exact.repair.records import read_record
    from exact.repair.owl import snapshot_from_axioms

    problem = read_record(problem_record)
    reports = {}
    for scope, axioms in (("source", problem.source_axioms), ("target", problem.target_axioms)):
        report = validate_owl2_dl_view(snapshot_from_axioms(axioms))
        reports[scope] = dict(
            issues=[
                dict(
                    rule_id=issue.rule_id,
                    severity=str(issue.severity),
                    message=issue.message,
                    constructor=issue.constructor,
                )
                for issue in report.issues
            ],
            datatype_axioms=[
                repr(axiom)
                for axiom in axioms
                if any(
                    isinstance(entity, owl.Datatype)
                    and entity.iri.value == "http://www.w3.org/2001/XMLSchema#date"
                    for entity in owl.signature(axiom)
                )
            ],
        )
    return reports


def qualify(manifest, pair_id, output, *, seconds=300.0, check_seconds=30.0, memory_mb=20000):
    """Bound construction plus each complete-source check; unknown stays unknown."""
    row = train_row(manifest, pair_id)  # Reject closed splits before any row/file access.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(seconds=float(seconds), check_seconds=float(check_seconds), memory_mb=memory_mb)
    if any(not math.isfinite(float(v)) or v <= 0 for v in settings.values()):
        raise ValueError("Finite positive Conference qualification budgets required")
    report_path = output / "qualification.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if any(
            previous.get(k) != v
            for k, v in dict(pair_id=pair_id, manifest=manifest, settings=settings).items()
        ):
            raise ValueError("Incompatible Conference qualification resume")
        if previous.get("input"):
            bound_bytes(previous["input"])
        if previous.get("completed_epoch"):
            return previous
        report = previous
        started = float(report["started_epoch"])
    else:
        started = time.time()
        report = dict(
            schema="exact-repair/conference-qualification/v1",
            pair_id=pair_id,
            split=row["split"],
            manifest=manifest,
            input_source=row,
            settings=settings,
            started_epoch=started,
            checks={},
            reference_rows_opened=False,
            intended_semantic_target="unavailable_requires_independent_grounding",
        )
        write_artifact(report_path, report)
    deadline = min(started + seconds, float(os.environ.get("EXACT_REPAIR_DEADLINE_EPOCH", "inf")))
    if "construction" not in report:
        construction = bounded_call(
            construct_native,
            manifest,
            pair_id,
            timeout=min(check_seconds, deadline - time.time() - 2),
            memory_mb=memory_mb,
        )
        report["construction"] = dataclasses.asdict(construction)
        write_artifact(report_path, report)
    construction = report["construction"]
    value = construction["value"] or {}
    if (
        construction["status"] == "complete"
        and construction["cleanup_complete"]
        and value.get("status") == "constructed"
    ):
        report["input"] = immutable(output / "input.json", value["problem"])
        for scope in ("source", "target", "union", "alignment"):
            if scope in report["checks"]:
                continue
            remaining = deadline - time.time() - 2
            if remaining <= 0:
                report["checks"][scope] = dict(status="unavailable_deadline")
                continue
            result = bounded_call(
                _check_native,
                value["problem"],
                scope,
                timeout=min(check_seconds, remaining),
                memory_mb=memory_mb,
            )
            report["checks"][scope] = dataclasses.asdict(result)
            write_artifact(report_path, report)
            if not result.cleanup_complete:
                break
    report.update(elapsed_seconds=time.time() - started, completed_epoch=time.time())
    write_artifact(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("previous")
    prep.add_argument("matcher_archive")
    prep.add_argument("output")
    check = sub.add_parser("qualify")
    check.add_argument("manifest")
    check.add_argument("pair_id")
    check.add_argument("output")
    check.add_argument("--seconds", type=float, default=300)
    check.add_argument("--check-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.operation == "prepare":
        print(json.dumps(prepare(args.previous, args.matcher_archive, args.output)))
    else:
        report = qualify(
            binding(args.manifest),
            args.pair_id,
            args.output,
            seconds=args.seconds,
            check_seconds=args.check_seconds,
        )
        print(
            json.dumps(
                {
                    "pair_id": args.pair_id,
                    "construction": report["construction"]["status"],
                    "checks": {k: v["status"] for k, v in report["checks"].items()},
                }
            )
        )


if __name__ == "__main__":
    main()
