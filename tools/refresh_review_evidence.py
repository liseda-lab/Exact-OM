"""Create a new review publication from retained literal provenance, without rescoring.

The original private index supplies exact semantic terms. Only unresolved literals
are reconsidered; identities, producer graph statuses and bridge history are retained.
This is an offline preparation operation, never a request-time enrichment path.
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

from exact_inspect.artifacts import atomic_json, publish_bundle, validate_bundle
from exact_inspect.context import OntologyContext
from exact_inspect.contracts import EntityRef, canonical_hash, file_hash
from exact_inspect.decisions import DecisionStore
from exact_inspect.evidence_metadata import filter_evidence_items, filter_pair_metadata
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.models import Publish
from exact_inspect.study.resources import ExplanationResource
from exact_inspect.study.workspace import build_workspace_scope


def recover_literals(items, contexts):
    """Resolve only exact retained literal terms; do not reconstruct missing features."""
    result = []
    for original in items:
        item = dict(original)
        terms = item.get("semantic_terms", {})
        if (
            not item.get("fact_ids")
            and "value" in terms
            and (terms.get("property_iri") or terms.get("prop_iri"))
        ):
            resolution = contexts[item["entity"]["ontology_version_id"]].resolve_feature(
                terms, item["entity"]
            )
            if resolution.get("fact_ids"):
                item.update(
                    fact_ids=resolution["fact_ids"],
                    status="available",
                    origin_resolution="resolved",
                    provenance_status="derived_from_saved_artifacts",
                    reason=None,
                )
        result.append(item)
    return result


def clone_file(source, target):
    if sys.platform == "darwin":
        subprocess.run(["cp", "-c", str(source), str(target)], check=True)
        return target
    return shutil.copy2(source, target)


def refresh(source, private_index, destination, revision):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Use a new independent output directory")
    raw = json.loads((source / "publication.json").read_bytes())
    publication = Publish.model_validate(raw)
    if publication.definition.purpose != "development_review":
        raise ValueError("This preparation is restricted to development review")
    old_bundle = validate_bundle(source / "package/package.json")
    private = DecisionStore(private_index)
    public_locator = next(iter(old_bundle.runs.values()))
    public = DecisionStore(source / "package" / public_locator)
    for key in (
        "run_id",
        "dependency_inventory_hash",
        "source_ontology_version_id",
        "target_ontology_version_id",
    ):
        if private.manifest()[key] != public.manifest()[key]:
            raise ValueError("Private and public indexes do not describe the same recorded run")
    shutil.copytree(source, destination, copy_function=clone_file)
    package = destination / "package"
    contexts = {v: OntologyContext(package / p) for v, p in old_bundle.ontologies.items()}
    policy = old_bundle.policy
    definition = raw["definition"]
    before, after, catalogue = Counter(), Counter(), []
    public_database = package / public_locator / "decisions.sqlite"
    with sqlite3.connect(public_database) as output, private._connect() as saved:
        for case in definition["cases"]:
            evidence, metadata = {}, {}
            pair_receipts = []
            for candidate in case["candidates"]:
                entity = candidate["entity"]
                row = saved.execute(
                    "SELECT id,candidate,trace,evidence FROM pairs WHERE source=? AND source_kind=? AND target=? AND target_kind=?",
                    (case["source"]["iri"], case["source"]["kind"], entity["iri"], entity["kind"]),
                ).fetchone()
                if row is None:
                    raise ValueError("Selected candidate missing from retained full pool")
                pid, _, trace, items = row
                trace, items = json.loads(trace), json.loads(items)
                old = output.execute(
                    "SELECT candidate,evidence FROM pairs WHERE id=?", (pid,)
                ).fetchone()
                if old is None:
                    raise ValueError("Public and private pair identities differ")
                old_candidate, old_items = json.loads(old[0]), json.loads(old[1])
                if {i["evidence_id"] for i in items} != {i["evidence_id"] for i in old_items}:
                    raise ValueError("Selected evidence identity changed")
                safe = filter_evidence_items(
                    recover_literals(items, contexts), contexts, policy, physically_filtered=True
                )
                for counts, rows in ((before, old_items), (after, safe)):
                    counts.update(i.get("origin_resolution", "not_recorded") for i in rows)
                trace["evidence_metadata"] = filter_pair_metadata(trace["evidence_metadata"], safe)
                # Decision events and exact-route history stay byte-equivalent to
                # the policy-admitted public trace. Only evidence metadata changes.
                public_trace = json.loads(
                    output.execute("SELECT trace FROM pairs WHERE id=?", (pid,)).fetchone()[0]
                )
                public_trace["evidence_metadata"] = trace["evidence_metadata"]
                old_candidate["evidence_counts"] = {
                    "visible_selected": sum(i["status"] == "available" for i in safe),
                    "scope": "prepared_visibility_policy",
                }
                old_candidate["evidence_status"] = (
                    "available" if any(i["fact_ids"] for i in safe) else "not_exported"
                )
                output.execute(
                    "UPDATE pairs SET candidate=?,trace=?,evidence=? WHERE id=?",
                    (json.dumps(old_candidate), json.dumps(public_trace), json.dumps(safe), pid),
                )
                cid = candidate["candidate_id"]
                evidence[cid] = safe
                metadata[cid] = {
                    "source": public_trace["source"],
                    "target": public_trace["target"],
                    "evidence_metadata": public_trace["evidence_metadata"],
                }
                channels = {}
                for item in safe:
                    if item["fact_ids"]:
                        channels.setdefault(item["channel"], set()).add(item["side"])
                pair_receipts.append(
                    {
                        "candidate_id": cid,
                        "pair_id": pid,
                        "selected": len(safe),
                        "origin_resolution": dict(Counter(i["origin_resolution"] for i in safe)),
                        "graph_projection": dict(Counter(i["graph_projection"] for i in safe)),
                        "resolved_channel_sides": {k: sorted(v) for k, v in channels.items()},
                    }
                )
            old_scope = next(
                s
                for s in definition["workspace_scopes"]
                if s["kind"] == "case" and s["case_id"] == case["case_id"]
            )
            with sqlite3.connect(source / old_scope["index_path"]) as db:
                explanations = [
                    json.loads(r[0])
                    for r in db.execute("SELECT payload FROM explanations ORDER BY id")
                ]
            scope = build_workspace_scope(
                destination,
                scope_id=case["case_id"] + "-literal-v3",
                kind="case",
                case=case,
                contexts=contexts,
                policy=policy,
                explanations=explanations,
                evidence=evidence,
                pair_metadata=metadata,
                components=old_scope["components"],
                reuse_prepared_contexts=True,
            )
            definition["workspace_scopes"][definition["workspace_scopes"].index(old_scope)] = (
                scope.model_dump(mode="json")
            )
            asset = next(
                a for a in definition["assets"] if a["asset_id"] == case["explanation_refs"][0]
            )
            asset_path = destination / asset["path"]
            prior = ExplanationResource.model_validate_json(asset_path.read_bytes())
            added = build_explanation_resource(
                contexts,
                [
                    EntityRef.model_validate(e)
                    for e in [case["source"], *(c["entity"] for c in case["candidates"])]
                ],
                policy=policy,
                evidence=evidence,
            )
            facts = {
                (f.fact_id, canonical_hash(f.subject)): f for f in [*prior.facts, *added.facts]
            }
            updated = prior.model_copy(
                update={"facts": list(facts.values()), "evidence": added.evidence}
            )
            # Revalidate cross references and retain every original reviewed claim.
            ExplanationResource.model_validate(updated.model_dump(mode="json"))
            asset_path.write_text(updated.model_dump_json())
            asset.update(sha256=file_hash(asset_path)[7:], size_bytes=asset_path.stat().st_size)
            catalogue.append({"case_id": case["case_id"], "pairs": pair_receipts})
    binding = {
        "schema": "exact-review-literal-recovery/1",
        "source_publication": file_hash(source / "publication.json"),
        "private_database": private.manifest()["database_hash"],
        "implementation": "review-selected-original-facts/3",
        "contexts": {v: c.manifest["artifacts"]["context.sqlite"] for v, c in contexts.items()},
        "before": dict(before),
        "after": dict(after),
        "cases": catalogue,
    }
    manifest = public.manifest()
    manifest.update(
        revision=canonical_hash(
            {"previous": manifest["revision"], "recovery": canonical_hash(binding)}
        ),
        database_hash=file_hash(public_database),
        literal_recovery=binding,
    )
    atomic_json(package / public_locator / "decisions.manifest.json", manifest)
    bundle_metadata = old_bundle.model_dump(mode="json", exclude={"artifacts", "package_id"})
    bundle_metadata["provenance"]["literal_recovery"] = canonical_hash(binding)
    publish_bundle(package, **bundle_metadata)
    definition["study_revision"] = revision
    raw = Publish.model_validate(raw).model_dump(mode="json")
    atomic_json(destination / "publication.json", raw)
    binding["publication_sha256"] = file_hash(destination / "publication.json")
    binding["package_sha256"] = file_hash(package / "package.json")
    atomic_json(destination / "literal-recovery.json", binding)
    return binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("private_index", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    result = refresh(args.source, args.private_index, args.destination, args.revision)
    print(json.dumps({"before": result["before"], "after": result["after"]}))
