"""Publish reviewed Luna prose using an additive manifest in the existing package root.

Unchanged artifacts keep their original paths and inodes. The old manifest/publication
remain valid. Exporting the new manifest still creates an ordinary self-contained archive.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from uuid import uuid4

from exact_inspect.artifacts import Artifact, InspectionBundle, validate_bundle
from exact_inspect.contracts import EntityRef, canonical_hash, file_hash
from exact_inspect.description_review import admit_independent_review
from exact_inspect.descriptions import validate_prepared_support
from exact_inspect.generation import FactPacket
from exact_inspect.readable_descriptions import presentation
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.models import Publish
from exact_inspect.study.workspace import build_workspace_scope
from tools.review_environment import AdmissionHashes, verify_assets, verify_workspaces
from tools.review_luna_bundle import contexts_for, freeze, read, verify_base


def revised_manifest(previous, package, replacements, binding, *, prefix="luna-20261012"):
    """Explicit artifacts avoid directory scanning or rewriting an existing manifest."""
    import re

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", prefix):
        raise ValueError("Invalid immutable revision name")
    metadata = previous.model_dump(mode="json")
    artifacts = {a["path"]: a for a in metadata["artifacts"]}
    dependencies = metadata["provenance"].setdefault("description_dependencies", {})
    for old_id, result in replacements.items():
        if result["explanation_id"] == old_id:
            continue
        rel = prefix + "/explanations/" + result["explanation_id"][7:] + ".json"
        freeze(package / rel, result)
        artifacts[rel] = Artifact(
            path=rel, sha256=file_hash(package / rel), size=(package / rel).stat().st_size
        ).model_dump()
        metadata["explanations"].pop(old_id)
        metadata["explanations"][result["explanation_id"]] = rel
        dependencies[result["explanation_id"]] = rel
    metadata["artifacts"] = [artifacts[k] for k in sorted(artifacts)]
    metadata["provenance"]["luna_prose_revision"] = binding
    metadata["package_id"] = canonical_hash(
        {k: v for k, v in metadata.items() if k != "package_id"}
    )
    return InspectionBundle.model_validate(metadata)


def case_scope(base, *, scope_id, **kwargs):
    """Install only a private small index; an interrupted identical install can resume."""
    staging_id = scope_id + ":staging:" + uuid4().hex
    staging_dir = base / "workspaces" / hashlib.sha256(staging_id.encode()).hexdigest()
    staging_index = staging_dir / "workspace.sqlite"
    target = (
        base / "workspaces" / hashlib.sha256(scope_id.encode()).hexdigest() / "workspace.sqlite"
    )
    try:
        scope = build_workspace_scope(
            base, scope_id=staging_id, reuse_prepared_contexts=True, **kwargs
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_hash(target) != "sha256:" + scope.index_sha256:
                raise ValueError("Interrupted case index differs; retained for inspection")
        else:
            staging_index.rename(target)
        return scope.model_copy(
            update={"scope_id": scope_id, "index_path": target.relative_to(base).as_posix()}
        )
    finally:
        # Only this invocation's newly allocated index can be removed here.
        staging_index.unlink(missing_ok=True)
        if staging_dir.exists():
            staging_dir.rmdir()


def publish(root):
    root = root.resolve()
    plan = read(root / "plan.json")
    complete = read(root / "generation-complete.json")
    if complete["plan_sha256"] != canonical_hash(plan):
        raise ValueError("Generation is incomplete or belongs to another plan")
    verify_base(plan)
    base = Path(plan["base"])
    package = base / "package"
    previous = validate_bundle(package / "package.json", verify_hashes=False)
    contexts = contexts_for(base, previous)
    selected, lineage, decisions = {}, {}, []
    for job in plan["jobs"]:
        old = job["old_id"]
        prior = read(package / previous.explanations[old])
        directory = root / "generation" / job["job_id"][7:]
        request_path = root / "review-requests" / (job["job_id"][7:] + ".json")
        review_path = root / "reviews" / (job["job_id"][7:] + ".json")
        if request_path.exists():
            if not review_path.exists():
                raise ValueError("Every reviewable draft needs a separate complete review")
            review = read(review_path)
            # Even an all-rejected receipt must bind every input and ordered claim.
            from exact_inspect.description_review import IndependentDescriptionReview

            review = IndependentDescriptionReview.model_validate(review).model_dump(mode="json")
            request = read(request_path)
            if any(review[k] != v for k, v in request["bindings"].items()):
                raise ValueError("Review does not bind this exact draft")
            if len(review["decisions"]) != len(request["claim_inventory"]) or any(
                d["claim_id"] != c["claim_id"] or d["claim_hash"] != c["claim_hash"]
                for d, c in zip(review["decisions"], request["claim_inventory"])
            ):
                raise ValueError("Review does not cover the exact candidate inventory")
            decisions.extend(review["decisions"])
            if any(d["decision"] == "approved" for d in review["decisions"]):
                result = admit_independent_review(
                    read(directory / "source.json"),
                    read(directory / "response.json")["response"],
                    review,
                )
                validate_prepared_support(result, contexts, previous.policy)
            else:
                result = prior
            reason = (
                "Separately reviewed Luna claims"
                if result is not prior
                else "No Luna claim approved; predecessor retained"
            )
            review_hash = canonical_hash(review)
        elif (directory / "abstained.json").exists() or (directory / "unreviewable.json").exists():
            result, reason, review_hash = (
                prior,
                "No reviewable Luna claims; predecessor retained",
                None,
            )
        else:
            raise ValueError("Draft has no complete preparation disposition")
        selected[old] = result
        lineage[old] = {
            "replacement": result["explanation_id"],
            "reason": reason,
            "luna_job": job["job_id"],
            "review_sha256": review_hash,
            "previous_support": prior["support_status"],
            "previous_claim_ids": [c["claim_id"] for c in prior["claims"]],
            "selected_claim_ids": [c["claim_id"] for c in result["claims"]],
        }
    binding = {
        "schema": "exact-reviewed-prose-overlay/1",
        "predecessor_package": plan["package_sha256"],
        "predecessor_publication": plan["publication_sha256"],
        "predecessor_composition": plan["composition_sha256"],
        "plan": canonical_hash(plan),
        "supersessions": canonical_hash(lineage),
    }
    receipt_path = root / "publication-receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["binding"] != binding:
            raise ValueError("Published review revision cannot be overwritten")
        for path, digest in receipt["files"].items():
            if file_hash(base / path) != digest:
                raise ValueError("Published revision changed")
        return receipt
    raw = copy.deepcopy(read(base / "publication.json"))
    raw["definition"]["study_revision"] = "luna-prose-20261012"
    catalogue = copy.deepcopy(read(base / "catalogue.json"))
    catalogue_cases = {c["case_id"]: c for c in catalogue["cases"]}
    scopes = {s["case_id"]: s for s in raw["definition"]["workspace_scopes"] if s["kind"] == "case"}
    raw["definition"]["workspace_scopes"] = [
        s for s in raw["definition"]["workspace_scopes"] if s["kind"] != "case"
    ]
    written = {}
    for case in raw["definition"]["cases"]:
        oldscope = scopes[case["case_id"]]
        with sqlite3.connect((base / oldscope["index_path"]).as_uri() + "?mode=ro", uri=True) as db:
            explanations = [
                selected[row[0]] for row in db.execute("SELECT id FROM explanations ORDER BY id")
            ]
            evidence = {}
            for cid, text in db.execute(
                "SELECT candidate_id,payload FROM evidence ORDER BY candidate_id,id"
            ):
                evidence.setdefault(cid, []).append(json.loads(text))
            pair_metadata = {
                cid: json.loads(text)
                for cid, text in db.execute("SELECT candidate_id,payload FROM pair_metadata")
            }
        checkpoint = root / "assembly" / f"{case['case_id']}.json"
        asset = next(
            a for a in raw["definition"]["assets"] if a["asset_id"] == case["explanation_refs"][0]
        )
        if checkpoint.exists():
            saved = read(checkpoint)
            if saved["binding"] != canonical_hash(binding):
                raise ValueError("Case assembly resume belongs to another revision")
            for path, digest in saved["files"].items():
                if file_hash(base / path) != digest:
                    raise ValueError("Completed case changed before resume")
            asset.update(saved["asset"])
            scope = saved["scope"]
        else:
            resource = build_explanation_resource(
                contexts,
                [
                    EntityRef.model_validate(e)
                    for e in [case["source"], *(c["entity"] for c in case["candidates"])]
                ],
                policy=previous.policy,
                packets=[FactPacket.model_validate(x["support_packet"]) for x in explanations],
                explanations=explanations,
                evidence=evidence,
                pair_metadata=pair_metadata,
            )
            asset["path"] = "luna-20261012/resources/" + case["case_id"] + ".json"
            freeze(base / asset["path"], resource.model_dump(mode="json"))
            asset.update(
                sha256=file_hash(base / asset["path"])[7:],
                size_bytes=(base / asset["path"]).stat().st_size,
            )
            scope = case_scope(
                base,
                scope_id="luna-20261012-" + case["case_id"],
                kind="case",
                case=case,
                contexts=contexts,
                policy=previous.policy,
                explanations=explanations,
                evidence=evidence,
                pair_metadata=pair_metadata,
                components=oldscope["components"],
            ).model_dump(mode="json")
            freeze(
                checkpoint,
                {
                    "binding": canonical_hash(binding),
                    "asset": asset,
                    "scope": scope,
                    "files": {
                        asset["path"]: "sha256:" + asset["sha256"],
                        scope["index_path"]: "sha256:" + scope["index_sha256"],
                    },
                },
            )
        written.update(
            {
                asset["path"]: "sha256:" + asset["sha256"],
                scope["index_path"]: "sha256:" + scope["index_sha256"],
            }
        )
        raw["definition"]["workspace_scopes"].append(scope)
        catalogue_cases[case["case_id"]]["descriptions"] = [
            {
                "id": x["explanation_id"],
                "kind": x["preparation_kind"],
                "status": x["manifest"]["status"],
                "support_status": x["support_status"],
                "presentation": presentation(x),
            }
            for x in explanations
        ]
        print("case frozen", case["case_id"], flush=True)
    summary = {
        "generation_mode": "luna-reviewed-overlay",
        "provider_jobs": len(plan["jobs"]),
        "model": "openai/gpt-6-luna",
        "provider_jobs_by_task": dict(
            Counter(j["identity"]["packet"]["task"] for j in plan["jobs"])
        ),
        "served_artifacts": len(selected),
        "luna_reviewed_artifacts": sum(v["replacement"] != k for k, v in lineage.items()),
        "retained_predecessor_artifacts": sum(v["replacement"] == k for k, v in lineage.items()),
        "independently_reviewed_artifacts": sum(
            x["support_status"] == "independently_reviewed" for x in selected.values()
        ),
        "deterministic_artifacts": sum(
            x["support_status"] == "typed_template_support" for x in selected.values()
        ),
        "factual_fallback_artifacts": sum(
            x["support_status"] == "fallback" for x in selected.values()
        ),
        "reviewed_draft_claims": len(decisions),
        "approved_luna_claims": sum(d["decision"] == "approved" for d in decisions),
        "rejected_luna_claims": sum(d["decision"] == "rejected" for d in decisions),
    }
    summary["provider_batch_status"] = (
        "independently_reviewed_prose_available"
        if summary["luna_reviewed_artifacts"]
        else "predecessor_retained"
    )
    about = raw["definition"].get("review_bundle")
    if about:
        about["prepared_at"] = plan["created_utc"]
        about.setdefault("delivered_layers", {})["luna_reviewed_prose"] = canonical_hash(binding)
        about["limitations"].append(
            "Luna prose was reviewed by a separate AI agent; it has not received a human clinical audit. "
            "Rejected or unusable drafts retain the preceding approved description or typed fallback."
        )
    manifest = revised_manifest(previous, package, selected, binding)
    if about:
        manifest.provenance["review_bundle"] = about
        manifest.provenance["delivered_layers"] = about["delivered_layers"]
    manifest.provenance["predecessor_generation_summary"] = previous.provenance.get(
        "generation_summary"
    )
    catalogue["predecessor_generation"] = {
        k: catalogue.get(k)
        for k in [
            "generation_attempt",
            "generation_mode",
            "provider_batch_status",
            "generation_summary",
        ]
    }
    catalogue.pop("generation_attempt", None)
    catalogue.update(
        generation_mode=summary["generation_mode"],
        provider_batch_status=summary["provider_batch_status"],
    )
    manifest.provenance["generation_summary"] = summary
    metadata = manifest.model_dump(mode="json")
    metadata["package_id"] = canonical_hash(
        {k: v for k, v in metadata.items() if k != "package_id"}
    )
    manifest = InspectionBundle.model_validate(metadata)
    manifest_path = package / "package-luna-20261012.json"
    freeze(manifest_path, manifest.model_dump(mode="json"))
    validate_bundle(manifest_path)
    raw = Publish.model_validate(raw).model_dump(mode="json")
    catalogue.update(
        publication_sha256=canonical_hash(raw),
        generation_summary=summary,
        luna_prose_revision=binding,
        luna_description_replacements=lineage,
    )
    hasher = AdmissionHashes()
    verify_assets(raw, base, hasher=hasher)
    verify_workspaces(raw, base, hasher=hasher)
    freeze(base / "catalogue-luna-20261012.json", catalogue)
    freeze(base / "publication-luna-20261012.json", raw)
    written.update(
        {
            str(p.relative_to(base)): file_hash(p)
            for p in [
                manifest_path,
                base / "catalogue-luna-20261012.json",
                base / "publication-luna-20261012.json",
            ]
        }
    )
    verify_base(plan)
    receipt = {
        "schema": "exact-luna-publication/1",
        "binding": binding,
        "files": written,
        "package_id": manifest.package_id,
        "summary": summary,
        "supersessions": lineage,
        "predecessor_unchanged": True,
        "existing_sessions_modified": False,
        "large_files_copied_or_relinked": 0,
        "artifact_root": str(base),
        "publication_file": "publication-luna-20261012.json",
        "package_manifest": "package/package-luna-20261012.json",
        "portable_export_supported": True,
    }
    freeze(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(publish(parser.parse_args().root)["summary"], sort_keys=True))
