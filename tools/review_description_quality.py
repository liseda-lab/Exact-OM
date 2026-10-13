"""Audit retained prose and stage targeted original-definition supplements offline.

Never edits an approved claim or calls a provider. Newly proposed claims remain
unserved until separately reviewed through description_review and published.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from exact_inspect.contracts import canonical_hash, file_hash
from exact_inspect.description_review import build_review_request
from exact_inspect.generation import FactPacket
from tools.review_luna_bundle import freeze, read, source_result

REVISION = "original-definition-supplement/1"


def audit(package, output):
    package, output = Path(package), Path(output)
    manifest = read(package)
    rows = []
    for old_id, location in sorted(manifest["explanations"].items()):
        result = read(package.parent / location)
        if result["task"] != "entity_profile":
            continue
        packet = FactPacket.model_validate(result["support_packet"])
        facts = {f["fact_id"]: f for f in packet.facts}
        cited = {fid for c in result["claims"] for fid in c["fact_ids"]}
        categories = {facts[fid]["category"] for fid in cited if fid in facts}
        definitions = [
            f
            for f in packet.facts
            if f["category"] == "definitions"
            and f["value"].get("term_type") == "literal"
            and f["value"].get("lexical_form")
        ]
        weak = bool(result["claims"]) and categories <= {"labels", "synonyms", "hierarchy"}
        row = {
            "explanation_id": old_id,
            "entity": result["entities"][0],
            "claims": result["claims"],
            "cited_categories": sorted(categories),
            "definition_fact_ids": [f["fact_id"] for f in definitions],
            "finding": (
                (
                    "preparation_omitted_available_definition"
                    if definitions
                    else "limited_readable_source"
                )
                if weak
                else "no_low_information_flag"
            ),
            "disposition": "preserve_approved_prose",
        }
        if weak and definitions and result.get("independent_review"):
            # Keep every approved clause and citation. Add one explicitly marked full
            # original definition, never a truncation, clinical paraphrase or model rerun.
            definition = definitions[0]
            text = "Original definition: “" + definition["value"]["lexical_form"] + "”"
            if len(text) > 4000:
                row["disposition"] = (
                    "definition_exceeds_claim_bound; original fact remains available"
                )
            else:
                claims = [
                    {k: c[k] for k in ("text", "fact_ids", "category")} for c in result["claims"]
                ]
                claims.append(
                    {"text": text, "fact_ids": [definition["fact_id"]], "category": "meaning"}
                )
                response = {
                    "model": "offline/" + REVISION,
                    "provider": "local deterministic preparation",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"claims": claims, "limitations": [], "relation": None}
                                )
                            }
                        }
                    ],
                }
                identity = {
                    "revision": REVISION,
                    "predecessor": old_id,
                    "predecessor_hash": file_hash(package.parent / location),
                    "packet": packet.model_dump(mode="json"),
                    "definition": definition["fact_id"],
                }
                job = {
                    "job_id": canonical_hash(identity),
                    "identity": {**identity, "payload": {"messages": [{"content": REVISION}]}},
                }
                source = source_result(job, response)
                source["manifest"]["requested_model"] = "offline/" + REVISION
                source["fixture_provenance"] = (
                    "Deterministic original-definition supplement to retained reviewed prose; awaiting separate review"
                )
                request = build_review_request(source, response)
                folder = output / "drafts" / job["job_id"][7:]
                for name, data in [
                    ("identity", identity),
                    ("source", source),
                    ("response", response),
                    ("review-request", request),
                ]:
                    freeze(folder / (name + ".json"), data)
                row.update(
                    disposition="draft_awaiting_independent_review",
                    draft_id=job["job_id"],
                    proposed_definition=definition["value"]["lexical_form"],
                )
        rows.append(row)
    report = {
        "schema": "exact-description-quality-audit/1",
        "revision": REVISION,
        "package": str(package),
        "package_sha256": file_hash(package),
        "profiles": rows,
        "provider_calls": 0,
        "replacement_policy": "preserve all approved clauses; supplement only separately reviewed full original definitions",
    }
    freeze(output / "audit.json", report)
    return report


def publish(package, publication, catalogue, root, revision):
    """New small revision; every changed claim needs an external review receipt."""
    import sqlite3

    from exact_inspect.artifacts import InspectionBundle, validate_bundle
    from exact_inspect.contracts import EntityRef
    from exact_inspect.description_review import admit_independent_review
    from exact_inspect.descriptions import validate_prepared_support
    from exact_inspect.readable_descriptions import presentation
    from exact_inspect.study.builder import build_explanation_resource
    from tools.publish_luna_review import case_scope, revised_manifest
    from tools.review_environment import (
        AdmissionHashes,
        verify_assets,
        verify_workspaces,
    )
    from tools.review_luna_bundle import contexts_for

    package, publication, catalogue, root = map(Path, (package, publication, catalogue, root))
    base = publication.parent
    old = validate_bundle(package, verify_hashes=False)
    audit = read(root / "audit.json")
    if file_hash(package) != audit["package_sha256"]:
        raise ValueError("Audited predecessor changed")
    contexts = contexts_for(base, old)
    selected = {eid: read(package.parent / loc) for eid, loc in old.explanations.items()}
    replacements = {}
    for row in audit["profiles"]:
        if not row.get("draft_id"):
            continue
        folder = root / "drafts" / row["draft_id"][7:]
        result = admit_independent_review(
            read(folder / "source.json"),
            read(folder / "response.json"),
            read(folder / "review.json"),
        )
        validate_prepared_support(result, contexts, old.policy)
        prior = selected[row["explanation_id"]]
        clauses = lambda r: [
            {k: c[k] for k in ("text", "fact_ids", "category")} for c in r["claims"]
        ]
        if clauses(result)[: len(prior["claims"])] != clauses(prior):
            raise ValueError("Definition supplement lost approved prose")
        selected[row["explanation_id"]] = result
        replacements[row["explanation_id"]] = result
    binding = {
        "schema": "exact-description-supplement-publication/1",
        "revision": revision,
        "predecessor_package": file_hash(package),
        "predecessor_publication": file_hash(publication),
        "audit": file_hash(root / "audit.json"),
        "reviews": {k: canonical_hash(v["independent_review"]) for k, v in replacements.items()},
    }
    raw = copy.deepcopy(read(publication))
    cat = copy.deepcopy(read(catalogue))
    files = {}
    raw["definition"]["study_revision"] = revision
    scopes = {s["case_id"]: s for s in raw["definition"]["workspace_scopes"] if s["kind"] == "case"}
    output_scopes = [s for s in raw["definition"]["workspace_scopes"] if s["kind"] != "case"]
    cat_cases = {c["case_id"]: c for c in cat["cases"]}
    for case in raw["definition"]["cases"]:
        oldscope = scopes[case["case_id"]]
        with sqlite3.connect((base / oldscope["index_path"]).as_uri() + "?mode=ro", uri=True) as db:
            ids = [row[0] for row in db.execute("SELECT id FROM explanations ORDER BY id")]
            if not any(i in replacements for i in ids):
                output_scopes.append(oldscope)
                continue
            explanations = [selected[i] for i in ids]
            evidence = {}
            for cid, payload in db.execute(
                "SELECT candidate_id,payload FROM evidence ORDER BY candidate_id,id"
            ):
                evidence.setdefault(cid, []).append(json.loads(payload))
            metadata = {
                cid: json.loads(v)
                for cid, v in db.execute("SELECT candidate_id,payload FROM pair_metadata")
            }
        resource = build_explanation_resource(
            contexts,
            [
                EntityRef.model_validate(e)
                for e in [case["source"], *(c["entity"] for c in case["candidates"])]
            ],
            policy=old.policy,
            packets=[FactPacket.model_validate(x["support_packet"]) for x in explanations],
            explanations=explanations,
            evidence=evidence,
            pair_metadata=metadata,
        )
        asset = next(
            a for a in raw["definition"]["assets"] if a["asset_id"] == case["explanation_refs"][0]
        )
        asset["path"] = revision + "/resources/" + case["case_id"] + ".json"
        freeze(base / asset["path"], resource.model_dump(mode="json"))
        asset.update(
            sha256=file_hash(base / asset["path"])[7:],
            size_bytes=(base / asset["path"]).stat().st_size,
        )
        scope = case_scope(
            base,
            scope_id=revision + "-" + case["case_id"],
            kind="case",
            case=case,
            contexts=contexts,
            policy=old.policy,
            explanations=explanations,
            evidence=evidence,
            pair_metadata=metadata,
            components=oldscope["components"],
        ).model_dump(mode="json")
        output_scopes.append(scope)
        files.update(
            {
                asset["path"]: "sha256:" + asset["sha256"],
                scope["index_path"]: "sha256:" + scope["index_sha256"],
            }
        )
        cat_cases[case["case_id"]]["descriptions"] = [
            {
                "id": x["explanation_id"],
                "kind": x["preparation_kind"],
                "status": x["manifest"]["status"],
                "support_status": x["support_status"],
                "presentation": presentation(x),
            }
            for x in explanations
        ]
    raw["definition"]["workspace_scopes"] = output_scopes
    about = raw["definition"]["review_bundle"]
    about.setdefault("delivered_layers", {})["definition_supplements"] = canonical_hash(binding)
    about["limitations"].append(
        "Selected low-information profiles retain their reviewed prose plus separately reviewed, explicitly quoted original definitions. This is local preparation, not another Luna generation batch."
    )
    manifest = revised_manifest(old, package.parent, replacements, binding, prefix=revision)
    data = manifest.model_dump(mode="json")
    data["provenance"]["review_bundle"] = about
    data["provenance"]["definition_supplements"] = binding
    # Retain the original Luna lineage under its original name.
    data["provenance"]["luna_prose_revision"] = old.provenance["luna_prose_revision"]
    data["package_id"] = canonical_hash({k: v for k, v in data.items() if k != "package_id"})
    manifest = InspectionBundle.model_validate(data)
    output_package = package.parent / ("package-" + revision + ".json")
    freeze(output_package, manifest.model_dump(mode="json"))
    validate_bundle(output_package)
    hasher = AdmissionHashes()
    verify_assets(raw, base, hasher=hasher)
    verify_workspaces(raw, base, hasher=hasher)
    for dest, value in [
        (base / ("publication-" + revision + ".json"), raw),
        (base / ("catalogue-" + revision + ".json"), cat),
    ]:
        freeze(dest, value)
        files[str(dest.relative_to(base))] = file_hash(dest)
    files[str(output_package.relative_to(base))] = file_hash(output_package)
    receipt = {
        "binding": binding,
        "files": files,
        "changed_profiles": len(replacements),
        "provider_calls": 0,
        "large_files_copied_or_relinked": 0,
        "package_id": manifest.package_id,
        "replacements": {k: v["explanation_id"] for k, v in replacements.items()},
    }
    freeze(root / "publication-receipt.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = audit(a.package, a.output)
    from collections import Counter

    print(json.dumps(dict(Counter(r["finding"] for r in report["profiles"])), indent=2))


if __name__ == "__main__":
    main()
