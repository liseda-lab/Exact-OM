"""Audit an additive Luna publication and exercise its real HTTPS/PostgreSQL surfaces.

Run with the shared storage guard. This makes development-review sessions only.
The API journey is separate evidence from the rendered frontend journey.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from exact_inspect.contracts import canonical_hash, file_hash
from exact_inspect.description_review import validate_reviewed_artifact
from tools import review_environment as environment
from tools.review_luna_bundle import read, verify_base
from tools.verify_composed_review import CombinedJourney
from tools.verify_review_environment import require


def static_audit(root):
    plan = read(root / "plan.json")
    delivery = read(root / "publication-receipt.json")
    base = Path(plan["base"])
    verify_base(plan)
    old = read(base / "package/package.json")
    new = read(base / delivery["package_manifest"])
    for key in ("ontologies", "runs", "policy"):
        require(old[key] == new[key], "unchanged_" + key)
    old_artifacts = {a["path"]: a for a in old["artifacts"]}
    new_artifacts = {a["path"]: a for a in new["artifacts"]}
    require(
        all(new_artifacts[k] == v for k, v in old_artifacts.items()),
        "all_old_artifact_locators_hashes_retained",
    )
    require(
        len(new["explanations"]) == len(old["explanations"]) == 260, "active_artifact_inventory"
    )
    require(
        set(new["explanations"]) == {v["replacement"] for v in delivery["supersessions"].values()},
        "selected_ids_match_lineage",
    )
    statuses, tasks, reviewed_claims, retained = Counter(), Counter(), 0, []
    old_claims = set()
    for old_id, lineage in delivery["supersessions"].items():
        original = read(base / "package" / old["explanations"][old_id])
        current = read(base / "package" / new["explanations"][lineage["replacement"]])
        require(current["entities"] == original["entities"], "unchanged_typed_description_identity")
        require(
            current["support_packet"]["task"] == original["support_packet"]["task"],
            "unchanged_description_task",
        )
        old_claims.update(
            c["claim_id"]
            for c in original["claims"]
            if original["support_status"] == "independently_reviewed"
        )
        statuses[current["support_status"]] += 1
        tasks[current["support_packet"]["task"]] += 1
        require(
            lineage["selected_claim_ids"] == [c["claim_id"] for c in current["claims"]],
            "selected_claim_inventory",
        )
        if current["support_status"] == "independently_reviewed":
            receipt = validate_reviewed_artifact(current)
            reviewed_claims += len(current["claims"])
            require(
                {d.claim_id for d in receipt.decisions if d.decision == "rejected"}.isdisjoint(
                    c["claim_id"] for c in current["claims"]
                ),
                "rejected_claims_not_served",
            )
        if old_id == lineage["replacement"]:
            require(current == original, "exact_predecessor_fallback")
            retained.append(old_id)
        else:
            require(
                current["independent_review"]["source_explanation_id"] == lineage["luna_job"],
                "review_bound_to_luna_job",
            )
    before = read(base / "publication.json")["definition"]
    after = read(base / delivery["publication_file"])["definition"]
    a, b = copy.deepcopy(before), copy.deepcopy(after)
    for k in ("study_revision", "assets", "workspace_scopes", "review_bundle"):
        a.pop(k, None)
        b.pop(k, None)
    require(a == b, "unchanged_study_protocol_cases_grading_schedules")
    previous_assets = {a["asset_id"]: a for a in before["assets"]}
    current_assets = {a["asset_id"]: a for a in after["assets"]}
    old_scopes = {s["case_id"]: s for s in before["workspace_scopes"] if s["kind"] == "case"}
    new_scopes = {s["case_id"]: s for s in after["workspace_scopes"] if s["kind"] == "case"}
    counts = Counter()
    for case in after["cases"]:
        asset_id = case["explanation_refs"][0]
        previous_resource = read(base / previous_assets[asset_id]["path"])
        resource = read(base / current_assets[asset_id]["path"])
        for key in ("entities", "pair_metadata", "hierarchy", "policy_hash"):
            require(resource[key] == previous_resource[key], "frozen_preserved_" + key)

        # Frozen evidence is identified by ID; scores are named records. The
        # builder may reorder those two collections when reading a SQLite scope.
        # Compare every value after only these order normalizations.
        def evidence_by_id(resource):
            evidence = copy.deepcopy(resource["evidence"])
            require(
                len({(e["candidate_id"], e["evidence_id"]) for e in evidence}) == len(evidence),
                "unique_candidate_evidence_ids",
            )
            for item in evidence:
                item["scores"] = sorted(item["scores"], key=canonical_hash)
            return {(e["candidate_id"], e["evidence_id"]): e for e in evidence}

        require(
            evidence_by_id(resource) == evidence_by_id(previous_resource),
            "frozen_all_evidence_values_preserved",
        )
        counts["resources_with_evidence_order_only_changes"] += (
            resource["evidence"] != previous_resource["evidence"]
        )
        old_facts = {
            (f["fact_id"], canonical_hash(f["subject"])): f for f in previous_resource["facts"]
        }
        new_facts = {(f["fact_id"], canonical_hash(f["subject"])): f for f in resource["facts"]}
        require(
            all(new_facts.get(k) == v for k, v in old_facts.items()),
            "original_frozen_facts_preserved",
        )
        previous_scope, scope = old_scopes[case["case_id"]], new_scopes[case["case_id"]]
        require(
            previous_scope["contexts"] == scope["contexts"]
            and previous_scope["components"] == scope["components"],
            "same_scope_contexts_components",
        )
        with sqlite3.connect(
            (base / previous_scope["index_path"]).as_uri() + "?mode=ro", uri=True
        ) as db1, sqlite3.connect(
            (base / scope["index_path"]).as_uri() + "?mode=ro", uri=True
        ) as db2:
            for table in ("evidence", "pair_metadata"):
                require(
                    list(db1.execute("SELECT * FROM " + table + " ORDER BY 1,2"))
                    == list(db2.execute("SELECT * FROM " + table + " ORDER BY 1,2")),
                    "scope_" + table + "_unchanged",
                )
            previous_ids = {r[0] for r in db1.execute("SELECT id FROM explanations")}
            ids = {r[0] for r in db2.execute("SELECT id FROM explanations")}
            require(
                ids == {delivery["supersessions"][i]["replacement"] for i in previous_ids},
                "scope_exact_prose_replacements",
            )
        counts["cases"] += 1
        counts["pairs"] += len(case["candidates"])
        counts["selected_records"] += len(resource["evidence"])
    require(counts["cases"] == 24 and counts["pairs"] == 120, "all_real_cases_checked")
    require(len(old_claims) == 68, "all_previous_approved_claims_retained_in_historical_artifacts")
    return {
        "status": "passed",
        "checks": dict(counts),
        "active_support": dict(statuses),
        "tasks": dict(tasks),
        "active_reviewed_claims": reviewed_claims,
        "retained_predecessor_ids": retained,
        "historical_approved_claims_preserved": len(old_claims),
        "old_artifacts_retained": len(old_artifacts),
        "new_artifacts": len(new_artifacts) - len(old_artifacts),
        "old_package": file_hash(base / "package/package.json"),
        "new_package": file_hash(base / delivery["package_manifest"]),
        "new_publication": file_hash(base / delivery["publication_file"]),
        "protected_predecessor_identity_unchanged": True,
    }


class LunaJourney(CombinedJourney):
    def __init__(self, config, *, receipt, combined, shared):
        super().__init__(
            config, receipt=receipt, combined=combined, evidence=combined, shared=shared
        )
        self.bundle = read(Path(config["package"]))

    def pair_metadata(self, prefix, case, capabilities):
        prior_ids = set(self.shared["descriptions"])
        super().pair_metadata(prefix, case, capabilities)
        for artifact_id in sorted(self.shared["descriptions"] - prior_ids):
            artifact = read(
                Path(self.config["package"]).parent / self.bundle["explanations"][artifact_id]
            )
            main = self.request(
                "GET",
                self.main + "/explanations/" + quote(artifact_id, safe=""),
                label="main_exact_selected_artifact",
            ).json()
            for key in ("claims", "support_status", "entities", "independent_review"):
                require(main.get(key) == artifact.get(key), "main_exact_prepared_" + key)
            packet = artifact["support_packet"]
            facts = {
                f["fact_id"]: f
                for f in packet["facts"] + packet["selection"].get("referenced_labels", [])
            }
            for claim in artifact["claims"]:
                for fact_id in claim["fact_ids"]:
                    f = facts[fact_id]
                    key = (prefix, canonical_hash(f["subject"]), fact_id)
                    if key in self.shared["citations"]:
                        continue
                    actual = self.request(
                        "GET",
                        prefix + "/facts/" + quote(fact_id, safe=""),
                        params=f["subject"],
                        label="exact_claim_citation_resolves",
                    ).json()
                    for field in ("fact_id", "subject", "category"):
                        require(actual[field] == f[field], "cited_original_" + field)
                    # Packets inline the original expression AST; the public Fact
                    # contract exposes its reference and has a separate axiom route.
                    value = dict(f["value"])
                    ast = value.pop("ast", None)
                    require(actual["value"] == value, "cited_original_value")
                    if ast is not None:
                        axiom = self.request(
                            "GET",
                            prefix + "/axioms/" + quote(f["axiom_ref"], safe=""),
                            params=f["subject"],
                            label="cited_expression_original_ast",
                        ).json()
                        require(axiom["ast"] == ast, "cited_original_expression_ast")
                        require(axiom["fact_id"] == fact_id, "cited_original_expression_identity")
                    self.shared["citations"].add(key)


def run(root, config, output, prior_static):
    output.mkdir(parents=True, exist_ok=False)
    baseline = read(prior_static)
    plan = read(root / "plan.json")
    require(
        baseline["publication_sha256"] == plan["publication_sha256"],
        "prior_taxonomy_receipt_publication_binding",
    )
    result = {
        "schema": "exact-luna-delivery-verification/1",
        "status": "running",
        "boundary": "Real-data preservation and certificate-verified HTTPS/PostgreSQL API; browser evidence is separate.",
        "verifier_sha256": file_hash(__file__),
        "static": static_audit(root),
        "reused_taxonomy_audit": {
            "path": str(prior_static),
            "sha256": file_hash(prior_static),
            "reason": "All old ontology/partial-navigation artifact hashes and file identities are unchanged.",
        },
    }
    environment.private_json(output / "static.json", result["static"])
    shared = {
        "pairs": set(),
        "cases": set(),
        "descriptions": set(),
        "citations": set(),
        "hierarchy": [],
        "basis_examples": [],
        "taxonomy": baseline["taxonomy"],
    }
    try:
        for index in range(4):
            journey = LunaJourney(
                config,
                receipt=output / f"journey-{index+1}.json",
                combined=Path(plan["base"]),
                shared=shared,
            )
            try:
                print("API journey", index + 1, "tutorial", flush=True)
                journey.prepare()
                journey.tutorial()
                current = journey.request(
                    "GET", "/api/v1/study/cases/current", label="allocated_schedule_probe"
                ).json()
                schedules = [
                    s
                    for s in journey.definition["schedules"]
                    if s["blocks"][0]["condition"] == current["condition"]
                    and current["case_id"] in s["blocks"][0]["case_ids"]
                ]
                require(len(schedules) == 1, "known_server_allocated_schedule")
                explanation_cases = set(
                    next(
                        b["case_ids"]
                        for b in schedules[0]["blocks"]
                        if b["condition"] == "explanation"
                    )
                )
                if explanation_cases <= shared["cases"]:
                    journey.report.update(
                        status="allocation_only",
                        limitation="Duplicate counterbalance test session retained at first case.",
                    )
                else:
                    print("API journey", index + 1, "24 cases", flush=True)
                    journey.cases()
                    journey.exports()
                    journey.report["status"] = "passed"
                with journey.study_log.open("rb") as stream:
                    stream.seek(journey.log_offset)
                    log = stream.read()
                counts = {
                    name: log.count(token)
                    for name, token in {
                        "content_length_protocol_errors": b"Too little data for declared Content-Length",
                        "protocol_errors": b"LocalProtocolError",
                        "asgi_exceptions": b"Exception in ASGI application",
                    }.items()
                }
                require(not any(counts.values()), "no_new_protocol_errors")
                journey.report["service_log_observation"] = counts
                journey.record()
            except Exception as exc:
                journey.report.update(
                    status="failed", failure=str(exc), failed_check=journey.checkpoint
                )
                journey.record()
                raise
            finally:
                journey.client.close()
            if len(shared["pairs"]) == 120:
                break
        require(
            len(shared["pairs"]) == 120 and len(shared["cases"]) == 24,
            "all_pairs_across_both_counterbalances",
        )
        require(len(shared["descriptions"]) == 260, "all_active_descriptions_served")
        verify_base(plan)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", failure=str(exc))
        raise
    finally:
        result.update(
            pairs=len(shared["pairs"]),
            cases=len(shared["cases"]),
            descriptions=len(shared["descriptions"]),
            scoped_citations=len(shared["citations"]),
            hierarchy=shared["hierarchy"],
            basis_examples=shared["basis_examples"],
        )
        environment.private_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-static", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, environment.load(args.environment), args.output, args.prior_static)
