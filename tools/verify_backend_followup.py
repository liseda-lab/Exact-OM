"""Preservation and actual-service verification of the 13 October backend correction."""

from __future__ import annotations
import argparse
import copy
import json
import sqlite3
from collections import Counter
from pathlib import Path

from exact_inspect.contracts import canonical_hash, file_hash
from tools import review_environment as environment
from tools.review_luna_bundle import read, verify_base
from tools.verify_luna_publication import LunaJourney
from tools.verify_review_environment import require


def preservation(root, base):
    receipt = read(root / "quality/publication-receipt.json")
    revision = receipt["binding"]["revision"]
    old = read(base / "package/package-luna-20261012.json")
    new = read(base / "package" / ("package-" + revision + ".json"))
    before = read(base / "publication-luna-20261012.json")["definition"]
    after = read(base / ("publication-" + revision + ".json"))["definition"]
    require(
        file_hash(base / "package/package-luna-20261012.json")
        == receipt["binding"]["predecessor_package"],
        "audited_package_identity",
    )
    require(
        file_hash(base / "publication-luna-20261012.json")
        == receipt["binding"]["predecessor_publication"],
        "audited_publication_identity",
    )
    for key in ("ontologies", "runs", "policy"):
        require(old[key] == new[key], "unchanged_" + key)
    artifacts = {x["path"]: x for x in new["artifacts"]}
    require(
        all(artifacts.get(x["path"]) == x for x in old["artifacts"]),
        "all_predecessor_artifacts_retained",
    )
    for key in ("cases", "schedules", "tutorial", "forms_version", "analysis_export_version"):
        require(before.get(key) == after.get(key), "unchanged_study_" + key)
    clause = lambda c: {k: c[k] for k in ("text", "category", "fact_ids")}
    counters = Counter()
    for eid, loc in old["explanations"].items():
        a = read(base / "package" / loc)
        b = read(base / "package" / new["explanations"][receipt["replacements"].get(eid, eid)])
        require(
            a["entities"] == b["entities"] and a["support_packet"] == b["support_packet"],
            "description_input_identity",
        )
        require(
            [clause(c) for c in b["claims"][: len(a["claims"])]]
            == [clause(c) for c in a["claims"]],
            "every_approved_clause_and_citation_preserved",
        )
        if eid not in receipt["replacements"]:
            require(a == b, "unaffected_artifact_identical")
        counters["prior_claims_preserved"] += len(a["claims"])
        counters["new_claims"] += len(b["claims"]) - len(a["claims"])
    oldassets = {x["asset_id"]: x for x in before["assets"]}
    newassets = {x["asset_id"]: x for x in after["assets"]}
    oldscopes = {s["case_id"]: s for s in before["workspace_scopes"] if s["kind"] == "case"}
    scopes = {s["case_id"]: s for s in after["workspace_scopes"] if s["kind"] == "case"}
    for case in after["cases"]:
        aid = case["explanation_refs"][0]
        a = read(base / oldassets[aid]["path"])
        b = read(base / newassets[aid]["path"])
        for key in ("entities", "pair_metadata", "hierarchy", "policy_hash"):
            require(a[key] == b[key], "frozen_" + key + "_preserved")

        def evidence(r):
            values = copy.deepcopy(r["evidence"])
            for x in values:
                x["scores"] = sorted(x["scores"], key=canonical_hash)
            return {(x["candidate_id"], x["evidence_id"]): x for x in values}

        require(evidence(a) == evidence(b), "all_frozen_feature_values_preserved")
        require(
            {(f["fact_id"], canonical_hash(f["subject"])): f for f in a["facts"]}
            == {(f["fact_id"], canonical_hash(f["subject"])): f for f in b["facts"]},
            "all_original_facts_preserved",
        )
        sa, sb = oldscopes[case["case_id"]], scopes[case["case_id"]]
        require(
            sa["contexts"] == sb["contexts"] and sa["components"] == sb["components"],
            "unchanged_contexts_partial_elk_components",
        )
        with sqlite3.connect(
            (base / sa["index_path"]).as_uri() + "?mode=ro", uri=True
        ) as da, sqlite3.connect((base / sb["index_path"]).as_uri() + "?mode=ro", uri=True) as db:
            for table in ("evidence", "pair_metadata"):
                require(
                    list(da.execute("SELECT * FROM " + table + " ORDER BY 1,2"))
                    == list(db.execute("SELECT * FROM " + table + " ORDER BY 1,2")),
                    "scope_" + table + "_identical",
                )
        counters["cases"] += 1
        counters["pairs"] += len(case["candidates"])
        counters["selected_records"] += len(b["evidence"])
    require(counters["new_claims"] == 17 and counters["pairs"] == 120, "expected_targeted_scope")
    verify_base(read(root.parent / "luna-bundle-20261012/plan.json"))
    return {
        "status": "passed",
        "checks": dict(counters),
        "package_id": new["package_id"],
        "source_publication_sha256": file_hash(base / ("publication-" + revision + ".json")),
        "protected_predecessor_stats_unchanged": True,
        "matcher_and_context_artifacts_unchanged": True,
    }


class FollowupJourney(LunaJourney):
    def complete_read_surfaces(self, prefix, capabilities):
        super().complete_read_surfaces(prefix, capabilities)
        oid = "sha256:26054e89dafcade53c821c0cc1a0b00574ec59becc81f6a930fc7474e44ab500"
        if self.shared.get("curated_checked") or oid not in capabilities["ontology_versions"]:
            return
        for basis in ("literal_asserted", "structural_navigation"):
            params = {"ontology_version_id": oid, "kind": "class", "basis": basis, "limit": 100}
            scoped, sp, first = self.pages(
                prefix + "/entry-points", params, "curated_complete_scoped_inventory"
            )
            main, mp, other = self.pages(
                self.main + "/ontologies/" + oid + "/entry-points",
                params,
                "curated_complete_main_inventory",
            )
            require(
                scoped == main and first["selection"] == other["selection"],
                "curated_main_study_parity",
            )
            require(
                first["returned_count"] == 1 and first["selection"]["kind"] == "curated_default",
                "curated_first_page",
            )
            require(
                first["default_entity"]["iri"] == "http://purl.obolibrary.org/obo/DOID_4",
                "declared_start_entity",
            )
            path = (
                Path(self.config["package"]).parent
                / self.bundle["ontologies"][oid]
                / "navigation.sqlite"
            )
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                expected = {
                    x[0]
                    for x in db.execute(
                        "SELECT iri FROM entries WHERE basis=? AND kind=?", (basis, "class")
                    )
                }
            require(
                {x["entity"]["iri"] for x in scoped} == expected and len(scoped) == len(expected),
                "all_roots_including_obsolete_accessible",
            )
            self.shared["hierarchy"].append(
                {"basis": basis, "curated_entries": len(scoped), "pages": sp, "main_pages": mp}
            )
        self.shared["curated_checked"] = True


def run(root, config, output):
    output.mkdir(parents=True, exist_ok=False)
    base = Path(config["assets"])
    result = {"schema": "exact-backend-followup-verification/1", "status": "running"}
    result["preservation"] = preservation(root, base)
    baseline = read(root.parent / "postmigration-20261011/api-verification-3/static.json")
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
        for i in range(4):
            j = FollowupJourney(
                config, receipt=output / f"journey-{i+1}.json", combined=base, shared=shared
            )
            try:
                print("API journey", i + 1, "training", flush=True)
                j.prepare()
                j.tutorial()
                current = j.request(
                    "GET", "/api/v1/study/cases/current", label="allocated_schedule"
                ).json()
                schedule = next(
                    s
                    for s in j.definition["schedules"]
                    if s["blocks"][0]["condition"] == current["condition"]
                    and current["case_id"] in s["blocks"][0]["case_ids"]
                )
                cases = set(
                    next(
                        b["case_ids"] for b in schedule["blocks"] if b["condition"] == "explanation"
                    )
                )
                if cases <= shared["cases"]:
                    j.report["status"] = "allocation_only"
                else:
                    j.cases()
                    j.exports()
                    j.report["status"] = "passed"
                j.record()
            except Exception as exc:
                j.report.update(status="failed", failure=str(exc), failed_check=j.checkpoint)
                j.record()
                raise
            finally:
                j.client.close()
            if len(shared["pairs"]) == 120:
                break
        require(
            len(shared["pairs"]) == 120 and len(shared["descriptions"]) == 260,
            "all_pairs_and_descriptions",
        )
        result["status"] = "passed"
    finally:
        result.update(
            pairs=len(shared["pairs"]),
            cases=len(shared["cases"]),
            descriptions=len(shared["descriptions"]),
            citations=len(shared["citations"]),
            hierarchy=shared["hierarchy"],
        )
        environment.private_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "environment", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.root, environment.load(a.environment), a.output), indent=2))
