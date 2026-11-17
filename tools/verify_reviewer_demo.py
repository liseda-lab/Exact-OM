"""Read-only acceptance of a deployed demo against its frozen study publication."""

import argparse
import json
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from exact_inspect.contracts import canonical_hash


def verify(origin, assets, publication, decisions):
    def get(path, **query):
        with urllib.request.urlopen(
            origin + path + ("?" + urllib.parse.urlencode(query) if query else ""), timeout=60
        ) as r:
            return json.load(r)

    def absent(path, method="GET"):
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    origin + path, method=method, data=b"{}" if method == "POST" else None
                ),
                timeout=10,
            )
        except urllib.error.HTTPError as e:
            assert e.code in (404, 405), (path, e.code)
        else:
            raise AssertionError("Exposed " + path)

    d = json.loads(publication.read_bytes())["definition"]
    recovery = json.loads(decisions.read_bytes())
    health = get("/api/v1/health")
    assert health["profile"] == "public_demo"
    run = get("/api/v1/runs")["items"]
    assert len(run) == 1
    prefix = "/api/v1/runs/" + run[0]["run_id"]
    sources = get(prefix + "/sources")
    assert sources["total_count"] == len(sources["items"]) == 10
    checked = 0
    seen = set()
    explanations = 0
    features = Counter()
    for row in sources["items"]:
        entity = row["entity"]
        case = next(c for c in d["cases"] if c["source"] == entity)
        seen.add(case["case_id"])
        actual = get(prefix + "/candidates", source=entity["iri"])["items"]
        assert len(actual) == 5
        scope = next(
            s
            for s in d["workspace_scopes"]
            if s.get("case_id") == case["case_id"] and s["kind"] == "case"
        )
        with sqlite3.connect(f'file:{assets/scope["index_path"]}?mode=ro', uri=True) as db:
            for c in case["candidates"]:
                a = next(x for x in actual if x["pair_id"] == c["candidate_id"])
                assert (
                    a["target"] == c["entity"]
                    and a["scores"][0]["value"] == c["score"]
                    and a["ordinal_ranks"]["candidate_joint_rank"] == c["display_position"]
                )
                trace = get(prefix + "/pair", pair_id=c["candidate_id"])
                assert trace["source"] == entity and trace["target"] == c["entity"]
                recovered = recovery["cases"][case["case_id"]]
                saved = next(
                    x for x in recovered["candidates"] if x["candidate_id"] == c["candidate_id"]
                )
                assert trace["ranking"]["rank"] == c["display_position"]
                assert trace["ranking"]["candidate_count"] == recovered["candidate_count"] == 100
                assert trace["ranking"]["artifact_ref"] == "sha256:" + recovery["submission_sha256"]
                assert trace["saved_alignment_member"] is None and trace["relation"] is None
                assert a["decision_status"] == "available"
                events = {e["stage"]: e for e in trace["events"]}
                assert len(events) == 10 and all(
                    e["status"] != "not_recorded" and e["artifact_refs"] for e in events.values()
                )
                assert (events["prefilter"]["reason_code"] == "exact_match_protected") == saved[
                    "protected_exact"
                ]
                assert events["selection"]["reason_code"] == "local_query_ranked"
                for stage in ["threshold", "cardinality", "relation_typing", "repair"]:
                    assert events[stage]["status"] == "not_run"
                score_values = {s["name"]: s["value"] for s in a["scores"]}
                for k in [
                    "S_base",
                    "S_struct",
                    "s_label",
                    "s_hier",
                    "s_attr",
                    "s_sim",
                    "s_diff",
                    "s_strsim",
                ]:
                    assert score_values.get(k) == saved["native_scores"].get(k)
                assert score_values.get("rank_utility") == saved["utility"]
                if saved["applied"]:
                    assert score_values["S_pair_final"] == saved["applied"]["S_pair_final"]
                    assert score_values.get("p_llm") == saved["applied"]["p_llm"]
                    assert (events["llm_signal"]["status"] == "completed") == saved["applied"][
                        "cached_response_reused"
                    ]
                    features["model_signal_applied"] += saved["applied"]["cached_response_reused"]
                evidence = get(prefix + "/pair-evidence", pair_id=c["candidate_id"], limit=100)
                assert evidence["total_count"] > 0
                features.update({"evidence_items": evidence["total_count"]})
                meta = trace["evidence_metadata"]
                features.update({"evidence_" + k: v for k, v in meta["counts"].items()})
                features["pairs_with_recorded_bridges"] += meta["bridge_status"] == "recorded"
                assert not any(
                    k in json.dumps([a, trace, evidence])
                    for k in ["acceptable_candidate_ids", "reference_target", "reference_rank"]
                )
                for e in [entity, c["entity"]]:
                    context = get("/api/v1/entity-context", **e)
                    assert context["entity"] == e
                checked += 1
            for eid, payload in db.execute("SELECT id,payload FROM explanations"):
                value = get("/api/v1/explanations/" + urllib.parse.quote(eid, safe=""))
                expected = json.loads(payload)
                assert (
                    value["claims"] == expected["claims"]
                    and value["support_packet"] == expected["support_packet"]
                )
                explanations += 1
    for path in [
        "/library/",
        "/api/v1/bundles",
        "/api/v1/jobs/new",
        "/api/v1/study/state",
        "/api/v1/admin/studies",
    ]:
        absent(path)
    absent("/api/v1/bundles/import", "POST")
    absent("/api/v1/bundles/import-jobs", "POST")
    for ontology in get("/api/v1/ontologies")["items"]:
        version = ontology["ontology_version_id"]
        for basis in ["literal_asserted", "structural_navigation", "elk_inferred_partial"]:
            entries = get(
                "/api/v1/ontologies/" + urllib.parse.quote(version, safe="") + "/entry-points",
                basis=basis,
            )
            assert entries["status"] in ["available", "partial", "not_exported", "absent_in_scope"]
    return {
        "source_revision": d["study_revision"],
        "package_id": health["package_id"],
        "sources": len(seen),
        "pairs_checked": checked,
        "description_reads_equal_latest_workspace": explanations,
        "reference_answers_absent": True,
        "import_endpoints_disabled": True,
        "hierarchy_bases_checked": True,
        "restored_trace_pairs": checked,
        "ranking_is_not_alignment": True,
        "feature_audit": dict(features),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origin", default="http://127.0.0.1:8011")
    for k in ["assets", "publication", "report", "decisions"]:
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    r = verify(a.origin, a.assets, a.publication, a.decisions)
    a.report.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r))
