"""Real offline production stages -> inspection API -> portable ZIP round trip.

Run with the qualified native/viz environment and a retained pytest --basetemp.
The test root contains staged_pipeline_paths.json, including the three prepared
packages and exported ZIPs for subsequent browser checks. No downloaded models,
hosted decisions, generated explanations, or fabricated pair features are used.
"""

import ast
from collections import Counter
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from tests.native_release_test import offline_native  # noqa: F401 -- shared pytest fixture


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11) or importlib.util.find_spec("torch") is None,
    reason="Requires the qualified Python 3.11+ native/viz environment with CPU Torch",
)

REPO = Path(__file__).resolve().parents[1]
WORDS = ("heart", "lung", "tissue", "disease")
TRAIN = tuple("urn:s:train-" + word for word in WORDS)
TARGETS = tuple("urn:t:" + word for word in (*WORDS, "organ"))
HEART, LUNG, EMPTY = ("urn:s:heldout-" + word for word in ("heart", "lung", "empty"))
PRIVATE_GOLD = "urn:PRIVATE:GOLD"
SCORES = {
    "S_final", "S_base", "S_pair_final", "S_struct", "U", "U_ind", "U_dis",
    "s_label", "s_hier", "s_attr", "s_sim", "s_diff", "s_strsim", "cand_sim",
    "repair_distinctive", "p_llm",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(paths):
    files = {p for root in paths for p in (root.rglob("*") if root.is_dir() else [root])
             if p.is_file()}
    return {str(p): _sha(p) for p in sorted(files)}


def _cli(root, module, *args):
    """Exercise the real dispatcher in a fresh, CPU-only offline interpreter."""
    logs = root / "cli-logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{len(list(logs.glob('*.json'))):02d}-{args[0]}.json"
    command = [sys.executable, "-m", module, *map(str, args)]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")])),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="7",
    )
    try:
        result = subprocess.run(command, cwd=REPO, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=180)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"CLI exceeded 180 seconds: {command!r}\n{exc}")
    _write_json(log, dict(command=command, returncode=result.returncode,
                         stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, (
        f"CLI failed; full log: {log}\n{command!r}\n"
        f"{result.stdout[-6000:]}\n{result.stderr[-12000:]}"
    )
    return result.stdout


def _exact(root, *args):
    return _cli(root, "exact.delivery.cli.main", *args)


def _ontology(path, side, entities):
    rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    owl = "http://www.w3.org/2002/07/owl#"
    rdfs = "http://www.w3.org/2000/01/rdf-schema#"
    tree = ET.Element(f"{{{rdf}}}RDF")
    ET.SubElement(tree, f"{{{owl}}}Ontology", {f"{{{rdf}}}about": f"urn:{side}:ontology"})
    for iri, label in entities:
        node = ET.SubElement(tree, f"{{{owl}}}Class", {f"{{{rdf}}}about": iri})
        ET.SubElement(node, f"{{{rdfs}}}label").text = label
        if iri != f"urn:{side}:organ":
            ET.SubElement(node, f"{{{rdfs}}}subClassOf",
                          {f"{{{rdf}}}resource": f"urn:{side}:organ"})
    path.write_bytes(ET.tostring(tree, encoding="utf-8", xml_declaration=True))


def _candidate_tsv(path, queries):
    # Deliberately false gold must never influence native scoring/application.
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["SrcEntity", "TgtEntity", "TgtCandidates"])
        writer.writerows((q["source"], PRIVATE_GOLD, repr(q["candidates"])) for q in queries)


def _inputs(f):
    root = f["root"]
    _ontology(f["source"], "s", [
        *zip(TRAIN, (word + " source" for word in WORDS)),
        (HEART, "heart"), (LUNG, "lung source"), (EMPTY, "source tissue disease"),
        ("urn:s:organ", "organ source"),
    ])
    _ontology(f["target"], "t", list(zip(TARGETS, (*WORDS, "organ"))))
    config = _json(f["config"])
    config["dataset"].update(filter_exact_matches=False, drop_exact_match_sources=False,
                             verbalization_mode="deterministic", reasoner="asserted")
    config["candidates"]["top_k"] = len(TARGETS)
    config["pipeline"][1]["params"]["calibration"].update(max_epochs=3, validation_folds=2)
    config["pipeline"][1]["params"].setdefault("llm", {})["enabled"] = False
    config["selector"]["enabled"] = True
    config["selector"]["rerank"].update(mode="pairwise", model="current_linear")
    config["matching"]["calibration"]["mode"] = "none"
    config["matching"]["nil"]["mode"] = "off"
    _write_json(f["config"], config)
    application_config = copy.deepcopy(config)
    application_config["dataset"]["filter_exact_matches"] = True
    application_config_path = root / "application-config.json"
    _write_json(application_config_path, application_config)
    train = [dict(qid=qid, source=s, candidates=list(reversed(TARGETS)), gold_target="urn:t:" + w)
             for qid, s, w in zip((103, 9, 405, 72), TRAIN, WORDS)]
    queries = [
        dict(qid=901, source=HEART, candidates=["urn:t:tissue", "urn:t:heart", "urn:t:lung"]),
        dict(qid=7, source=LUNG, candidates=["urn:t:disease", "urn:t:lung", "urn:t:heart"]),
        dict(qid=42, source=HEART, candidates=["urn:t:lung", "urn:t:heart"]),
        dict(qid=6001, source=EMPTY, candidates=[]),
        dict(qid=63, source=HEART, candidates=["urn:t:heart"]),
    ]
    train_queries, queries_path = root / "train-queries.json", root / "application-queries.json"
    _write_json(train_queries, train)
    _write_json(queries_path, queries)
    train_candidates, app_candidates = root / "train-candidates.tsv", root / "application-candidates.tsv"
    _candidate_tsv(train_candidates, train)
    _candidate_tsv(app_candidates, queries)
    positives = root / "train-positive.tsv"
    with positives.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["SrcEntity", "TgtEntity"])
        writer.writerows((q["source"], q["gold_target"]) for q in train)
    folds = root / "train-folds.json"
    _write_json(folds, {source: i % 2 for i, source in enumerate(TRAIN)})
    return dict(train_queries=train_queries, queries=queries_path, train_candidates=train_candidates,
                app_candidates=app_candidates, application_config=application_config_path,
                positives=positives, folds=folds)


class _DeterministicCacheEngine:
    """Only the decision-cache seed is a test double; no Elephant model is loaded."""

    def predict(self, payload):
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).digest()
        return {"probability": 0.2 + digest[0] / 255 * 0.6,
                "local_model_call_attempted": False, "live_provider_call_attempted": False}


def _decisions(root, native, policy):
    from exact.decisions.pipeline import run

    directory = root / (native.name + "-decisions")
    args = ["decisions", "prepare", "--candidates", native / "pairs.jsonl",
            "--sources", native / "sources.txt", "--output", directory, "--policy", policy]
    if policy == "local-application":
        args += ["--anchors", native / "anchors.json"]
    _exact(root, *args)
    plan, cache = directory / "plan.json", directory / "answers.sqlite"
    requests = _rows(directory / "requests.jsonl")
    assert requests and len(requests) == _json(plan)["request_count"]
    seeded = run(plan, cache, engine=_DeterministicCacheEngine())
    assert seeded["completed"] == len(requests) and seeded["provider_calls"] == 0
    _exact(root, "decisions", "run", "--plan", plan, "--cache", cache, "--cache-only")
    verified = _json(directory / "status.json")
    assert verified["completed"] == verified["cached_response_reused"] == len(requests)
    assert verified["local_model_calls"] == verified["provider_calls"] == 0
    fused = directory / "scored-pairs.jsonl"
    _exact(root, "decisions", "apply", "--candidates", native / "pairs.jsonl",
           "--plan", plan, "--cache", cache, "--output", fused)
    native_rows = {(r["Src"], r["Tgt"]): r for r in _rows(native / "pairs.jsonl")}
    fused_rows = {(r["Src"], r["Tgt"]): r for r in _rows(fused)}
    requested = {tuple(r["pair"]): r for r in requests}
    assert fused_rows.keys() == native_rows.keys()
    for key, row in fused_rows.items():
        original = native_rows[key]
        assert row.get("selected_attributions") == original.get("selected_attributions")
        assert not row["local_model_call_attempted"] and not row["live_provider_call_attempted"]
        assert row["llm_used"] is (key in requested)
        assert row["cached_response_reused"] is (key in requested)
        probability = (_DeterministicCacheEngine().predict(requested[key]["payload"])["probability"]
                       if key in requested else None)
        weight = min(1., max(0., .8 * original["U"])) if probability is not None else 0.
        score = ((1 - weight) * original["S_base"] + weight * probability
                 if probability is not None else original["S_base"])
        assert row["p_llm"] == probability and row["llm_weight"] == weight
        assert row["S_final"] == row["S_pair_final"] == score
    return plan, fused


def _get(client, url, **params):
    result = client.get(url, params=params)
    assert result.status_code == 200, (url, params, result.text)
    return result.json()


def _pages(client, url, **params):
    result, cursors = [], set()
    total = None
    while True:
        page = _get(client, url, limit=2, **params)
        assert page["returned_count"] == len(page["items"])
        if total is None:
            total = page["total_count"]
        assert page["total_count"] == total
        result.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            assert len(result) == total
            return result
        assert cursor not in cursors and page["items"], "Pagination must make progress"
        cursors.add(cursor)
        params["cursor"] = cursor


def _api_snapshot(client):
    runs = _pages(client, "/api/v1/runs")
    assert len(runs) == 1
    run = runs[0]
    prefix = "/api/v1/runs/" + run["run_id"]
    sources = _pages(client, prefix + "/sources")
    pairs, facts = {}, {}
    for source in sources:
        params = dict(source=source["entity"]["iri"])
        if "query_id" in source:
            params["query_id"] = source["query_id"]
        candidates = _pages(client, prefix + "/candidates", **params)
        if "candidate_count" in source:
            assert len(candidates) == source["candidate_count"]
        for candidate in candidates:
            pid = candidate["pair_id"]
            assert pid not in pairs, "Query occurrences must have distinct pair identities"
            trace = _get(client, prefix + "/pair", pair_id=pid)
            evidence = _pages(client, prefix + "/pair-evidence", pair_id=pid)
            for item in evidence:
                ontology = item["entity"]["ontology_version_id"]
                originals = facts.setdefault(ontology, {})
                for fact_id in item["fact_ids"]:
                    if fact_id not in originals:
                        originals[fact_id] = _get(client, "/api/v1/axioms/" + fact_id,
                                                  ontology_version_id=ontology)
                    axiom = originals[fact_id]
                    assert axiom["id"] == axiom["fact_id"] == axiom["axiom_id"] == fact_id
                    assert axiom["ontology_version_id"] == ontology
                    assert axiom["availability"] == "available"
            pairs[pid] = dict(candidate=candidate, trace=trace, evidence=evidence)
    assert len(sources) == run["counts"]["sources"]
    assert len(pairs) == run["counts"]["pairs"]
    assert sum(len(record["evidence"]) for record in pairs.values()) == run["counts"]["selected_evidence"]
    return dict(run=run, sources=sources, pairs=pairs, facts=facts)


def _selected(row):
    """Independent inventory of the saved native selected items, including both sides."""
    attrs = row.get("selected_attributions", {})
    for channel, payload in attrs.get("triple_attributions", {}).items():
        groups = payload.items() if channel == "hierarchy" else [(None, payload)]
        for family, sides in groups:
            for side in ("source", "target"):
                for item in sides[side]:
                    yield (channel, family, side), item
    for side, items in attrs.get("attributes", {}).items():
        for item in items:
            yield ("attributes", None, side), item


def _semantic_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _semantic_terms(item):
    # The public identity contract excludes labels, display text and item aliases.
    terms = {key: item[key] for key in (
        "subject_iri", "object_iri", "rel_iri", "property_iri", "prop_iri", "entity_iri",
        "value", "literal_terms", "literal_lexical_form", "datatype", "language",
        "type_closure", "role", "source_axiom_refs", "derivation",
    ) if item.get(key) is not None}
    if terms.get("rel_iri") == "http://subclassof":
        terms["rel_iri"] = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    if "grouped_semantic_features" in item:
        terms["grouped_semantic_features"] = sorted(
            [_semantic_terms(feature) for feature in item["grouped_semantic_features"]],
            key=_semantic_hash,
        )
    return terms


def _strings(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_strings(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_strings(v) for v in value))
    return set()


def _expected_evidence(candidate, row):
    expected = {}
    for (channel, family, side), original in _selected(row):
        terms = _semantic_terms(original)
        assert (terms.get("subject_iri") and terms.get("object_iri")) or (
            (terms.get("property_iri") or terms.get("prop_iri")) and "value" in terms
        ), "The native fixture must retain semantic evidence identities"
        if terms.get("rel_iri") in {"domain", "range"}:
            terms["role"] = "property_" + terms["rel_iri"]
            terms["rel_iri"] = "http://www.w3.org/2000/01/rdf-schema#" + terms["rel_iri"]
        feature_id = _semantic_hash(dict(
            ontology_version_id=candidate[side]["ontology_version_id"],
            entity_kind=candidate[side]["kind"], interpretation=original.get("interpretation", "projected"),
            family=family, terms=terms,
        ))
        evidence_id = _semantic_hash(dict(side=side, channel=channel, feature_id=feature_id, legacy=None))
        assert evidence_id not in expected, "The tiny fixture should have unique semantic features"
        expected[evidence_id] = dict(feature_id=feature_id, channel=channel, family=family, side=side,
                                    terms=terms, original=original)
    return expected


def _assert_common(snapshot, rows):
    resolved = 0
    for record in snapshot["pairs"].values():
        candidate, trace, evidence = (record[k] for k in ("candidate", "trace", "evidence"))
        key = candidate["source"]["iri"], candidate["target"]["iri"]
        row = rows[key]
        assert candidate["source"] == trace["source"] and candidate["target"] == trace["target"]
        assert trace["pair_id"] == candidate["pair_id"]
        scores = {s["name"]: s["value"] for s in candidate["scores"]}
        expected = {name: row[name] for name in SCORES if row.get(name) is not None}
        expected["w_llm"] = row["llm_weight"]
        assert {name: scores[name] for name in expected} == expected
        events = {e["stage"]: e for e in trace["events"]}
        assert events["llm_signal"]["reason_code"] == (
            "staged_decision_applied" if row["llm_used"] else "staged_decision_not_requested")
        assert events["llm_signal"]["status"] == ("completed" if row["llm_used"] else "not_run")
        selected = _expected_evidence(candidate, row)
        assert Counter(e["evidence_id"] for e in evidence) == Counter(list(selected))
        assert trace["evidence_metadata"]["counts"]["selected"] == len(evidence)
        for item in evidence:
            saved = selected[item["evidence_id"]]
            original, terms = saved["original"], saved["terms"]
            for name in ("channel", "family", "side"):
                assert item[name] == saved[name]
            assert item["values"] == {name: original[name] for name in (
                "support", "importance", "weight", "edge_ic", "unsupported_mass",
            ) if name in original}
            # The policy export intentionally withholds raw strings even for resolved items.
            assert item["historical_item_alias"] is None
            assert item["semantic_terms"] == item["display"] == {}
            assert item["entity"] == candidate[item["side"]]
            if item["status"] == "available":
                assert item["feature_id"] == saved["feature_id"] and item["fact_ids"]
                assert item["origin_resolution"] == "resolved"
                originals = snapshot["facts"][item["entity"]["ontology_version_id"]]
                linked = [originals[fid] for fid in item["fact_ids"]]
                assert all(item["entity"]["iri"] in _strings(fact["ast"]) for fact in linked)
                if terms.get("subject_iri") and terms.get("object_iri"):
                    endpoints = {terms["subject_iri"], terms["object_iri"]}
                    assert any(endpoints <= _strings(fact["ast"]) for fact in linked)
                    if terms.get("rel_iri") == "http://www.w3.org/2000/01/rdf-schema#subClassOf":
                        assert any(fact["ast"]["type"] == "SubClassOf"
                                   and terms["subject_iri"] in _strings(fact["ast"]["sub_class"])
                                   and terms["object_iri"] in _strings(fact["ast"]["super_class"])
                                   for fact in linked)
                else:
                    literal = {terms.get("property_iri") or terms["prop_iri"], str(terms["value"])}
                    assert any(literal <= _strings(fact["ast"]) for fact in linked)
                if original.get("source_axiom_refs"):
                    assert set(original["source_axiom_refs"]) <= {
                        fact["original_axiom_digest"] for fact in linked}
                resolved += 1
            else:
                assert item["feature_id"] is None and not item["fact_ids"]
    assert resolved > 0, "Real native selected ontology evidence must survive preparation"
    assert PRIVATE_GOLD not in json.dumps(snapshot)


def _assert_global(snapshot, rows, acceptance):
    with (acceptance / "mappings.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    expected = {(r["SrcEntity"], r["TgtEntity"]): float(r["Score"]) for r in saved}
    assert len(expected) == len(saved) == _json(acceptance / "completion.json")["mappings"]
    ns = {"a": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"}
    resource = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
    rdf = {}
    for cell in ET.parse(acceptance / "alignment.rdf").findall(".//a:Cell", ns):
        key = (cell.find("a:entity1", ns).get(resource), cell.find("a:entity2", ns).get(resource))
        assert key not in rdf and cell.findtext("a:relation", namespaces=ns) == "="
        rdf[key] = float(cell.findtext("a:measure", namespaces=ns))
    assert rdf == expected
    seen, members = set(), {}
    for record in snapshot["pairs"].values():
        candidate, trace = record["candidate"], record["trace"]
        key = candidate["source"]["iri"], candidate["target"]["iri"]
        assert key not in seen
        seen.add(key)
        assert candidate["saved_alignment_member"] is (key in expected)
        assert trace["saved_alignment_member"] is (key in expected)
        assert trace["ranking"] is None and candidate["ranking"] is None
        assert trace["relation"] == ("equivalent" if key in expected else None)
        scores = {s["name"]: s["value"] for s in candidate["scores"]}
        assert ("P_match" in scores) is (key in expected)
        if key in expected:
            members[key] = scores["P_match"]
        provenance = candidate["membership_provenance"]
        assert provenance["artifact_hash"] == "sha256:" + _sha(acceptance / "mappings.csv")
    assert seen == rows.keys() and members == expected
    assert {s["entity"]["iri"] for s in snapshot["sources"]} == {s for s, _ in rows}
    assert snapshot["run"]["counts"]["queries"] == 0
    assert seen - expected.keys(), "The viewer must also retain unselected candidates"
    _assert_common(snapshot, rows)


def _assert_local(snapshot, rows, queries, rankings, anchors):
    saved = _json(rankings.with_suffix(".scores.json"))
    original = _json(queries)
    assert [q["qid"] for q in saved["queries"]] == [q["qid"] for q in original]
    assert {q["source"] for q in original}.isdisjoint(TRAIN)
    sources = {s["query_id"]: (s["entity"]["iri"], s["candidate_count"]) for s in snapshot["sources"]}
    assert len(sources) == len(snapshot["sources"]) == len(original)
    assert sources == {q["qid"]: (q["source"], len(q["candidates"])) for q in original}
    positions = {q["qid"]: i for i, q in enumerate(original)}
    assert {s["query_id"]: s["query_position"] for s in snapshot["sources"]} == positions
    assert snapshot["run"]["counts"]["sources"] == saved["original_query_count"] == 5
    assert len({s["entity"]["iri"] for s in snapshot["sources"]}) == saved["source_count"] == 3
    assert sources[6001] == (EMPTY, 0) and positions[6001] == 3
    expected = {}
    with rankings.open(newline="") as stream:
        published = list(csv.DictReader(stream, delimiter="\t"))
    for query, result, tsv in zip(original, saved["queries"], published, strict=True):
        assert query["source"] == result["source"] == tsv["SrcEntity"]
        ordered = [c["target"] for c in result["candidates"]]
        assert Counter(ordered) == Counter(query["candidates"])
        assert ast.literal_eval(tsv["TgtCandidates"]) == ordered
        for position, candidate in enumerate(result["candidates"], 1):
            expected[query["qid"], query["source"], candidate["target"]] = (position, candidate)
    actual = {}
    protected = {(r["Src"], r["Tgt"]) for r in _json(anchors)}
    assert (HEART, "urn:t:heart") in protected
    anchor_occurrences = 0
    for record in snapshot["pairs"].values():
        candidate, trace = record["candidate"], record["trace"]
        ranking = candidate["ranking"]
        assert ranking == trace["ranking"]
        source, target = candidate["source"]["iri"], candidate["target"]["iri"]
        key = ranking["query_id"], source, target
        assert ranking["query_position"] == positions[key[0]]
        assert key not in actual
        actual[key] = candidate["pair_id"]
        position, saved_candidate = expected[key]
        assert ranking["rank"] == candidate["ordinal_ranks"]["candidate_joint_rank"] == position
        assert ranking["candidate_count"] == sources[key[0]][1]
        assert ranking["saved_output_member"] is True
        assert ranking["artifact_ref"] == "sha256:" + _sha(rankings)
        assert candidate["saved_alignment_member"] is None
        assert trace["saved_alignment_member"] is None and trace["relation"] is None
        scores = {s["name"]: s["value"] for s in candidate["scores"]}
        assert scores["S_rank"] == saved_candidate["score"] and "P_match" not in scores
        events = {e["stage"]: e for e in trace["events"]}
        assert events["selection"]["reason_code"] == "local_ranking_applied"
        if (source, target) in protected:
            anchor_occurrences += 1
            assert saved_candidate["protected_exact"] is True and scores["S_rank"] == 1.
            assert "rank_utility" not in scores
            assert events["prefilter"]["reason_code"] == "exact_match_protected"
            # Protection at ranking does not erase an actual native scoring pass.
            scored = "selected_attributions" in rows[source, target]
            assert events["pair_scoring"]["status"] == ("completed" if scored else "not_run")
            assert events["pair_scoring"]["reason_code"] == (
                "native_pair_scores_recorded" if scored else "exact_match_protected")
        else:
            assert scores["rank_utility"] == saved_candidate["utility"]
    assert anchor_occurrences == 3
    assert actual.keys() == expected.keys()
    assert len(actual) == saved["candidate_occurrences"] == 9
    assert snapshot["run"]["counts"]["queries"] == len(original)
    _assert_common(snapshot, rows)


def test_real_offline_staged_pipeline_and_portable_import(offline_native):
    from fastapi.testclient import TestClient
    from exact_inspect.service import create_prepared_app

    f = offline_native
    root = f["root"]
    inputs = _inputs(f)
    manifest_path = root / "staged_pipeline_paths.json"
    paths = dict(root=str(root), source_ontology=str(f["source"]), target_ontology=str(f["target"]),
                 inputs={key: str(path) for key, path in inputs.items()}, packages={}, archives={},
                 libraries={}, stages={}, status="building")
    _write_json(manifest_path, paths)
    stages = {}
    for name, mode, config, candidates, policy in (
        ("global", "global", f["config"], None, "global"),
        ("train", "local", f["config"], inputs["train_candidates"], "local-train"),
        ("local", "local", inputs["application_config"], inputs["app_candidates"], "local-application"),
    ):
        native = root / (name + "-native")
        args = ["score", "--source", f["source"], "--target", f["target"], "--config", config,
                "--output", native, "--mode", mode, "--device", "cpu", "--sources-per-shard", "2"]
        args += (["--candidates", candidates] if candidates else
                 ["--source-prefix", "urn:s:", "--target-prefix", "urn:t:"])
        _exact(root, *args)
        receipt = _json(native / "complete.json")
        assert receipt["status"] == "complete" and receipt["mode"] == mode
        assert receipt["provider_calls"] == 0 and receipt["reference_labels_loaded"] is False
        native_rows = _rows(native / "pairs.jsonl")
        assert len(native_rows) == receipt["candidate_count"]
        assert len({(r["Src"], r["Tgt"]) for r in native_rows}) == len(native_rows)
        assert PRIVATE_GOLD not in json.dumps(native_rows)
        plan, fused = _decisions(root, native, policy)
        stages[name] = dict(native=native, plan=plan, fused=fused)
        paths["stages"][name] = {key: str(path) for key, path in stages[name].items()}
        _write_json(manifest_path, paths)

    train_rows = _rows(stages["train"]["fused"])
    assert {(r["Src"], r["Tgt"]) for r in train_rows} == {(s, t) for s in TRAIN for t in TARGETS}
    assert all(r["selected_attributions"] and "selector_evidence_items" in r for r in train_rows)
    assert not any(r["exact_anchor"] for r in train_rows)
    acceptance_model = root / "acceptance-model"
    _exact(root, "acceptance", "fit", "--candidates", stages["global"]["fused"],
           "--train", inputs["positives"], "--output", acceptance_model, "--family", "logistic",
           "--folds", "2", "--fold-membership", inputs["folds"], "--threads", "1")
    outputs = {}
    for policy in ("full", "partial"):
        directory = root / ("acceptance-" + policy)
        _exact(root, "acceptance", "apply", "--candidates", stages["global"]["fused"],
               "--model", acceptance_model, "--policy", policy, "--output", directory)
        outputs["global_" + policy] = (stages["global"], directory, acceptance_model / "model.json")
    rank_model = root / "rank-model.json"
    _exact(root, "rank", "fit", "--candidates", stages["train"]["fused"],
           "--queries", inputs["train_queries"], "--config", f["config"], "--model", rank_model,
           "--fold-membership", inputs["folds"], "--confirmed-negatives")
    rank = _json(rank_model)
    assert set(rank["fit_provenance"]["training_sources"]) == set(TRAIN)
    assert len(rank["folds"]) == 2
    for fold in rank["folds"]:
        assert len(fold["train_sources"]) == len(fold["heldout_sources"]) == 2
        assert set(fold["train_sources"]).isdisjoint(fold["heldout_sources"])
    rankings = root / "rankings.tsv"
    rank_args = ["rank", "apply", "--candidates", stages["local"]["fused"],
                 "--queries", inputs["queries"], "--model", rank_model,
                 "--anchors", stages["local"]["native"] / "anchors.json", "--output", rankings]
    _exact(root, *rank_args)
    publication = [rankings, rankings.with_suffix(".scores.json"), rankings.with_suffix(".receipt.json")]
    repeated = _snapshot([rank_model, *publication])
    _exact(root, *rank_args)
    assert _snapshot([rank_model, *publication]) == repeated
    outputs["local_ranking"] = (stages["local"], rankings, rank_model)

    producer_paths = [f["source"], f["target"], f["config"], *inputs.values(), acceptance_model,
                      rank_model, *publication]
    for stage in stages.values():
        producer_paths.extend([stage["native"], stage["plan"].parent])
    producer_paths.extend(value[1] for key, value in outputs.items() if key.startswith("global"))
    before = _snapshot(producer_paths)
    for name, (stage, final, model) in outputs.items():
        prepared = root / ("prepared-" + name)
        args = ["prepare", "--source-ontology", f["source"], "--target-ontology", f["target"],
                "--native-dir", stage["native"], "--scored-pairs", stage["fused"],
                "--decision-plan", stage["plan"], "--model-manifest", model, "--output", prepared]
        args += (["--rankings", final, "--queries", inputs["queries"]] if name == "local_ranking"
                 else ["--acceptance-dir", final])
        stdout = _cli(root, "exact_inspect.cli", *args)
        package = Path(json.loads(stdout)["package"])
        archive = root / (name + ".zip")
        _cli(root, "exact_inspect.cli", "export", "--package", package, "--output", archive)
        library = root / ("library-" + name)
        paths["packages"][name], paths["archives"][name] = str(package), str(archive)
        paths["libraries"][name] = str(library)
        _write_json(manifest_path, paths)  # Leave usable browser artifacts even if an API check fails.
        with TestClient(create_prepared_app(package)) as client:
            snapshot = _api_snapshot(client)
            if name == "local_ranking":
                missing_query = client.get("/api/v1/runs/" + snapshot["run"]["run_id"] + "/candidates",
                                           params={"source": HEART})
                assert missing_query.status_code == 422
        rows = {(r["Src"], r["Tgt"]): r for r in _rows(stage["fused"])}
        if name == "local_ranking":
            _assert_local(snapshot, rows, inputs["queries"], final, stage["native"] / "anchors.json")
        else:
            _assert_global(snapshot, rows, final)
        # Import into a fresh empty library; compare all paginated API records.
        with TestClient(create_prepared_app(None, library_dir=library)) as client:
            response = client.post("/api/v1/bundles/import", content=archive.read_bytes(),
                                   headers={"Content-Type": "application/zip"})
            assert response.status_code == 200, response.text
            package_id = response.json()["package_id"]
            assert package_id == _json(package)["package_id"]
            assert client.post("/api/v1/bundles/" + package_id + "/select").status_code == 200
            assert _get(client, "/api/v1/health")["package_id"] == package_id
            assert _api_snapshot(client) == snapshot
        assert _snapshot(producer_paths) == before, "Inspection must not mutate producer artifacts"
    paths["status"] = "complete"
    _write_json(manifest_path, paths)
    print(f"Staged pipeline browser artifacts: {manifest_path}")
