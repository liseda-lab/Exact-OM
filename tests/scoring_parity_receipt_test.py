import copy

import pytest

from tools.compare_scoring_parity import compare_records


def rows():
    return [
        {
            "src_iri": "s",
            "tgt_iri": t,
            "confidences": {"S_final": 0.5},
            "prediction": {"global_match": True},
            "context_triples": ["fact1", "fact2"],
        }
        for t in ("a", "b")
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_identical_nonfinite_outputs_cannot_pass_parity(value):
    baseline, changed = rows(), rows()
    # S_final can remain finite while an evidence support becomes invalid.
    baseline[0]["support"] = changed[0]["support"] = value
    result = compare_records(baseline, changed)
    assert result["status"] == "failed"
    assert result["numeric_mismatch_count"] == 1


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0])
def test_invalid_tolerance_cannot_disable_numeric_checks(tolerance):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_records(rows(), rows(), tolerance=tolerance)


def test_ordinary_only_excludes_synthetic_replays_without_hiding_natural_evidence():
    baseline = rows()
    baseline[0]["experiment_diagnostics"] = {
        "difference": {"natural_support": 0.25, "controlled_perturbations": {"synthetic": 0.4}},
        "difference_replay": [{"synthetic": 0.5}],
        "other_diagnostic": "retained",
    }
    changed = copy.deepcopy(baseline)
    del changed[0]["experiment_diagnostics"]["difference"]["controlled_perturbations"]
    del changed[0]["experiment_diagnostics"]["difference_replay"]
    original = copy.deepcopy(baseline)
    assert compare_records(baseline, changed)["status"] == "failed"
    result = compare_records(baseline, changed, ordinary_only=True)
    assert result["status"] == "passed"
    assert len(result["excluded_diagnostic_paths"]) == 2
    assert baseline == original  # Normalization never edits the saved evidence.
    changed[0]["experiment_diagnostics"]["difference"]["natural_support"] = 0.3
    assert compare_records(baseline, changed, ordinary_only=True)["status"] == "failed"
    changed = copy.deepcopy(baseline)
    changed[0]["experiment_diagnostics"]["other_diagnostic"] = "changed"
    assert compare_records(baseline, changed, ordinary_only=True)["status"] == "failed"


def test_tolerance_does_not_hide_changed_winner_or_fact_order():
    baseline, changed = rows(), rows()
    changed[1]["confidences"]["S_final"] += 1e-7
    result = compare_records(baseline, changed)
    assert result["numeric_mismatch_count"] == 0
    assert result["discrete_mismatch_count"] == 1
    assert result["status"] == "failed"
    changed = copy.deepcopy(baseline)
    changed[0]["context_triples"].reverse()
    assert compare_records(baseline, changed)["status"] == "failed"


def test_gate_membership_and_exact_ties_checked_separately_from_scores():
    baseline, changed = rows(), rows()
    assert compare_records(baseline, changed)["status"] == "passed"
    changed[0]["prediction"]["global_match"] = False
    assert compare_records(baseline, changed)["discrete_mismatch_count"] == 1


def test_rankings_are_compared_within_each_original_query():
    baseline = rows() + rows()
    for index, row in enumerate(baseline):
        row["original_query_id"] = "q1" if index < 2 else "q2"
    changed = copy.deepcopy(baseline)
    changed[1]["confidences"]["S_final"] += 1e-7
    changed[2]["confidences"]["S_final"] -= 1e-7
    result = compare_records(baseline, changed)
    assert result["numeric_mismatch_count"] == 0
    assert result["discrete_mismatch_count"] == 2


def test_empty_outputs_cannot_establish_parity():
    assert compare_records([], [])["status"] == "failed"


@pytest.mark.parametrize("passes", [True, False])
def test_cli_exit_status_matches_durable_parity_receipt(tmp_path, monkeypatch, passes):
    import json
    import sys
    from exact.runs.store import ExplanationStore, _atomic_json
    from tools.compare_scoring_parity import main

    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for path in (baseline, candidate):
        records = rows()
        for row in records:
            row["original_query_id"] = "q1"
        if path == candidate and not passes:
            records[0]["context_triples"].reverse()
        ExplanationStore(path / "cold/ordinary-evidence").append(records)
        _atomic_json(path / "cold.json", {"chunks": [{"rows": 2, "query_ids": ["q1"]}]})
    output = tmp_path / "parity.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_scoring_parity",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
    )
    if passes:
        main()
    else:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
    assert json.loads(output.read_text())["status"] == ("passed" if passes else "failed")


def fitted_outputs(tmp_path):
    from exact.runs.store import ExplanationStore, _atomic_json
    from exact.utils.provenance import sha256_file

    manifest = tmp_path / "fitted-manifest.json"
    workload = tmp_path / "workload.json"
    _atomic_json(workload, {"queries": [{"qid": "q1", "source": "s", "candidates": ["a", "b"]}]})
    _atomic_json(
        manifest,
        {
            "namespace": "qualification_only",
            "workload": {"path": str(workload), "sha256": sha256_file(workload)},
        },
    )
    paths = (tmp_path / "baseline", tmp_path / "candidate")
    for root in paths:
        records = rows()
        for row in records:
            row["original_query_id"] = "q1"
        ExplanationStore(root / "cold/ordinary-evidence").append(records)
        _atomic_json(
            root / "cold.json",
            {"chunks": [{"rows": 2, "query_ids": ["q1"], "exact_prefiltered_rows": 0}]},
        )
        _atomic_json(
            root / "measurement.json",
            {
                "namespace": "qualification_only",
                "paid_calls": 0,
                "fixture_outputs_promotable": False,
                "fitted_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            },
        )
        _atomic_json(
            root / "cold/fitted-decisions.json",
            {
                "namespace": "qualification_only",
                "selector_scope": "bounded_cohort",
                "hosted_quality_established": False,
                "policy": {
                    "threshold": 0.7,
                    "source_cardinality": 1,
                    "target_cardinality": 1,
                    "extraction": {"mode": "mutual_best"},
                },
                "protected_pairs": [],
                "mappings": [["s", "a"]],
                "rows": [{"Src": "s", "Tgt": target, "S_final": 0.5} for target in ("a", "b")],
            },
        )
        _atomic_json(root / "cold/selector-evidence.json", records)
        _atomic_json(
            root / "cold/hosted-fixtures.json",
            [
                {"role": "decision", "prompt": "support=0.867", "probability": 0.5},
            ],
        )
    return paths


def test_fitted_comparison_checks_saved_decisions_and_exact_prompts(tmp_path):
    from tools.compare_scoring_parity import compare

    baseline, candidate = fitted_outputs(tmp_path)
    result = compare(baseline, candidate, tmp_path / "parity.json", fitted=True)
    assert result["status"] == result["fitted_selector_mapping_parity"] == "passed"
    assert result["fitted"]["historical_paid_prediction_parity_established"] is False


@pytest.mark.parametrize("mode", ["warm", "partial", "replay"])
def test_fitted_parity_reads_the_requested_cache_mode(tmp_path, mode):
    from tools.compare_scoring_parity import compare

    baseline, candidate = fitted_outputs(tmp_path)
    for root in (baseline, candidate):
        (root / "cold").rename(root / mode)
        (root / "cold.json").rename(root / (mode + ".json"))
    result = compare(baseline, candidate, tmp_path / "parity.json", fitted=True, mode=mode)
    assert result["status"] == "passed"
    assert result["compared_mode"] == mode


@pytest.mark.parametrize("change", ["rank", "mapping", "policy", "anchors", "prompt", "evidence"])
def test_fitted_comparison_does_not_hide_discrete_drift(tmp_path, change):
    import json
    from exact.runs.store import _atomic_json
    from tools.compare_scoring_parity import compare

    baseline, candidate = fitted_outputs(tmp_path)
    filename = {"prompt": "hosted-fixtures", "evidence": "selector-evidence"}.get(
        change, "fitted-decisions"
    )
    path = candidate / "cold" / (filename + ".json")
    value = json.loads(path.read_text())
    if change == "rank":
        value["rows"][1]["S_final"] += 1e-7  # Within tolerance, but a different winner.
    elif change == "mapping":
        value["mappings"] = [["s", "b"]]
    elif change == "policy":
        value["policy"]["threshold"] += 1e-7
    elif change == "anchors":
        value["protected_pairs"] = [["s", "anchor"]]
    elif change == "prompt":
        value[0]["prompt"] = "support=0.868"
    else:
        value[0]["context_triples"].reverse()
    _atomic_json(path, value)
    result = compare(baseline, candidate, tmp_path / "parity.json", fitted=True)
    assert result["status"] == result["fitted_selector_mapping_parity"] == "failed"


def test_fitted_parity_rejects_different_cohorts_and_changed_bindings(tmp_path):
    import json
    from exact.runs.store import _atomic_json
    from tools.compare_scoring_parity import compare

    baseline, candidate = fitted_outputs(tmp_path)
    path = candidate / "cold.json"
    original = json.loads(path.read_text())
    _atomic_json(path, {"chunks": [{"rows": 2, "query_ids": ["different"]}]})
    with pytest.raises(ValueError, match="equal complete cohorts"):
        compare(baseline, candidate, tmp_path / "parity.json", fitted=True)
    _atomic_json(path, original)
    (tmp_path / "fitted-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed fitted qualification manifest"):
        compare(baseline, candidate, tmp_path / "parity.json", fitted=True)


def test_fitted_parity_rejects_unreported_candidate_tail(tmp_path):
    from exact.runs.store import ExplanationStore
    from tools.compare_scoring_parity import compare

    baseline, candidate = fitted_outputs(tmp_path)
    extra = {**rows()[0], "src_iri": "extra", "original_query_id": "extra"}
    ExplanationStore(candidate / "cold/ordinary-evidence").append([extra])
    with pytest.raises(ValueError, match="complete durable ordinary store"):
        compare(baseline, candidate, tmp_path / "parity.json", fitted=True)


@pytest.mark.parametrize("damage", ["subset", "reordered", "score", "policy", "query"])
def test_fitted_parity_rejects_identical_incomplete_outputs(tmp_path, damage):
    import json
    from exact.runs.store import _atomic_json
    from tools.compare_scoring_parity import compare

    baseline, candidate = fitted_outputs(tmp_path)
    for root in (baseline, candidate):
        path = root / "cold/fitted-decisions.json"
        result = json.loads(path.read_text())
        evidence_path = root / "cold/selector-evidence.json"
        evidence = json.loads(evidence_path.read_text())
        if damage == "subset":
            result["rows"] = result["rows"][:1]
            evidence = evidence[:1]
        elif damage == "reordered":
            result["rows"].reverse()
        elif damage == "score":
            del result["rows"][0]["S_final"]
        elif damage == "query":
            evidence[0]["original_query_id"] = "wrong"
        else:
            del result["policy"]["extraction"]
        _atomic_json(path, result)
        _atomic_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="cohort|finite final scores|Incomplete fitted"):
        compare(baseline, candidate, tmp_path / "parity.json", fitted=True)


def test_mixed_legacy_query_stamps_cannot_silently_repair_wrong_ids(tmp_path):
    from exact.runs.store import ExplanationStore, _atomic_json
    from tools.compare_scoring_parity import compare

    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for root in (baseline, candidate):
        records = rows()
        records[0]["original_query_id"] = "wrong"
        ExplanationStore(root / "cold/ordinary-evidence").append(records)
        _atomic_json(root / "cold.json", {"chunks": [{"rows": 2, "query_ids": ["q1"]}]})
    with pytest.raises(ValueError, match="Incomplete query identities"):
        compare(baseline, candidate, tmp_path / "parity.json")


def test_identically_truncated_receipts_cannot_hide_an_admitted_candidate(tmp_path):
    import json
    from exact.runs.store import _atomic_json
    from exact.utils.provenance import sha256_file
    from tools.compare_scoring_parity import compare

    baseline, candidate = fitted_outputs(tmp_path)
    workload_path = tmp_path / "workload.json"
    workload = json.loads(workload_path.read_text())
    workload["queries"][0]["candidates"].append("c")
    _atomic_json(workload_path, workload)
    manifest_path = tmp_path / "fitted-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["workload"]["sha256"] = sha256_file(workload_path)
    _atomic_json(manifest_path, manifest)
    for root in (baseline, candidate):
        path = root / "measurement.json"
        measurement = json.loads(path.read_text())
        measurement["fitted_manifest"]["sha256"] = sha256_file(manifest_path)
        _atomic_json(path, measurement)
    # Both implementations and their receipts agree, but both omit candidate c.
    with pytest.raises(ValueError, match="every admitted candidate"):
        compare(baseline, candidate, tmp_path / "parity.json", fitted=True)


def test_identical_mappings_outside_the_cohort_cannot_establish_parity(tmp_path):
    import json
    from exact.runs.store import _atomic_json
    from tools.compare_scoring_parity import compare

    baseline, candidate = fitted_outputs(tmp_path)
    for root in (baseline, candidate):
        path = root / "cold/fitted-decisions.json"
        result = json.loads(path.read_text())
        result["mappings"].append(["s", "unscored"])
        _atomic_json(path, result)
    with pytest.raises(ValueError, match="outside the admitted cohort"):
        compare(baseline, candidate, tmp_path / "parity.json", fitted=True)
