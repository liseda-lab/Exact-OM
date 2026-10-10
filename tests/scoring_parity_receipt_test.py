import copy

import pytest

from tools.compare_scoring_parity import compare_records


def rows():
    return [{"src_iri": "s", "tgt_iri": t, "confidences": {"S_final": .5},
             "prediction": {"global_match": True}, "context_triples": ["fact1", "fact2"]}
            for t in ("a", "b")]


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
    monkeypatch.setattr(sys, "argv", ["compare_scoring_parity", "--baseline", str(baseline),
                                    "--candidate", str(candidate), "--output", str(output)])
    if passes:
        main()
    else:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
    assert json.loads(output.read_text())["status"] == ("passed" if passes else "failed")
