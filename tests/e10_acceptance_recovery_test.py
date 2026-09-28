"""Recovery retains failed costs and immutable qualification/source boundaries."""

import copy

import pytest

from tools import recover_e10_acceptance as recovery


def test_failed_planning_cost_retains_every_prior_row_and_reservation():
    state = {
        "limits": {"node_hours_cap": 336, "final_requests_reserved": 5000},
        "work": {
            "historical/G0": {"status": "reserved", "seconds": 43200},
            "preparation/e10-acceptance-01/E10": {"end": 100},
            "unknown-request": {"status": "failed", "requests": 1},
        },
        "intervals": [[0, 100]],
    }
    before = copy.deepcopy(state)
    terminal = {
        "status": "blocked",
        "recorded_at": "1970-01-01T00:02:30+00:00",
        "error": {
            "message": "E10: artifact-dependent plan requires completed producer outputs; inspect or resume those producers first"
        },
    }
    result = recovery.reconcile(state, terminal)
    assert state == before
    assert result["limits"] == before["limits"]
    assert all(result["work"][key] == value for key, value in before["work"].items())
    assert result["work"][recovery.FAILED_WORK]["seconds"] == 50
    assert result["node_seconds"] == result["elapsed_seconds"] == 150
    with pytest.raises(ValueError, match="already accounted"):
        recovery.reconcile(result, terminal)
    terminal["status"] = "interrupted"
    with pytest.raises(ValueError, match="Unexpected parent failure"):
        recovery.reconcile(state, terminal)


def test_reuse_rejects_unreviewed_numerical_or_native_changes():
    old = dict(
        commit=recovery.OLD_SOURCE,
        native_code_sha256={"native": "same"},
        extraction_code={"files": {path: "old" for path in recovery.CHANGED_NUMERICAL_SCOPE}},
    )
    new = copy.deepcopy(old)
    new["extraction_code"]["files"] = {path: "new" for path in recovery.CHANGED_NUMERICAL_SCOPE}
    recovery.verify_source(new, old)
    new["extraction_code"]["files"]["exact/impl/trainer/fitting.py"] = "changed"
    with pytest.raises(ValueError, match="dependency changes"):
        recovery.verify_source(new, old)
    new = copy.deepcopy(old)
    new["native_code_sha256"]["native"] = "changed"
    with pytest.raises(ValueError, match="native packages changed"):
        recovery.verify_source(new, old)
