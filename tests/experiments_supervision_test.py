"""The hourly checker must be conservative about liveness and repair identities."""

import copy
import json

import pytest

from exact.experiments.supervision import assess_runs, inspect_runs


def _runs():
    return [
        {"id": "E09", "step_id": "14372.5"},
        {"id": "B1", "step_id": "14372.6", "depends_on": ["E09"]},
    ]


def _observed():
    return {
        "E09": {"status": {"status": "running", "recorded_at": "2020-01-01T00:00:00Z"}},
        "B1": {"status": {"status": "waiting_for_verified_E09"}},
    }


def test_quiet_live_projection_and_waiting_dependency_are_healthy():
    result = assess_runs(
        _runs(), _observed(), step_states={"14372.5": "RUNNING", "14372.6": "RUNNING"}
    )
    assert result["incidents"] == []
    assert [r["status"] for r in result["findings"]] == ["healthy", "waiting"]


def test_dead_step_overrides_stale_running_status_and_focuses_upstream():
    result = assess_runs(_runs(), _observed(), step_states={"14372.6": "RUNNING"})
    assert result["status"] == "needs_attention"
    assert len(result["incidents"]) == 1
    assert result["incidents"][0]["kind"] == "step_missing"
    assert result["incidents"][0]["run_ids"] == ["E09", "B1"]


def test_missing_all_receipts_still_detects_dead_step():
    result = assess_runs(_runs()[:1], {}, step_states={})
    assert result["incidents"][0]["kind"] == "step_missing"


def test_dependency_failure_deduplicates_but_independent_failure_does_not():
    observed = _observed()
    observed["E09"]["status"].update(
        status="blocked", error={"type": "ValueError", "message": "budget"}
    )
    observed["B1"]["status"].update(status="blocked", error={"message": "Upstream E09 stopped"})
    result = assess_runs(_runs(), observed, step_states={})
    assert len(result["incidents"]) == 1
    assert result["incidents"][0]["focus_run_id"] == "E09"
    observed["B1"]["status"]["error"] = {"message": "Checksum mismatch"}
    assert len(assess_runs(_runs(), observed, step_states={})["incidents"]) == 2


def test_same_failure_keeps_identity_across_timestamps_and_resubmitted_step():
    observed = _observed()
    observed["E09"]["status"].update(
        status="blocked",
        error={"type": "Error", "message": "Failed at 2026-09-25T03:15:19.13+00:00"},
    )
    before = assess_runs(_runs(), observed, step_states={})
    changed = copy.deepcopy(observed)
    changed["E09"]["status"]["recorded_at"] = "later"
    changed["E09"]["status"]["error"]["message"] = "Failed at 2026-09-25T05:15:19.13+00:00"
    runs = _runs()
    runs[0]["step_id"] = "14372.7"
    after = assess_runs(runs, changed, step_states={})
    assert before["incidents"][0]["id"] == after["incidents"][0]["id"]


@pytest.mark.parametrize("exit_code", [1, 130, 137])
def test_nonzero_launcher_exit_requires_attention(exit_code):
    observed = {"E09": {"exit_code": exit_code}}
    result = assess_runs(_runs()[:1], observed, step_states={})
    assert result["incidents"][0]["kind"] == "launcher_failed"


def test_zero_exit_without_success_receipt_is_incomplete():
    observed = {"E09": {"exit_code": 0}}
    assert (
        assess_runs(_runs()[:1], observed, step_states={})["incidents"][0]["kind"] == "step_missing"
    )


def test_completion_waits_until_slurm_step_releases_before_next_batch():
    observed = {name: {"completion": {"status": "complete"}} for name in ("E09", "B1")}
    live = assess_runs(_runs(), observed, step_states={"14372.6": "COMPLETING"})
    assert live["incidents"] == []
    done = assess_runs(_runs(), observed, step_states={})
    assert done["incidents"][0]["kind"] == "next_batch"
    observed["E09"]["completion"]["recorded_at"] = "later"
    assert done["incidents"] == assess_runs(_runs(), observed, step_states={})["incidents"]
    assert assess_runs([], {}, step_states={})["incidents"] == []


@pytest.mark.parametrize("step_states", [None, {"14372.5": "UNKNOWN"}])
def test_scheduler_error_cannot_trigger_repair(step_states):
    result = assess_runs(_runs()[:1], _observed(), step_states=step_states)
    assert result["retryable"]
    assert result["incidents"] == []


def test_malformed_receipt_is_retryable_not_failure(tmp_path):
    status = tmp_path / "status.json"
    status.write_text('{"status":')
    runs = [{**_runs()[0], "status_path": str(status)}]
    result = inspect_runs(runs, step_states={})
    assert result["retryable"]
    assert result["incidents"] == []
    status.write_text(json.dumps({"status": "running"}))
    assert inspect_runs(runs, step_states={})["incidents"][0]["kind"] == "step_missing"


@pytest.mark.parametrize(
    "completion", [{"status": "running"}, {"status": "complete", "exit_code": "unknown"}]
)
def test_incomplete_receipt_does_not_authorize_next_batch(completion):
    observed = {"E09": {"completion": completion}}
    result = assess_runs(_runs()[:1], observed, step_states={})
    assert result["retryable"]
    assert result["incidents"] == []


def test_disabled_historical_launch_is_not_repaired():
    runs = _runs()
    runs[0]["enabled"] = False
    result = assess_runs(runs, _observed(), step_states={"14372.6": "RUNNING"})
    assert [r["run_id"] for r in result["findings"]] == ["B1"]
    assert result["incidents"] == []


@pytest.mark.parametrize(
    "runs",
    [
        [{"id": "same", "step_id": "1"}, {"id": "same", "step_id": "2"}],
        [{"id": "x", "step_id": "1", "depends_on": ["missing"]}],
        [{"id": "x", "step_id": "1", "depends_on": ["x"]}],
    ],
)
def test_invalid_registry_is_rejected(runs):
    with pytest.raises(ValueError):
        assess_runs(runs, {}, step_states={})


def test_oversized_metadata_is_bounded_and_retryable(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(" " * (1024 * 1024 + 1))
    result = inspect_runs([{**_runs()[0], "completion_path": str(path)}], step_states={})
    assert result["retryable"] and not result["incidents"]


@pytest.mark.parametrize(
    "completion", [{"status": "failed"}, {"status": "complete", "exit_code": 1}]
)
def test_explicit_failed_completion_requires_diagnosis(completion):
    result = assess_runs(_runs()[:1], {"E09": {"completion": completion}}, step_states={})
    assert result["incidents"][0]["kind"] == "completion_failed"
