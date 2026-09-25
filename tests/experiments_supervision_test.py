"""The hourly checker must be conservative about liveness and repair identities."""

import copy
import json

import pytest

from exact.experiments.supervision import assess_runs, inspect_runs, inspection_incident


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


def _recovery_runs():
    return [
        {
            "id": "E10-original",
            "step_id": "14372.12",
            "status_path": "/experiment/old-root/status.json",
            "enabled": False,
            "superseded_by": "E10-recovery-01",
        },
        {
            "id": "E10-recovery-01",
            "step_id": "14372.14",
            "status_path": "/experiment/recovered-root/status.json",
        },
    ]


def _failure_id(runs, message, *, error_type="ValueError"):
    active = next(run for run in runs if run.get("enabled", True))
    observed = {
        active["id"]: {
            "status": {"status": "blocked", "error": {"type": error_type, "message": message}}
        }
    }
    return assess_runs(runs, observed, step_states={})["incidents"][0]["id"]


def test_recovery_keeps_failure_identity_across_registered_ids_paths_and_steps():
    before = _recovery_runs()[:1]
    before[0].pop("superseded_by")
    before[0]["enabled"] = True
    first = _failure_id(
        before,
        "E10-original step 14372.12: invalid /experiment/old-root/E10/config.json at "
        "2026-09-25T03:15:19.13+00:00",
    )
    after = _failure_id(
        _recovery_runs(),
        "E10-recovery-01 step 14372.14: invalid /experiment/recovered-root/E10/config.json at "
        "2026-09-25T05:15:19.13+00:00",
    )
    assert first == after


def test_recovery_chain_includes_disabled_ancestors():
    runs = _recovery_runs()
    first = _failure_id(runs, "E05: base config changed after screening")
    runs[-1].update(enabled=False, superseded_by="E10-recovery-02")
    runs.append({"id": "E10-recovery-02", "step_id": "14372.15"})
    assert _failure_id(runs, "E05: base config changed after screening") == first
    incident = assess_runs(runs, {}, step_states={})["incidents"][0]
    assert incident["scope_run_id"] == "E10-original"
    assert incident["focus_run_id"] == "E10-recovery-02"


@pytest.mark.parametrize(
    "message,error_type",
    [
        ("E06: base config changed after screening", "ValueError"),
        ("E05: base config changed after screening", "KeyError"),
        ("E05: model fingerprint changed after screening", "ValueError"),
    ],
)
def test_different_diagnoses_do_not_share_retry_limits(message, error_type):
    runs = _recovery_runs()
    first = _failure_id(runs, "E05: base config changed after screening")
    assert _failure_id(runs, message, error_type=error_type) != first


def test_unrelated_runs_with_the_same_error_remain_independent():
    first = _failure_id([{"id": "E01", "step_id": "1.1"}], "Checksum mismatch")
    second = _failure_id([{"id": "E10", "step_id": "1.2"}], "Checksum mismatch")
    assert first != second


def test_normalization_preserves_unregistered_paths_and_distinct_files():
    runs = _recovery_runs()
    first = _failure_id(runs, "invalid /experiment/recovered-root/E10/config.json")
    assert _failure_id(runs, "invalid /experiment/recovered-root/E10/budget.json") != first
    assert _failure_id(runs, "invalid /experiment/recovered-root-other/E10/config.json") != first
    assert _failure_id(runs, "invalid /external/dataset-01/config.json") != _failure_id(
        runs, "invalid /external/dataset-02/config.json"
    )
    assert _failure_id(runs, "value 14372.140 is invalid") != _failure_id(
        runs, "value 14372.141 is invalid"
    )


@pytest.mark.parametrize(
    "runs",
    [
        [{"id": "a", "step_id": "1.1", "superseded_by": "missing"}],
        [{"id": "a", "step_id": "1.1", "superseded_by": "a"}],
        [
            {"id": "a", "step_id": "1.1", "superseded_by": "b"},
            {"id": "b", "step_id": "1.2", "superseded_by": "a"},
        ],
        [
            {"id": "a", "step_id": "1.1", "superseded_by": "c"},
            {"id": "b", "step_id": "1.2", "superseded_by": "c"},
            {"id": "c", "step_id": "1.3"},
        ],
    ],
)
def test_invalid_recovery_lineages_are_rejected(runs):
    with pytest.raises(ValueError):
        assess_runs(runs, {}, step_states={})


def test_waiting_child_follows_replaced_upstream_incident():
    runs = _recovery_runs()
    runs.append({"id": "E01", "step_id": "14372.13", "depends_on": ["E10-original"]})
    observed = {
        "E10-recovery-01": {"status": {"status": "blocked", "error": "Configuration mismatch"}},
        "E01": {"status": {"status": "waiting_dependency"}},
    }
    result = assess_runs(runs, observed, step_states={"14372.13": "RUNNING"})
    assert len(result["incidents"]) == 1
    assert result["incidents"][0]["run_ids"] == ["E10-recovery-01", "E01"]
    assert result["findings"][1]["incident_id"] == result["incidents"][0]["id"]


def test_recovery_cannot_introduce_effective_dependency_cycle():
    runs = _recovery_runs()
    runs[-1]["depends_on"] = ["E01"]
    runs.append({"id": "E01", "step_id": "14372.13", "depends_on": ["E10-original"]})
    with pytest.raises(ValueError, match="Recovered run dependencies contain a cycle"):
        assess_runs(runs, {}, step_states={})


def test_different_reason_without_exception_does_not_share_retry_limit():
    runs = [{"id": "E01", "step_id": "1.1"}]
    one = assess_runs(
        runs, {"E01": {"status": {"status": "blocked", "reason": "budget"}}}, step_states={}
    )
    two = assess_runs(
        runs,
        {"E01": {"status": {"status": "blocked", "reason": "scientific decision"}}},
        step_states={},
    )
    assert one["incidents"][0]["id"] != two["incidents"][0]["id"]


def test_unreadable_evidence_identity_survives_recovery_and_distinguishes_errors():
    runs = _recovery_runs()
    before = inspection_incident(
        runs,
        {
            "run_id": "E10-original",
            "reason": "Receipt temporarily unreadable",
            "errors": ["status: PermissionError: /experiment/old-root/status.json"],
        },
    )
    finding = {
        "run_id": "E10-recovery-01",
        "reason": "Receipt temporarily unreadable",
        "errors": ["status: PermissionError: /experiment/recovered-root/status.json"],
    }
    after = inspection_incident(runs, finding)
    assert before["id"] == after["id"]
    assert after["scope_run_id"] == "E10-original"
    finding["errors"] = ["status: ValueError: metadata exceeds 1 MiB"]
    assert inspection_incident(runs, finding)["id"] != before["id"]
