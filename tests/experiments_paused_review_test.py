"""A single admitted validation can request review without resuming the campaign."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from exact.experiments import notifications
from exact.experiments.dispatch import dispatch_ready
from exact.experiments.paused_reviews import notify_paused_reviews


def bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "paused_review_supervisor", Path(__file__).parents[1] / "tools/supervise_experiments.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    run = {
        "id": "isolated-validation", "step_id": "123.2", "dispatch_nonce": "a" * 32,
        "status_path": str(tmp_path / "worker-status.json"),
        "completion_path": str(tmp_path / "completion.json"),
        "exit_path": str(tmp_path / "exit-code"),
        "require_result_review": True, "selection_eligible": False,
        "review_report_path": str(tmp_path / "results.json"),
    }
    cli.write(tmp_path / "recipe.json", {"scientific_execution": False, "maximum_pairs": 600})
    run["review_recipe"] = bind(tmp_path / "recipe.json")
    registry = {
        "runs": [run], "pending_batches": [], "remaining_work_status": "needs_user",
        "paused_monitor_allowlist": [{"run_id": run["id"], "dispatch_nonce": run["dispatch_nonce"]}],
    }
    cli.write(tmp_path / "registry.json", registry)
    (tmp_path / "PAUSE").write_text("Manual approval required; never resume automatically.")
    policy = {
        "allocation": "123", "max_attempts_per_incident": 2,
        "notifications": {"enabled": True, "transport": "command", "command": ["/test/mail"],
                          "recipient": "owner@example.org", "sender": "monitor@example.org"},
    }
    state, steps, clock = {"incidents": {}, "agent_runs": []}, {"123.0": "RUNNING", "123.2": "RUNNING"}, [1000.0]
    monkeypatch.setattr(cli, "slurm_steps", lambda _: dict(steps))
    monkeypatch.setattr(cli.time, "time", lambda: clock[0])
    monkeypatch.setattr(cli, "authenticate", lambda *a: pytest.fail("PAUSE forbids model authentication"))
    monkeypatch.setattr(cli, "run_agent", lambda *a: pytest.fail("PAUSE forbids repair"))
    monkeypatch.setattr(notifications, "_deliver", lambda *a: pytest.fail("Only enqueue during check"))

    def check(act=True):
        return cli.check(tmp_path, policy, state, act=act)

    def finish(status="complete", code=0):
        cli.write(tmp_path / "results.json", {"algorithmic_invariants": "passed", "pairs_per_second": 4.25})
        receipt = {"step_id": run["step_id"], "dispatch_nonce": run["dispatch_nonce"],
                   "recipe_sha256": run["review_recipe"]["sha256"], "status": status, "exit_code": code,
                   "review_report": bind(tmp_path / "results.json")}
        cli.write(tmp_path / "completion.json", receipt)
        steps.pop("123.2", None)
        return receipt

    return cli, run, registry, policy, state, steps, clock, check, finish


def alerts(tmp_path):
    return [json.loads(path.read_text()) for path in (tmp_path / "alerts").glob("*.json")]


def test_terminal_review_deduplicates_across_checks_and_restart_without_unpause(monitor, tmp_path, monkeypatch):
    cli, _, _, policy, state, _, _, check, finish = monitor
    before = (tmp_path / "registry.json").read_bytes()
    assert check()["paused_review_notifications"] == []
    finish()
    for _ in range(4):
        assert check()["status"] == "paused"
        state.clear()
        state.update(cli.read(tmp_path / "state.json"))
    queued = alerts(tmp_path)
    assert len(queued) == 1 and queued[0]["outcome"] == "approval_needed"
    assert "finished" in queued[0]["summary"] and "does not establish scientific admission" in queued[0]["summary"]
    assert queued[0]["attempts"] == 0 and state["agent_runs"] == []
    sent = []
    monkeypatch.setattr(notifications, "_deliver", lambda *a: sent.append(a))
    notifications.flush_notifications(tmp_path, policy["notifications"])
    check()
    notifications.flush_notifications(tmp_path, policy["notifications"])
    assert len(sent) == 1
    assert (tmp_path / "PAUSE").exists() and (tmp_path / "registry.json").read_bytes() == before
    assert dispatch_ready(tmp_path, "123", {"123.0": "RUNNING"}) == {"status": "paused"}


@pytest.mark.parametrize("change", ["not_allowed", "wrong_nonce", "no_review", "scientific", "foreign_allocation"])
def test_unapproved_runs_never_request_review(monitor, tmp_path, change):
    cli, run, registry, _, _, _, _, check, finish = monitor
    if change == "not_allowed":
        registry["paused_monitor_allowlist"] = []
    elif change == "wrong_nonce":
        registry["paused_monitor_allowlist"][0]["dispatch_nonce"] = "b" * 32
    elif change == "no_review":
        run["require_result_review"] = False
    elif change == "scientific":
        run["selection_eligible"] = True
    else:
        run["step_id"] = "999.2"
    cli.write(tmp_path / "registry.json", registry)
    finish()
    assert check()["status"] == "paused"
    assert alerts(tmp_path) == []


def test_owned_failure_requires_action_but_does_not_restart(monitor, tmp_path):
    _, _, _, _, state, _, _, check, finish = monitor
    finish("failed", 1)
    for _ in range(3):
        assert check()["status"] == "paused"
    assert len(alerts(tmp_path)) == 1 and "failed" in alerts(tmp_path)[0]["summary"]
    assert state["agent_runs"] == []


def test_missing_step_requires_running_then_two_consecutive_absences(monitor, tmp_path):
    _, _, _, _, _, steps, clock, check, _ = monitor
    steps.pop("123.2")
    for _ in range(3):
        check()
        clock[0] += 60
    assert alerts(tmp_path) == []
    steps["123.2"] = "RUNNING"
    check()
    steps.pop("123.2")
    check()
    clock[0] += 61
    steps["123.2"] = "UNKNOWN"
    check()  # A scheduler failure breaks consecutive absence evidence.
    steps.pop("123.2")
    check()
    assert alerts(tmp_path) == []
    clock[0] += 61
    check()
    assert len(alerts(tmp_path)) == 1
    assert "no automatic retry" in alerts(tmp_path)[0]["summary"]


@pytest.mark.parametrize("damage", ["nonce", "step", "recipe", "report_path", "report_hash", "nan", "no_status"])
def test_unusable_terminal_receipt_never_claims_finished(monitor, tmp_path, damage):
    cli, _, _, _, _, _, clock, check, finish = monitor
    check()
    receipt = finish()
    if damage == "nonce":
        receipt["dispatch_nonce"] = "b" * 32
    elif damage == "step":
        receipt["step_id"] = "123.3"
    elif damage == "recipe":
        (tmp_path / "recipe.json").write_text('{}')
    elif damage == "report_path":
        receipt["review_report"]["path"] = str(tmp_path / "unexpected.json")
    elif damage == "report_hash":
        receipt["review_report"]["sha256"] = None
    elif damage == "nan":
        (tmp_path / "results.json").write_text('{"measurement":{"value":1e999}}')
        receipt["review_report"] = bind(tmp_path / "results.json")
    else:
        receipt.pop("status")
    cli.write(tmp_path / "completion.json", receipt)
    check()
    assert alerts(tmp_path) == []
    clock[0] += 61
    check()
    assert len(alerts(tmp_path)) == 1
    assert "could not be verified" in alerts(tmp_path)[0]["summary"]
    assert "validation finished" not in alerts(tmp_path)[0]["summary"]


def test_read_only_and_lost_allocation_never_enqueue(monitor, tmp_path):
    _, _, _, _, _, steps, _, check, finish = monitor
    finish()
    assert check(act=False)["status"] == "paused"
    assert alerts(tmp_path) == []
    steps.clear()
    with pytest.raises(ValueError, match="Retained allocation"):
        check()
    assert alerts(tmp_path) == [] and (tmp_path / "PAUSE").exists()


@pytest.mark.parametrize("damage", ["nonce", "step", "allowlist", "allowlist_entry"])
def test_malformed_review_opt_in_cannot_derail_monitor(monitor, tmp_path, damage):
    _, run, registry, policy, state, _, _, _, _ = monitor
    if damage == "nonce":
        run["dispatch_nonce"] = None
        registry["paused_monitor_allowlist"][0]["dispatch_nonce"] = None
    elif damage == "step":
        run["step_id"] = []
    elif damage == "allowlist":
        registry["paused_monitor_allowlist"] = {"not": "a list"}
    else:
        registry["paused_monitor_allowlist"] = [None, [], 1]
    observation = {"findings": [{"run_id": run["id"], "step_id": run["step_id"]}]}
    assert notify_paused_reviews(tmp_path, policy, registry, observation, state, now=0) == []
    assert alerts(tmp_path) == []
