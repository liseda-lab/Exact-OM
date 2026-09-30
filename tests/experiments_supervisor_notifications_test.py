"""Intervention notifications must not stall independent experiments or read-only checks."""

import importlib.util
import subprocess
import json
from pathlib import Path

import pytest

from exact.experiments import notifications


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/supervise_experiments.py"
    spec = importlib.util.spec_from_file_location("notifying_supervisor_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Preserve retained jobs and scientific controls.")
    policy = {
        "allocation": "14372",
        "repository": str(tmp_path),
        "instructions": str(instructions),
        "instructions_sha256": cli.digest(instructions),
        "max_attempts_per_incident": 2,
        "max_agent_runs_per_day": None,
        "agent_timeout_seconds": 60,
        "notifications": {
            "transport": "command",
            "command": ["/example/cluster-mail"],
            "recipient": "owner@example.org",
            "sender": "cluster@example.org",
        },
    }
    state = {"incidents": {}, "agent_runs": []}
    incident = {
        "id": "failure",
        "kind": "run_failed",
        "run_ids": ["E10"],
        "reason": "Repeated extraction failure",
    }
    observation = {"status": "needs_attention", "findings": [], "incidents": [incident]}
    cli.write(tmp_path / "registry.json", {"runs": []})
    monkeypatch.setattr(cli, "slurm_steps", lambda _: {"14372.0": "RUNNING"})
    monkeypatch.setattr(cli, "authenticate", lambda _: None)
    monkeypatch.setattr(cli, "inspect_runs", lambda *args, **kwargs: observation)
    return cli, policy, state, observation


def _result(outcome):
    return {
        "status": "complete",
        "result": {
            "outcome": outcome,
            "summary": "Choose eligible ontology scope",
            "handoff": "note.md",
        },
    }


def test_decision_alert_is_immediate_and_not_resent(monitor, tmp_path, monkeypatch):
    cli, policy, state, _ = monitor
    sent = []
    monkeypatch.setattr(
        notifications.subprocess, "run", lambda *args, **kwargs: sent.append(kwargs)
    )
    monkeypatch.setattr(cli, "run_agent", lambda *args: _result("needs_user"))
    status = cli.check(tmp_path, policy, state, act=True)
    assert status["notification"]["delivery"] == "pending"
    assert sent == []  # Detection and approval alerts do not delay repair.
    notifications.flush_notifications(tmp_path, policy["notifications"])
    assert status["notification"]["handoff"] == "note.md"
    assert state["incidents"]["failure"]["needs_user"]
    cli.check(tmp_path, policy, state, act=True)
    notifications.flush_notifications(tmp_path, policy["notifications"])
    assert len(sent) == 1 and len(state["agent_runs"]) == 1


def test_last_failed_attempt_notifies_in_same_check(monitor, tmp_path, monkeypatch):
    cli, policy, state, _ = monitor
    state["incidents"]["failure"] = {"attempts": 1, "observations": 2}
    alerts = []
    monkeypatch.setattr(cli, "run_agent", lambda *args: {"status": "failed"})
    monkeypatch.setattr(
        cli,
        "notify_intervention",
        lambda *args, **kwargs: alerts.append(args) or {"delivery": "sent"},
    )
    status = cli.check(tmp_path, policy, state, act=True)
    assert [alert[2] for alert in alerts] == ["problem_detected", "incident_attempt_limit"]
    assert status["notification"]["delivery"] == "sent"
    assert state["incidents"]["failure"]["attempts"] == 2


def test_delivery_failure_retries_without_another_agent(monitor, tmp_path, monkeypatch):
    cli, policy, state, _ = monitor
    state["incidents"]["failure"] = {"attempts": 1, "observations": 2, "needs_user": True}
    monkeypatch.setattr(cli, "run_agent", lambda *args: pytest.fail("Needs user decision"))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("mail", 15)

    monkeypatch.setattr(notifications.subprocess, "run", timeout)
    first = cli.check(tmp_path, policy, state, act=True)
    assert first["notification"]["delivery"] == "pending"
    assert notifications.flush_notifications(tmp_path, policy["notifications"])["counts"] == {"failed": 1, "suppressed": 1}
    monkeypatch.setattr(notifications.subprocess, "run", lambda *args, **kwargs: None)
    original_time = notifications.time.time()
    monkeypatch.setattr(notifications.time, "time", lambda: original_time + 61)
    assert notifications.flush_notifications(tmp_path, policy["notifications"])["counts"] == {"sent": 1, "suppressed": 1}
    second = cli.check(tmp_path, policy, state, act=True)
    assert second["notification"]["delivery"] == "sent"
    assert second["notification"]["attempts"] == 2
    assert state["agent_runs"] == []


def test_independent_incident_proceeds_while_first_needs_user(monitor, tmp_path, monkeypatch):
    cli, policy, state, observation = monitor
    state["incidents"]["failure"] = {"attempts": 1, "observations": 2, "needs_user": True}
    observation["incidents"].append(
        {"id": "independent", "kind": "run_failed", "run_ids": ["E12"], "reason": "Worker failed"}
    )
    prompts = []
    monkeypatch.setattr(cli, "notify_intervention", lambda *args, **kwargs: {"delivery": "failed"})

    def repaired(_policy, _directory, prompt, _stop):
        prompts.append(prompt)
        return _result("repaired")

    monkeypatch.setattr(cli, "run_agent", repaired)
    status = cli.check(tmp_path, policy, state, act=True)
    assert status["outcome"] == "repaired" and len(prompts) == 1
    assert state["agent_runs"][0]["incident"] == "independent"
    assert state["incidents"]["failure"]["attempts"] == 1


def test_read_only_check_does_not_send_or_repair(monitor, tmp_path, monkeypatch):
    cli, policy, state, _ = monitor
    state["incidents"]["failure"] = {"attempts": 2, "observations": 2, "needs_user": True}
    monkeypatch.setattr(
        cli, "notify_intervention", lambda *args, **kwargs: pytest.fail("Read only")
    )
    monkeypatch.setattr(cli, "run_agent", lambda *args: pytest.fail("Read only"))
    assert cli.check(tmp_path, policy, state, act=False)["status"] == "needs_attention"
    assert not (tmp_path / "alerts").exists()


def test_recovery_requires_healthy_successor_without_email(monitor, tmp_path, monkeypatch):
    cli, policy, state, observation = monitor
    monkeypatch.setattr(cli, "run_agent", lambda *args: _result("repaired"))
    cli.check(tmp_path, policy, state, act=True)
    observation.update(status="healthy", incidents=[], findings=[])
    cli.check(tmp_path, policy, state, act=True)
    assert not state["incidents"]["failure"].get("recovered")
    cli.write(tmp_path / "registry.json", {"runs": [
        {"id": "E10", "superseded_by": "E10-recovery"}, {"id": "E10-recovery"}
    ]})
    observation["findings"] = [{"run_id": "E10-recovery", "status": "healthy"}]
    for _ in range(2):
        cli.check(tmp_path, policy, state, act=True)
    assert state["incidents"]["failure"]["recovered"]
    alerts = [json.loads(p.read_text()) for p in (tmp_path / "alerts").glob("*.json")]
    assert sorted(alert["outcome"] for alert in alerts) == ["problem_detected", "recovered"]
    assert all(alert["delivery"] == "suppressed" and alert["attempts"] == 0 for alert in alerts)


def test_uncertain_mail_is_diagnosed_by_bounded_repair(monitor, tmp_path, monkeypatch):
    cli, policy, state, observation = monitor
    alert = notifications.notify_intervention(
        tmp_path, {"id": "prior", "reason": "decision", "run_ids": []},
        "requires_user", "Decide", config=policy["notifications"], defer=True,
    )
    alert["delivery"] = "ambiguous"
    cli.write(Path(alert["path"]), alert)
    observation.update(status="healthy", incidents=[])
    prompts = []
    def repair(_policy, _directory, prompt, _stop):
        prompts.append(prompt)
        return _result("needs_user")
    monkeypatch.setattr(cli, "run_agent", repair)
    status = cli.check(tmp_path, policy, state, act=True)
    assert status["outcome"] == "needs_user"
    assert "notification_delivery_uncertain" in prompts[0]
    assert alert["path"] in prompts[0]
    assert len(state["agent_runs"]) == 1


def test_failed_repair_is_local_until_same_incident_attempts_are_exhausted(monitor, tmp_path, monkeypatch):
    cli, policy, state, _ = monitor
    monkeypatch.setattr(cli, "run_agent", lambda *args: {"status": "failed"})
    sent = []
    monkeypatch.setattr(notifications, "_deliver", lambda *args: sent.append(args))
    cli.check(tmp_path, policy, state, act=True)
    assert notifications.flush_notifications(tmp_path, policy["notifications"])["counts"] == {"suppressed": 1}
    assert not sent and state["incidents"]["failure"]["attempts"] == 1
    cli.check(tmp_path, policy, state, act=True)
    assert notifications.flush_notifications(tmp_path, policy["notifications"])["counts"] == {"suppressed": 1, "sent": 1}
    assert len(sent) == 1 and state["incidents"]["failure"]["attempts"] == 2


def test_supervisor_outage_requires_three_checks_and_fifteen_minutes(monitor):
    cli, _, state, _ = monitor
    error = "RuntimeError: Unable to query Slurm steps"
    incident, requires_user = cli.record_supervisor_error(state, error, 1000)
    assert not requires_user and incident["observations"] == 1
    _, requires_user = cli.record_supervisor_error(state, error, 1899)
    assert not requires_user
    _, requires_user = cli.record_supervisor_error(state, error, 1899)
    assert not requires_user
    incident, requires_user = cli.record_supervisor_error(state, error, 1900)
    assert requires_user and incident["first_seen_epoch"] == 1000
    # A distinct transient error does not inherit the previous outage window.
    incident, requires_user = cli.record_supervisor_error(state, "OSError: temporary NAS error", 2000)
    assert not requires_user and incident["observations"] == 1
    assert incident["first_seen_epoch"] == 2000
    # A slow second check is still insufficient on its own.
    _, requires_user = cli.record_supervisor_error(state, "OSError: temporary NAS error", 4000)
    assert not requires_user


def test_monitor_loop_queues_only_persistent_outage_and_resets_after_recovery(
    monitor, tmp_path, monkeypatch
):
    cli, policy, _, _ = monitor
    policy["interval_seconds"] = 450
    cli.write(tmp_path / "policy.json", policy)
    monkeypatch.setattr(cli.sys, "argv", ["supervisor", "--directory", str(tmp_path)])
    monkeypatch.setenv("SLURM_JOB_ID", policy["allocation"])
    monkeypatch.setenv("SLURM_STEP_ID", "1")
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    monkeypatch.setattr(cli.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(notifications, "_deliver", lambda *args: pytest.fail("Live mail"))
    clock = [1000.0]
    monkeypatch.setattr(cli.time, "time", lambda: clock[0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    checks = []

    def observe(*args, **kwargs):
        checks.append(clock[0])
        if len(checks) == 4:
            return {"status": "healthy"}
        if len(checks) == 5:
            (tmp_path / "STOP").touch()
        raise RuntimeError("controller temporarily unavailable")

    monkeypatch.setattr(cli, "check", observe)
    assert cli.main() == 0
    alerts = [json.loads(path.read_text()) for path in (tmp_path / "alerts").glob("*.json")]
    actionable, = [alert for alert in alerts if alert["delivery"] == "pending"]
    assert actionable["outcome"] == "supervisor_unavailable"
    assert actionable["handoff"] == str(tmp_path / "status.json")
    assert "Inspect the saved error" in actionable["summary"]
    assert all(alert["outcome"] != "supervisor_error" for alert in alerts)
    state = cli.read(tmp_path / "state.json")
    assert state["supervisor_error"]["observations"] == 1
    assert checks == [1000, 1450, 1900, 2350, 2800]
