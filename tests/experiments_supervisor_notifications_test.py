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
    assert len(sent) == 2 and len(state["agent_runs"]) == 1


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
    assert notifications.flush_notifications(tmp_path, policy["notifications"])["counts"] == {"failed": 2}
    monkeypatch.setattr(notifications.subprocess, "run", lambda *args, **kwargs: None)
    original_time = notifications.time.time()
    monkeypatch.setattr(notifications.time, "time", lambda: original_time + 61)
    assert notifications.flush_notifications(tmp_path, policy["notifications"])["counts"] == {"sent": 2}
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


def test_recovery_requires_healthy_successor_and_notifies_once(monitor, tmp_path, monkeypatch):
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
    outcomes = [json.loads(p.read_text())["outcome"] for p in (tmp_path / "alerts").glob("*.json")]
    assert sorted(outcomes) == ["problem_detected", "recovered"]
