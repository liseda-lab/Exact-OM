"""A finished queue asks for the next study once, without diagnostic model calls."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from exact.experiments import notifications
from exact.experiments.completion import idle_completion


def completed():
    registry = dict(
        campaign="fixture", remaining_work_status="terminal", pending_batches=[], runs=[]
    )
    observation = dict(
        status="needs_attention",
        incidents=[],
        findings=[
            dict(run_id="done", step_id="123.4", status="complete", scheduler_state="MISSING")
        ],
    )
    return registry, observation


@pytest.mark.parametrize(
    "condition", ["pending", "unfinished", "live", "failed", "incident", "launching", "empty"]
)
def test_incomplete_queues_do_not_emit_completion(condition):
    registry, observation = completed()
    dispatch = {}
    if condition == "pending":
        registry["pending_batches"] = [dict(id="later")]
    if condition == "unfinished":
        registry["remaining_work_status"] = "pending"
    if condition == "live":
        observation["findings"][0]["scheduler_state"] = "RUNNING"
    if condition == "failed":
        observation["findings"][0]["status"] = "failed"
    if condition == "incident":
        observation["incidents"] = [dict(id="failure")]
    if condition == "launching":
        dispatch["next"] = dict(status="starting")
    if condition == "empty":
        observation["findings"] = []
    assert idle_completion(registry, observation, dispatch) is None


def test_monitor_queues_one_actionable_email_across_ticks_and_restart(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "idle_supervisor", Path(__file__).parents[1] / "tools/supervise_experiments.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    registry, observation = completed()
    cli.write(tmp_path / "registry.json", registry)
    monkeypatch.setattr(cli, "slurm_steps", lambda _: {"123.0": "RUNNING"})
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: copy.deepcopy(observation))
    monkeypatch.setattr(cli, "authenticate", lambda *a: pytest.fail("No model authentication"))
    monkeypatch.setattr(cli, "run_agent", lambda *a: pytest.fail("No model calls"))
    policy = dict(
        allocation="123",
        notify_when_idle=True,
        notifications=dict(
            enabled=True,
            transport="command",
            command=["/test/mail"],
            recipient="owner@example.org",
            sender="monitor@example.org",
        ),
    )
    state = dict(incidents={}, agent_runs=[])
    assert (
        cli.check(tmp_path, policy, state, act=False)["status"] == "completed_waiting_for_decision"
    )
    assert not list((tmp_path / "alerts").glob("*.json"))
    (tmp_path / "PAUSE").touch()
    assert cli.check(tmp_path, policy, state, act=True)["status"] == "paused"
    assert not list((tmp_path / "alerts").glob("*.json"))
    (tmp_path / "PAUSE").unlink()
    for _ in range(3):
        state = json.loads((tmp_path / "state.json").read_text())
        result = cli.check(tmp_path, policy, state, act=True)
        assert result["notification"]["outcome"] == "approval_needed"
    sent = []
    monkeypatch.setattr(notifications, "_deliver", lambda msg, conf: sent.append(msg))
    notifications.flush_notifications(tmp_path, policy["notifications"])
    cli.check(tmp_path, policy, state, act=True)
    notifications.flush_notifications(tmp_path, policy["notifications"])
    assert len(sent) == 1 and "queue has finished" in sent[0].get_content()
    assert len(list((tmp_path / "alerts").glob("*.json"))) == 1
    observation["findings"].append(dict(run_id="new-study", step_id="123.5", status="complete"))
    cli.check(tmp_path, policy, state, act=True)
    notifications.flush_notifications(tmp_path, policy["notifications"])
    assert len(sent) == 2
