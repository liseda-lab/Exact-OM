"""Exercise unattended control without starting Codex, Slurm, or numerical work."""

import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "tools/supervise_experiments.py"
    spec = importlib.util.spec_from_file_location("tested_supervisor_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controller(cli, tmp_path, monkeypatch):
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Preserve allocation, budgets and valid checkpoints.")
    policy = {
        "allocation": "14372",
        "repository": str(tmp_path),
        "instructions": str(instructions),
        "instructions_sha256": cli.digest(instructions),
        "codex": "/example/codex",
        "max_attempts_per_incident": 2,
        "max_agent_runs_per_day": 3,
        "agent_timeout_seconds": 60,
    }
    cli.write(tmp_path / "registry.json", {"runs": []})
    monkeypatch.setattr(cli, "slurm_steps", lambda _: {"14372.0": "RUNNING"})
    monkeypatch.setattr(cli, "authenticate", lambda _: None)
    state = {"incidents": {}, "agent_runs": []}
    return policy, state


def _observation(kind="run_failed"):
    return {
        "status": "needs_attention",
        "findings": [],
        "incidents": [{"id": "failure", "kind": kind, "run_ids": ["E09"]}],
        "retryable": False,
    }


def _result(outcome="repaired"):
    return {"outcome": outcome, "summary": "Checked", "steps": [], "commits": [], "handoff": ""}


def test_healthy_check_never_authenticates_or_invokes_agent(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    monkeypatch.setattr(
        cli, "inspect_runs", lambda *a, **k: {"status": "healthy", "incidents": [], "findings": []}
    )

    def forbidden(*args):
        pytest.fail("Healthy runs must consume no model invocation")

    monkeypatch.setattr(cli, "authenticate", forbidden)
    monkeypatch.setattr(cli, "run_agent", forbidden)
    assert cli.check(tmp_path, policy, state, act=True)["status"] == "healthy"
    assert state["agent_runs"] == []


def test_scheduler_failure_never_invokes_agent(cli, controller, tmp_path, monkeypatch):
    policy, state = controller

    def unavailable(_):
        raise RuntimeError("controller temporarily unavailable")

    monkeypatch.setattr(cli, "slurm_steps", unavailable)
    monkeypatch.setattr(cli, "run_agent", lambda *a: pytest.fail("Scheduler query failed"))
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        cli.check(tmp_path, policy, state, act=True)
    assert not state["agent_runs"]


@pytest.mark.parametrize(
    "record, prior_runs, expected",
    [
        ({"attempts": 0, "observations": 2, "needs_user": True}, [], "requires_user"),
        ({"attempts": 2, "observations": 2}, [], "incident_attempt_limit"),
        (
            {"attempts": 0, "observations": 2},
            [{"started_epoch": 100}] * 3,
            "daily_agent_limit",
        ),
        ({"attempts": 0, "observations": 1}, [], "confirm_missing_step_on_next_check"),
    ],
)
def test_attempt_limits_are_enforced(cli, controller, record, prior_runs, expected):
    policy, state = controller
    incident = {"id": "failure", "kind": "step_missing"}
    state["incidents"]["failure"] = record
    state["agent_runs"] = prior_runs
    assert cli.eligible(state, incident, policy, now=101) == (False, expected)


def test_old_attempts_age_out_of_daily_limit(cli, controller):
    policy, state = controller
    state["incidents"]["failure"] = {"attempts": 0, "observations": 2, "first_seen_epoch": 0}
    state["agent_runs"] = [{"started_epoch": 0}] * 3
    assert cli.eligible(state, {"id": "failure", "kind": "step_missing"}, policy, 86401)[0]


def test_explicit_pause_prevents_repair(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: _observation())
    monkeypatch.setattr(cli, "run_agent", lambda *a: pytest.fail("Explicitly paused"))
    (tmp_path / "PAUSE").touch()
    assert cli.check(tmp_path, policy, state, act=True)["status"] == "paused"
    assert state["agent_runs"] == []


def test_failed_agent_consumes_attempt_and_keeps_a_report(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: _observation())

    def failed(*args):
        raise RuntimeError("CLI child failed")

    monkeypatch.setattr(cli, "run_agent", failed)
    status = cli.check(tmp_path, policy, state, act=True)
    assert state["incidents"]["failure"]["attempts"] == 1
    assert state["agent_runs"][0]["status"] == "failed"
    assert cli.read(Path(status["report"]))["error"] == "RuntimeError: CLI child failed"


def test_needs_user_prevents_another_invocation(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: _observation())
    calls = []

    def report(*args):
        calls.append(args)
        return {"status": "complete", "result": _result("needs_user")}

    monkeypatch.setattr(cli, "run_agent", report)
    cli.check(tmp_path, policy, state, act=True)
    status = cli.check(tmp_path, policy, state, act=True)
    assert len(calls) == 1
    assert status["action"] == "requires_user"


def test_child_environment_has_no_api_fallback_or_experiment_secrets(cli, monkeypatch):
    for key in (*cli.API_OVERRIDES, "OPENROUTER_API_KEY", "EXACT_STOP_PATH", "PYTHONPATH"):
        monkeypatch.setenv(key, "must-not-propagate")
    monkeypatch.setenv("SLURM_JOB_ID", "14372")
    environment = cli.environment()
    assert environment["SLURM_JOB_ID"] == "14372"
    assert not set(cli.API_OVERRIDES) & set(environment)
    assert "OPENROUTER_API_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert not any(key.startswith("EXACT_") for key in environment)


def _fake_child(cli, monkeypatch, directory, *, result, code=0, completed=True, running=False):
    class Child:
        def __init__(self, command, **kwargs):
            self.stdin = io.StringIO()
            self.returncode = None if running else code
            self.signals = []
            self.command = command
            event = {
                "type": "turn.completed" if completed else "turn.failed",
                "usage": {"input_tokens": 9},
            }
            kwargs["stdout"].write(json.dumps(event) + "\n")
            (directory / "result.json").write_text(result)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def poll(self):
            return self.returncode

        def send_signal(self, value):
            self.signals.append(value)
            self.returncode = -2

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(cli.subprocess, "Popen", Child)


@pytest.mark.parametrize(
    "result,code,completed,expected",
    [
        (json.dumps(_result()), 0, True, "complete"),
        (json.dumps(_result()), 1, True, "failed"),
        (json.dumps(_result()), 0, False, "failed"),
        (json.dumps(["invalid result"]), 0, True, "failed"),
        (json.dumps({**_result(), "unexpected": True}), 0, True, "failed"),
        ('{"outcome":', 0, True, "failed"),
    ],
)
def test_agent_result_must_be_valid_and_keep_usage(
    cli, controller, tmp_path, monkeypatch, result, code, completed, expected
):
    policy, _ = controller
    _fake_child(cli, monkeypatch, tmp_path, result=result, code=code, completed=completed)
    report = cli.run_agent(policy, tmp_path, "Authorized task", lambda: False)
    assert report["status"] == expected
    assert report["exit_code"] == code
    assert report["usage"] == ([{"input_tokens": 9}] if completed else [])
    assert cli.read(tmp_path / "report.json") == report


def test_interrupted_child_records_exit_and_usage(cli, controller, tmp_path, monkeypatch):
    policy, _ = controller
    _fake_child(cli, monkeypatch, tmp_path, result=json.dumps(_result()), running=True)
    report = cli.run_agent(policy, tmp_path, "Authorized task", lambda: True)
    assert report["status"] == "failed"
    assert report["interrupted"] is True
    assert report["exit_code"] == -2
    assert report["usage"] == [{"input_tokens": 9}]


@pytest.mark.parametrize("invalid_result", [["invalid result"], "text", 42])
def test_invalid_child_result_cannot_break_controller_accounting(
    cli, controller, tmp_path, monkeypatch, invalid_result
):
    policy, state = controller
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: _observation())
    monkeypatch.setattr(cli, "run_agent", lambda *a: {"status": "failed", "result": invalid_result})
    assert cli.check(tmp_path, policy, state, act=True)["status"] == "intervention_finished"
    assert state["agent_runs"][0]["status"] == "failed"


def test_experiment_stop_file_prevents_repair(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    stopped = tmp_path / "experiment-STOP"
    stopped.touch()
    cli.write(tmp_path / "registry.json", {"runs": [], "pause_paths": [str(stopped)]})
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: _observation())
    monkeypatch.setattr(cli, "run_agent", lambda *a: pytest.fail("Experiment was stopped"))
    status = cli.check(tmp_path, policy, state, act=True)
    assert status["status"] == "paused"
    assert str(stopped) in status["paused_by"]
    assert state["agent_runs"] == []


def test_missing_step_requires_separated_consecutive_observations(
    cli, controller, tmp_path, monkeypatch
):
    policy, state = controller
    observations = iter(
        [
            _observation("step_missing"),
            _observation("step_missing"),
            {"status": "healthy", "incidents": [], "findings": []},
            _observation("step_missing"),
            _observation("step_missing"),
        ]
    )
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: next(observations))
    times = iter([100, 101, 102, 200, 261])
    monkeypatch.setattr(cli.time, "time", lambda: next(times))
    for _ in range(2):
        cli.check(tmp_path, policy, state)
    incident = {"id": "failure", "kind": "step_missing"}
    assert not cli.eligible(state, incident, policy, 101)[0]
    cli.check(tmp_path, policy, state)
    assert state["incidents"]["failure"]["observations"] == 0
    cli.check(tmp_path, policy, state)
    assert not cli.eligible(state, incident, policy, 200)[0]
    cli.check(tmp_path, policy, state)
    assert cli.eligible(state, incident, policy, 261)[0]


def test_interrupted_repair_blocks_automatic_retry(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: _observation())
    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda *a: {"status": "failed", "interrupted": True, "result": None},
    )
    cli.check(tmp_path, policy, state, act=True)
    assert state["incidents"]["failure"]["needs_user"] is True
    assert cli.check(tmp_path, policy, state, act=True)["action"] == "requires_user"


def test_changed_instructions_do_not_charge_or_start_an_attempt(
    cli, controller, tmp_path, monkeypatch
):
    policy, state = controller
    Path(policy["instructions"]).write_text("Modified after deployment")
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: _observation())
    monkeypatch.setattr(cli, "run_agent", lambda *a: pytest.fail("Unreviewed instructions"))
    with pytest.raises(ValueError, match="instructions changed"):
        cli.check(tmp_path, policy, state, act=True)
    assert state["incidents"]["failure"]["attempts"] == 0
    assert state["agent_runs"] == []
