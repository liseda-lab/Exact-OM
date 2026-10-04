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
        (json.dumps({**_result(), "steps": ["14408.334: legacy launch explanation"]}),
         0, True, "complete"),
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
    # Guide new responses without invalidating historical prose-bearing results;
    # preparation credit is decided separately from receipt evidence.
    steps = cli.read(tmp_path / "schema.json")["properties"]["steps"]
    assert steps["items"] == {"type": "string"}
    assert "Slurm step IDs" in steps["description"]
    assert "14408.334" in steps["description"] and "Use []" in steps["description"]


def test_interrupted_child_records_exit_and_usage(cli, controller, tmp_path, monkeypatch):
    policy, _ = controller
    _fake_child(cli, monkeypatch, tmp_path, result=json.dumps(_result()), running=True)
    report = cli.run_agent(policy, tmp_path, "Authorized task", lambda: True)
    assert report["status"] == "failed"
    assert report["interrupted"] is True
    assert report["exit_code"] == -2
    assert report["usage"] == [{"input_tokens": 9}]


def test_long_intervention_keeps_five_minute_observations(cli, controller, tmp_path, monkeypatch):
    policy, _ = controller
    policy.update(interval_seconds=300, agent_timeout_seconds=3600)
    _fake_child(cli, monkeypatch, tmp_path, result=json.dumps(_result()), running=True)
    clock = iter([0, 301, 301])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    observed = []
    monkeypatch.setattr(cli, "refresh_progress", lambda p, d: observed.append(d))
    cli.run_agent(policy, tmp_path, "Authorized task", lambda: True)
    assert observed == [tmp_path.parent.parent]


def test_intervention_observation_cannot_invoke_model(cli, controller, tmp_path, monkeypatch):
    policy, _ = controller
    cli.write(tmp_path / "status.json", {"status": "repairing", "intervention": "saved"})
    monkeypatch.setattr(
        cli, "run_agent", lambda *a: pytest.fail("Observation must remain deterministic")
    )
    monkeypatch.setattr(
        cli, "authenticate", lambda *a: pytest.fail("No model preflight on observation")
    )
    cli.refresh_progress(policy, tmp_path)
    assert cli.read(tmp_path / "status.json")["status"] == "repairing"
    assert (tmp_path / "health.json").exists()


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


@pytest.mark.parametrize("explicit_null", [False, True])
def test_distinct_incidents_have_no_default_daily_limit(cli, controller, explicit_null):
    policy, state = controller
    policy.pop("max_agent_runs_per_day")
    if explicit_null:
        policy["max_agent_runs_per_day"] = None
    state["agent_runs"] = [{"started_epoch": 100}] * 100
    state["incidents"]["new-error"] = {"attempts": 0, "observations": 1}
    assert cli.eligible(state, {"id": "new-error", "kind": "run_failed"}, policy, 101) == (
        True,
        "eligible",
    )
    state["incidents"]["new-error"]["attempts"] = 2
    assert cli.eligible(state, {"id": "new-error", "kind": "run_failed"}, policy, 101) == (
        False,
        "incident_attempt_limit",
    )


@pytest.mark.parametrize("limit", [None, 3])
def test_policy_accepts_optional_daily_limit(cli, controller, limit):
    policy, _ = controller
    policy.update(interval_seconds=300, max_agent_runs_per_day=limit)
    cli.validate_policy(policy)
    policy.pop("max_agent_runs_per_day")
    cli.validate_policy(policy)


@pytest.mark.parametrize("limit", [0, -1, True, "unlimited"])
def test_policy_rejects_invalid_daily_limit(cli, controller, limit):
    policy, _ = controller
    policy.update(interval_seconds=300, max_agent_runs_per_day=limit)
    with pytest.raises(ValueError, match="Daily repair limit"):
        cli.validate_policy(policy)


@pytest.fixture
def authentication_policy(cli, tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('model = "gpt-6-astra"\nmodel_reasoning_effort = "max"\n')
    policy = {
        "codex": "/example/codex",
        "codex_config": str(config),
        "codex_config_sha256": cli.digest(config),
        "model_reasoning_effort": "xhigh",
    }
    calls = []

    def login(command, **kwargs):
        calls.append(command)
        return cli.subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")

    monkeypatch.setattr(cli.subprocess, "run", login)
    return policy, config, calls


def test_semantic_config_accepts_only_inert_or_overridden_changes(cli, authentication_policy):
    policy, config, calls = authentication_policy
    policy["codex_config_semantic_sha256"] = cli.config_fingerprint(policy)
    config.write_text(
        '# Reordered and reformatted by the CLI\n'
        'model_reasoning_effort="xhigh"\nmodel="gpt-6-astra"\n'
        '[tui]\nnotifications=false\n[notice]\nhide_model_warning=true\n'
    )
    cli.authenticate(policy)
    assert calls == [[
        "/example/codex", "-c", 'forced_login_method="chatgpt"',
        "-c", 'model_provider="openai"', "login", "status",
    ]]


@pytest.mark.parametrize("change", [
    'model = "different-model"\n',
    'model_provider = "another-provider"\n',
    'service_tier = "priority"\n',
    'forced_login_method = "api"\n',
    'approval_policy = "never"\n',
    'sandbox_mode = "danger-full-access"\n',
    'personality = "friendly"\n',
    'sqlite_home = "/different/history"\n',
    'unknown_future_setting = true\n',
    '[projects."/repo"]\ntrust_level = "trusted"\n',
    '[model_providers.custom]\nbase_url = "https://example.test"\n',
    '[plugins.example]\nenabled = true\n',
    '[memories]\nenabled = true\n',
])
def test_semantic_config_rejects_other_changes_before_login(cli, authentication_policy, change):
    policy, config, calls = authentication_policy
    policy["codex_config_semantic_sha256"] = cli.config_fingerprint(policy)
    original = config.read_text()
    config.write_text(change if change.startswith('model =') else original + change)
    with pytest.raises(ValueError, match="Codex configuration changed"):
        cli.authenticate(policy)
    assert calls == []


@pytest.mark.parametrize("explicit_first", [False, True])
def test_semantic_config_accepts_explicit_default_tier(cli, authentication_policy, explicit_first):
    policy, config, calls = authentication_policy
    original = config.read_text()
    explicit = original + 'service_tier = "default"\n'
    config.write_text(explicit if explicit_first else original)
    policy["codex_config_semantic_sha256"] = cli.config_fingerprint(policy)
    config.write_text(original if explicit_first else explicit)
    cli.authenticate(policy)
    assert len(calls) == 1


def test_semantic_config_retains_inherited_effort(cli, authentication_policy):
    policy, config, calls = authentication_policy
    policy.pop("model_reasoning_effort")
    policy["codex_config_semantic_sha256"] = cli.config_fingerprint(policy)
    config.write_text(config.read_text().replace('"max"', '"xhigh"'))
    with pytest.raises(ValueError, match="Codex configuration changed"):
        cli.authenticate(policy)
    assert calls == []


def test_semantic_config_binds_effective_effort_override(cli, authentication_policy):
    policy, _, calls = authentication_policy
    policy["codex_config_semantic_sha256"] = cli.config_fingerprint(policy)
    policy["model_reasoning_effort"] = "low"
    with pytest.raises(ValueError, match="Codex configuration changed"):
        cli.authenticate(policy)
    assert calls == []


def test_legacy_config_pin_still_requires_exact_file(cli, authentication_policy):
    policy, config, calls = authentication_policy
    cli.authenticate(policy)
    calls.clear()
    config.write_text(config.read_text() + "# Harmless but not explicitly migrated\n")
    with pytest.raises(ValueError, match="Codex configuration changed"):
        cli.authenticate(policy)
    assert calls == []


def test_semantic_config_still_requires_chatgpt_login(cli, authentication_policy, monkeypatch):
    policy, _, _ = authentication_policy
    policy["codex_config_semantic_sha256"] = cli.config_fingerprint(policy)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k:
                        cli.subprocess.CompletedProcess(a[0], 0, "Logged in using API key", ""))
    with pytest.raises(ValueError, match="API fallback is disabled"):
        cli.authenticate(policy)


@pytest.mark.parametrize("remaining", ["pending", "complete", "deferred", "needs_user", None])
def test_empty_pending_queue_only_falls_back_for_explicit_unfinished_scope(
    cli, controller, tmp_path, monkeypatch, remaining
):
    policy, state = controller
    cli.write(tmp_path / "registry.json", {
        "runs": [], "pending_batches": [], "capacity": {}, "remaining_work_status": remaining
    })
    observation = {"status": "complete", "findings": [], "incidents": [
        {"id": "next", "kind": "next_batch", "run_ids": [], "reason": "All complete"}
    ]}
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: observation)
    cli.check(tmp_path, policy, state)
    assert bool(cli.read(tmp_path / "health.json")["incidents"]) == (remaining == "pending")


def test_prepared_batch_handoff_never_invokes_model(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    cli.write(tmp_path / 'registry.json', {
        'runs': [], 'pending_batches': [{'id': 'prepared', 'launch': {'reviewed': True}}],
        'capacity': {},
    })
    monkeypatch.setattr(cli, 'authenticate', lambda *a: pytest.fail('Prepared dispatch needs no model'))
    monkeypatch.setattr(cli, 'run_agent', lambda *a: pytest.fail('Prepared dispatch needs no model'))
    result = cli.check(tmp_path, policy, state, act=True)
    assert not state['agent_runs']
    assert result['status'] != 'repairing'


def test_dispatch_poll_is_independent_and_retries_errors(cli, controller, tmp_path, monkeypatch):
    policy, _ = controller
    calls = []

    class Stop:
        def is_set(self):
            return len(calls) == 2

        def wait(self, seconds):
            assert seconds == 15

    def dispatch(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise OSError('temporary storage failure')
        return {'status': 'registered', 'step_id': '14372.36'}

    monkeypatch.setattr(cli, 'dispatch_ready', dispatch)
    monkeypatch.setattr(cli, 'authenticate', lambda *a: pytest.fail('Dispatch cannot invoke a model'))
    cli.dispatch_worker(tmp_path, policy, Stop())
    assert len(calls) == 2
    assert cli.read(tmp_path / 'dispatch-status.json')['status'] == 'registered'


def test_ready_prepared_launch_defers_unprepared_science_but_allows_metadata(
    cli, controller, tmp_path, monkeypatch
):
    policy, state = controller
    cli.write(tmp_path / 'registry.json', {
        'runs': [], 'capacity': {'gpus': 1, 'cpus': 6},
        'pending_batches': [
            {'id': 'unprepared-science', 'resources': {'gpus': 1}},
            {'id': 'prepared-science', 'resources': {'gpus': 1}, 'launch': {'reviewed': True}},
            {'id': 'metadata', 'preparation_only': True, 'resources': {'gpus': 0, 'cpus': 1}},
        ],
    })
    prompts = []
    monkeypatch.setattr(cli, 'run_agent', lambda p, d, prompt, stop: (
        prompts.append(prompt) or {'status': 'complete', 'result': _result('no_change')}
    ))
    cli.check(tmp_path, policy, state, act=True)
    health = cli.read(tmp_path / 'health.json')
    assert [item['batch_id'] for item in health['incidents']] == ['metadata']
    assert len(prompts) == 1
    assert state['agent_runs'][0]['incident'] == health['incidents'][0]['id']


@pytest.mark.parametrize('reservation,suppressed', [
    ('reserved', True), ('starting', True), ('failed', True),
    ('registered', False), ('resolved', False),
])
def test_unregistered_prepared_reservation_holds_science_admission(
    cli, controller, tmp_path, reservation, suppressed
):
    policy, state = controller
    cli.write(tmp_path / 'registry.json', {
        'runs': [], 'capacity': {'gpus': 1},
        'pending_batches': [{'id': 'unprepared-science', 'resources': {'gpus': 1}}],
    })
    cli.write(tmp_path / 'dispatch-state.json', {
        'already-reserved': {'status': reservation, 'descriptor_sha256': 'bound', 'error': 'launch fault'},
    })
    cli.check(tmp_path, policy, state)
    incidents = cli.read(tmp_path / 'health.json')['incidents']
    assert any(item['kind'] == 'next_batch' for item in incidents) is not suppressed
    if reservation == 'failed':
        assert any(item['kind'] == 'dispatch_failed' for item in incidents)


def test_prepared_priority_preserves_real_failure_repair(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    cli.write(tmp_path / 'registry.json', {
        'runs': [], 'capacity': {'gpus': 1},
        'pending_batches': [
            {'id': 'unprepared-science', 'resources': {'gpus': 1}},
            {'id': 'prepared-science', 'resources': {'gpus': 1}, 'launch': {'reviewed': True}},
        ],
    })
    monkeypatch.setattr(cli, 'inspect_runs', lambda *a, **k: _observation())
    called = []
    monkeypatch.setattr(cli, 'run_agent', lambda *args: (
        called.append(args) or {'status': 'complete', 'result': _result()}
    ))
    cli.check(tmp_path, policy, state, act=True)
    assert len(called) == 1
    assert state['agent_runs'][0]['incident'] == 'failure'
    assert cli.read(tmp_path / 'health.json')['incidents'][0]['kind'] == 'run_failed'


def test_unready_prepared_descriptor_does_not_starve_independent_science(cli, controller, tmp_path):
    policy, state = controller
    cli.write(tmp_path / 'registry.json', {
        'runs': [], 'capacity': {'gpus': 1},
        'pending_batches': [
            {'id': 'unprepared-science', 'resources': {'gpus': 1}},
            {'id': 'prepared-science', 'depends_on': ['missing-parent'],
             'resources': {'gpus': 1}, 'launch': {'reviewed': True}},
        ],
    })
    cli.check(tmp_path, policy, state)
    assert [item['batch_id'] for item in cli.read(tmp_path / 'health.json')['incidents']] == ['unprepared-science']


def test_reserved_prepared_launch_suppresses_generic_empty_queue_continuation(
    cli, controller, tmp_path, monkeypatch
):
    policy, state = controller
    cli.write(tmp_path / 'registry.json', {
        'runs': [], 'capacity': {}, 'pending_batches': [], 'remaining_work_status': 'pending',
    })
    cli.write(tmp_path / 'dispatch-state.json', {'reserved': {'status': 'starting'}})
    monkeypatch.setattr(cli, 'inspect_runs', lambda *a, **k: _observation('next_batch'))
    cli.check(tmp_path, policy, state)
    assert not cli.read(tmp_path / 'health.json')['incidents']


@pytest.mark.parametrize('stopped,reason', [(False, 'timeout'), (True, 'stop_requested')])
def test_agent_reports_timeout_separately_from_stop_and_persists_deadline(
    cli, controller, tmp_path, monkeypatch, stopped, reason
):
    policy, _ = controller
    _fake_child(cli, monkeypatch, tmp_path, result=json.dumps(_result()), running=True)
    clock = iter([0, 61])
    monkeypatch.setattr(cli.time, 'monotonic', lambda: next(clock))
    (tmp_path / 'HANDOFF.md').write_text('Prepared work; not submitted.')
    report = cli.run_agent(policy, tmp_path, 'Authorized repair', lambda: stopped)
    assert report['interruption_reason'] == reason
    assert report['interrupted'] and report['status'] == 'failed'
    assert report['handoff'] == str(tmp_path / 'HANDOFF.md')
    prompt = (tmp_path / 'prompt.md').read_text()
    assert report['deadline_at'] in prompt
    assert 'Begin final handoff and result by:' in prompt
    assert str(tmp_path / 'HANDOFF.md') in prompt
    assert 'Distinguish saved preparation from queued/submitted work' in prompt


def test_timeout_uses_remaining_attempt_for_saved_work_reconciliation(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    monkeypatch.setattr(cli, 'inspect_runs', lambda *a, **k: _observation())
    prompts = []

    def repair(policy, directory, prompt, stopped):
        prompts.append(prompt)
        if len(prompts) == 1:
            (directory / 'HANDOFF.md').write_text('Committed repair; descriptor not queued.')
            return {'status': 'failed', 'interrupted': True, 'interruption_reason': 'timeout', 'result': None}
        assert str(tmp_path / 'interventions/failure-1') in prompt
        assert 'reconcile its saved HANDOFF.md' in prompt
        assert '"interruption_reason": "timeout"' in prompt
        return {'status': 'complete', 'result': _result()}

    monkeypatch.setattr(cli, 'run_agent', repair)
    cli.check(tmp_path, policy, state, act=True)
    assert state['incidents']['failure']['attempts'] == 1
    assert not state['incidents']['failure'].get('needs_user')
    cli.check(tmp_path, policy, state, act=True)
    assert len(prompts) == 2
    assert state['incidents']['failure']['attempts'] == 2


def test_repeated_timeouts_escalate_at_existing_attempt_limit(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    monkeypatch.setattr(cli, 'inspect_runs', lambda *a, **k: _observation())
    calls = []
    monkeypatch.setattr(cli, 'run_agent', lambda *args: calls.append(args) or {
        'status': 'failed', 'interrupted': True, 'interruption_reason': 'timeout', 'result': None,
    })
    cli.check(tmp_path, policy, state, act=True)
    result = cli.check(tmp_path, policy, state, act=True)
    assert result['notification']['outcome'] == 'incident_attempt_limit'
    assert cli.check(tmp_path, policy, state, act=True)['action'] == 'incident_attempt_limit'
    assert len(calls) == state['incidents']['failure']['attempts'] == 2


@pytest.mark.parametrize('reason', ['stop_requested', None])
def test_user_stop_and_unknown_interruption_still_require_intervention(
    cli, controller, tmp_path, monkeypatch, reason
):
    policy, state = controller
    monkeypatch.setattr(cli, 'inspect_runs', lambda *a, **k: _observation())
    calls = []
    monkeypatch.setattr(cli, 'run_agent', lambda *args: calls.append(args) or {
        'status': 'failed', 'interrupted': True, 'interruption_reason': reason, 'result': None,
    })
    cli.check(tmp_path, policy, state, act=True)
    assert cli.check(tmp_path, policy, state, act=True)['action'] == 'requires_user'
    assert len(calls) == 1


def test_explicit_decision_remains_blocking_even_if_repair_timed_out(cli, controller, tmp_path, monkeypatch):
    policy, state = controller
    monkeypatch.setattr(cli, 'inspect_runs', lambda *a, **k: _observation())
    monkeypatch.setattr(cli, 'run_agent', lambda *args: {
        'status': 'failed', 'interrupted': True, 'interruption_reason': 'timeout',
        'result': _result('needs_user'),
    })
    cli.check(tmp_path, policy, state, act=True)
    assert state['incidents']['failure']['needs_user'] is True


@pytest.fixture
def queued_recovery(cli, controller, tmp_path, monkeypatch):
    import hashlib

    policy, state = controller
    worker = tmp_path / 'reviewed-worker.sh'
    worker.write_text('#!/bin/bash\nexit 0\n')
    cli.write(tmp_path / 'old-status.json', {'status': 'blocked', 'error': 'saved error'})
    cli.write(tmp_path / 'busy-status.json', {'status': 'running'})
    failed = {'id': 'failed', 'step_id': '14372.34', 'pending_recovery': 'recovery',
              'status_path': str(tmp_path / 'old-status.json')}
    busy = {'id': 'busy', 'step_id': '14372.38', 'status_path': str(tmp_path / 'busy-status.json'),
            'resources': {'gpus': 1, 'cpus': 6}}
    batch = {'id': 'recovery', 'resources': {'gpus': 1, 'cpus': 6}, 'depends_on': ['busy'], 'launch': {
        'run': {'id': 'recovery', 'status_path': str(tmp_path / 'new-status.json'),
                'completion_path': str(tmp_path / 'new-complete.json'), 'exit_path': str(tmp_path / 'exit')},
        'argv': ['/usr/bin/srun', '--jobid=14372', '/bin/bash', str(worker)],
        'nonce': 'reviewed-recovery-001', 'tmux_socket': str(tmp_path / 'socket'),
        'step_path': str(tmp_path / 'step.json'), 'launcher_log': str(tmp_path / 'launcher.log'),
        'bindings': [{'path': str(worker), 'sha256': hashlib.sha256(worker.read_bytes()).hexdigest()}],
    }}
    registry = {'runs': [failed, busy], 'pending_batches': [batch], 'capacity': {'gpus': 1, 'cpus': 6}}
    cli.write(tmp_path / 'registry.json', registry)
    monkeypatch.setattr(cli, 'slurm_steps', lambda _: {'14372.0': 'RUNNING', '14372.38': 'RUNNING'})
    return policy, state, registry, batch


def test_verified_queued_recovery_waits_without_repair_or_false_recovery_alert(
    cli, queued_recovery, tmp_path, monkeypatch
):
    policy, state, registry, _ = queued_recovery
    incidents = cli.inspect_runs(registry['runs'], step_states={'14372.38': 'RUNNING'})['incidents']
    incident = incidents[0]
    state['incidents'][incident['id']] = {'attempts': 1, 'observations': 1, 'alerted': True, 'incident': incident}
    monkeypatch.setattr(cli, 'run_agent', lambda *a: pytest.fail('Verified replacement is already queued'))
    monkeypatch.setattr(cli, 'notify_blocker', lambda *a, **k: pytest.fail('Waiting is neither failure nor recovery'))
    result = cli.check(tmp_path, policy, state, act=True)
    assert result['status'] == 'waiting'
    health = cli.read(tmp_path / 'health.json')
    assert not health['incidents']
    failed = next(row for row in health['findings'] if row['run_id'] == 'failed')
    assert failed['status'] == 'waiting' and failed['pending_recovery'] == 'recovery'
    assert failed['phase'] == 'blocked'  # Original terminal evidence is preserved.
    assert state['incidents'][incident['id']]['attempts'] == 1
    assert not state['incidents'][incident['id']].get('recovered')


@pytest.mark.parametrize('invalid', [
    'missing', 'hash', 'run_id', 'allocation', 'dependency', 'self_dependency',
    'disabled', 'decision', 'capacity', 'dispatch_failed', 'live_old_worker', 'malformed_pointer',
])
def test_unverified_queued_recovery_leaves_original_failure_actionable(
    cli, queued_recovery, tmp_path, monkeypatch, invalid
):
    policy, state, registry, batch = queued_recovery
    if invalid == 'missing':
        registry['pending_batches'] = []
    elif invalid == 'hash':
        batch['launch']['bindings'][0]['sha256'] = 'wrong'
    elif invalid == 'run_id':
        batch['launch']['run']['id'] = 'unrelated'
    elif invalid == 'allocation':
        batch['launch']['argv'][1] = '--jobid=99999'
    elif invalid == 'dependency':
        batch['depends_on'] = ['missing-parent']
    elif invalid == 'self_dependency':
        batch['depends_on'] = ['failed']
    elif invalid == 'disabled':
        batch['enabled'] = False
    elif invalid == 'decision':
        batch['needs_user'] = True
    elif invalid == 'capacity':
        batch['resources']['gpus'] = 2
    elif invalid == 'dispatch_failed':
        cli.write(tmp_path / 'dispatch-state.json', {
            'recovery': {'status': 'failed', 'descriptor_sha256': 'bound', 'error': 'launch failure'},
        })
    elif invalid == 'live_old_worker':
        monkeypatch.setattr(cli, 'slurm_steps', lambda _: {'14372.0': 'RUNNING', '14372.34': 'RUNNING'})
    else:
        registry['runs'][0]['pending_recovery'] = {'malformed': True}
    cli.write(tmp_path / 'registry.json', registry)
    cli.check(tmp_path, policy, state)
    incidents = cli.read(tmp_path / 'health.json')['incidents']
    assert any(row['kind'] == 'run_failed' and row['focus_run_id'] == 'failed' for row in incidents)


def test_queued_recovery_chain_does_not_spawn_a_second_repair(
    cli, queued_recovery, tmp_path, monkeypatch
):
    import copy

    policy, state, registry, recovery = queued_recovery
    parent = copy.deepcopy(recovery)
    parent.update(id='queued-main', depends_on=['busy'])
    parent['launch']['run']['id'] = 'queued-main'
    parent['launch']['nonce'] = 'reviewed-queued-main-001'
    registry['pending_batches'].append(parent)
    recovery['depends_on'] = ['queued-main']
    cli.write(tmp_path / 'registry.json', registry)
    incident = cli.inspect_runs(registry['runs'], step_states={'14372.38': 'RUNNING'})['incidents'][0]
    state['incidents'][incident['id']] = {
        'attempts': 1, 'observations': 1, 'alerted': True, 'incident': incident,
    }
    monkeypatch.setattr(cli, 'run_agent', lambda *a: pytest.fail('Queued recovery chain owns repair'))
    monkeypatch.setattr(cli, 'notify_blocker', lambda *a, **k: pytest.fail('Still waiting'))
    assert cli.check(tmp_path, policy, state, act=True)['status'] == 'waiting'
    health = cli.read(tmp_path / 'health.json')
    assert not health['incidents']
    assert state['incidents'][incident['id']]['attempts'] == 1
    assert not state['incidents'][incident['id']].get('recovered')
    assert cli.read(tmp_path / 'registry.json') == registry
    assert not (tmp_path / 'dispatch-state.json').exists()
    assert not Path(recovery['launch']['step_path']).exists()


def test_qualified_nested_failure_with_queued_recovery_does_not_invoke_second_repair(
    cli, queued_recovery, tmp_path, monkeypatch
):
    policy, state, registry, _ = queued_recovery
    observed = cli.inspect_runs(registry['runs'], step_states={'14372.38': 'RUNNING'})
    incident = observed['incidents'][0]
    incident['kind'] = 'scientific_software_error'
    state['incidents'][incident['id']] = {
        'attempts': 1, 'observations': 1, 'incident': incident, 'alerted': True,
    }
    monkeypatch.setattr(cli, 'inspect_runs', lambda *a, **k: observed)
    monkeypatch.setattr(cli, 'run_agent', lambda *a: pytest.fail('Qualified replacement is already queued'))
    monkeypatch.setattr(cli, 'notify_blocker', lambda *a, **k: pytest.fail('Prepared recovery needs no alert'))
    current = cli.check(tmp_path, policy, state, act=True)
    assert current['status'] == 'waiting'
    assert not cli.read(tmp_path / 'health.json')['incidents']
    assert state['incidents'][incident['id']]['attempts'] == 1


def test_command_index_migration_conserves_all_attempts_and_blocks_third_repair(
    cli, controller, tmp_path, monkeypatch
):
    from exact.experiments.supervision import _incident
    policy, state = controller
    original = dict(id='original', step_id='14372.1', enabled=False, superseded_by='replacement',
                    status_path=str(tmp_path / 'original.json'))
    replacement = dict(id='replacement', step_id='14372.2', status_path=str(tmp_path / 'replacement.json'))
    cli.write(tmp_path / 'registry.json', {'runs': [original, replacement]})
    for row, index in [(original, 2), (replacement, 1)]:
        detail = dict(type='RuntimeError', message=f'command {index} exited 1: RuntimeError: cleanup failed')
        cli.write(Path(row['status_path']), dict(status='failed', error=detail))
        identity = _incident(row['id'], 'run_failed', '', detail, scope='original')['id']
        state['incidents'][identity] = dict(attempts=1, unsuccessful_attempts=1, observations=1)
        state['agent_runs'].append(dict(incident=identity, attempt=1, directory=f'original-report-{index}',
                                        started_epoch=0))
    archived = json.loads(json.dumps(state['agent_runs']))
    monkeypatch.setattr(cli, 'run_agent', lambda *a: pytest.fail('Two same-cause failures exhausted retries'))
    monkeypatch.setattr(cli, 'notify_blocker', lambda *a, **kw: {})
    result = cli.check(tmp_path, policy, state, act=True)
    assert result['action'] == 'incident_attempt_limit'
    health = cli.read(tmp_path / 'health.json')
    identity = health['incidents'][0]['id']
    merged = state['incidents'][identity]
    assert merged['attempts'] == merged['unsuccessful_attempts'] == 2
    assert len(merged['merged_incident_ids']) == 2
    assert state['agent_runs'] == archived
    assert all(state['incidents'][row['incident']]['attempts'] == 1 for row in archived)
    cli.check(tmp_path, policy, state, act=True)
    assert merged['attempts'] == merged['unsuccessful_attempts'] == 2
    assert state['agent_runs'] == archived


@pytest.mark.parametrize('invalid', [None, 'nonce', 'cause', 'status_only'])
def test_failed_manual_replacements_bound_retries_without_model_calls(
    cli, controller, tmp_path, monkeypatch, invalid
):
    policy, state = controller
    runs = []
    for index in range(3):
        row = dict(id=f'run{index}', step_id=f'14372.{index+1}', dispatch_nonce=f'owner{index}',
                   status_path=str(tmp_path / f'status{index}.json'),
                   completion_path=str(tmp_path / f'completion{index}.json'))
        if index < 2:
            row.update(enabled=False, superseded_by=f'run{index+1}')
        detail = dict(type='RuntimeError', message=f'command {index} exited 1: RuntimeError: cleanup failed')
        if invalid == 'cause' and index == 1:
            detail['message'] += ' in an unrelated subsystem'
        cli.write(Path(row['status_path']), dict(status='failed', error=detail))
        if invalid != 'status_only' or index != 1:
            cli.write(Path(row['completion_path']), dict(
                status='failed', exit_code=1, error=detail,
                step_id=row['step_id'], dispatch_nonce='wrong' if invalid == 'nonce' and index == 1
                else row['dispatch_nonce']))
        runs.append(row)
    cli.write(tmp_path / 'registry.json', {'runs': runs})
    monkeypatch.setattr(cli, 'run_agent', lambda *a: pytest.fail('Read-only test must not launch a model'))
    result = cli.check(tmp_path, policy, state, act=False)
    incident = cli.read(tmp_path / 'health.json')['incidents'][0]
    record = state['incidents'][incident['id']]
    assert record['attempts'] == 0 and state['agent_runs'] == []
    assert record['unsuccessful_attempts'] == (2 if invalid is None else 1)
    assert (result['status'] == 'blocked') == (invalid is None)
    # A model invocation already charged for the same failed replacement is not
    # charged again by the receipt lower bound on the following observation.
    record['unsuccessful_attempts'] = 2
    cli.check(tmp_path, policy, state, act=False)
    assert record['unsuccessful_attempts'] == 2


@pytest.mark.parametrize('replacement_state', ['queued', 'running', 'complete', 'failed', 'wrong_nonce', 'wrong_binding'])
def test_post_intervention_reobserves_owned_recovery_before_limit_email(
    cli, controller, tmp_path, monkeypatch, replacement_state
):
    policy, state = controller
    old_dir, new_dir = tmp_path / 'old', tmp_path / 'new'
    old_dir.mkdir()
    new_dir.mkdir()
    error = dict(type='RuntimeError', message='command 0 exited 1: same failure')
    prior = dict(id='old', logical_id='logical', step_id='14372.1', dispatch_nonce='old-owner',
                 status_path=str(old_dir / 'status.json'), completion_path=str(old_dir / 'completion.json'))
    cli.write(Path(prior['status_path']), dict(status='failed', error=error))
    cli.write(Path(prior['completion_path']), dict(status='failed', step_id=prior['step_id'],
              dispatch_nonce='old-owner', error=error))
    registry = dict(runs=[prior], pending_batches=[], capacity=dict(cpus=2))
    cli.write(tmp_path / 'registry.json', registry)
    incident = cli.inspect_runs(registry['runs'], step_states={})['incidents'][0]
    state['incidents'][incident['id']] = dict(attempts=1, unsuccessful_attempts=1, observations=2)
    steps = {'14372.0': 'RUNNING'}
    monkeypatch.setattr(cli, 'slurm_steps', lambda _: steps)
    clock = ['before-intervention']
    monkeypatch.setattr(cli, 'timestamp', lambda: clock[0])
    alerts = []
    monkeypatch.setattr(cli, 'notify_blocker', lambda *a, **kw: alerts.append(a[3]) or {})

    def repair(*args):
        clock[0] = 'after-intervention'
        successor = dict(id='new', logical_id='logical', recovery_of='old', repair_attempt=2,
                         max_repairs=2, recovery_completion_sha256=cli.digest(Path(prior['completion_path'])),
                         status_path=str(new_dir / 'status.json'), completion_path=str(new_dir / 'completion.json'),
                         exit_path=str(new_dir / 'exit_code'))
        if replacement_state in {'queued', 'wrong_binding'}:
            if replacement_state == 'wrong_binding':
                successor['recovery_completion_sha256'] = 'not-the-receipt'
            prior['pending_recovery'] = 'new'
            script = tmp_path / 'worker.sh'
            script.write_text('true\n')
            registry['pending_batches'] = [dict(id='new', depends_on=[], resources=dict(cpus=2), launch=dict(
                run=successor, nonce='new-owner-qualified',
                argv=['/usr/bin/srun', '--jobid=14372', '/bin/bash', str(script)],
                tmux_socket=str(tmp_path / 'socket'), step_path=str(new_dir / 'step.json'),
                launcher_log=str(new_dir / 'launcher.log'),
                bindings=[dict(path=str(script), sha256=cli.digest(script))]))]
        else:
            prior.update(enabled=False, superseded_by='new')
            successor.update(step_id='14372.2', dispatch_nonce='new-owner')
            registry['runs'].append(successor)
            cli.write(new_dir / 'step.json', dict(step_id='14372.2',
                      dispatch_nonce='wrong' if replacement_state == 'wrong_nonce' else 'new-owner'))
            if replacement_state in {'running', 'wrong_nonce'}:
                steps['14372.2'] = 'RUNNING'
                cli.write(new_dir / 'status.json', dict(status='running'))
            else:
                receipt = dict(status=replacement_state, step_id='14372.2', dispatch_nonce='new-owner')
                if replacement_state == 'failed':
                    receipt['error'] = {**error, 'message': error['message'].replace('command 0', 'command 1')}
                cli.write(new_dir / 'status.json', receipt)
                cli.write(new_dir / 'completion.json', receipt)
        cli.write(tmp_path / 'registry.json', registry)
        return dict(status='complete', result=_result('submitted'), result_valid=True)

    monkeypatch.setattr(cli, 'run_agent', repair)
    result = cli.check(tmp_path, policy, state, act=True)
    verified = replacement_state in {'queued', 'running', 'complete'}
    assert ('incident_attempt_limit' not in alerts) == verified
    assert (result['status'] != 'blocked') == verified
    assert result['checked_at'] == 'after-intervention'
    assert state['incidents'][incident['id']]['unsuccessful_attempts'] == 2
    if verified:
        assert result['replacement_progress']['status'] == replacement_state
