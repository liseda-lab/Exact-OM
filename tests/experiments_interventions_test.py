"""Controller regressions for expanding queues and productive preparation phases."""
import importlib.util
import json
from pathlib import Path

import pytest

from exact.experiments.interventions import (
    MAX_PROMPT_CHARACTERS, account_intervention, intervention_prompt,
    migrate_retry_accounting, preparation_progress,
)


def result(steps=(), outcome="submitted"):
    return {"status": "complete", "result_valid": True,
            "result": {"outcome": outcome, "steps": list(steps), "commits": [],
                       "handoff": "", "summary": "Prepared the next phase"}}


def test_large_registry_is_bound_to_complete_immutable_snapshot(tmp_path):
    registry = {"runs": [{"id": f"run-{i}", "payload": "large" * 2000} for i in range(200)],
                "pending_batches": [{"id": "next", "depends_on": ["run-199"]}]}
    context = {"registry": registry, "incident": {"id": "incident", "kind": "next_batch", "batch_id": "next"},
               "supervisor_directory": str(tmp_path), "allocation": "1", "repair_attempt": 3,
               "max_attempts": 2, "health": {"findings": []}}
    instructions = "Pinned complete policy." * 5000
    prompt = intervention_prompt(tmp_path, instructions, context)
    assert len(prompt) < MAX_PROMPT_CHARACTERS - 4096
    assert '"focused_run_ids": [\n    "run-199"' in prompt
    snapshot = tmp_path / "context.json"
    assert snapshot.stat().st_size > 1_048_576
    document = json.loads(snapshot.read_text())
    assert document["registry"] == registry
    assert document["instructions"] == instructions
    import hashlib
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() in prompt
    with pytest.raises(FileExistsError):
        intervention_prompt(tmp_path, instructions, context)


def test_two_successful_phases_do_not_consume_unsuccessful_repair_limit(tmp_path):
    record = {"attempts": 0, "unsuccessful_attempts": 0}
    incident = {"kind": "next_batch", "batch_id": "prepare"}
    registry = {"runs": []}
    for number in range(1, 4):
        new = {"runs": registry["runs"] + [{"id": f"row-{number}", "step_id": f"1.{number}"}]}
        record["attempts"] += 1
        assert account_intervention(record, incident, result([f"1.{number}"]), registry, new)
        registry = new
    assert record["attempts"] == 3
    assert record["unsuccessful_attempts"] == 0
    # Repeating the same submitted step is not another successful phase.
    assert not account_intervention(record, incident, result(["1.3"]), registry, registry)
    assert record["unsuccessful_attempts"] == 1


@pytest.mark.parametrize("outcome", ["submitted", "repaired", "no_change"])
def test_claimed_success_without_progress_consumes_failure_attempt(outcome):
    record = {"unsuccessful_attempts": 0}
    for _ in range(2):
        assert not account_intervention(record, {"kind": "next_batch"}, result(outcome=outcome), {}, {})
    assert record["unsuccessful_attempts"] == 2


def test_unrelated_dispatch_is_not_preparation_progress():
    before = {"runs": [], "pending_batches": [{"id": "other", "launch": {"nonce": "old"}}]}
    after = {"runs": [{"id": "other", "step_id": "1.7"}], "pending_batches": []}
    assert preparation_progress(before, after, {"kind": "next_batch", "batch_id": "target"}, result()) == []


def test_prepared_descriptors_require_bound_files_and_new_identity(tmp_path):
    import hashlib
    script = tmp_path / "launch.sh"
    script.write_text("true\n")
    batch = {"id": "prepared", "launch": {"bindings": [
        {"path": str(script), "sha256": hashlib.sha256(script.read_bytes()).hexdigest()}]}}
    before, after = {"pending_batches": []}, {"pending_batches": [batch]}
    incident = {"kind": "next_batch", "batch_id": "target"}
    assert preparation_progress(before, after, incident, result())
    assert not preparation_progress(after, after, incident, result())
    script.write_text("changed")
    assert not preparation_progress(before, after, incident, result())


def test_migration_proves_success_and_preserves_unproven_failures(tmp_path):
    incident = {"kind": "next_batch"}
    state = {"incidents": {"planning": {"attempts": 4, "incident": incident},
                           "worker": {"attempts": 2, "incident": {"kind": "run_failed"}}}, "agent_runs": []}
    for number, report in enumerate([result(["1.1"]), result(["1.2"]), result(["1.2"]),
                                     {"status": "failed", "result": None}], 1):
        directory = tmp_path / str(number)
        directory.mkdir()
        (directory / "report.json").write_text(json.dumps(report))
        state["agent_runs"].append({"incident": "planning", "attempt": number, "directory": str(directory)})
    registry = {"runs": [{"step_id": "1.1"}, {"step_id": "1.2"}]}
    migrate_retry_accounting(state, registry)
    planning = state["incidents"]["planning"]
    assert planning["attempts"] == 4
    assert planning["unsuccessful_attempts"] == 2
    assert len(planning["retry_accounting"]["verified_preparation"]) == 2
    assert state["incidents"]["worker"]["unsuccessful_attempts"] == 2
    # Migration is idempotent and cannot forgive earlier real failures later.
    migrate_retry_accounting(state, registry)
    assert planning["unsuccessful_attempts"] == 2


def test_worker_error_attempts_remain_cumulative_across_replacement_jobs():
    record = {"unsuccessful_attempts": 0}
    before = {"runs": [{"id": "original", "step_id": "1.1"}]}
    for number in (2, 3):
        incident = {"id": "same-cause", "kind": "run_failed", "focus_run_id": f"replacement-{number}"}
        after = {"runs": before["runs"] + [{"id": f"replacement-{number}", "step_id": f"1.{number}"}]}
        account_intervention(record, incident, result([f"1.{number}"]), before, after)
        before = after
    assert record["unsuccessful_attempts"] == 2


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "tools/supervise_experiments.py"
    spec = importlib.util.spec_from_file_location("bounded_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_control_exposes_blockers_and_never_calls_model_or_mail(cli, tmp_path, monkeypatch):
    cli.write(tmp_path / "registry.json", {"runs": []})
    (tmp_path / "MANUAL_CONTROL").write_text("User-authorized maintenance")
    incident = {"id": "blocked", "kind": "next_batch", "batch_id": "target"}
    monkeypatch.setattr(cli, "slurm_steps", lambda _: {"1.0": "RUNNING"})
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: {
        "status": "healthy", "findings": [], "incidents": [incident]})
    monkeypatch.setattr(cli, "run_agent", lambda *a: pytest.fail("manual controller owns repair"))
    monkeypatch.setattr(cli, "notify_intervention", lambda *a, **k: pytest.fail("manual controller owns mail"))
    state = {"incidents": {"blocked": {"attempts": 2, "observations": 1}}, "agent_runs": []}
    policy = {"allocation": "1", "max_attempts_per_incident": 2}
    current = cli.check(tmp_path, policy, state, act=True)
    assert current["status"] == "manual_control"
    assert current["monitoring_status"] == "blocked"
    assert current["blocked_incidents"][0]["reason"] == "incident_attempt_limit"
    assert current["dispatch"] == "continues"
    assert cli.read(tmp_path / "health.json")["status"] == "blocked"
    assert cli.notify_blocker(tmp_path, policy, incident, "requires_user")["status"] == "suppressed_manual_control"


def test_oversized_external_prompt_rejected_before_child_launch(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "authenticate", lambda _: None)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: pytest.fail("must fail preflight"))
    with pytest.raises(ValueError, match="before launch"):
        cli.run_agent({"agent_timeout_seconds": 60}, tmp_path, "x" * MAX_PROMPT_CHARACTERS, lambda: False)


def test_controller_continues_productive_preparation_and_stops_repeated_noops(cli, tmp_path, monkeypatch):
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Preserve costs and the allocation")
    policy = {"allocation": "1", "max_attempts_per_incident": 2,
              "instructions": str(instructions), "instructions_sha256": cli.digest(instructions)}
    state = {"incidents": {}, "agent_runs": []}
    cli.write(tmp_path / "registry.json", {"runs": []})
    incident = {"id": "planning", "kind": "next_batch", "batch_id": "prepare", "run_ids": []}
    monkeypatch.setattr(cli, "slurm_steps", lambda _: {"1.0": "RUNNING"})
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: {
        "status": "healthy", "findings": [], "incidents": [incident.copy()]})
    monkeypatch.setattr(cli, "authenticate", lambda _: None)
    calls = []

    def prepare(_policy, directory, prompt, stop):
        number = len(calls) + 1
        calls.append(directory)
        # The durable count is reserved before launching, even if this process
        # dies while work is being done or before the final report is read.
        saved = cli.read(tmp_path / "state.json")["incidents"]["planning"]
        assert saved["attempts"] == number
        assert saved["unsuccessful_attempts"] >= 1
        if number <= 3:
            registry = cli.read(tmp_path / "registry.json")
            registry["runs"].append({"id": f"child-{number}", "step_id": f"1.{number}"})
            cli.write(tmp_path / "registry.json", registry)
            return result([f"1.{number}"])
        return result(["1.3"])  # a false success claim repeating an old child

    monkeypatch.setattr(cli, "run_agent", prepare)
    for _ in range(3):
        current = cli.check(tmp_path, policy, state, act=True)
        assert current["status"] == "intervention_finished"
        assert current["progress_verified"] is True
        assert current["unsuccessful_attempts"] == 0
    cli.check(tmp_path, policy, state, act=True)
    current = cli.check(tmp_path, policy, state, act=True)
    assert current["status"] == "blocked"
    assert current["action"] == "incident_attempt_limit"
    assert current["blocked_incidents"][0]["incident_id"] == "planning"
    assert cli.read(tmp_path / "health.json")["status"] == "blocked"
    cli.check(tmp_path, policy, state, act=True)
    assert len(calls) == 5
    assert state["incidents"]["planning"]["attempts"] == 5
    assert state["incidents"]["planning"]["unsuccessful_attempts"] == 2


def test_manual_control_leaves_deterministic_dispatch_active(cli, tmp_path, monkeypatch):
    (tmp_path / "MANUAL_CONTROL").touch()
    monkeypatch.setattr(cli, "slurm_steps", lambda _: {"1.0": "RUNNING"})
    calls = []
    monkeypatch.setattr(cli, "dispatch_ready", lambda *a, **k: calls.append(a) or {"status": "dispatched"})

    class Once:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, seconds):
            self.stopped = True

    cli.dispatch_worker(tmp_path, {"allocation": "1"}, Once())
    assert len(calls) == 1
    assert cli.read(tmp_path / "dispatch-status.json")["status"] == "dispatched"
    monkeypatch.setattr(cli, "flush_notifications", lambda *a: pytest.fail("mail is suppressed"))
    cli.notification_worker(tmp_path, {}, Once())


def test_observations_during_model_work_preserve_other_blocked_stages(cli, tmp_path, monkeypatch):
    blocked = {"incident_id": "old", "batch_id": "pending", "kind": "next_batch",
               "reason": "incident_attempt_limit"}
    cli.write(tmp_path / "registry.json", {"runs": [], "pending_batches": [{"id": "pending"}]})
    cli.write(tmp_path / "status.json", {"status": "repairing", "blocked_incidents": [blocked]})
    monkeypatch.setattr(cli, "slurm_steps", lambda _: {"1.0": "RUNNING"})
    cli.refresh_progress({"allocation": "1"}, tmp_path)
    assert cli.read(tmp_path / "health.json")["status"] == "blocked"
    assert cli.read(tmp_path / "status.json")["blocked_incidents"] == [blocked]


def test_failed_prompt_preflight_preserves_evidence_without_stranding_next_attempt(cli, tmp_path, monkeypatch):
    instructions = tmp_path / "instructions.md"
    instructions.write_text("Pinned instructions")
    policy = {"allocation": "1", "max_attempts_per_incident": 2,
              "instructions": str(instructions), "instructions_sha256": cli.digest(instructions)}
    state = {"incidents": {}, "agent_runs": []}
    cli.write(tmp_path / "registry.json", {"runs": []})
    incident = {"id": "prepare", "kind": "next_batch", "run_ids": []}
    monkeypatch.setattr(cli, "slurm_steps", lambda _: {"1.0": "RUNNING"})
    monkeypatch.setattr(cli, "inspect_runs", lambda *a, **k: {
        "status": "healthy", "findings": [], "incidents": [incident.copy()]})
    monkeypatch.setattr(cli, "authenticate", lambda _: None)
    original = cli.intervention_prompt

    def fail(directory, *args):
        (directory / "context.json").write_text("partial serialization")
        raise ValueError("preflight failed")

    monkeypatch.setattr(cli, "intervention_prompt", fail)
    with pytest.raises(ValueError, match="preflight failed"):
        cli.check(tmp_path, policy, state, act=True)
    assert state["incidents"]["prepare"]["attempts"] == 0
    assert state["agent_runs"] == []
    assert not (tmp_path / "interventions/prepare-1").exists()
    archive = next((tmp_path / "preflight-failures").iterdir())
    assert (archive / "context.json").read_text() == "partial serialization"
    assert cli.read(archive / "preflight-error.json")["model_launched"] is False
    monkeypatch.setattr(cli, "intervention_prompt", original)
    monkeypatch.setattr(cli, "run_agent", lambda *a: result(outcome="no_change"))
    assert cli.check(tmp_path, policy, state, act=True)["unsuccessful_attempts"] == 1
    assert state["incidents"]["prepare"]["attempts"] == 1


def test_alias_migration_preserves_observation_and_result_history():
    from exact.experiments.interventions import migrate_incident_aliases
    old = dict(attempts=2, unsuccessful_attempts=2, observations=9, first_seen_epoch=1,
               last_seen='2026-01-01', last_result={'handoff': 'original'}, progress_witnesses=['saved'],
               needs_user=True, alerted=True)
    state = {'incidents': {'old': dict(old)}, 'agent_runs': [{'incident': 'old', 'attempt': 1}]}
    incident = dict(id='new', legacy_incident_ids=['old'], unsuccessful_recovery_run_ids=['replacement'])
    migrate_incident_aliases(state, [incident])
    current = state['incidents']['new']
    assert current['merged_history']['old'] == old
    for key in ('attempts', 'unsuccessful_attempts', 'observations', 'first_seen_epoch', 'last_seen',
                'last_result', 'progress_witnesses', 'needs_user', 'alerted'):
        assert current[key] == old[key]
    migrate_incident_aliases(state, [incident])
    assert current['attempts'] == current['unsuccessful_attempts'] == 2
    assert state['agent_runs'] == [{'incident': 'old', 'attempt': 1}]
