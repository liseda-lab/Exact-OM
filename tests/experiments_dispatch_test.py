"""Prepared handoffs require receipts and retain ownership across controller crashes."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from exact.experiments import dispatch

REAL_RUN = subprocess.run


def save(path, value):
    path.write_text(json.dumps(value))


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def queue(tmp_path, monkeypatch):
    calls = []

    def spawn(argv, **kwargs):
        # Ownership must be durable before the existing external server launches.
        state = load(tmp_path / "dispatch-state.json")
        assert any(row["may_have_started"] for row in state.values())
        assert argv[:3] == ["/usr/bin/tmux", "-N", "-S"]
        assert "new-session" in argv
        calls.append(argv)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(dispatch.subprocess, "run", spawn)
    monkeypatch.setattr(
        dispatch,
        "_tmux_server",
        lambda launch: (["/usr/bin/tmux", "-N", "-S", launch["tmux_socket"]], 12345),
    )
    worker = tmp_path / "worker.sh"
    worker.write_text("#!/bin/bash\nexit 0\n")
    batch = {
        "id": "E18",
        "resources": {"gpus": 1, "cpus": 4, "memory_gib": 80},
        "launch": {
            "argv": [
                "/usr/bin/srun",
                "--jobid=14372",
                "--overlap",
                "--time=0",
                "/bin/bash",
                str(worker),
            ],
            "nonce": "test-dispatch-nonce-001",
            "tmux_socket": str(tmp_path / "tmux.sock"),
            "bindings": [
                {"path": str(worker), "sha256": hashlib.sha256(worker.read_bytes()).hexdigest()}
            ],
            "step_path": str(tmp_path / "step.json"),
            "launcher_log": str(tmp_path / "launcher.log"),
            "run": {
                "id": "E18",
                "status_path": str(tmp_path / "status.json"),
                "exit_path": str(tmp_path / "exit"),
                "completion_path": str(tmp_path / "complete.json"),
            },
        },
    }
    registry = {
        "runs": [],
        "pending_batches": [batch],
        "capacity": {"gpus": 1, "cpus": 6, "memory_gib": 125},
    }
    save(tmp_path / "registry.json", registry)
    return registry, batch, calls


def tick(tmp_path, steps=None):
    return dispatch.dispatch_ready(
        tmp_path, "14372", steps or {"14372.0": "RUNNING"}, supervisor_step="35"
    )


def receipt(batch, step="14372.36"):
    save(
        Path(batch["launch"]["step_path"]),
        {"step_id": step, "dispatch_nonce": batch["launch"]["nonce"]},
    )


def test_launch_registers_verified_step_without_duplicate_spawn(tmp_path, queue):
    registry, batch, calls = queue
    assert tick(tmp_path)["status"] == "starting"
    assert load(tmp_path / "registry.json")["runs"] == []
    assert tick(tmp_path)["status"] == "no_ready_launch"
    receipt(batch)
    result = tick(tmp_path, {"14372.0": "RUNNING", "14372.36": "RUNNING"})
    assert result == {"status": "registered", "batch_id": "E18", "step_id": "14372.36"}
    current = load(tmp_path / "registry.json")
    assert current["pending_batches"] == []
    assert current["runs"][0]["dispatch_nonce"] == batch["launch"]["nonce"]
    assert len(calls) == 1
    assert tick(tmp_path)["status"] == "no_ready_launch"


def test_restart_reconciles_existing_worker_and_registration_crash(tmp_path, queue):
    _, batch, calls = queue
    tick(tmp_path)
    # A new supervisor can only reconcile durable receipts, not inherit a handle.
    receipt(batch)
    assert tick(tmp_path, {"14372.0": "RUNNING", "14372.36": "RUNNING"})["status"] == "registered"
    state = load(tmp_path / "dispatch-state.json")
    state["E18"]["status"] = "starting"  # Crash after registry write, before state write.
    save(tmp_path / "dispatch-state.json", state)
    tick(tmp_path)
    assert load(tmp_path / "dispatch-state.json")["E18"]["status"] == "registered"
    assert len(calls) == 1


@pytest.fixture
def queued_recovery(tmp_path, queue):
    registry, batch, calls = queue
    registry["runs"] = [
        {
            "id": "failed",
            "step_id": "14372.33",
            "pending_recovery": batch["id"],
            "enabled": True,
            "reason": "Retain failed attempt and its accounting",
            "budget_path": str(tmp_path / "historic-budget.json"),
        }
    ]
    batch["recovery_for"] = "failed"
    save(tmp_path / "registry.json", registry)
    return registry, batch, calls


@pytest.mark.parametrize("explicit_parent", [True, False])
def test_recovery_lineage_requires_verified_registration(
    tmp_path, queued_recovery, explicit_parent
):
    registry, batch, calls = queued_recovery
    if not explicit_parent:  # Existing pending-only recovery declarations remain supported.
        batch.pop("recovery_for")
        save(tmp_path / "registry.json", registry)
    observation = dispatch.inspect_runs(registry["runs"], step_states={"14372.0": "RUNNING"})
    assert waiting_recoveries(registry, observation) == {"failed": "E18"}
    assert tick(tmp_path)["status"] == "starting"
    assert tick(tmp_path)["status"] == "no_ready_launch"
    assert load(tmp_path / "registry.json") == registry

    receipt(batch)
    steps = {"14372.0": "RUNNING", "14372.36": "RUNNING"}
    assert tick(tmp_path, steps)["status"] == "registered"
    current = load(tmp_path / "registry.json")
    expected_parent = {
        **registry["runs"][0],
        "enabled": False,
        "superseded_by": "E18",
        "retain_accounting": True,
    }
    expected_parent.pop("pending_recovery")
    assert current["runs"][0] == expected_parent
    assert current["runs"][1]["step_id"] == "14372.36"
    assert current["runs"][1]["dispatch_nonce"] == batch["launch"]["nonce"]
    assert current["pending_batches"] == []
    dispatch.inspect_runs(current["runs"], step_states=steps)  # Registry invariants still hold.

    # Registration is authoritative if the controller dies before writing dispatch state.
    state = load(tmp_path / "dispatch-state.json")
    state["E18"]["status"] = "starting"
    save(tmp_path / "dispatch-state.json", state)
    assert tick(tmp_path, steps)["status"] == "no_ready_launch"
    assert load(tmp_path / "dispatch-state.json")["E18"]["status"] == "registered"
    assert load(tmp_path / "registry.json") == current
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["nonce", "nonnumeric_step"])
def test_recovery_rejects_forged_receipt_without_promoting_parent(
    tmp_path, queued_recovery, damage
):
    registry, batch, calls = queued_recovery
    assert tick(tmp_path)["status"] == "starting"
    proof = {"step_id": "14372.36", "dispatch_nonce": batch["launch"]["nonce"]}
    proof["dispatch_nonce" if damage == "nonce" else "step_id"] = (
        "another-dispatch-nonce" if damage == "nonce" else "14372.extern"
    )
    save(Path(batch["launch"]["step_path"]), proof)
    tick(tmp_path, {"14372.0": "RUNNING", "14372.36": "RUNNING"})
    assert load(tmp_path / "registry.json") == registry
    assert load(tmp_path / "dispatch-state.json")["E18"]["status"] == "failed"
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["ambiguous", "mismatched", "missing", "superseded"])
def test_conflicting_queued_recovery_never_spawns(tmp_path, queued_recovery, damage):
    registry, batch, calls = queued_recovery
    parent = registry["runs"][0]
    registry["runs"].append({"id": "other", "step_id": "14372.34", "enabled": False})
    if damage == "ambiguous":
        registry["runs"][1]["pending_recovery"] = batch["id"]
    elif damage == "mismatched":
        batch["recovery_for"] = "other"
    elif damage == "missing":
        parent.pop("pending_recovery")
    else:
        parent["superseded_by"] = "other"
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["status"] == "failed"
    assert load(tmp_path / "registry.json") == registry
    assert load(tmp_path / "dispatch-state.json")["E18"]["may_have_started"] is False
    assert not calls


@pytest.mark.parametrize("damage", ["ambiguous", "changed", "superseded"])
def test_inflight_recovery_conflicts_preserve_registry(tmp_path, queued_recovery, damage):
    registry, batch, calls = queued_recovery
    assert tick(tmp_path)["status"] == "starting"
    registry["runs"].append({"id": "other", "step_id": "14372.34", "enabled": False})
    if damage == "ambiguous":
        registry["runs"][1]["pending_recovery"] = batch["id"]
    elif damage == "changed":
        registry["runs"][0]["pending_recovery"] = "another-recovery"
    else:
        registry["runs"][0]["superseded_by"] = "other"
    save(tmp_path / "registry.json", registry)
    receipt(batch)
    steps = {"14372.0": "RUNNING", "14372.36": "RUNNING"}
    tick(tmp_path, steps)
    tick(tmp_path, steps)
    assert load(tmp_path / "registry.json") == registry
    assert load(tmp_path / "dispatch-state.json")["E18"]["status"] == "failed"
    assert len(calls) == 1


def test_recovery_dependency_cycle_cannot_partially_mutate_registry(tmp_path, queued_recovery):
    registry, batch, calls = queued_recovery
    save(tmp_path / "parent-complete.json", {"status": "complete", "passed": True})
    (tmp_path / "parent-exit").write_text("0")
    registry["runs"].append(
        {
            "id": "dependent",
            "step_id": "14372.34",
            "depends_on": ["failed"],
            "completion_path": str(tmp_path / "parent-complete.json"),
            "exit_path": str(tmp_path / "parent-exit"),
        }
    )
    # The disabled historical prerequisite is valid; replacing it with its descendant is not.
    registry["runs"][0]["enabled"] = False
    batch["depends_on"] = ["dependent"]
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["status"] == "starting"
    receipt(batch)
    tick(tmp_path, {"14372.0": "RUNNING", "14372.36": "RUNNING"})
    assert load(tmp_path / "registry.json") == registry
    assert "cycle" in dispatch.dispatch_incidents(tmp_path)[0]["reason"]
    assert len(calls) == 1


def test_short_worker_requires_matching_terminal_receipt(tmp_path, queue):
    _, batch, calls = queue
    tick(tmp_path)
    receipt(batch)
    save(tmp_path / "complete.json", {"status": "complete"})
    (tmp_path / "exit").write_text("0")
    assert tick(tmp_path)["status"] != "registered"
    save(
        tmp_path / "complete.json",
        {
            "status": "complete",
            "exit_code": 0,
            "step_id": "14372.36",
            "dispatch_nonce": batch["launch"]["nonce"],
        },
    )
    assert tick(tmp_path)["status"] == "registered"
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["worker", "allocation", "inline", "nonce"])
def test_unreviewed_or_invalid_descriptor_never_spawns(tmp_path, queue, change):
    registry, batch, calls = queue
    launch = batch["launch"]
    if change == "worker":
        Path(launch["argv"][-1]).write_text("changed")
    elif change == "allocation":
        launch["argv"][1] = "--jobid=99999"
    elif change == "inline":
        launch["argv"][-2:] = ["/bin/bash", "-c"]
    else:
        launch["nonce"] = "short"
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["status"] == "failed"
    assert not calls
    assert dispatch.dispatch_incidents(tmp_path)[0]["kind"] == "dispatch_failed"


def test_stale_or_forged_step_receipt_is_not_registered(tmp_path, queue):
    _, batch, calls = queue
    tick(tmp_path)
    receipt(batch, "99999.4")
    tick(tmp_path)
    assert not load(tmp_path / "registry.json")["runs"]
    assert "nonce" in dispatch.dispatch_incidents(tmp_path)[0]["reason"]
    assert len(calls) == 1


def test_receipt_timeout_is_actionable_and_never_blindly_retried(tmp_path, queue, monkeypatch):
    _, _, calls = queue
    monkeypatch.setattr(dispatch.time, "time", lambda: 100)
    tick(tmp_path)
    monkeypatch.setattr(dispatch.time, "time", lambda: 191)
    tick(tmp_path)
    tick(tmp_path)
    assert len(calls) == 1
    assert "90s" in dispatch.dispatch_incidents(tmp_path)[0]["reason"]


def test_pending_dependency_resolves_after_parent_registered_and_complete(tmp_path, queue):
    registry, batch, calls = queue
    child = json.loads(json.dumps(batch))
    child.update(id="E16", depends_on=["E18"])
    child["launch"]["run"]["id"] = "E16"
    child["launch"]["nonce"] = "test-dispatch-nonce-002"
    child["launch"]["step_path"] = str(tmp_path / "child-step.json")
    registry["pending_batches"].append(child)
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["batch_id"] == "E18"
    receipt(batch)
    tick(tmp_path, {"14372.0": "RUNNING", "14372.36": "RUNNING"})
    assert (
        tick(tmp_path, {"14372.0": "RUNNING", "14372.36": "RUNNING"})["status"] == "no_ready_launch"
    )
    save(tmp_path / "complete.json", {"status": "complete", "passed": True})
    (tmp_path / "exit").write_text("0")
    assert tick(tmp_path)["batch_id"] == "E16"
    assert len(calls) == 2


def test_gpu_reservation_does_not_block_available_cpu_preparation(tmp_path, queue):
    registry, batch, calls = queue
    prep = json.loads(json.dumps(batch))
    prep.update(id="prepare", resources={"gpus": 0, "cpus": 1, "memory_gib": 2})
    prep["launch"]["run"]["id"] = "prepare"
    prep["launch"]["nonce"] = "test-dispatch-prep-001"
    prep["launch"]["step_path"] = str(tmp_path / "prep-step.json")
    registry["pending_batches"].append(prep)
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["batch_id"] == "E18"
    assert tick(tmp_path)["batch_id"] == "prepare"
    assert len(calls) == 2


@pytest.mark.parametrize("guard", ["pause", "stop", "unknown_step", "lock"])
def test_admission_guards(tmp_path, queue, guard):
    _, _, calls = queue
    if guard in {"pause", "stop"}:
        (tmp_path / guard.upper()).touch()
        assert tick(tmp_path)["status"] == "paused"
    elif guard == "unknown_step":
        assert (
            tick(tmp_path, {"14372.0": "RUNNING", "14372.77": "RUNNING"})["status"]
            == "unregistered_steps_present"
        )
    else:
        with (tmp_path / "registry.json.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert tick(tmp_path)["status"] == "registry_busy"
    assert not calls


def test_no_allocation_cannot_submit(tmp_path, queue):
    with pytest.raises(ValueError, match="allocation unavailable"):
        dispatch.dispatch_ready(tmp_path, "14372", {})
    assert not queue[2]


def test_late_receipt_recovers_uncertain_launch_without_resubmitting(tmp_path, queue, monkeypatch):
    _, batch, calls = queue
    monkeypatch.setattr(dispatch.time, "time", lambda: 100)
    tick(tmp_path)
    monkeypatch.setattr(dispatch.time, "time", lambda: 191)
    tick(tmp_path)
    assert dispatch.dispatch_incidents(tmp_path)
    receipt(batch)
    assert tick(tmp_path, {"14372.0": "RUNNING", "14372.36": "RUNNING"})["status"] == "registered"
    assert not dispatch.dispatch_incidents(tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "server_group,accepted",
    [
        ("0::/system.slice/slurmstepd.scope/allocation/step_extern/user/task_0", True),
        ("0::/system.slice/slurmstepd.scope/allocation/step_35/user/task_0", False),
        ("0::/system.slice/slurmstepd.scope/other/step_extern/user/task_0", False),
        ("0::/user.slice/user-1001.slice/session-4.scope", False),
    ],
)
def test_tmux_server_must_be_in_same_allocation_extern(monkeypatch, server_group, accepted):
    pid = os.getpid()
    commands = []
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::/system.slice/slurmstepd.scope/allocation/step_35/user/task_0\n"
        if str(path) == f"/proc/{pid}/cgroup":
            return server_group + "\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    def query(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(pid))

    monkeypatch.setattr(dispatch.subprocess, "run", query)
    if accepted:
        assert dispatch._tmux_server({"tmux_socket": "/tmp/existing.sock"})[1] == pid
    else:
        with pytest.raises(ValueError, match="extern cgroup"):
            dispatch._tmux_server({"tmux_socket": "/tmp/existing.sock"})
    assert commands == [
        ["/usr/bin/tmux", "-N", "-S", "/tmp/existing.sock", "display-message", "-p", "#{pid}"]
    ]


def test_missing_tmux_server_cannot_autocreate(monkeypatch):
    calls = []

    def missing(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(dispatch.subprocess, "run", missing)
    with pytest.raises(ValueError, match="never create"):
        dispatch._tmux_server({"tmux_socket": "/tmp/missing.sock"})
    assert len(calls) == 1 and "-N" in calls[0]


def test_wrapper_preserves_literal_argv_and_records_launcher_exit(tmp_path):
    sentinel = tmp_path / "should-not-exist"
    literal = f"$(touch {sentinel}) `touch {sentinel}` spaces"
    worker = tmp_path / "literal-worker.sh"
    worker.write_text('printf "%s\\n" "$1"\nexit 7\n')
    wrapper, exit_path = dispatch._detached_wrapper(
        {
            "argv": ["/bin/bash", str(worker), literal],
            "launcher_log": str(tmp_path / "a log.txt"),
        }
    )
    result = subprocess.run(["/bin/bash", str(wrapper)], capture_output=True)
    assert result.returncode == 7
    assert exit_path.read_text().strip() == "7"
    assert (tmp_path / "a log.txt").read_text().strip() == literal
    assert not sentinel.exists()


def test_launcher_exit_becomes_failure_without_repeating_submission(tmp_path, queue):
    tick(tmp_path)
    state = load(tmp_path / "dispatch-state.json")
    Path(state["E18"]["launcher_exit_path"]).write_text("1\n")
    tick(tmp_path)
    assert "Launcher exited" in dispatch.dispatch_incidents(tmp_path)[0]["reason"]
    assert len(queue[2]) == 1


@pytest.fixture
def recovery_chain(queue):
    import copy

    registry, recovery, calls = queue
    recovery["depends_on"] = ["main"]
    main = copy.deepcopy(recovery)
    main.update(id="main", depends_on=["complete"])
    main["launch"]["run"]["id"] = "main"
    main["launch"]["nonce"] = "test-main-nonce-001"
    registry["pending_batches"].append(main)
    registry["runs"] = [
        {"id": "failed", "step_id": "14372.33", "pending_recovery": recovery["id"]},
        {"id": "complete", "step_id": "14372.34"},
    ]
    observation = {
        "findings": [
            {"run_id": "failed", "status": "needs_attention"},
            {"run_id": "complete", "status": "complete"},
        ]
    }
    return registry, observation, main, calls


def waiting_recoveries(registry, observation, state=None, steps=None):
    return dispatch.pending_recoveries(
        registry, observation, "14372", steps or {"14372.0": "RUNNING"}, state or {}
    )


def test_reviewed_queued_dependencies_wait_without_mutating_or_launching(recovery_chain):
    import copy

    registry, observation, _, calls = recovery_chain
    before = copy.deepcopy((registry, observation))
    assert waiting_recoveries(registry, observation) == {"failed": "E18"}
    assert (registry, observation) == before
    assert not calls
    assert dispatch.pending_batches(registry, observation)[0]["batch_id"] == "main"
    assert not any(x["batch_id"] == "E18" for x in dispatch.pending_batches(registry, observation))


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "disabled",
        "needs_user",
        "hash",
        "allocation",
        "resources",
        "empty_resources",
        "parents_type",
        "parent_type",
        "self_cycle",
        "queue_cycle",
        "original_cycle",
        "failed_dependency",
        "missing_finding",
        "dispatch_failed",
        "dispatch_resolved",
        "dispatch_registered",
        "dispatch_nonce",
        "dispatch_hash",
        "dispatch_resources",
    ],
)
def test_invalid_queued_ancestor_leaves_original_failure_actionable(recovery_chain, damage):
    registry, observation, main, calls = recovery_chain
    state = {}
    if damage == "missing":
        registry["pending_batches"].remove(main)
    elif damage == "disabled":
        main["enabled"] = False
    elif damage == "needs_user":
        main["needs_user"] = True
    elif damage == "hash":
        main["launch"]["bindings"][0]["sha256"] = "wrong"
    elif damage == "allocation":
        main["launch"]["argv"][1] = "--jobid=99999"
    elif damage == "resources":
        main["resources"]["gpus"] = 2
    elif damage == "empty_resources":
        main["resources"] = {}
    elif damage == "parents_type":
        main["depends_on"] = "complete"
    elif damage == "parent_type":
        main["depends_on"] = [{}]
    elif damage == "self_cycle":
        main["depends_on"] = ["main"]
    elif damage == "queue_cycle":
        main["depends_on"] = ["E18"]
    elif damage == "original_cycle":
        main["depends_on"] = ["failed"]
    elif damage == "failed_dependency":
        observation["findings"][1]["status"] = "needs_attention"
    elif damage == "missing_finding":
        observation["findings"].pop()
    else:
        record = {
            "status": "starting",
            "nonce": main["launch"]["nonce"],
            "descriptor_sha256": dispatch._identity(main),
            "resources": main["resources"],
        }
        if damage in {"dispatch_failed", "dispatch_resolved", "dispatch_registered"}:
            record["status"] = damage.removeprefix("dispatch_")
        elif damage == "dispatch_nonce":
            record["nonce"] = "different-nonce"
        elif damage == "dispatch_hash":
            record["descriptor_sha256"] = "changed"
        else:
            record["resources"] = {"gpus": 0}
        state["main"] = record
    assert waiting_recoveries(registry, observation, state) == {}
    assert not calls


@pytest.mark.parametrize("status", ["healthy", "waiting", "complete"])
def test_registered_parent_states_remain_valid(recovery_chain, status):
    registry, observation, _, _ = recovery_chain
    observation["findings"][1]["status"] = status
    assert waiting_recoveries(registry, observation) == {"failed": "E18"}


@pytest.mark.parametrize("status", ["reserved", "starting"])
def test_matching_inflight_queued_ancestor_remains_waiting(recovery_chain, status):
    registry, observation, main, _ = recovery_chain
    state = {
        "main": {
            "status": status,
            "nonce": main["launch"]["nonce"],
            "descriptor_sha256": dispatch._identity(main),
            "resources": main["resources"],
        }
    }
    assert waiting_recoveries(registry, observation, state) == {"failed": "E18"}


@pytest.mark.parametrize("invalid", [None, "cycle", "live", "broken"])
def test_failed_parent_requires_verified_replacement_chain(recovery_chain, invalid):
    import copy

    registry, observation, main, _ = recovery_chain
    parent = registry["runs"][1]
    parent["pending_recovery"] = "parent-repair"
    observation["findings"][1]["status"] = "needs_attention"
    repair = copy.deepcopy(main)
    repair.update(id="parent-repair", depends_on=[])
    repair["launch"]["run"]["id"] = "parent-repair"
    repair["launch"]["nonce"] = "test-parent-repair-001"
    registry["pending_batches"].append(repair)
    steps = {"14372.0": "RUNNING"}
    if invalid == "cycle":
        repair["depends_on"] = ["main"]
    elif invalid == "live":
        steps[parent["step_id"]] = "RUNNING"
    elif invalid == "broken":
        repair["needs_user"] = True
    result = waiting_recoveries(registry, observation, steps=steps)
    assert result == ({} if invalid else {"failed": "E18", "complete": "parent-repair"})


@pytest.mark.parametrize("cycle", [False, True])
def test_registered_supersession_chains_are_checked(recovery_chain, cycle):
    registry, observation, _, _ = recovery_chain
    registry["runs"][1]["superseded_by"] = "successor"
    registry["runs"].append({"id": "successor", "step_id": "14372.35"})
    observation["findings"].append({"run_id": "successor", "status": "complete"})
    if cycle:
        registry["runs"][-1]["superseded_by"] = "complete"
    assert waiting_recoveries(registry, observation) == ({} if cycle else {"failed": "E18"})


def test_optional_guard_validation_runs_under_lock_before_spawn(tmp_path, queue):
    validated = []

    def guard(launch):
        with (tmp_path / "registry.json.lock").open("a") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validated.append(launch)
        raise ValueError("Required storage guard missing")

    result = dispatch.dispatch_ready(
        tmp_path, "14372", {"14372.0": "RUNNING"}, supervisor_step="35", validate_launch=guard
    )
    assert result["status"] == "failed"
    assert validated and not queue[2]
    state = load(tmp_path / "dispatch-state.json")["E18"]
    assert state["may_have_started"] is False
    assert "storage guard" in dispatch.dispatch_incidents(tmp_path)[0]["reason"]


def test_new_guard_does_not_invalidate_already_started_receipts(tmp_path, queue):
    _, batch, calls = queue
    tick(tmp_path)
    receipt(batch)
    result = dispatch.dispatch_ready(
        tmp_path,
        "14372",
        {"14372.0": "RUNNING", "14372.36": "RUNNING"},
        supervisor_step="35",
        validate_launch=lambda launch: pytest.fail("already launched"),
    )
    assert result["status"] == "registered"
    assert len(calls) == 1


def _two_device_queue(tmp_path, queue, *, first="5090", second="2080"):
    import copy

    registry, batch, calls = queue
    registry["capacity"] = {"gpus": 2, "cpus": 14, "memory_gib": 100}
    registry["gpu_devices"] = {
        "5090": {"gres": "gpu:rtx5090:1"},
        "2080": {"gres": "gpu:rtx2080ti:1"},
    }
    batch["resources"]["memory_gib"] = 40
    other = copy.deepcopy(batch)
    other["id"] = other["launch"]["run"]["id"] = "independent"
    other["launch"]["nonce"] = "test-independent-nonce"
    other["launch"]["step_path"] = str(tmp_path / "second-step.json")
    other["launch"]["launcher_log"] = str(tmp_path / "second-launcher.log")
    for row, device in ((batch, first), (other, second)):
        if device is not None:
            row["gpu_devices"] = [device]
            row["launch"]["argv"].insert(2, "--gres=" + registry["gpu_devices"][device]["gres"])
    registry["pending_batches"].append(other)
    save(tmp_path / "registry.json", registry)
    return registry, batch, other, calls


def test_distinct_gpu_reservations_can_overlap_before_first_receipt(tmp_path, queue):
    registry, first, second, calls = _two_device_queue(tmp_path, queue)
    assert tick(tmp_path)["batch_id"] == first["id"]
    assert tick(tmp_path)["batch_id"] == second["id"]
    assert len(calls) == 2
    state = load(tmp_path / "dispatch-state.json")
    assert state[first["id"]]["gpu_devices"] == ["5090"]
    assert state[second["id"]]["gpu_devices"] == ["2080"]


@pytest.mark.parametrize("first,second", [("5090", "5090"), (None, "2080"), ("5090", None)])
def test_shared_or_unspecified_gpu_owner_remains_exclusive(tmp_path, queue, first, second):
    _, _, _, calls = _two_device_queue(tmp_path, queue, first=first, second=second)
    assert tick(tmp_path)["status"] == "starting"
    assert tick(tmp_path)["status"] == "no_ready_launch"
    assert len(calls) == 1


def test_registered_gpu_ownership_survives_receipt_and_admits_other_gpu(tmp_path, queue):
    _, first, second, calls = _two_device_queue(tmp_path, queue)
    tick(tmp_path)
    receipt(first)
    steps = {"14372.0": "RUNNING", "14372.36": "RUNNING"}
    assert tick(tmp_path, steps)["status"] == "registered"
    assert load(tmp_path / "registry.json")["runs"][0]["gpu_devices"] == ["5090"]
    assert tick(tmp_path, steps)["batch_id"] == second["id"]
    assert len(calls) == 2


@pytest.mark.parametrize("resource,limit", [("cpus", 7), ("memory_gib", 79)])
def test_distinct_gpus_do_not_bypass_aggregate_cpu_or_ram(tmp_path, queue, resource, limit):
    registry, _, _, calls = _two_device_queue(tmp_path, queue)
    registry["capacity"][resource] = limit
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["status"] == "starting"
    assert tick(tmp_path)["status"] == "no_ready_launch"
    assert len(calls) == 1


def test_declared_device_must_match_actual_slurm_gres(tmp_path, queue):
    registry, first, _, calls = _two_device_queue(tmp_path, queue)
    first["gpu_devices"] = ["2080"]
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["status"] == "failed"
    assert not calls
    assert "does not match" in load(tmp_path / "dispatch-state.json")[first["id"]]["error"]


def test_named_profile_must_match_cpu_ram_slurm_flags(tmp_path, queue):
    registry, first, _, calls = _two_device_queue(tmp_path, queue)
    first["resources"] = {"cpus": 4, "gpus": 1, "memory_mb": 24576}
    first["resource_profile"] = "training"
    registry["resource_profiles"] = {"training": dict(first["resources"])}
    registry["capacity"]["memory_mb"] = 102400
    first["launch"]["argv"][2:2] = ["--cpus-per-task=4", "--mem=32768"]
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["status"] == "failed"
    assert not calls
    assert "CPU/RAM" in load(tmp_path / "dispatch-state.json")[first["id"]]["error"]


def test_primary_priority_and_future_window_are_deterministic(tmp_path, queue, monkeypatch):
    registry, first, second, _ = _two_device_queue(tmp_path, queue)
    monkeypatch.setattr(dispatch.time, "time", lambda: 100)
    second.update(priority=10, not_before_epoch=100, deadline_epoch=200)
    first.update(priority=20, not_before_epoch=110)
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["batch_id"] == second["id"]


def test_expired_unstarted_batch_retains_denominator_and_costs_without_launch(
    tmp_path, queue, monkeypatch
):
    registry, batch, calls = queue
    monkeypatch.setattr(dispatch.time, "time", lambda: 100)
    ledger = tmp_path / "costs.json"
    saved_costs = {"attempts": {"old-failure": {"elapsed_seconds": 123}}, "cumulative": 123}
    save(ledger, saved_costs)
    batch.update(deadline_epoch=99, deadline_policy="defer", budget_path=str(ledger))
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path) == {"status": "deferred_deadline", "batch_id": batch["id"]}
    current = load(tmp_path / "registry.json")
    assert current["pending_batches"] == [] and not calls
    row = current["deferred_batches"][0]
    assert row["batch"] == batch and row["status"] == "unavailable_deadline"
    assert row["costs_reset"] is False and row["attempt_launched"] is False
    assert load(ledger) == saved_costs


def test_deadline_cannot_discard_inflight_ownership_or_extend_contract(
    tmp_path, queue, monkeypatch
):
    registry, batch, calls = queue
    clock = [80]
    monkeypatch.setattr(dispatch.time, "time", lambda: clock[0])
    batch.update(deadline_epoch=90, deadline_policy="defer")
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["status"] == "starting"
    clock[0] = 100
    receipt(batch)
    assert tick(tmp_path, {"14372.0": "RUNNING", "14372.36": "RUNNING"})["status"] == "registered"
    current = load(tmp_path / "registry.json")
    assert current["runs"][0]["deadline_epoch"] == 90
    assert not current.get("deferred_batches") and len(calls) == 1


def test_dispatch_reserves_bound_cumulative_budget_before_detached_spawn(
    tmp_path, queue, monkeypatch
):
    registry, descriptor, calls = queue
    monkeypatch.setattr(dispatch.time, "time", lambda: 100)
    descriptor["resources"] = {"cpus": 1, "gpus": 0, "memory_mb": 512}
    registry["capacity"] = dict(descriptor["resources"])
    ledger = tmp_path / "costs.json"
    save(
        ledger,
        {
            "limit_worker_seconds": None,
            "attempts": {},
            "stage_limits": {"calibration": {"elapsed_seconds": 50}},
        },
    )
    job = {
        "id": "science",
        "seconds": 100,
        "resources": descriptor["resources"],
        "budget_stages": ["calibration"],
    }
    manifest = tmp_path / "batch.json"
    code = Path(__file__).resolve().parents[1]
    adapter = code / "tools/repair/batch.py"
    save(
        manifest,
        {
            "jobs": [job],
            "ledger": str(ledger),
            "python": sys.executable,
            "code": str(code),
            "frozen_files": {str(adapter): hashlib.sha256(adapter.read_bytes()).hexdigest()},
        },
    )
    detached = dispatch.subprocess.run

    def run_adapter(argv, **kwargs):
        if "reserve-launch" in argv:
            assert kwargs["timeout"] == 30 and kwargs["cwd"] == code
            return REAL_RUN(argv, **kwargs)
        return detached(argv, **kwargs)

    monkeypatch.setattr(dispatch.subprocess, "run", run_adapter)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "batch.sha256").write_text(digest)
    descriptor["launch"]["bindings"].append({"path": str(manifest), "sha256": digest})
    descriptor["budget_reservation"] = {
        "batch": str(manifest),
        "job_id": "science",
        "attempt": str(tmp_path),
    }
    save(tmp_path / "registry.json", registry)
    assert tick(tmp_path)["status"] == "starting"
    row = load(ledger)["attempts"][str(tmp_path)]
    assert row["reserved_seconds"] == 50 and row["started_epoch"] > 0
    assert load(tmp_path / "dispatch-state.json")[descriptor["id"]]["reserved_worker_seconds"] == 50
    assert len(calls) == 1
    assert tick(tmp_path)["status"] == "no_ready_launch"
    assert len(load(ledger)["attempts"]) == 1
