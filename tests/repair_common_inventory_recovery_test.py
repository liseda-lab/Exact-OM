"""Recovery must retain failures, deny replay and bind the original execution."""
import copy
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair import common_inventory_recovery as recovery
from tools.repair.expanded_corpus import binding
from tools.repair.expanded_profile import checkpoint

STEP = "14408.380"


def fixture():
    row = dict(id="row5", seconds=300, cpu_seconds=600)
    saved = dict(row=row, status="error", detail=recovery.CAUSE, cleanup_complete=True,
        result=None, payloads=[], elapsed_seconds=237.3, resources={"cpu_seconds": 37.02},
        identity="original", content_hash="old")
    guard = dict(row=copy.deepcopy(row), identity="original")
    owner = dict(step_id=STEP, surviving_processes=0, cgroup_absent=True, signals_sent=0)
    return saved, guard, owner


def test_reconciliation_preserves_unknown_error_costs_and_originals():
    saved, guard, owner = fixture()
    original = copy.deepcopy(saved)
    result = recovery.reconciled_unknown(saved, guard, {"path": "old"}, {"path": "proof"}, owner, STEP)
    assert saved == original
    assert result["status"] == "unknown_after_cleanup_reconciliation" and result["result"] is None
    assert result["original_status"] == "error" and result["original_detail"] == recovery.CAUSE
    for key in ("resources", "elapsed_seconds", "payloads", "row"):
        assert result[key] == original[key]
    assert result["additional_scientific_seconds"] == 0 and not result["prior_costs_reset"]


@pytest.mark.parametrize("change", ["other_error", "positive_result", "outer_cleanup", "different_row",
    "identity", "live_owner", "wrong_step", "signals", "cgroup"])
def test_reject_unqualified_evidence(change):
    saved, guard, owner = fixture()
    if change == "other_error": saved["detail"] = "ValueError: unrelated crash"
    elif change == "positive_result": saved["result"] = {"assignment": [0]}
    elif change == "outer_cleanup": saved["cleanup_complete"] = False
    elif change == "different_row": guard["row"]["id"] = "other"
    elif change == "identity": guard["identity"] = "different"
    elif change == "live_owner": owner["surviving_processes"] = 1
    elif change == "wrong_step": owner["step_id"] = "14408.0"
    elif change == "signals": owner["signals_sent"] = 1
    else: owner["cgroup_absent"] = False
    with pytest.raises(ValueError):
        recovery.reconciled_unknown(saved, guard, {}, {}, owner, STEP)


@pytest.mark.parametrize("change", ["replay_failed", "skip_row", "cost_reset", "wrong_count"])
def test_reject_changed_slice(change):
    plan = dict(original_slice=[0, 16], failed_index=5, only_unstarted_slice=[6, 16],
                original_row_count=16, prior_costs_reset=False)
    if change == "replay_failed": plan["only_unstarted_slice"] = [5, 16]
    elif change == "skip_row": plan["failed_index"] = 6
    elif change == "cost_reset": plan["prior_costs_reset"] = True
    else: plan["original_row_count"] = 15
    with pytest.raises(ValueError): recovery.slice_bounds(plan)


@pytest.mark.parametrize("change", [None, "nonce", "command", "missing_prefix", "row_identity",
    "guard_identity", "spent_next_row", "spent_next_payload", "finished_guard",
    "wrong_command_index", "other_error", "different_attempt", "job_command", "payload"])
def test_original_validation_binds_prefix_guard_and_unstarted_rows(tmp_path, change):
    work = tmp_path / "work" / "inventory"
    rows = [dict(id=f"row{i}") for i in range(3)]
    schedule = tmp_path / "schedule.json"; write_artifact(schedule, dict(rows=rows))
    argv = ["python", "-m", "tools.repair.common_inventory", str(schedule), str(work),
            "--start", "0", "--stop", "3"]
    frozen = tmp_path / "batch.json"
    write_artifact(frozen, dict(python="python", code="/original/code", jobs=[dict(id="job",
        commands=[argv[:-1] + ["4"] if change == "job_command" else argv])]))
    plan = dict(original_slice=[0, 3], failed_index=1, only_unstarted_slice=[2, 3],
        original_row_count=3, prior_costs_reset=False, original_step=STEP,
        frozen_batch=binding(frozen), source_root="/original/code", schedule=binding(schedule))
    receipt = tmp_path / "completion.json"
    write_artifact(receipt, dict(status="failed", exit_code=1, step_id=STEP, dispatch_nonce="nonce",
        error={"message": "other failure" if change == "other_error" else recovery.COMMAND_ERROR},
        batch=str(frozen), work=str(work.parent), job_id="job"))
    owner = tmp_path / "step.json"
    write_artifact(owner, dict(step_id=STEP, dispatch_nonce="wrong" if change == "nonce" else "nonce"))
    command = (tmp_path / "other-attempt" if change == "different_attempt" else tmp_path) / (
        "command-1.json" if change == "wrong_command_index" else "command-0.json")
    write_artifact(command, dict(cwd="/wrong" if change == "command" else "/original/code", argv=argv))
    refs = []
    payload = work / "payload.json"; write_artifact(payload, {})
    for i in range(2):
        path = work / "rows" / f"row{i}.json"
        identity = canonical_hash(("science", rows[i]))
        checkpoint(path, "changed" if change == "row_identity" else identity,
            row=rows[i], status="complete" if i == 0 else "error", detail="" if i == 0 else recovery.CAUSE,
            cleanup_complete=True, payloads=[binding(payload)], result=None)
        refs.append(dict(index=i, receipt=binding(path)))
    guard = work / "inflight" / "row1.json"
    checkpoint(guard, "changed" if change == "guard_identity" else canonical_hash(("science", rows[1])), row=rows[1])
    if change == "spent_next_row": write_artifact(work / "rows" / "row2.json", {})
    if change == "spent_next_payload": (work / "payloads" / "row2").mkdir(parents=True)
    if change == "finished_guard": write_artifact(work / "inflight" / "row0.json", {})
    if change == "payload": write_artifact(payload, {"changed": True})
    evidence = dict(completion=binding(receipt), ownership=binding(owner), command=binding(command),
                    rows=refs[1:] if change == "missing_prefix" else refs, guard=binding(guard))
    if change:
        with pytest.raises(ValueError): recovery.validate_original(plan, evidence, "science")
    else:
        assert set(recovery.validate_original(plan, evidence, "science")) == {0, 1}


def test_runner_reuses_prefix_preserves_guard_and_denominator(tmp_path, monkeypatch):
    from exact.experiments.science_health import inspect_science
    from exact.repair import study
    from tools.repair import common_inventory as science, batch
    from tools.repair.batch import sha

    source = tmp_path / "source"; tool = source / "tools/repair"; tool.mkdir(parents=True)
    for name in ("common_inventory.py", "fresh_evaluation.py"): (tool / name).write_text("original source")
    monkeypatch.setattr(science, "__file__", str(tool / "common_inventory.py"))
    monkeypatch.setattr(science.fresh, "__file__", str(tool / "fresh_evaluation.py"))
    monkeypatch.setattr(study, "runtime_manifest", lambda: {"fixture": "same"})
    monkeypatch.setattr(science, "validate_schedule", lambda s: None)
    work = tmp_path / "work"; work.mkdir()
    rows = [{"id": f"row{i}"} for i in range(16)]
    schedule = tmp_path / "schedule.json"; write_artifact(schedule, {"rows": rows, "followup": "xr21-expanded-evaluation-001"})
    completion = tmp_path / "failed-completion.json"; write_artifact(completion, {"work": str(work)})
    previous, guard, owner = fixture(); previous["row"] = rows[5]; guard["row"] = rows[5]
    prior = {}
    for index in range(5):
        payload = work / f"result-{index}.json"
        write_artifact(payload, {"row_id": f"row{index}", "status": "timeout"})
        path = work / f"original-{index}.json"
        saved = checkpoint(path, "old", row=rows[index], status="complete", result=binding(payload))
        prior[index] = (binding(path), saved)
    failed = work / "failed.json"; write_artifact(failed, previous); prior[5] = (binding(failed), previous)
    guard_path = work / "guard.json"; write_artifact(guard_path, guard)
    evidence = tmp_path / "evidence.json"
    write_artifact(evidence, {"guard": binding(guard_path), "completion": binding(completion)})
    originals = {p: Path(p).read_bytes() for p in [guard_path, *[ref["path"] for ref, _ in prior.values()]]}
    monkeypatch.setattr(recovery, "validate_original", lambda p, e, i: prior)
    requested, executed = [], []
    def continue_rows(plan, output, start, stop):
        requested.append((start, stop)); refs = []
        for index in range(start, stop):
            path = output / f"row{index}.json"
            if not path.exists():
                executed.append(index); checkpoint(path, "original", row=rows[index], status="timeout", result=None)
            refs.append(dict(binding(path), row_id=f"row{index}", status="timeout"))
        return dict(scheduled=stop-start, recorded=stop-start, rows=refs)
    monkeypatch.setattr(science, "run", continue_rows)
    frozen = tmp_path / "batch.json"; write_artifact(frozen, {"code": str(source)})
    monkeypatch.setattr(batch, "checked_batch", lambda p: {"code": str(source)})
    identity = canonical_hash((sha(schedule), sha(science.__file__), sha(science.fresh.__file__), {"fixture": "same"}))
    output = work / "inventory-revision-002"; plan = tmp_path / "plan.json"
    write_artifact(plan, dict(runner=binding(recovery.__file__), output=str(output), source_root=str(source),
        runtime={"fixture": "same"}, schedule=binding(schedule), scientific_identity=identity,
        evidence=binding(evidence), frozen_batch=binding(frozen), original_step=STEP,
        original_slice=[0, 16], failed_index=5, only_unstarted_slice=[6, 16], original_row_count=16, prior_costs_reset=False))
    first = recovery.run(plan, output, owner_check=lambda s: owner)
    second = recovery.run(plan, output, owner_check=lambda s: owner)
    assert first == second and requested == [(6, 16)] * 2 and executed == list(range(6, 16))
    assert first["scheduled"] == first["recorded"] == 16
    assert first["outcomes"] == {"timeout": 15, "unknown_after_cleanup_reconciliation": 1}
    assert all(Path(path).read_bytes() == raw for path, raw in originals.items())
    assert not first["supervision_admitted"] and not first["generation"] and not first["gates_passed"]
    assert first["common_inventory_regret"] is None
    assert first["followup"] == "xr21-expanded-evaluation-001"
    attempt = tmp_path / "attempt"; attempt.mkdir()
    write_artifact(attempt / "outputs.json", {"inventory-revision-002/science-health.json": sha(output / "science-health.json")})
    run = dict(step_id="14408.999", dispatch_nonce="nonce", science_report_relative="inventory-revision-002/science-health.json",
               completion_path=str(attempt / "completion.json"))
    complete = dict(status="complete", step_id=run["step_id"], dispatch_nonce="nonce", work=str(work))
    assert inspect_science(run, complete) == dict(failures=[], errors=[])
