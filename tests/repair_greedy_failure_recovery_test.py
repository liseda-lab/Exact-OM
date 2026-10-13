import copy
import time

import pytest

from exact.repair.api import write_artifact
from exact.repair.workers import CallResult
from tests.repair_primary_evaluation_test import fixture_manifest
from tools.repair.batch import read
from tools.repair.greedy_failure_recovery import recovery_state
from tools.repair.historical_regression import binding
from tools.repair.primary_evaluation import execute_rows


def prepared(tmp_path, monkeypatch, status):
    manifest = fixture_manifest(tmp_path)
    greedy = copy.deepcopy(manifest["rows"][0])
    greedy["id"] = "greedy"
    greedy["adapter"]["operation"] = "observable_score_greedy"
    manifest["rows"].append(greedy)
    outcomes = iter(
        [
            CallResult(
                status, value={"logical_status": "UNKNOWN"} if status == "complete" else None
            ),
            CallResult("error", detail="StopIteration: "),
        ]
    )
    monkeypatch.setattr("exact.repair.workers.bounded_call", lambda *a, **kw: next(outcomes))
    work = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="StopIteration"):
        execute_rows(manifest, work)
    write_artifact(tmp_path / "manifest.json", manifest)
    documents = {
        "batch": dict(jobs=[dict(id="job", commands=[[str(tmp_path / "manifest.json")]])]),
        "step": dict(step_id="1.2", dispatch_nonce="old"),
        "completion": dict(
            step_id="1.2",
            dispatch_nonce="old",
            status="failed",
            exit_code=1,
            job_id="job",
            batch=str(tmp_path / "batch.json"),
            work=str(work),
            finished_epoch=time.time() - 1,
        ),
        "outputs": {str(p.relative_to(work)): binding(p)["sha256"] for p in work.rglob("*.json")},
        "retirement": dict(
            retired_step="1.2",
            live_steps=[],
            all_owned_processes_retired=True,
            observed_epoch=time.time(),
        ),
    }
    for name, value in documents.items():
        write_artifact(tmp_path / (name + ".json"), value)
    receipt = {
        k: binding(tmp_path / (k + ".json")) for k in ("batch", "step", "completion", "outputs")
    }
    receipt.update(step_id="1.2", dispatch_nonce="old", expected_status="failed")
    write_artifact(
        tmp_path / "recovery.json",
        dict(
            schema="exact-repair/primary-greedy-boundary-recovery/v1",
            manifest=binding(tmp_path / "manifest.json"),
            receipt=receipt,
            retirement=binding(tmp_path / "retirement.json"),
            failed_row_id="greedy",
        ),
    )
    return dict(
        manifest,
        source_commit="fixed",
        dispatch_nonce="new",
        greedy_boundary_recovery=binding(tmp_path / "recovery.json"),
    )


@pytest.mark.parametrize("status", ["complete", "timeout"])
def test_completed_unknown_or_timeout_is_never_replayed(tmp_path, monkeypatch, status):
    manifest = prepared(tmp_path, monkeypatch, status)
    costs, retained = recovery_state(manifest)
    old = read(tmp_path / "failed/rows/native/completion.json")
    assert retained["native"]["result"] == old["result"]
    assert retained["native"]["charged_seconds"] == old["charged_seconds"]
    calls = []

    def finish(function, row, *a, **kw):
        calls.append((row["id"], kw["timeout"]))
        return CallResult("complete", value={"logical_status": "UNKNOWN"})

    monkeypatch.setattr("exact.repair.workers.bounded_call", finish)
    report = execute_rows(manifest, tmp_path / "recovered")
    assert len(calls) == 1 and calls[0][0] == "greedy"
    assert calls[0][1] <= 35 - costs["greedy"]
    assert report["rows"][0]["retained_row"] == binding(
        tmp_path / "failed/rows/native/completion.json"
    )
    assert report["rows"][1]["prior_charged_seconds"] == costs["greedy"]
    assert execute_rows(manifest, tmp_path / "recovered") == report and len(calls) == 1
    changed = copy.deepcopy(manifest)
    changed["rows"][1]["case_seconds"] += 1
    with pytest.raises(ValueError, match="scientific inputs or budgets"):
        recovery_state(changed)


def test_retained_receipt_cannot_be_rewritten(tmp_path, monkeypatch):
    manifest = prepared(tmp_path, monkeypatch, "complete")
    path = tmp_path / "failed/rows/native/completion.json"
    value = read(path)
    value["result"]["logical_status"] = "VERIFIED_FEASIBLE"
    write_artifact(path, value)
    with pytest.raises(ValueError, match="dependency changed"):
        recovery_state(manifest)
