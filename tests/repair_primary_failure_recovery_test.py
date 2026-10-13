import copy
import time

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tests.repair_primary_evaluation_test import fixture_manifest
from tools.repair.historical_regression import binding
from tools.repair.primary_evaluation import execute_rows
from tools.repair.primary_failure_recovery import CAUSE, prior_costs


def recovery(tmp_path, *, detail=CAUSE, cleanup=True):
    original = fixture_manifest(tmp_path)
    work = tmp_path / "failed"
    row = original["rows"][0]
    saved = dict(
        identity=canonical_hash((canonical_hash(original), row)),
        status="error",
        result=None,
        detail=detail,
        cleanup_complete=cleanup,
        charged_seconds=7.0,
        reserved_seconds=row["case_seconds"],
        artifacts=[],
    )
    relative = "rows/" + row["id"] + "/completion.json"
    documents = {
        "manifest.json": original,
        "step.json": dict(step_id="1.2", dispatch_nonce="old"),
        "batch.json": dict(jobs=[dict(id="job", commands=[[str(tmp_path / "manifest.json")]])]),
        "retirement.json": dict(
            retired_step="1.2",
            all_owned_processes_retired=True,
            live_steps=[],
            observed_epoch=time.time(),
        ),
    }
    write_artifact(work / relative, saved)
    write_artifact(
        work / "report.json",
        dict(manifest_identity=canonical_hash(original), rows=[dict(id=row["id"], **saved)]),
    )
    documents["outputs.json"] = {n: binding(work / n)["sha256"] for n in (relative, "report.json")}
    documents["completion.json"] = dict(
        status="failed",
        exit_code=1,
        job_id="job",
        step_id="1.2",
        dispatch_nonce="old",
        batch=str(tmp_path / "batch.json"),
        work=str(work),
        finished_epoch=time.time() - 1,
    )
    for name, value in documents.items():
        write_artifact(tmp_path / name, value)
    receipt = {
        k: binding(tmp_path / (k + ".json")) for k in ("completion", "step", "outputs", "batch")
    }
    receipt.update(step_id="1.2", dispatch_nonce="old", expected_status="failed")
    write_artifact(
        tmp_path / "recovery.json",
        dict(
            schema="exact-repair/primary-feature-boundary-recovery/v1",
            manifest=binding(tmp_path / "manifest.json"),
            receipt=receipt,
            retirement=binding(tmp_path / "retirement.json"),
        ),
    )
    return dict(
        original,
        source_commit="fixed",
        dispatch_nonce="new",
        feature_boundary_recovery=binding(tmp_path / "recovery.json"),
    )


def test_failed_row_only_uses_remaining_budget_and_keeps_prior_cost(tmp_path, monkeypatch):
    manifest = recovery(tmp_path)
    assert prior_costs(manifest) == {"native": 7.0}
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return CallResult("timeout", detail="remaining budget spent")

    monkeypatch.setattr("exact.repair.workers.bounded_call", evaluate)
    result = execute_rows(manifest, tmp_path / "recovered")
    assert len(calls) == 1 and 0 < calls[0] <= 28
    row = result["rows"][0]
    assert row["prior_charged_seconds"] == 7 and row["charged_seconds"] >= 7
    assert execute_rows(manifest, tmp_path / "recovered") == result and len(calls) == 1
    changed = copy.deepcopy(manifest)
    changed["rows"][0]["case_seconds"] += 10
    with pytest.raises(ValueError, match="scientific inputs or budgets"):
        prior_costs(changed)


@pytest.mark.parametrize("detail,cleanup", [("ordinary timeout", True), (CAUSE, False)])
def test_unknown_cleanup_and_undiagnosed_outcomes_cannot_renew_a_row(tmp_path, detail, cleanup):
    with pytest.raises(ValueError, match="bounded, cleaned-up"):
        prior_costs(recovery(tmp_path, detail=detail, cleanup=cleanup))
