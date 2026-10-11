"""A cleanup recovery preserves fixed rows, authenticated failure and prior costs."""

import copy

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.historical_regression import binding
from tools.repair.representative_capacity import recovery_rows


def recovery_fixture(tmp_path):
    row = dict(id="fixed", case_seconds=40)
    prior = dict(rows=[row], source_commit="old", dispatch_nonce="old", models={"fixed": "weights"})
    write_artifact(tmp_path / "manifest.json", prior)
    saved = dict(
        identity=canonical_hash((canonical_hash(prior), row)),
        status="cleanup_incomplete",
        cleanup_complete=False,
        detail="nested worker cleanup incomplete",
        charged_seconds=17,
    )
    work = tmp_path / "work"
    write_artifact(work / "rows/fixed/completion.json", saved)
    write_artifact(
        work / "report.json",
        dict(manifest_identity=canonical_hash(prior), rows=[dict(id="fixed", **saved)]),
    )
    write_artifact(
        tmp_path / "outputs.json",
        {str(p.relative_to(work)): binding(p)["sha256"] for p in work.rglob("*.json")},
    )
    write_artifact(
        tmp_path / "batch.json",
        dict(jobs=[dict(id="failed", commands=[[str(tmp_path / "manifest.json")]])]),
    )
    completion = dict(
        status="failed",
        exit_code=1,
        step_id="14451.129",
        dispatch_nonce="old",
        batch=str(tmp_path / "batch.json"),
        job_id="failed",
        work=str(work),
        finished_epoch=100,
    )
    write_artifact(tmp_path / "completion.json", completion)
    write_artifact(tmp_path / "step.json", dict(step_id="14451.129", dispatch_nonce="old"))
    write_artifact(
        tmp_path / "retirement.json",
        dict(
            retired_step="14451.129",
            live_steps=["14451.128"],
            observed_epoch=101,
            all_owned_processes_retired=True,
        ),
    )
    recovery = dict(
        manifest=binding(tmp_path / "manifest.json"),
        failed_row_id="fixed",
        retirement=binding(tmp_path / "retirement.json"),
        receipt=dict(
            **{
                key: binding(tmp_path / (key + ".json"))
                for key in ("batch", "completion", "step", "outputs")
            },
            expected_status="failed",
            step_id="14451.129",
            dispatch_nonce="old"
        ),
    )
    write_artifact(tmp_path / "recovery.json", recovery)
    return dict(
        prior,
        source_commit="fixed",
        dispatch_nonce="new",
        technical_recovery=binding(tmp_path / "recovery.json"),
    )


def test_recovery_returns_original_charge_and_rejects_changed_rows(tmp_path):
    manifest = recovery_fixture(tmp_path)
    retained = recovery_rows(manifest)["fixed"]
    assert retained["charged_seconds"] == retained["prior_charged_seconds"] == 17
    assert retained["status"] == "prior_implementation_failure" and retained["result"] is None
    assert retained["prior_cleanup_complete"] is False and retained["cleanup_complete"] is True
    changed = copy.deepcopy(manifest)
    changed["rows"][0]["case_seconds"] = 50
    with pytest.raises(ValueError, match="scientific rows or limits"):
        recovery_rows(changed)


@pytest.mark.parametrize("damage", ["live", "nonce", "payload", "scientific_timeout"])
def test_recovery_rejects_unretired_or_unbound_or_scientific_attempt(tmp_path, damage):
    from tools.repair.batch import read

    manifest = recovery_fixture(tmp_path)
    recovery = read(tmp_path / "recovery.json")
    if damage == "live":
        retirement = read(tmp_path / "retirement.json")
        retirement["live_steps"].append("14451.129")
        write_artifact(tmp_path / "retirement.json", retirement)
        recovery["retirement"] = binding(tmp_path / "retirement.json")
    elif damage == "nonce":
        recovery["receipt"]["dispatch_nonce"] = "wrong"
    elif damage == "payload":
        (tmp_path / "work/rows/fixed/completion.json").write_text("{}")
    else:
        path = tmp_path / "work/rows/fixed/completion.json"
        saved = read(path)
        saved.update(status="timeout", cleanup_complete=True)
        write_artifact(path, saved)
        report = read(tmp_path / "work/report.json")
        report["rows"] = [dict(id="fixed", **saved)]
        write_artifact(tmp_path / "work/report.json", report)
        write_artifact(
            tmp_path / "outputs.json",
            {
                str(p.relative_to(tmp_path / "work")): binding(p)["sha256"]
                for p in (tmp_path / "work").rglob("*.json")
            },
        )
        recovery["receipt"]["outputs"] = binding(tmp_path / "outputs.json")
    write_artifact(tmp_path / "recovery.json", recovery)
    manifest["technical_recovery"] = binding(tmp_path / "recovery.json")
    with pytest.raises((ValueError, RuntimeError)):
        recovery_rows(manifest)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_recovery_never_replays_expired_case_and_runs_remaining_rows(tmp_path, monkeypatch, device):
    from tests.repair_primary_evaluation_test import fixture_manifest
    from exact.repair.workers import CallResult
    from tools.repair.primary_evaluation import execute_rows

    manifest = fixture_manifest(tmp_path)
    manifest["device"] = device
    manifest.update(scope="representative_development", ledger_path=str(tmp_path / "ledger.json"))
    write_artifact(tmp_path / "ledger.json", dict(stages=dict(learning=dict(started_epoch=100))))
    manifest["deadline_epoch"] = 1000
    monkeypatch.setattr(
        "tools.repair.representative_capacity.validate_manifest", lambda *a, **k: None
    )
    original = manifest["rows"][0]
    manifest["rows"].append(dict(original, id="unattempted"))
    retained = dict(
        identity=canonical_hash((canonical_hash(manifest), original)),
        status="prior_implementation_failure",
        result=None,
        cleanup_complete=True,
        charged_seconds=17,
        prior_charged_seconds=17,
        artifacts=[],
    )
    monkeypatch.setattr(
        "tools.repair.representative_capacity.recovery_rows", lambda m: {"native": retained}
    )
    monkeypatch.setattr("tools.repair.evaluation_ownership.validate_owner", lambda *a: None)
    monkeypatch.setattr("tools.repair.evaluation_ownership.external_pools", lambda m: {})
    monkeypatch.setattr("tools.repair.primary_evaluation.time.time", lambda: 100)
    calls = []
    monkeypatch.setattr(
        "exact.repair.workers.bounded_call",
        lambda *a, **k: calls.append(k) or CallResult("timeout"),
    )
    report = execute_rows(manifest, tmp_path / "run", now=lambda: 100)
    assert len(calls) == 1  # Only the unattempted case runs.
    assert calls[0]["timeout"] == 35  # 40 original -5 case cleanup
    assert calls[0]["cleanup_seconds"] == (2.0 if device == "cuda" else 0.2)
    assert report["rows"][0]["charged_seconds"] == 17
    assert report["rows"][0]["prior_charged_seconds"] == 17
