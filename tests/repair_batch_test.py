"""Frozen batch integrity and cumulative reservation accounting, without Slurm jobs."""

import json

import pytest

from tools.repair.batch import charge, checked_batch, run


def test_unlimited_campaign_preserves_costs_and_finite_worker_reservations(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"limit_worker_seconds": None, "attempts": {}}))
    job = {"id": "one", "seconds": 50, "resources": {"cpus": 2, "gpus": 0, "memory_mb": 512}}
    assert charge(ledger, job, "first") == 50
    charge(ledger, job, "first", elapsed=20, cpu_seconds=10)
    assert charge(ledger, job, "retry") == 30
    with pytest.raises(TimeoutError):
        charge(ledger, job, "duplicate-work")
    assert charge(ledger, {**job, "id": "independent"}, "next") == 50
    value = json.loads(ledger.read_text())
    assert value["cumulative"]["worker_seconds"] == 100
    assert value["attempts"]["first"]["elapsed_seconds"] == 20


def test_replacement_costs_and_unsettled_reservations_are_never_reset(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"limit_worker_seconds": 100, "attempts": {}}))
    job = {"id": "label", "seconds": 50, "resources": {"cpus": 2, "gpus": 1, "memory_mb": 4096}}
    assert charge(ledger, job, "one") == 50
    charge(ledger, job, "one", elapsed=12, cpu_seconds=5, peak_rss_mb=100)
    assert charge(ledger, job, "replacement") == 38
    with pytest.raises(TimeoutError, match="budget"):
        charge(ledger, job, "third")
    totals = json.loads(ledger.read_text())["cumulative"]
    assert totals["worker_seconds"] == 50
    assert totals["allocated_gpu_seconds"] == 50
    assert totals["allocated_cpu_seconds"] == 100
    assert totals["measured_cpu_seconds"] == 5


def test_changed_frozen_manifest_is_rejected_before_commands(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text('{"changed": true}')
    (tmp_path / "batch.sha256").write_text("invalid")
    with pytest.raises(ValueError, match="manifest changed"):
        checked_batch(path)


def test_worker_preserves_failure_receipt_and_cost(tmp_path, monkeypatch):
    import sys

    from tools.repair import batch

    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"limit_worker_seconds": 30, "attempts": {}}))
    job = {
        "id": "fixture",
        "seconds": 10,
        "resources": {"cpus": 1, "gpus": 0, "memory_mb": 512},
        "commands": [["{python}", "-c", "raise ValueError('confirmed test failure')"]],
    }
    manifest = dict(
        jobs=[job],
        campaign=str(tmp_path),
        ledger=str(ledger),
        code=str(tmp_path),
        protocol="unused",
        python=sys.executable,
    )
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(manifest))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "2")
    monkeypatch.setattr(batch, "checked_batch", lambda _: manifest)
    assert run(path, "fixture", attempt) == 1
    completion = json.loads((attempt / "completion.json").read_text())
    assert completion["status"] == "failed" and completion["step_id"] == "123.2"
    assert "confirmed test failure" in completion["error"]["message"]
    value = json.loads(ledger.read_text())
    assert value["cumulative"]["worker_seconds"] > 0
    assert value["attempts"][str(attempt)]["status"] == "settled"


def test_freeze_records_export_runtime_instead_of_callers_editable_installation(tmp_path):
    import subprocess
    import sys

    from tools.repair.batch import freeze

    repo = tmp_path / "repository"
    package = repo / "exact" / "repair"
    package.mkdir(parents=True)
    (repo / "exact" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "study.py").write_text(
        "def runtime_manifest():\n    return {'identity': 'committed-export-runtime'}\n"
    )
    (repo / "protocol.json").write_text("{}")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Fixture source",
        ],
        cwd=repo,
        check=True,
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            dict(repository=str(repo), protocol_source="protocol.json", python=sys.executable)
        )
    )
    destination = tmp_path / "export"
    freeze(spec, destination)
    assert json.loads((destination / "runtime.json").read_text()) == {
        "identity": "committed-export-runtime"
    }
