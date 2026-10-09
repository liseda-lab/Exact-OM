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
        "commands": [
            [
                "{python}",
                "-c",
                "import json; from pathlib import Path; "
                "Path('{work}/report.json').write_text(json.dumps(dict(status='blocked_external'))); "
                "raise ValueError('confirmed test failure')",
            ]
        ],
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
    assert "report.json" in json.loads((attempt / "outputs.json").read_text())


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


def _staged(tmp_path, limits):
    ledger = tmp_path / "stages.json"
    ledger.write_text(
        json.dumps({"limit_worker_seconds": None, "attempts": {}, "stage_limits": limits})
    )
    job = {
        "id": "one",
        "seconds": 100,
        "budget_stages": list(limits),
        "resources": {"cpus": 2, "gpus": 1, "memory_mb": 512},
    }
    return ledger, job


def test_elapsed_stage_clock_and_logical_cost_survive_replacement(tmp_path, monkeypatch):
    from tools.repair import batch

    clock = [100]
    monkeypatch.setattr(batch.time, "time", lambda: clock[0])
    ledger, job = _staged(tmp_path, {"learning": {"elapsed_seconds": 50}})
    job["logical_id"] = "fixed-scientific-work"
    assert charge(ledger, job, "first") == 50
    clock[0] = 110
    charge(ledger, job, "first", elapsed=10)
    replacement = {**job, "id": "replacement-worker"}
    assert charge(ledger, replacement, "second") == 40
    clock[0] = 120
    charge(ledger, replacement, "second", elapsed=10)
    clock[0] = 151
    with pytest.raises(TimeoutError, match="stage"):
        charge(ledger, replacement, "third")
    state = json.loads(ledger.read_text())
    assert state["stages"]["learning"]["started_epoch"] == 100
    assert state["cumulative"]["worker_seconds"] == 20
    assert len(state["attempts"]) == 2


def test_overlapping_gpu_and_worker_envelopes_include_pending_reservations(tmp_path):
    ledger, job = _staged(
        tmp_path, {"learning": {"worker_seconds": 150}, "fitting": {"gpu_seconds": 100}}
    )
    job["resources"]["gpus"] = 2
    assert charge(ledger, job, "first") == 50
    with pytest.raises(TimeoutError):
        charge(ledger, {**job, "id": "another"}, "second")
    charge(ledger, job, "first", elapsed=10)
    single = {**job, "id": "single", "resources": {**job["resources"], "gpus": 1}}
    assert charge(ledger, single, "single") == 80
    current = json.loads(ledger.read_text())
    assert current["stages"]["fitting"]["cumulative"]["gpu_seconds"] == 100
    assert current["stages"]["learning"]["cumulative"]["worker_seconds"] == 90


def test_concurrent_reservations_cannot_overspend_stage(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    ledger, job = _staged(tmp_path, {"development": {"worker_seconds": 100}})
    job["seconds"] = 20

    def reserve(index):
        try:
            return charge(ledger, {**job, "id": f"job-{index}"}, f"attempt-{index}")
        except TimeoutError:
            return 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        reserved = list(pool.map(reserve, range(8)))
    assert sum(reserved) == 100 and reserved.count(20) == 5
    assert len(json.loads(ledger.read_text())["attempts"]) == 5


def test_dispatch_queue_time_is_claimed_once_and_settled_without_double_count(
    tmp_path, monkeypatch
):
    from tools.repair import batch

    clock = [100]
    monkeypatch.setattr(batch.time, "time", lambda: clock[0])
    ledger, job = _staged(tmp_path, {"acquisition": {"elapsed_seconds": 50}})
    assert charge(ledger, job, "queued") == 50
    clock[0] = 112
    assert charge(ledger, job, "queued", claim=True, worker_started_epoch=110) == 40
    with pytest.raises(ValueError, match="already claimed"):
        charge(ledger, job, "queued", claim=True)
    clock[0] = 118
    charge(ledger, job, "queued", elapsed=8)
    row = json.loads(ledger.read_text())["attempts"]["queued"]
    assert row["elapsed_seconds"] == 18  # 10 queued + 8 in worker, including verification.


def test_explicit_deadlines_and_frozen_stage_limits_cannot_be_extended(tmp_path, monkeypatch):
    from tools.repair import batch

    clock = [100]
    monkeypatch.setattr(batch.time, "time", lambda: clock[0])
    ledger, job = _staged(
        tmp_path, {"calibration": {"elapsed_seconds": 1000, "deadline_epoch": 130}}
    )
    job["deadline_epoch"] = 120
    assert charge(ledger, job, "first") == 20
    charge(ledger, job, "first", elapsed=5)
    with pytest.raises(ValueError, match="logical-job"):
        charge(ledger, {**job, "deadline_epoch": 150}, "changed")
    value = json.loads(ledger.read_text())
    value["stage_limits"]["calibration"]["deadline_epoch"] = 150
    ledger.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="limits cannot change"):
        charge(ledger, {**job, "id": "independent"}, "changed-stage")


def test_settlement_never_reduces_old_costs_or_erases_api_spend(tmp_path):
    ledger, job = _staged(tmp_path, {"stage": {"worker_seconds": 1000}})
    value = json.loads(ledger.read_text())
    value["cumulative"] = {"external_api_cost_usd": 7.25}
    ledger.write_text(json.dumps(value))
    charge(ledger, job, "a")
    charge(ledger, job, "a", elapsed=10, cpu_seconds=7)
    with pytest.raises(ValueError, match="reduce"):
        charge(ledger, job, "a", elapsed=9, cpu_seconds=7)
    with pytest.raises(ValueError, match="CPU cost"):
        charge(ledger, job, "a", elapsed=10, cpu_seconds=6)
    assert json.loads(ledger.read_text())["cumulative"]["external_api_cost_usd"] == 7.25


def test_shared_commit_export_is_reused_and_corruption_rejected(tmp_path):
    import subprocess
    from tools.repair.batch import source_snapshot

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "input.txt").write_text("committed")
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
            "source",
        ],
        cwd=repo,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    code, manifest = source_snapshot(repo, commit, tmp_path / "sources")
    inode = (code / "input.txt").stat().st_ino
    again, same_manifest = source_snapshot(repo, commit, tmp_path / "sources")
    assert again == code and same_manifest == manifest
    assert (again / "input.txt").stat().st_ino == inode
    (code / "input.txt").write_text("modified")
    with pytest.raises(ValueError, match="snapshot changed"):
        source_snapshot(repo, commit, tmp_path / "sources")


def test_prepared_descriptor_carries_budget_device_profile_and_deadline(tmp_path, monkeypatch):
    import sys
    from tools.repair import batch

    ledger, job = _staged(tmp_path, {"stage": {"gpu_seconds": 100}})
    job.update(gpu_devices=["physical-5090"], resource_profile="training", deadline_epoch=200)
    job["resources"]["gres"] = "gpu:rtx5090:1"
    manifest = dict(
        jobs=[job],
        code=str(tmp_path),
        python=sys.executable,
        allocation="14451",
        ledger=str(ledger),
    )
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(batch, "checked_batch", lambda _: manifest)
    descriptor = batch.prepare_dispatch(
        path, job["id"], tmp_path / "attempt", tmux_socket=tmp_path / "tmux.sock"
    )
    assert (
        descriptor["gpu_devices"] == ["physical-5090"]
        and descriptor["resource_profile"] == "training"
    )
    assert descriptor["budget_reservation"] == dict(
        batch=str(path), job_id=job["id"], attempt=str(tmp_path / "attempt")
    )
    assert descriptor["deadline_epoch"] == 200
    assert "--gres=gpu:rtx5090:1" in descriptor["launch"]["argv"]
    assert {x["path"] for x in descriptor["launch"]["bindings"]} == {
        str(path),
        str(tmp_path / "attempt/worker.sh"),
    }
