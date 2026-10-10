"""Resource separation and real pool handoff without TEST data or hosted clients."""

import copy
import json
import operator
import time
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from tests.repair_primary_evaluation_test import fixture_manifest, schedule
from tools.repair.batch import sha
from tools.repair.evaluation_ownership import (
    PROFILE,
    capacity,
    lane_for,
    partition,
    resources,
    validate_lane,
    validate_owner,
)
from tools.repair.historical_regression import binding
from tools.repair.primary_evaluation import compile_rows, execute_rows
from tools.repair.primary_runtime import DEV_GPU, FIT_GPU


def test_partition_preserves_science_and_bounds_actual_parallelism(monkeypatch):
    value = schedule()
    rows = compile_rows(value)

    def closed(*args, **kwargs):
        raise AssertionError("TEST payload opened")

    monkeypatch.setattr(Path, "open", closed)
    shards = partition(value, rows)
    assert len(shards) == 54
    assert sorted(r["id"] for s in shards for r in s["rows"]) == sorted(r["id"] for r in rows)
    assert all(lane_for(r) == "inference" for r in rows if r["experiment"] == "E3")
    ends = {}
    periods = []
    for shard in shards:
        start = max((ends[d] for d in shard["depends_on"]), default=0)
        end = start + shard["seconds"]
        ends[shard["id"]] = end
        periods.append((start, end, shard))
        if shard["lane"] != "pool":
            # Every consumed pool producer is finished before its consumer starts.
            for row in shard["rows"]:
                if row["pool"]:
                    producer = next(
                        s for s in shards if any(r["id"] == row["pool"] for r in s["rows"])
                    )
                    assert ends[producer["id"]] <= start
    for t in {a for a, _, _ in periods}:
        active = [s for a, b, s in periods if a <= t < b]
        assert len(active) <= 2
        assert sum(s["resources"]["gpus"] for s in active) <= 1
        assert sum(s["resources"]["cpus"] for s in active) <= 6
        assert sum(s["resources"]["memory_mb"] for s in active) <= 40000
    projection = capacity(shards, value)
    assert max(ends.values()) == projection["critical_path_seconds"] == 79920
    assert projection["gpu_seconds"] == 74400 < 100800
    assert projection["worker_seconds"] == 115320
    assert projection["elapsed_projection_with_handoffs_seconds"] == 96120
    assert projection["dispatch_handoff_allowance_seconds"] == 54 * 300
    assert projection["projection_passed"] and not projection["capacity_qualified"]
    assert projection["conference_reserved_worker_seconds"] >= 2 * 144000 / 3
    assert len(value["semantic_slots"]) == 128
    assert len(value["comparisons"]) == 518


def lane_manifest(lane):
    row = next(r for r in compile_rows(schedule()) if lane_for(r) == lane)
    return dict(
        rows=[row],
        ownership_lane=lane,
        cpus=3,
        memory_mb=20000,
        gpu_uuid=DEV_GPU if lane == "inference" else None,
        device="cuda" if lane == "inference" else "cpu",
        concurrent_load_profile=PROFILE,
        dispatch_nonce="owner",
    )


@pytest.mark.parametrize("change", ["model", "device", "profile", "memory"])
def test_native_worker_cannot_adopt_an_inference_row_or_device(change):
    m = lane_manifest("native")
    if change == "model":
        m["rows"] = lane_manifest("inference")["rows"]
    elif change == "device":
        m["gpu_uuid"], m["device"] = DEV_GPU, "cuda"
    elif change == "profile":
        m["concurrent_load_profile"] = "different"
    else:
        m["memory_mb"] = 10000
    with pytest.raises(ValueError, match="lane|profile"):
        validate_lane(m, check_visibility=False)


def test_cpu_requires_zero_visible_gpus(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="zero Slurm-visible"):
        validate_lane(lane_manifest("native"))


def test_inference_uses_exact_uuid_not_device_index_or_alternate_gpu(monkeypatch):
    seen = []
    monkeypatch.setattr("tools.repair.evaluation_ownership.owned_device", seen.append)
    m = lane_manifest("inference")
    validate_lane(m)
    assert seen == [DEV_GPU]
    m["gpu_uuid"] = FIT_GPU
    with pytest.raises(ValueError, match="profile"):
        validate_lane(m)


@pytest.mark.parametrize("lane", ["native", "inference"])
def test_runtime_checks_charged_owner_nonce_device_and_stage(tmp_path, monkeypatch, lane):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "4")
    attempt = tmp_path / "attempt"
    write_artifact(attempt / "step.json", dict(step_id="123.4", dispatch_nonce="owner"))
    m = lane_manifest(lane)
    owner = dict(
        status="running",
        resources=resources(lane),
        gpu_devices=[DEV_GPU] if lane == "inference" else [],
        budget_stages=(
            ["evaluation", "secondary_evaluation"] if lane == "inference" else ["evaluation"]
        ),
    )
    stage = dict(attempts={str(attempt): owner})
    validate_owner(m, stage)
    m["dispatch_nonce"] = "other"
    with pytest.raises(ValueError, match="nonce differs"):
        validate_owner(m, stage)
    m["dispatch_nonce"] = "owner"
    owner["budget_stages"] = []
    with pytest.raises(ValueError, match="nonce differs"):
        validate_owner(m, stage)


def fake_pool(row, case, public, protocol, model, pool, directory, deadline, device):
    from tools.repair.shared_release import bound

    path = Path(directory) / "pool.json"
    write_artifact(path, bound(public)["input"])
    return dict(status="pool_complete", pool=binding(path), logical_status="NOT_APPLICABLE")


def prepared_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 120))
    producer = fixture_manifest(tmp_path)
    producer.update(
        ownership_lane="pool", dispatch_nonce="pool-nonce", source_commit="fixture", schedule=None
    )
    row = producer["rows"][0]
    row.update(id="shared-pool", experiment="E2_pool", input_identity="fixture-input")
    row["adapter"]["operation"] = "pool"
    manifest_path = tmp_path / "producer.json"
    write_artifact(manifest_path, producer)
    work = tmp_path / "producer-work"
    execute_rows(producer, work, evaluator=fake_pool)
    attempt = tmp_path / "producer-attempt"
    batch_path = tmp_path / "batch.json"
    write_artifact(
        batch_path,
        dict(
            frozen_files={str(manifest_path): sha(manifest_path)},
            jobs=[dict(id="producer", commands=[["python", str(manifest_path)]])],
        ),
    )
    batch_path.with_name("batch.sha256").write_text(sha(batch_path))
    terminal = dict(
        status="complete",
        exit_code=0,
        job_id="producer",
        work=str(work),
        dispatch_nonce="pool-nonce",
        step_id="123.1",
        batch=str(batch_path),
    )
    write_artifact(attempt / "completion.json", terminal)
    write_artifact(attempt / "step.json", dict(dispatch_nonce="pool-nonce", step_id="123.1"))
    write_artifact(
        attempt / "outputs.json",
        {str(p.relative_to(work)): sha(p) for p in work.rglob("*") if p.is_file()},
    )
    consumer = copy.deepcopy(producer)
    consumer.pop("ownership_lane")
    consumer["rows"][0].update(id="consumer", experiment="E2")
    consumer["rows"][0]["adapter"].update(
        operation="native_deletion", pool_dependency="shared-pool"
    )
    consumer["pool_producers"] = {
        "shared-pool": dict(
            manifest=binding(manifest_path),
            attempt=str(attempt),
            work=str(work),
            job_id="producer",
            dispatch_nonce="pool-nonce",
        )
    }
    return consumer, producer, work, attempt


def test_terminal_pool_reused_by_real_native_consumer_without_regeneration(tmp_path, monkeypatch):
    consumer, producer, work, attempt = prepared_pool(tmp_path, monkeypatch)
    first = execute_rows(consumer, tmp_path / "consumer")
    assert first["rows"][0]["result"]["logical_status"] == "VERIFIED_FEASIBLE"
    assert first["external_pool_cost_references"] == ["shared-pool"]
    assert first["rows"][0]["shared_cost_reference"] == "shared-pool"
    # A callback that cannot execute proves both successful rows resume without replay.
    second = execute_rows(consumer, tmp_path / "consumer", evaluator=operator.add)
    assert first == second
    assert len(list((work / "rows").iterdir())) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "nonce",
        "step",
        "running",
        "input",
        "artifact",
        "digest",
        "missing_contract",
        "unbound_manifest",
    ],
)
def test_pool_handoff_rejects_incompatible_or_live_producer(tmp_path, monkeypatch, damage):
    consumer, producer, work, attempt = prepared_pool(tmp_path, monkeypatch)
    if damage in {"nonce", "step", "running"}:
        record = json.loads((attempt / "completion.json").read_text())
        record[{"nonce": "dispatch_nonce", "step": "step_id", "running": "status"}[damage]] = (
            "wrong"
        )
        write_artifact(attempt / "completion.json", record)
    elif damage == "input":
        consumer["public_inputs"]["fixture"] = dict(path="wrong", sha256="wrong")
    elif damage == "artifact":
        (work / "rows/shared-pool/payload/pool.json").write_text("tampered")
    elif damage == "digest":
        write_artifact(attempt / "outputs.json", {})
    elif damage == "missing_contract":
        consumer["pool_producers"] = {}
    else:
        batch = json.loads((tmp_path / "batch.json").read_text())
        batch["frozen_files"] = {}
        write_artifact(tmp_path / "batch.json", batch)
        (tmp_path / "batch.sha256").write_text(sha(tmp_path / "batch.json"))
    with pytest.raises(ValueError):
        execute_rows(consumer, tmp_path / "bad", evaluator=operator.add)
    assert not (tmp_path / "bad/rows/consumer/started.json").exists()


def test_unavailable_pool_is_retained_without_retry_or_duplicate_cost(tmp_path, monkeypatch):
    consumer, producer, work, attempt = prepared_pool(tmp_path, monkeypatch)
    path = work / "rows/shared-pool/completion.json"
    saved = json.loads(path.read_text())
    saved.update(status="timeout", result=None, artifacts=[])
    write_artifact(path, saved)
    outputs = json.loads((attempt / "outputs.json").read_text())
    outputs["rows/shared-pool/completion.json"] = sha(path)
    report_path = work / "report.json"
    report = json.loads(report_path.read_text())
    report["rows"] = [dict(id="shared-pool", **saved)]
    write_artifact(report_path, report)
    outputs["report.json"] = sha(report_path)
    write_artifact(attempt / "outputs.json", outputs)
    result = execute_rows(consumer, tmp_path / "consumer", evaluator=operator.add)
    assert result["rows"][0]["status"] == "unavailable_shared_pool"
    assert result["charged_seconds"] == 0
    assert result["external_pool_cost_references"] == ["shared-pool"]


def test_complete_disabled_preparation_audits_and_rejects_changed_dispatch(tmp_path, monkeypatch):
    import hashlib

    from tools.repair import prepare_evaluation_capacity as preparation

    campaign, output = tmp_path / "campaign", tmp_path / "prepared"
    immutable_schedule = campaign / "schedule.json"
    declared = schedule()
    declared.update(models_freeze_epoch=1791997200.0, measurements_deadline_epoch=1792191600.0)
    write_artifact(immutable_schedule, declared)
    prior = dict(
        schedule=binding(immutable_schedule),
        endpoint_projection=dict(path="sealed-endpoint", sha256="unchanged"),
        teacher_incident_id="existing-incident",
    )
    write_artifact(campaign / "prior.json", prior)
    write_artifact(campaign / "completed.json", dict(audit_receipt={}))
    write_artifact(
        campaign / "supervisor/registry.json",
        dict(
            primary_evaluation_preparation=dict(
                preparation=binding(campaign / "prior.json"),
                completion=binding(campaign / "completed.json"),
            )
        ),
    )
    monkeypatch.setattr(
        preparation, "validate_completion", lambda receipt: (dict(work="unused"), {}, {}, {})
    )
    monkeypatch.setattr(
        preparation,
        "source_identity",
        lambda: dict(revision="fixture", dirty_hash=hashlib.sha256(b"").hexdigest()),
    )
    saved = preparation.prepare(campaign, output)
    audited = preparation.audit(output / "prepared.json", tmp_path / "audit")
    assert audited["expected_rows"] == 1535 and audited["ownership_shards"] == 54
    assert audited["lane_rows"] == dict(pool=71, native=497, inference=967)
    assert not audited["test_execution_admitted"]
    assert saved["endpoint_projection_unchanged"] == prior["endpoint_projection"]
    assert all(
        not json.loads(Path(r["path"]).read_text())["execution_authorized"]
        for r in saved["execution_manifests"]
    )
    saved["dispatch"][-1]["depends_on"] = []
    write_artifact(output / "prepared.json", saved)
    with pytest.raises(ValueError, match="Ownership shard changed"):
        preparation.audit(output / "prepared.json", tmp_path / "invalid")


def test_unattempted_pool_tail_remains_unavailable_without_crash(tmp_path, monkeypatch):
    consumer, producer, work, attempt = prepared_pool(tmp_path, monkeypatch)
    report_path = work / "report.json"
    report = json.loads(report_path.read_text())
    report["rows"] = [dict(id="shared-pool", status="not_attempted", result=None)]
    write_artifact(report_path, report)
    outputs = json.loads((attempt / "outputs.json").read_text())
    outputs = {k: v for k, v in outputs.items() if not k.startswith("rows/")}
    outputs["report.json"] = sha(report_path)
    write_artifact(attempt / "outputs.json", outputs)
    result = execute_rows(consumer, tmp_path / "consumer", evaluator=operator.add)
    assert result["rows"][0]["status"] == "unavailable_shared_pool"
    assert result["charged_seconds"] == 0
