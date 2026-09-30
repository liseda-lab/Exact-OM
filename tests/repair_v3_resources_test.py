"""CPU/GPU budgets survive replacement; worker CPU limits kill actual busy work."""

import copy
import json
import math
from pathlib import Path

import pytest

from exact.repair.checkpointing import CumulativeBudget
from exact.repair.protocol import load_protocol_v3
from exact.repair.workers import bounded_call

SAMPLE = (
    Path(__file__).resolve().parents[1] / "specs/exact-repair/protocol/xr21-review-conformance.json"
)


def _busy_worker(marker):
    Path(marker).write_text("started")
    value = 1
    while True:
        value = (value * 17 + 11) % 65537


def _small_worker():
    return sum(range(1000))


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="CPU supervision requires Linux")
def test_worker_cpu_ceiling_stops_busy_work_and_records_usage(tmp_path):
    marker = tmp_path / "busy-started"
    result = bounded_call(_busy_worker, str(marker), timeout=6.0, cpu_seconds=1.0)
    assert marker.read_text() == "started"
    assert result.status == "cpu_limit", result
    usage = dict(result.resource_usage)
    assert usage["cpu_seconds"] >= 1.0
    assert 0 < usage["wall_seconds"] < 6.5
    assert usage["peak_sampled_tree_rss_bytes"] > 0
    assert all(math.isfinite(value) and value >= 0 for value in usage.values())
    assert result.cleanup_complete


def test_successful_worker_retains_positive_measured_cpu_and_wall():
    result = bounded_call(_small_worker, timeout=6.0, cpu_seconds=3.0)
    assert result.status == "complete", result
    assert result.value == 499500
    usage = dict(result.resource_usage)
    assert 0 < usage["cpu_seconds"] < 3.0
    assert usage["wall_seconds"] > 0


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_worker_cpu_limit_rejected(value):
    with pytest.raises(ValueError, match="positive and finite"):
        bounded_call(_small_worker, timeout=1.0, cpu_seconds=value)


def test_cpu_gpu_replacement_charges_retained_reservations_once(tmp_path, monkeypatch):
    import exact.repair.checkpointing as module

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    path = tmp_path / "budget.json"
    options = dict(cpu_seconds=100.0, gpu_hours=0.02, allocated_gpus=2)
    with CumulativeBudget(path, "frozen", 100.0, **options) as budget:
        assert budget.remaining == pytest.approx(36.0)
        assert budget.begin(10.0, cpu_seconds=20.0) == 10.0
        assert budget.remaining_cpu == 20.0
        now[0] += 3.0
        budget.finish(cpu_seconds=4.0)
        assert budget.state["spent_seconds"] == 3.0
        assert budget.state["spent_cpu_seconds"] == 4.0
        assert budget.state["spent_gpu_hours"] == pytest.approx(6 / 3600)
        assert budget.remaining == pytest.approx(33.0)
        assert budget.remaining_cpu == 96.0
        budget.begin(8.0, cpu_seconds=12.0)
        # Closing without finish simulates process loss after durable reservation.
        budget.close()
        budget.started = None
    with CumulativeBudget(path, "frozen", 100.0, **options) as budget:
        assert budget.state["spent_seconds"] == 11.0
        assert budget.state["spent_cpu_seconds"] == 16.0
        assert budget.state["spent_gpu_hours"] == pytest.approx(22 / 3600)
        assert budget.remaining == pytest.approx(25.0)
        assert budget.remaining_cpu == 84.0
        assert budget.state["attempts"][-1]["status"] == "lost_reserved_cost"
    with CumulativeBudget(path, "frozen", 100.0, **options) as budget:
        assert len(budget.state["attempts"]) == 2
        assert budget.begin(100.0, cpu_seconds=100.0) == pytest.approx(25.0)
        assert budget.remaining_cpu == 84.0
        now[0] += 26.0
        # Overshoot remains recorded; costs never disappear by clipping to a cap.
        budget.finish("cpu_limit", cpu_seconds=90.0)
        assert budget.state["spent_seconds"] == 37.0
        assert budget.state["spent_cpu_seconds"] == 106.0
        assert budget.state["spent_gpu_hours"] == pytest.approx(74 / 3600)
        assert budget.remaining == 0.0
        assert budget.remaining_cpu == 0.0
        with pytest.raises(TimeoutError):
            budget.begin()


def test_allocation_only_ledger_tracks_gpu_time_without_a_cap(tmp_path, monkeypatch):
    import exact.repair.checkpointing as module

    now = [20.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    path = tmp_path / "allocation.json"
    with CumulativeBudget(path, "allocated", 30.0, allocated_gpus=2) as budget:
        budget.begin(10.0)
        now[0] += 4.0
        budget.finish()
        assert budget.state["limit_gpu_hours"] is None
        assert budget.state["spent_gpu_hours"] == pytest.approx(8 / 3600)
        assert budget.remaining == 26.0
    with CumulativeBudget(path, "allocated", 30.0, allocated_gpus=2) as budget:
        assert budget.state["spent_gpu_hours"] == pytest.approx(8 / 3600)
    with pytest.raises(ValueError, match="resource policy changed"):
        CumulativeBudget(path, "allocated", 30.0, allocated_gpus=1)


def test_unknown_cpu_usage_charges_reservation_and_zero_gpu_cap_blocks(tmp_path, monkeypatch):
    import exact.repair.checkpointing as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    with CumulativeBudget(tmp_path / "cpu.json", "cpu", 10.0, cpu_seconds=5.0) as budget:
        budget.begin(cpu_seconds=2.0)
        budget.finish("interrupted")
        assert budget.state["spent_cpu_seconds"] == 2.0
        attempt = budget.state["attempts"][-1]
        assert attempt["observed_cpu_seconds"] is None
        assert attempt["cpu_accounting"] == "reserved_upper_bound"
    with CumulativeBudget(
        tmp_path / "gpu.json", "gpu", 10.0, allocated_gpus=2, gpu_hours=0.0
    ) as budget:
        assert budget.remaining == 0.0
        with pytest.raises(TimeoutError):
            budget.begin()


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "2"])
@pytest.mark.parametrize("field", ["seconds", "cpu_seconds"])
def test_invalid_begin_reservation_cannot_mutate_ledger(tmp_path, value, field):
    path = tmp_path / "budget.json"
    with CumulativeBudget(path, "frozen", 10.0, cpu_seconds=10.0) as budget:
        before = path.read_bytes(), copy.deepcopy(budget.state)
        with pytest.raises(ValueError):
            budget.begin(**{field: value})
        assert (path.read_bytes(), budget.state) == before
        assert budget.started is None


@pytest.mark.parametrize("usage", [-1, True, float("nan"), float("inf"), "1"])
def test_invalid_finish_cannot_drop_active_reservation(tmp_path, usage):
    path = tmp_path / "budget.json"
    with CumulativeBudget(path, "frozen", 10.0, cpu_seconds=10.0) as budget:
        budget.begin(2.0, cpu_seconds=3.0)
        before = path.read_bytes(), copy.deepcopy(budget.state), budget.started
        with pytest.raises(ValueError):
            budget.finish(cpu_seconds=usage)
        assert (path.read_bytes(), budget.state, budget.started) == before
        budget.finish(cpu_seconds=0.0)


def test_failed_finish_publication_keeps_reservation(tmp_path, monkeypatch):
    import exact.repair.checkpointing as module

    path = tmp_path / "budget.json"
    budget = CumulativeBudget(path, "frozen", 10.0, cpu_seconds=10.0)
    budget.begin(2.0, cpu_seconds=3.0)
    before = path.read_bytes(), copy.deepcopy(budget.state), budget.started
    original = module.write_artifact
    monkeypatch.setattr(module, "write_artifact", lambda *a: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(OSError, match="full"):
        budget.finish(cpu_seconds=1.0)
    assert (path.read_bytes(), budget.state, budget.started) == before
    monkeypatch.setattr(module, "write_artifact", original)
    budget.close()
    with CumulativeBudget(path, "frozen", 10.0, cpu_seconds=10.0) as replacement:
        assert replacement.state["spent_seconds"] == 2.0
        assert replacement.state["spent_cpu_seconds"] == 3.0


def test_legacy_wall_ledger_keeps_history_and_can_add_observed_cpu(tmp_path, monkeypatch):
    import exact.repair.checkpointing as module

    now = [1.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    path = tmp_path / "legacy.json"
    old = {
        "identity": "old",
        "limit_seconds": 10.0,
        "spent_seconds": 2.0,
        "attempts": [{"status": "complete", "elapsed_seconds": 2.0}],
    }
    path.write_text(json.dumps(old))
    with CumulativeBudget(path, "old", 10.0) as budget:
        assert budget.state == old
        assert budget.remaining_cpu is None
        budget.begin(2.0)
        now[0] += 1.0
        budget.finish(cpu_seconds=0.5)
        assert budget.state["attempts"][0] == old["attempts"][0]
        assert budget.state["spent_cpu_seconds"] == 0.5
        assert "limit_cpu_seconds" not in budget.state
    with CumulativeBudget(path, "old", 10.0) as budget:
        assert budget.remaining == 7.0
        assert budget.remaining_cpu is None


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_protocol_gpu_allocation_is_a_strict_nonnegative_integer(tmp_path, value):
    manifest = json.loads(SAMPLE.read_text())
    manifest["resources"]["allocated_gpus"] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_protocol_v3(path)


def test_stage_cpu_caps_cannot_each_reuse_campaign_allowance(tmp_path):
    manifest = json.loads(SAMPLE.read_text())
    resources = manifest["resources"]
    for stage in resources["stage_cpu_seconds"]:
        resources["stage_cpu_seconds"][stage] = resources["campaign_cpu_seconds"]
    path = tmp_path / "multiplied-budget.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Stage allocations exceed campaign budget"):
        load_protocol_v3(path)


@pytest.mark.parametrize("status", [None, 1, ""])
def test_invalid_finish_status_keeps_active_reservation(tmp_path, status):
    path = tmp_path / "budget.json"
    with CumulativeBudget(path, "frozen", 10.0) as budget:
        budget.begin(2.0)
        before = path.read_bytes(), copy.deepcopy(budget.state), budget.started
        with pytest.raises(ValueError, match="status"):
            budget.finish(status)
        assert (path.read_bytes(), budget.state, budget.started) == before
