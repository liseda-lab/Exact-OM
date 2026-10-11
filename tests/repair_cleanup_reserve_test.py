"""CUDA teardown reserves time inside the existing call and ancestor deadlines."""

import operator
from types import SimpleNamespace

import pytest

from exact.repair import workers


def test_slow_killed_descendant_needs_longer_bounded_grace(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(workers.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        workers.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(workers, "_tree_pids", lambda pid: [pid, pid + 1])
    monkeypatch.setattr(workers.os, "kill", lambda *args: None)
    monkeypatch.setattr(workers.os, "waitpid", lambda *args: (100, 9))
    monkeypatch.setattr(
        workers.Path, "read_text", lambda path: "1 (child) " + ("D" if clock[0] < 0.4 else "Z")
    )
    assert not workers._stop(100, False)
    clock[0] = 0.0
    assert workers._stop(100, False, 2.0)
    assert 0.4 <= clock[0] < 0.5
    workers._STOP_EXIT_CODES.pop(100, None)


@pytest.mark.parametrize("inherited,timeout", [(None, 1.0), ("101", 10.0)])
def test_cleanup_reserve_prevents_work_without_enough_time(monkeypatch, inherited, timeout):
    monkeypatch.setattr(workers, "_NESTED_CLEANUP_COMPLETE", True)
    monkeypatch.setattr(workers.time, "time", lambda: 100.0)
    if inherited is None:
        monkeypatch.delenv("EXACT_REPAIR_DEADLINE_EPOCH", raising=False)
    else:
        monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", inherited)
    monkeypatch.setattr(workers.os, "fork", lambda: pytest.fail("insufficient cleanup reserve"))
    result = workers.bounded_call(operator.add, 1, 2, timeout=timeout, cleanup_seconds=2.0)
    assert result.status == "timeout" and result.cleanup_complete


@pytest.mark.parametrize("device,reserve", [("cpu", 0.2), ("cuda", 2.0), ("cuda:0", 2.0)])
def test_checkpoint_boundary_reserves_device_cleanup(monkeypatch, device, reserve):
    from exact.repair.pipeline import bounded_freeze_checkpoint

    calls = []
    monkeypatch.setattr(workers, "bounded_call", lambda *a, **k: calls.append(k))
    bounded_freeze_checkpoint(SimpleNamespace(), "unused", seconds=5, device=device)
    assert calls[0]["cleanup_seconds"] == reserve
    assert calls[0]["timeout"] == 5 and calls[0]["device"] == device


@pytest.mark.parametrize("reserve", [0, 0.1, True, float("inf"), float("nan")])
def test_invalid_cleanup_reserve_rejected(reserve):
    with pytest.raises(ValueError, match="cleanup_seconds"):
        workers.bounded_call(operator.add, 1, 2, timeout=5, cleanup_seconds=reserve)


@pytest.mark.parametrize("device,reserve", [("cpu", 0.2), ("cuda", 2.0), ("cuda:0", 2.0)])
def test_staged_dev_reserves_serialization_and_device_cleanup(monkeypatch, device, reserve):
    from tools.repair import train

    clock = [10.0]
    calls = []
    monkeypatch.setattr(train.time, "monotonic", lambda: clock[0])

    def save(path, state):
        path.write_bytes(b"checkpoint")
        clock[0] += 1.25

    monkeypatch.setattr(train, "save_training_state", save)
    monkeypatch.setattr(train, "bounded_call", lambda *a, **k: calls.append(k))
    model = SimpleNamespace(revision="v3", config={}, metadata={}, state_dict=lambda: {})
    train._staged_training_model(
        None, model, seconds=8, elementary_seconds=3, selection_options={}, device=device
    )
    assert calls[0]["timeout"] == calls[0]["total_seconds"] == 6.75
    assert calls[0]["elementary_seconds"] == 3
    assert calls[0]["cleanup_seconds"] == reserve
    assert calls[0]["generation_options"]["device"] == device
