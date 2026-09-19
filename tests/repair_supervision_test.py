"""Process deadlines cover nested compiler workers and transport cleanup."""

import os
import time
from pathlib import Path

import pytest

from exact.repair.workers import bounded_call


def identity(value):
    return value


def nested():
    return bounded_call(identity, 17, timeout=3)


def delayed_write(path):
    Path(path + ".started").write_text(str(os.getpid()))
    time.sleep(2)
    Path(path).write_text("escaped supervisor")


def nested_slow(path):
    return bounded_call(delayed_write, path, timeout=10)


def test_nested_worker_is_supported_and_returns_its_own_status():
    outcome = bounded_call(nested, timeout=5)
    assert outcome.status == "complete"
    assert outcome.value.status == "complete"
    assert outcome.value.value == 17


@pytest.mark.skipif(os.name != "posix", reason="POSIX descendant-group cleanup")
def test_outer_deadline_cancels_live_nested_worker(tmp_path):
    path = str(tmp_path / "escaped")
    started = time.monotonic()
    outcome = bounded_call(nested_slow, path, timeout=1.5)
    assert outcome.status == "timeout"
    assert time.monotonic() - started < 2.1
    assert Path(path + ".started").exists(), "exercise cancellation after child startup"
    time.sleep(1.0)
    assert not Path(path).exists()


def allocate_and_wait():
    data = bytearray(64 * 1024 * 1024)
    time.sleep(5)
    return len(data)


@pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="Linux RSS supervision")
def test_memory_limit_returns_unknown_stage_status_and_cleans_worker():
    outcome = bounded_call(allocate_and_wait, timeout=5, memory_mb=40)
    assert outcome.status == "memory_limit"
    assert "RSS" in outcome.detail
