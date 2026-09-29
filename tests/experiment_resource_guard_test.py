"""Finalization uses allocation headroom without relaxing STOP or hosted limits."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments import campaign, harness
from tools import experiment_resources as resources
from tools.measured_once import require_reuse_only


def memory_environment(monkeypatch, *, cgroup_bytes=132859822080, per_node=None):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=135015358464))
    for key in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        monkeypatch.delenv(key, raising=False)
    if per_node is not None:
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(per_node))
    data = {
        "/proc/self/cgroup": "0::/job/step\n",
        "/sys/fs/cgroup/job/step/memory.max": "max",
        "/sys/fs/cgroup/job/memory.max": str(cgroup_bytes),
        "/sys/fs/cgroup/memory.max": "max",
    }
    read, is_file = Path.read_text, Path.is_file
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: data[str(path)] if str(path) in data else read(path, *a, **kw),
    )
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) in data or is_file(path))


def test_observed_node_allocation_sets_111_gib_guard(monkeypatch):
    memory_environment(monkeypatch)
    assert resources.memory_limit_bytes() == 119573839872
    assert resources.memory_limit_bytes() > 56 * 1024**3


def test_smaller_actual_slurm_allocation_is_still_respected(monkeypatch):
    memory_environment(monkeypatch, per_node=64 * 1024)
    assert resources.memory_limit_bytes() == int(64 * 1024**3 * 0.9)


@pytest.fixture
def execution(tmp_path, monkeypatch):
    memory_environment(monkeypatch)
    monkeypatch.setattr(resources, "process_tree_rss_bytes", lambda: 64 * 1024**3)
    caps = {
        "EXACT_OPENROUTER_REQUEST_CAP": "100",
        "EXACT_OPENROUTER_TOKEN_CAP": "1000",
        "EXACT_OPENROUTER_RETRY_UNKNOWN": "0",
    }
    for key, value in caps.items():
        monkeypatch.setenv(key, value)
    calls = []

    def subprocess(command, **kwargs):
        calls.append(kwargs)
        return 0, 0.1, 1

    monkeypatch.setattr(harness, "_run_subprocess", subprocess)
    return SimpleNamespace(
        wave=tmp_path,
        caps=caps,
        calls=calls,
        original=subprocess,
        stop=tmp_path / "runtime/exact-om-focused-v2/STOP",
    )


def test_more_than_56_gib_is_allowed_and_wire_caps_remain_frozen(execution, monkeypatch):
    def execute(*args, **kwargs):
        assert kwargs["jobs"] == 1 and kwargs["resume"] is True
        harness._run_subprocess(
            [],
            cwd=execution.wave,
            stdout_path=execution.wave / "out",
            stderr_path=execution.wave / "err",
            env={key: "99999" for key in execution.caps},
        )
        return 0

    monkeypatch.setattr(campaign, "execute_campaign", execute)
    resources.guarded_execute(Path("campaign.yaml"), execution.wave, Path("source"))
    assert execution.calls[0]["env"] == execution.caps
    assert not execution.stop.exists()
    assert harness._run_subprocess is execution.original


def test_allocation_guard_stops_cooperatively_before_execution(execution, monkeypatch):
    monkeypatch.setattr(resources, "process_tree_rss_bytes", lambda: 120 * 1024**3)
    monkeypatch.setattr(
        campaign, "execute_campaign", lambda *a, **kw: pytest.fail("must not dispatch")
    )
    with pytest.raises(ValueError, match="RSS guard"):
        resources.guarded_execute(Path("campaign.yaml"), execution.wave, Path("source"))
    assert "119573839872" in execution.stop.read_text()
    assert execution.calls == []
    assert harness._run_subprocess is execution.original


def test_user_stop_is_preserved(execution, monkeypatch):
    execution.stop.parent.mkdir(parents=True)
    execution.stop.write_text("user requested pause\n")
    monkeypatch.setattr(
        campaign, "execute_campaign", lambda *a, **kw: pytest.fail("must not dispatch")
    )
    with pytest.raises(ValueError, match="Existing STOP"):
        resources.guarded_execute(Path("campaign.yaml"), execution.wave, Path("source"))
    assert execution.stop.read_text() == "user requested pause\n"


def test_stop_during_finalization_is_not_reported_as_success(execution, monkeypatch):
    def execute(*a, **kw):
        execution.stop.parent.mkdir(parents=True)
        execution.stop.write_text("user requested pause\n")
        return 0

    monkeypatch.setattr(campaign, "execute_campaign", execute)
    with pytest.raises(ValueError, match="retained STOP"):
        resources.guarded_execute(Path("campaign.yaml"), execution.wave, Path("source"))
    assert execution.stop.read_text() == "user requested pause\n"


def test_reuse_only_worker_prohibition_survives_adapter(execution, monkeypatch):
    def execute(*a, **kw):
        harness._run_subprocess(
            [],
            cwd=execution.wave,
            stdout_path=execution.wave / "out",
            stderr_path=execution.wave / "err",
        )

    monkeypatch.setattr(campaign, "execute_campaign", execute)
    with pytest.raises(RuntimeError, match="must not repeat"):
        with require_reuse_only():
            resources.guarded_execute(Path("campaign.yaml"), execution.wave, Path("source"))
    assert execution.calls == []
    assert harness._run_subprocess is execution.original


def test_external_pause_checked_before_dispatch(execution, monkeypatch):
    def paused():
        raise ValueError("registered experiment paused")

    monkeypatch.setattr(
        campaign, "execute_campaign", lambda *a, **kw: pytest.fail("must not dispatch")
    )
    with pytest.raises(ValueError, match="experiment paused"):
        resources.guarded_execute(
            Path("campaign.yaml"), execution.wave, Path("source"), check_pause=paused
        )
