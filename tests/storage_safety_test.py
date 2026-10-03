"""Quota prevention fails before dispatch/write while immutable bytes are stable."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.utils import fitted_artifacts as artifacts
from tools import storage_guard


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setenv("EXACT_STORAGE_MIN_FREE_BYTES", "0")
    monkeypatch.setenv("EXACT_JSON_MAX_BYTES", str(8 * 1024**3))


def test_streaming_preserves_canonical_bytes_and_existing_identity(tmp_path, monkeypatch):
    payload = {"unicode": "á🥕", "rows": [{"x": 0.123, "s": "test"}] * 50000}
    expected = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    monkeypatch.setattr(artifacts.json, "dumps", lambda *a, **k: pytest.fail("whole encode"))
    path = tmp_path / "scores.json"
    assert artifacts.freeze_json(path, payload) is payload
    assert path.read_bytes() == expected
    assert artifacts.freeze_json(path, payload) is payload
    with pytest.raises(ValueError, match="identity conflict"):
        artifacts.freeze_json(path, {"changed": True})
    assert path.read_bytes() == expected


def test_artifact_limit_cleans_partial_and_does_not_publish(tmp_path):
    path = tmp_path / "oversized.json"
    with pytest.raises(ValueError, match="safety limit"):
        artifacts.freeze_json(path, {"rows": ["x" * 20000] * 100}, max_bytes=1024**2)
    assert not path.exists()
    assert not path.with_suffix(".json.partial").exists()


def test_free_reserve_is_rechecked_during_streaming(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_STORAGE_MIN_FREE_BYTES", "100")
    values = iter([10 * 1024**2, 50])
    monkeypatch.setattr(
        artifacts.shutil, "disk_usage", lambda _: SimpleNamespace(free=next(values))
    )
    path = tmp_path / "scores.json"
    with pytest.raises(OSError, match="Storage reserve"):
        artifacts.freeze_json(path, {"rows": ["x" * 20000] * 120})
    assert not path.exists()
    assert not path.with_suffix(".json.partial").exists()


def test_invalid_json_does_not_leave_partial(tmp_path):
    path = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        artifacts.freeze_json(path, {"x": float("nan")})
    assert list(tmp_path.iterdir()) == []


def test_tree_usage_deduplicates_hardlinks_and_does_not_follow_symlinks(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    one = root / "one"
    one.write_bytes(b"x" * 10000)
    before = storage_guard.tree_bytes(root)
    os.link(one, root / "two")
    assert storage_guard.tree_bytes(root) == before
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large").write_bytes(b"x" * 100000)
    (root / "linked").symlink_to(outside, target_is_directory=True)
    assert storage_guard.tree_bytes(root) < 100000


@pytest.mark.parametrize(
    "free,max_used,reason", [(1, 100000, "free reserve"), (100000, 1, "usage ceiling")]
)
def test_guard_refuses_admission_and_preserves_user_pause(
    tmp_path, monkeypatch, free, max_used, reason
):
    monkeypatch.setattr(storage_guard.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    monkeypatch.setattr(
        storage_guard.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    pause = tmp_path / "PAUSE"
    pause.write_text("user pause\n")
    stop = tmp_path / "STOP"
    assert (
        storage_guard.run(
            ["worker"],
            root=tmp_path,
            pause_paths=[pause, stop],
            min_free=10,
            max_used=max_used,
            growth_reserve=0,
        )
        == 75
    )
    assert pause.read_text() == "user pause\n"
    assert reason in stop.read_text()


def test_guard_pauses_running_worker_before_filesystem_exhaustion(tmp_path, monkeypatch):
    free = iter([1000, 1])
    monkeypatch.setattr(
        storage_guard.shutil, "disk_usage", lambda _: SimpleNamespace(free=next(free))
    )
    monkeypatch.setattr(storage_guard, "tree_bytes", lambda _: 100)
    worker = SimpleNamespace(
        wait=lambda **kw: (_ for _ in ()).throw(
            storage_guard.subprocess.TimeoutExpired("worker", 0)
        )
    )
    launched, stopped = [], []

    def launch(*args, **kwargs):
        launched.append(kwargs)
        return worker

    monkeypatch.setattr(storage_guard.subprocess, "Popen", launch)
    monkeypatch.setattr(storage_guard, "stop_owned_worker", stopped.append)
    assert (
        storage_guard.run(
            ["worker"],
            root=tmp_path,
            pause_paths=[tmp_path / "PAUSE"],
            min_free=10,
            max_used=1000,
            growth_reserve=0,
            interval=0.01,
        )
        == 75
    )
    assert launched[0]["start_new_session"] is True
    assert launched[0]["env"]["EXACT_STORAGE_MIN_FREE_BYTES"] == "10"
    assert stopped == [worker]


def test_only_scientific_descendant_is_interrupted_before_accounting_settles(monkeypatch):
    events = []
    scientist = SimpleNamespace(
        cmdline=lambda: ["python", "-m", "exact.delivery.cli.main"],
        send_signal=lambda sig: events.append(("scientific", sig)),
    )
    accountant = SimpleNamespace(
        cmdline=lambda: ["python", "tools/prepared_batch.py"],
        send_signal=lambda sig: pytest.fail("accounting must settle"),
    )
    owner = SimpleNamespace(children=lambda recursive: [accountant, scientist])
    monkeypatch.setattr(storage_guard.psutil, "Process", lambda pid: owner)
    process = SimpleNamespace(pid=123, wait=lambda **kwargs: events.append(("settled", kwargs)))
    storage_guard.stop_owned_worker(process)
    assert events[0] == ("scientific", storage_guard.signal.SIGINT)
    assert events[1] == ("settled", {"timeout": 30})


def test_worker_exit_code_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_guard, "check_storage", lambda *a, **kw: 0)
    worker = SimpleNamespace(wait=lambda **kw: 7)
    monkeypatch.setattr(storage_guard.subprocess, "Popen", lambda *a, **kw: worker)
    assert storage_guard.run(["worker"], root=tmp_path, pause_paths=[tmp_path / "PAUSE"]) == 7
    assert not (tmp_path / "PAUSE").exists()


@pytest.mark.parametrize("exit_code", [0, 1])
def test_completed_cleanup_runs_only_after_successful_worker_exit(tmp_path, monkeypatch, exit_code):
    events = []
    monkeypatch.setattr(storage_guard, "check_storage", lambda *a, **kw: 0)

    def wait(**kwargs):
        events.append("worker-exited")
        return exit_code

    monkeypatch.setattr(
        storage_guard.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(wait=wait)
    )
    monkeypatch.setattr(
        storage_guard, "deduplicate_completed", lambda *a: events.append(("dedup", a))
    )
    assert (
        storage_guard.run(
            ["worker"],
            root=tmp_path,
            pause_paths=[],
            dedup_helper="helper",
            completed_run_root=tmp_path,
        )
        == exit_code
    )
    expected = ["worker-exited"]
    if exit_code == 0:
        expected.append(("dedup", ("helper", tmp_path)))
    assert events == expected


def test_optional_cleanup_failure_keeps_scientific_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(storage_guard, "check_storage", lambda *a, **kw: 0)
    monkeypatch.setattr(
        storage_guard.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(wait=lambda **kw: 0)
    )

    def fail(command, **kwargs):
        assert command[-2:] == ["--receipt", str(tmp_path / "storage-dedup.json")]
        raise storage_guard.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(storage_guard.subprocess, "run", fail)
    assert (
        storage_guard.run(
            ["worker"],
            root=tmp_path,
            pause_paths=[],
            dedup_helper="helper",
            completed_run_root=tmp_path,
        )
        == 0
    )
    assert "deduplication skipped" in capsys.readouterr().err


@pytest.fixture
def launch_policy(tmp_path):
    worker = tmp_path / "original.sh"
    worker.write_text("#!/bin/bash\nexit 0\n")
    source = tmp_path / "guard.py"
    source.write_text("# reviewed guard\n")
    policy = {
        "storage_guard": {
            "python": "/usr/bin/python3",
            "source": storage_guard._binding(source),
            "usage_root": str(tmp_path),
            "min_free_bytes": 100,
            "max_used_bytes": 600,
            "growth_reserve_bytes": 16,
        }
    }
    launch = {
        "argv": ["/usr/bin/srun", "--jobid=14372", "/bin/bash", str(worker)],
        "bindings": [storage_guard._binding(worker)],
        "pause_paths": [str(tmp_path / "runtime" / "STOP")],
    }
    return launch, policy


def test_guard_wrapper_preserves_original_worker_and_binds_configuration(tmp_path, launch_policy):
    launch, policy = launch_policy
    original = launch["argv"][-1]
    guarded = storage_guard.guard_launch(launch, policy, tmp_path)
    assert launch["argv"][-1] == original
    assert guarded["argv"][:-1] == launch["argv"][:-1]
    assert guarded["storage_guard"]["worker"]["path"] == original
    assert guarded["storage_guard"]["policy_sha256"] == storage_guard._policy_hash(
        policy["storage_guard"]
    )
    storage_guard.validate_launch(guarded, policy, tmp_path)
    assert storage_guard.guard_launch(guarded, policy, tmp_path) == guarded
    assert Path(original).read_text() == "#!/bin/bash\nexit 0\n"


@pytest.mark.parametrize(
    "tamper", ["missing", "policy", "source", "worker", "wrapper", "bypass", "stop"]
)
def test_future_launch_cannot_bypass_storage_policy(tmp_path, launch_policy, tamper):
    launch, policy = launch_policy
    guarded = storage_guard.guard_launch(launch, policy, tmp_path)
    if tamper == "missing":
        guarded.pop("storage_guard")
    elif tamper == "policy":
        policy["storage_guard"]["max_used_bytes"] += 1
    elif tamper == "source":
        Path(policy["storage_guard"]["source"]["path"]).write_text("changed")
    elif tamper == "worker":
        Path(guarded["storage_guard"]["worker"]["path"]).write_text("changed")
    elif tamper == "wrapper":
        path = Path(guarded["argv"][-1])
        path.write_text("#!/bin/bash\nexit 0\n")
        guarded["bindings"] = [
            storage_guard._binding(path) if item["path"] == str(path) else item
            for item in guarded["bindings"]
        ]
    elif tamper == "bypass":
        guarded["argv"][-1] = launch["argv"][-1]
    else:
        guarded["storage_guard"]["stop_path"] = str(tmp_path / "unregistered-stop")
    with pytest.raises(ValueError):
        storage_guard.validate_launch(guarded, policy, tmp_path)


def test_optional_policy_preserves_old_deployments(launch_policy, tmp_path):
    launch, _ = launch_policy
    assert storage_guard.guard_launch(launch, {}, tmp_path) is launch
    assert storage_guard.validate_launch(launch, {}, tmp_path) is None


def test_new_policy_uses_fresh_wrapper_and_binds_completed_cleanup(launch_policy, tmp_path):
    launch, policy = launch_policy
    old = storage_guard.guard_launch(launch, policy, tmp_path)
    old_bytes = Path(old["argv"][-1]).read_bytes()
    helper = tmp_path / "deduplicate_fitting.py"
    helper.write_text("# reviewed helper\n")
    policy["storage_guard"]["completed_fitting_dedup"] = {"source": storage_guard._binding(helper)}
    with pytest.raises(ValueError, match="completion.json"):
        storage_guard.guard_launch(launch, policy, tmp_path)
    launch["run"] = {"completion_path": str(tmp_path / "completion.json")}
    new = storage_guard.guard_launch(launch, policy, tmp_path)
    assert new["argv"][-1] != old["argv"][-1]
    assert Path(old["argv"][-1]).read_bytes() == old_bytes
    assert "--completed-run-root " + str(tmp_path) in Path(new["argv"][-1]).read_text()
    assert storage_guard._binding(helper) in new["bindings"]
    storage_guard.validate_launch(new, policy, tmp_path)
    helper.write_text("# changed\n")
    with pytest.raises(ValueError, match="source binding changed"):
        storage_guard.validate_launch(new, policy, tmp_path)


def test_storage_pause_incident_is_stable_across_multiple_markers(tmp_path):
    plain, first, second = [tmp_path / name for name in ("USER", "PAUSE", "STOP")]
    plain.write_text("User requested maintenance\n")
    first.write_text("Storage safety guard: free reserve 1 < 100\n")
    second.write_text(first.read_text())
    assert storage_guard.pause_incident([plain]) is None
    incident = storage_guard.pause_incident([plain, first, second])
    assert incident == storage_guard.pause_incident([second, first])
    assert incident["kind"] == "storage_safety"


def test_preparing_new_batch_automatically_binds_storage_guard(
    tmp_path, monkeypatch, launch_policy
):
    from tests.prepared_batch_history_test import _fixture
    from tools import prepared_batch

    recipe, _, _ = _fixture(tmp_path)
    _, policy = launch_policy
    policy.update(repository=str(tmp_path), allocation="14372")
    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    prepared_batch.write(supervisor / "policy.json", policy)
    prepared_batch.write(supervisor / "registry.json", {"runs": []})
    environment = tmp_path / "environment.json"
    prepared_batch.write(environment, {})
    code = tmp_path / "code"
    (code / "tools").mkdir(parents=True)
    (code / "tools/prepared_batch.py").write_text("# pinned worker\n")
    monkeypatch.setattr(prepared_batch.subprocess, "check_output", lambda *a, **kw: "1" * 40)
    batch = {
        "id": "E18",
        "scientific_step": "E18",
        "binding_path": recipe["group"]["path"],
        "depends_on": [],
    }
    launch = prepared_batch.prepare(
        tmp_path / "next",
        batch,
        base_campaign=Path(recipe["base_campaign"]["path"]),
        code=code,
        supervisor=supervisor,
        environment_path=environment,
        checks=["test"],
    )
    storage_guard.validate_launch(launch, policy, supervisor)
    assert Path(launch["argv"][-1]).name.startswith("worker-entry.storage-guard-")
    assert launch["storage_guard"]["worker"]["path"].endswith("worker-entry.sh")


@pytest.mark.parametrize("failed_stderr", [False, True])
def test_failed_pause_write_still_stops_owned_worker(tmp_path, monkeypatch, failed_stderr):
    checks = iter([10, OSError("Disk quota exceeded")])

    def check(*args, **kwargs):
        value = next(checks)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(storage_guard, "check_storage", check)
    monkeypatch.setattr(storage_guard, "pause", lambda *a: (_ for _ in ()).throw(OSError("quota")))
    if failed_stderr:
        monkeypatch.setattr(
            storage_guard,
            "print",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("log quota")),
            raising=False,
        )
    worker = SimpleNamespace(
        wait=lambda **kw: (_ for _ in ()).throw(
            storage_guard.subprocess.TimeoutExpired("worker", 0)
        )
    )
    monkeypatch.setattr(storage_guard.subprocess, "Popen", lambda *a, **kw: worker)
    stopped = []
    monkeypatch.setattr(storage_guard, "stop_owned_worker", stopped.append)
    assert storage_guard.run(["worker"], root=tmp_path, pause_paths=[tmp_path / "PAUSE"]) == 75
    assert stopped == [worker]


def test_recreated_storage_pause_is_a_new_notification_episode(tmp_path):
    pause = tmp_path / "PAUSE"
    reason = "Storage safety guard: Disk quota exceeded\n"
    pause.write_text(reason)
    first = storage_guard.pause_incident([pause])
    assert storage_guard.pause_incident([pause]) == first
    stamp = pause.stat().st_mtime_ns
    pause.unlink()
    pause.write_text(reason)
    os.utime(pause, ns=(stamp + 1000000000, stamp + 1000000000))
    second = storage_guard.pause_incident([pause])
    assert second["reason"] == first["reason"]
    assert second["id"] != first["id"]
    assert storage_guard.pause_incident([pause]) == second
