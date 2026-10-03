"""A real numeric Slurm guard reports itself before slow storage admission."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.storage_safety_test import launch_policy  # noqa: F401
from tools import storage_guard

NONCE = "reviewed-dispatch-1234"


@pytest.fixture
def slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "14372")
    monkeypatch.setenv("SLURM_STEP_ID", "92")
    read_text = Path.read_text
    group = ["0::/system.slice/slurmstepd.scope/opaque/step_92/user/task_0\n"]

    def read(path, *args, **kwargs):
        return group[0] if path == Path("/proc/self/cgroup") else read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    return group


def test_receipt_is_durable_before_storage_scan_and_normal_worker_remains_compatible(
    tmp_path, monkeypatch, slurm
):
    path = tmp_path / "step.json"
    expected = {"step_id": "14372.92", "dispatch_nonce": NONCE}
    events = []

    def scan(*args, **kwargs):
        assert json.loads(path.read_text()) == expected
        assert not list(tmp_path.glob(".step-receipt-*"))
        events.append("scan")
        return 0

    def worker(*args, **kwargs):
        assert events == ["scan"]
        # The unchanged worker writes the same fields through atomic replacement.
        pending = path.with_suffix(".pending")
        pending.write_text(json.dumps(expected))
        os.replace(pending, path)
        events.append("worker")
        return SimpleNamespace(wait=lambda **kw: 0)

    monkeypatch.setattr(storage_guard, "check_storage", scan)
    monkeypatch.setattr(storage_guard.subprocess, "Popen", worker)
    assert (
        storage_guard.run(
            ["worker"], root=tmp_path, pause_paths=[], step_path=path, dispatch_nonce=NONCE
        )
        == 0
    )
    assert events == ["scan", "worker"]
    storage_guard.publish_step_receipt(path, NONCE)
    assert json.loads(path.read_text()) == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("SLURM_JOB_ID", ""),
        ("SLURM_STEP_ID", ""),
        ("SLURM_STEP_ID", "extern"),
        ("SLURM_STEP_ID", "batch"),
    ],
)
def test_no_receipt_without_numeric_slurm(tmp_path, monkeypatch, slurm, field, value):
    monkeypatch.setenv(field, value)
    path = tmp_path / "step.json"
    with pytest.raises(ValueError, match="numeric Slurm"):
        storage_guard.publish_step_receipt(path, NONCE)
    assert not path.exists()


def test_numeric_environment_outside_matching_slurm_cgroup_is_rejected(tmp_path, slurm):
    slurm[0] = "0::/user.slice/step_extern/task_0\n"
    with pytest.raises(ValueError, match="matching Slurm"):
        storage_guard.publish_step_receipt(tmp_path / "step.json", NONCE)


@pytest.mark.parametrize("nonce", ["", "too-short", "not a safe nonce", None])
def test_invalid_nonce_never_publishes_receipt(tmp_path, slurm, nonce):
    with pytest.raises(ValueError, match="dispatch nonce"):
        storage_guard.publish_step_receipt(tmp_path / "step.json", nonce)
    assert not (tmp_path / "step.json").exists()


def test_conflicting_receipt_is_never_overwritten(tmp_path, slurm):
    path = tmp_path / "step.json"
    old = '{"step_id":"14372.91","dispatch_nonce":"another-dispatch"}\n'
    path.write_text(old)
    with pytest.raises(ValueError, match="conflicts"):
        storage_guard.publish_step_receipt(path, NONCE)
    assert path.read_text() == old
    assert not list(tmp_path.glob(".step-receipt-*"))


def test_storage_rejection_leaves_authentic_receipt_and_never_starts_science(
    tmp_path, monkeypatch, slurm
):
    path = tmp_path / "step.json"

    def reject(*args, **kwargs):
        assert json.loads(path.read_text())["step_id"] == "14372.92"
        raise ValueError("Storage admission refused")

    monkeypatch.setattr(storage_guard, "check_storage", reject)
    monkeypatch.setattr(
        storage_guard.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("No worker before admission"),
    )
    assert (
        storage_guard.run(
            ["worker"],
            root=tmp_path,
            pause_paths=[tmp_path / "STOP"],
            step_path=path,
            dispatch_nonce=NONCE,
        )
        == 75
    )
    assert "Storage admission refused" in (tmp_path / "STOP").read_text()


def test_wrapper_binds_step_receipt_and_rejects_changed_nonce(tmp_path, request):
    launch, policy = request.getfixturevalue("launch_policy")
    launch.update(step_path=str(tmp_path / "step.json"), nonce=NONCE)
    guarded = storage_guard.guard_launch(launch, policy, tmp_path)
    text = Path(guarded["argv"][-1]).read_text()
    assert "--step-path " + str(tmp_path / "step.json") in text
    assert "--dispatch-nonce " + NONCE in text
    storage_guard.validate_launch(guarded, policy, tmp_path)
    guarded["nonce"] = "different-dispatch-1234"
    with pytest.raises(ValueError, match="required storage guard"):
        storage_guard.validate_launch(guarded, policy, tmp_path)
