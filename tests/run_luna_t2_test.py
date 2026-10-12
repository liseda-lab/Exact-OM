"""Ownership and failure handoff for the optional remote Luna replay."""

import fcntl
import os
from pathlib import Path

import pytest

from tools import run_luna_t2 as launcher


@pytest.fixture
def plan(tmp_path, monkeypatch):
    supervisor, output = tmp_path / "supervisor", tmp_path / "output"
    root = output / "dispatch"
    supervisor.mkdir()
    root.mkdir(parents=True)
    launcher.write(supervisor / "registry.json", {"runs": [], "pause_paths": []})
    environment = tmp_path / "environment.json"
    launcher.write(environment, {"EXACT_HOSTED_EXPERIMENT_ID": "E07"})
    credential = tmp_path / "api_key"
    credential.write_text("current-fixture-key\n")
    value = {
        "dispatch_root": str(root),
        "output": str(output),
        "supervisor": str(supervisor),
        "bindings": [launcher.binding(environment)],
        "environment": launcher.binding(environment),
        "code_root": str(tmp_path),
        "api_key_path": str(credential),
        "nonce": "test-launch-nonce",
        "coordinator_allocation": "14372",
        "remote_argv": ["fixture-srun"],
    }
    path = tmp_path / "plan.json"
    launcher.write(path, value)
    monkeypatch.setenv("SLURM_JOB_ID", "14372")
    monkeypatch.setattr(
        launcher,
        "publish_step_receipt",
        lambda path, nonce: launcher.write(path, {"step_id": "14372.99", "dispatch_nonce": nonce}),
    )
    return path, value


def test_remote_environment_removes_inherited_allocation_and_credentials():
    assert launcher.remote_environment(
        {
            "SLURM_JOB_ID": "14372",
            "SLURM_STEP_ID": "99",
            "SLURM_JOB_GPUS": "0",
            "EXACT_HOSTED_EXPERIMENT_ID": "E17",
            "CUDA_VISIBLE_DEVICES": "0",
            "OPENROUTER_API_KEY": "old-fixture",
            "OPENAI_API_KEY": "other-fixture",
            "PATH": "/bin",
        }
    ) == {"PATH": "/bin"}


def test_competing_owner_does_not_change_receipts(plan, monkeypatch):
    path, value = plan
    root = Path(value["dispatch_root"])
    launcher.write(root / "status.json", {"status": "running", "owner": "original"})
    before = (root / "status.json").read_bytes()
    with (root / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert launcher.coordinate(path) == 73
    assert (root / "status.json").read_bytes() == before
    assert not (root / "exit-code").exists()


@pytest.mark.parametrize("receipt", ["remote-step.json", "dispatch-claim.json"])
def test_old_remote_receipt_prevents_duplicate_launch(plan, monkeypatch, receipt):
    path, value = plan
    root = Path(value["dispatch_root"])
    launcher.write(root / receipt, {"step_id": "14465.0"})
    monkeypatch.setattr(
        launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate launch")
    )
    with pytest.raises(ValueError, match="already has an owner"):
        launcher.coordinate(path)


def test_wrong_allocation_cannot_poison_step_receipt(plan, monkeypatch):
    path, value = plan
    monkeypatch.setenv("SLURM_JOB_ID", "14451")
    with pytest.raises(ValueError, match="outside its retained allocation"):
        launcher.coordinate(path)
    assert not (Path(value["dispatch_root"]) / "step.json").exists()


@pytest.mark.parametrize("remote_code", [0, 9])
def test_remote_result_propagates_to_supervisor_receipts(plan, monkeypatch, remote_code):
    path, value = plan
    root, output = Path(value["dispatch_root"]), Path(value["output"])

    class Process:
        def __init__(self, argv, *, cwd, env):
            assert argv == ["fixture-srun"] and cwd == value["code_root"]
            assert "SLURM_JOB_ID" not in env
            assert (root / "dispatch-claim.json").exists()
            launcher.write(
                root / "remote-step.json", {"step_id": "14466.0", "dispatch_nonce": value["nonce"]}
            )
            if remote_code == 0:
                for name in ("manifest.json", "predictions.jsonl"):
                    (output / name).write_text("{}\n")
                launcher.write(
                    output / "comparison.json",
                    {
                        "status": "complete",
                        "manifest": launcher.binding(output / "manifest.json"),
                        "predictions": launcher.binding(output / "predictions.jsonl"),
                    },
                )
                launcher.write(
                    output / "completion.json",
                    {
                        "status": "complete",
                        "report": launcher.binding(output / "comparison.json"),
                    },
                )

        def wait(self):
            return remote_code

        def poll(self):
            return remote_code

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    if remote_code:
        with pytest.raises(RuntimeError, match="exited 9"):
            launcher.coordinate(path)
    else:
        assert launcher.coordinate(path) == 0
    receipt = launcher.read(root / "completion.json")
    assert receipt["status"] == ("failed" if remote_code else "complete")
    assert receipt["step_id"] == "14372.99"
    assert int((root / "exit-code").read_text()) == remote_code


def test_pause_prevents_remote_submission(plan, monkeypatch):
    path, value = plan
    (Path(value["supervisor"]) / "PAUSE").touch()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("paused launch"))
    with pytest.raises(RuntimeError, match="pause/STOP"):
        launcher.coordinate(path)


def test_remote_worker_refreshes_repo_key_and_e07_environment(plan, monkeypatch):
    path, value = plan
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "liseda-t2")
    monkeypatch.setenv("OPENROUTER_API_KEY", "expired-fixture")
    monkeypatch.setenv("EXACT_HOSTED_EXPERIMENT_ID", "E17")
    from tools import compare_luna_e07

    def run(output):
        assert output == value["output"]
        assert os.environ["OPENROUTER_API_KEY"] == "current-fixture-key"
        assert os.environ["EXACT_HOSTED_EXPERIMENT_ID"] == "E07"
        return {"status": "complete"}

    monkeypatch.setattr(compare_luna_e07, "run", run)
    assert launcher.worker(path) == 0
    assert "fixture-key" not in (Path(value["dispatch_root"]) / "remote-step.json").read_text()
