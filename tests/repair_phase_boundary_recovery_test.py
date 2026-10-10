"""Module-mode boundary handling and fail-closed receipt-only recovery."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.repair.historical_regression import binding
from tools.repair.phase_boundary_recovery import validate_recovery


@pytest.mark.parametrize(
    "mode", ["fit", "development", "qualify", "recover", "recover_mutated", "unguarded"]
)
def test_module_main_catches_training_boundary(tmp_path, mode):
    # Use the real module entry point and a canonical training-side import. GPU,
    # data and training work are stubbed; the duplicate __main__ import is real.
    script = r"""
import json, os, runpy, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools.repair import train
from tools.repair.historical_regression import binding
from tools.repair.primary_runtime import FIT_GPU, DEV_GPU

root, requested = Path(sys.argv[1]), sys.argv[2]
mode = "fit" if requested in {"recover", "recover_mutated", "unguarded"} else requested
manifest_path = root / "manifest.json"
manifest_path.write_text("{}")
attempt = root / "attempt"
attempt.mkdir()
(attempt / "step.json").write_text('{"step_id":"1.1"}')
manifest = dict(schema="exact-repair/primary-phase-runtime/v1", execution_authorized=True,
    admission={}, protocol={}, preparation={}, shared_weak_labels={}, seed=13,
    phase_gpu_uuids=dict(fit=FIT_GPU, development=DEV_GPU), state_directory=str(root / "state"))
recovery = None
if requested.startswith("recover"):
    import torch
    manifest["phase_boundary_recovery"] = {"path":"bound-contract"}
    state = root / "state/training-state.pt"
    state.parent.mkdir()
    torch.save({}, state)
    recovery = dict(status="phase_complete", next_phase="development", state=binding(state),
        optimizer_updates=630, optimizer_updates_replayed=0)
gates = dict.fromkeys(("teacher_qualified", "shared_weak_labels_qualified", "weak_scale_frozen",
    "measured_common_endpoint_admitted", "gpu_and_stage_projection_passed"), True)
config = dict(target_basis="symbolic", profile={}, development_case_ids=[],
    development_epochs=[5, 6], decode_seconds=1)
ledger = dict(stage_limits={"model-symbolic-13":dict(gpu_seconds=14400)},
    attempts={str(attempt):dict(status="reserved", started_epoch=0, reserved_seconds=1,
        budget_stages=["learning", "primary_training", "model-symbolic-13"], resources=dict(gpus=1))})
def read(path):
    if Path(path).name == "step.json": return dict(step_id="1.1")
    if Path(path).name == "ledger.json": return ledger
    return manifest
def boundary(*args, **kwargs):
    from tools.repair.primary_runtime import PhaseBoundary
    if requested == "recover_mutated": state.write_bytes(b"changed")
    if requested in {"recover", "recover_mutated", "unguarded"}:
        raise ValueError("Fit cannot bypass a pending DEV pass")
    raise PhaseBoundary("fit" if mode == "development" else "development", 5, 630)
os.environ.update(SLURM_JOB_ID="1", SLURM_STEP_ID="1", EXACT_REPAIR_DEADLINE_EPOCH="9999999999",
    EXACT_REPAIR_STAGE_LEDGER=str(root / "ledger.json"))
patches = {
    "tools.repair.batch.read": read,
    "tools.repair.shared_release.bound": lambda x: gates,
    "tools.repair.shared_release.authenticate": lambda x: manifest_path,
    "tools.repair.prepare.load_preparation": lambda x: ([], {}, {}),
    "exact.repair.protocol.load_protocol_v3": lambda *a, **k: {},
    "exact.repair.protocol.training_projection_v3": lambda x: x,
    "exact.repair.semantic_fidelity.read_fidelity_training_artifact": lambda *a: {},
    "tools.repair.common_training.validate_closed_preparation": lambda *a, **k: None,
    "tools.repair.training_readiness.require_ready": lambda *a: None,
    "tools.repair.train._protocol_arguments": lambda x: dict(config),
    "tools.repair.train._protocol_profile": lambda x: {},
    "tools.repair.train.scheduled_development": lambda *a, **k: [],
    "tools.repair.train._train_payload": boundary,
    "tools.repair.train.train_cases": boundary,
    "tools.repair.phase_boundary_recovery.validate_recovery": lambda *a: recovery,
    "torch.cuda.device_count": lambda: 1,
    "torch.cuda.get_device_properties": lambda x: SimpleNamespace(uuid=DEV_GPU if mode == "development" else FIT_GPU),
    "torch.cuda.reset_peak_memory_stats": lambda: None,
    "torch.cuda.max_memory_allocated": lambda: 0,
}
for name, value in patches.items(): patch(name, value).start()
sys.argv = ["tools.repair.primary_runtime", mode, str(manifest_path), str(root / "output")]
runpy.run_module("tools.repair.primary_runtime", run_name="__main__", alter_sys=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), mode],
        text=True,
        capture_output=True,
        timeout=45,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    if mode in {"recover_mutated", "unguarded"}:
        assert result.returncode != 0
        assert not (tmp_path / "output/report.json").exists()
        assert (
            "changed the checkpoint" if mode == "recover_mutated" else "pending DEV"
        ) in result.stderr
        return
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["status"] == "phase_complete"
    assert report["next_phase"] == ("fit" if mode == "development" else "development")
    if mode == "recover":
        assert report["optimizer_updates_replayed"] == 0


def recovery_fixture(tmp_path):
    def artifact(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return binding(path)

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    manifest = dict(state_directory=str(state_dir))
    original = artifact("manifest.json", manifest)
    step = dict(step_id="14451.102", dispatch_nonce="nonce")
    failed = dict(
        **step,
        status="failed",
        exit_code=1,
        job_id="fit5",
        error=dict(
            message="command 0 exited 1: tools.repair.primary_runtime.PhaseBoundary: development"
        ),
    )
    rng = dict(cpu=torch.tensor([1], dtype=torch.uint8), python=(1,), cuda=[])
    saved = dict(
        schema="exact-repair/training-state/v3",
        recovery_revision="exact-phase-resume/v3.1",
        identity="frozen-inputs-and-source",
        phase="development_loss",
        next_epoch=4,
        next_offset=1,
        optimized=1,
        optimizer_updates=5,
        optimizer_case_counts={"a": 5},
        epoch_order=["a"],
        development_progress=dict(epoch=4, losses={}, decoded={}, generated={}),
        development_case_ids=["d"],
        history=[dict(epoch=e, development_status="not_scheduled") for e in range(1, 5)],
        first_started_epoch=100,
        execution_count=1,
        fit_rng=rng,
        cpu_rng=rng["cpu"],
        python_rng=rng["python"],
        cuda_rng=rng["cuda"],
        model={"w": torch.tensor([1.0])},
        optimizer={"state": {0: {"step": torch.tensor(5), "exp_avg": torch.tensor([0.1])}}},
        train_loss=0.5,
    )
    live, snapshot = state_dir / "training-state.pt", tmp_path / "snapshot.pt"

    def freeze(value=saved, failure=failed):
        torch.save(value, live)
        snapshot.write_bytes(live.read_bytes())
        contract = dict(
            schema="exact-repair/phase-boundary-recovery/v1",
            original_manifest=original,
            completion=artifact("completion.json", failure),
            step=artifact("step.json", step),
            failed_run_id="fit5",
            live_state=binding(live),
            checkpoint_snapshot=binding(snapshot),
            checkpoint_identity="frozen-inputs-and-source",
            first_started_epoch=100,
        )
        return {**manifest, "phase_boundary_recovery": artifact("recovery.json", contract)}

    return freeze, saved, failed, live


def test_recovery_preserves_checkpoint_and_counts(tmp_path):
    freeze, _, _, live = recovery_fixture(tmp_path)
    manifest = freeze()
    before = live.read_bytes()
    report = validate_recovery(
        manifest,
        "fit",
        dict(development_epochs=[5, 6], development_case_ids=["d"]),
        [(SimpleNamespace(case_id="a"), None)],
    )
    assert report["optimizer_updates"] == 5
    assert report["optimizer_updates_replayed"] == 0
    assert report["original_failure_preserved"] is True
    assert live.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["partial", "dev_started", "rng", "nan", "nonce", "live", "contract"]
)
def test_recovery_rejects_incompatible_or_changed_evidence(tmp_path, change):
    freeze, saved, failed, live = recovery_fixture(tmp_path)
    saved, failed = copy.deepcopy(saved), copy.deepcopy(failed)
    if change == "partial":
        saved["optimizer_updates"] -= 1
    if change == "dev_started":
        saved["development_progress"]["generated"]["d"] = {}
    if change == "rng":
        saved["cpu_rng"] = torch.tensor([2], dtype=torch.uint8)
    if change == "nan":
        saved["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    if change == "nonce":
        failed["dispatch_nonce"] = "different"
    manifest = freeze(saved, failed)
    if change == "live":
        live.write_bytes(b"changed")
    if change == "contract":
        manifest["seed"] = 37
    with pytest.raises(ValueError):
        validate_recovery(
            manifest,
            "fit",
            dict(development_epochs=[5, 6], development_case_ids=["d"]),
            [(SimpleNamespace(case_id="a"), None)],
        )
