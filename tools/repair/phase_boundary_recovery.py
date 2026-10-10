"""Authenticate an unpublished first-DEV boundary without replaying optimization."""

from __future__ import annotations

import math
from pathlib import Path

from tools.repair.historical_regression import binding
from tools.repair.phase_boundary import PhaseBoundary
from tools.repair.shared_release import authenticate, bound


def validate_recovery(manifest, phase, config, fitting):
    """Return a candidate receipt; the runner must still validate exact resume.

    The original failed receipt stays failed. An immutable checkpoint copy pins
    what survived the failure; its live counterpart must remain byte-identical
    through recovery. Only the first scheduled DEV boundary is supported.
    """
    import torch

    contract = bound(manifest["phase_boundary_recovery"])
    original = bound(contract["original_manifest"])
    if (
        contract.get("schema") != "exact-repair/phase-boundary-recovery/v1"
        or phase != "fit"
        or {k: v for k, v in manifest.items() if k != "phase_boundary_recovery"} != original
    ):
        raise ValueError("Boundary recovery changed the original phase contract")
    failed, step = bound(contract["completion"]), bound(contract["step"])
    if (
        failed.get("status") != "failed"
        or failed.get("exit_code") != 1
        or failed.get("error", {}).get("message")
        != "command 0 exited 1: tools.repair.primary_runtime.PhaseBoundary: development"
        or failed.get("dispatch_nonce") != step.get("dispatch_nonce")
        or failed.get("step_id") != step.get("step_id")
        or failed.get("job_id") != contract["failed_run_id"]
    ):
        raise ValueError("Boundary recovery requires the diagnosed nonce-bound failure")
    live = Path(manifest["state_directory"]) / "training-state.pt"
    if binding(live) != contract["live_state"]:
        raise ValueError("Boundary recovery live checkpoint changed")
    snapshot = authenticate(contract["checkpoint_snapshot"])
    if contract["checkpoint_snapshot"]["sha256"] != contract["live_state"]["sha256"]:
        raise ValueError("Boundary recovery snapshot differs from the live checkpoint")
    saved = torch.load(snapshot, weights_only=True, map_location="cpu")
    epoch = config["development_epochs"][0]
    ids = {case.case_id for case, _ in fitting}
    rng = saved.get("fit_rng") or {}
    history = saved.get("history", [])
    if not (
        saved.get("schema") == "exact-repair/training-state/v3"
        and saved.get("recovery_revision") == "exact-phase-resume/v3.1"
        and saved.get("identity") == contract["checkpoint_identity"]
        and saved.get("phase") == "development_loss"
        and saved.get("next_epoch") == epoch - 1
        and saved.get("next_offset") == saved.get("optimized") == len(ids)
        and saved.get("optimizer_updates") == epoch * len(ids)
        and saved.get("optimizer_case_counts") == dict.fromkeys(ids, epoch)
        and len(saved.get("epoch_order", ())) == len(ids)
        and set(saved.get("epoch_order", ())) == ids
        and saved.get("development_progress")
        == dict(epoch=epoch - 1, losses={}, decoded={}, generated={})
        and saved.get("development_case_ids") == config["development_case_ids"]
        and [row.get("epoch") for row in history] == list(range(1, epoch))
        and all(row.get("development_status") == "not_scheduled" for row in history)
        and saved.get("best_epoch") is None
        and saved.get("best_state") is None
        and not saved.get("pending_acquisition")
        and saved.get("first_started_epoch") == contract["first_started_epoch"]
        and saved.get("execution_count") == 1
        and set(rng) == {"cpu", "python", "cuda"}
        and torch.equal(rng["cpu"], saved["cpu_rng"])
        and rng["python"] == saved["python_rng"]
        and len(rng["cuda"]) == len(saved["cuda_rng"])
        and all(torch.equal(a, b) for a, b in zip(rng["cuda"], saved["cuda_rng"]))
        and bool(saved.get("model"))
        and bool(saved.get("optimizer", {}).get("state"))
    ):
        raise ValueError("Checkpoint is not the untouched first scheduled DEV boundary")

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return True

    if not all(finite(saved[key]) for key in ("model", "optimizer", "train_loss", "history")):
        raise ValueError("Boundary recovery requires finite model and optimizer state")
    return dict(
        **PhaseBoundary("development", epoch, saved["optimizer_updates"]).receipt,
        state=contract["live_state"],
        recovery=manifest["phase_boundary_recovery"],
        recovered_from_step=step["step_id"],
        recovered_from_run=contract["failed_run_id"],
        optimizer_updates_replayed=0,
        checkpoint_unchanged=True,
        original_failure_preserved=True,
    )
