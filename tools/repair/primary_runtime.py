"""Owned, process-separated primary fit and DEV segments.

A phase boundary is a successful durable checkpoint, never a trained model or a
permission to retain the GPU. The dispatcher must finish the step before handing
that checkpoint to the next device owner. Primary execution requires a separate
resolved admission; engineering qualification never produces selectable weights.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import time
from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, immutable

FIT_GPU = "GPU-19b25b77-7f5b-79a9-0c39-843ea57bd484"
DEV_GPU = "GPU-7b3f1042-3c17-9c54-5653-415de63df835"


class PhaseBoundary(Exception):
    def __init__(self, next_phase, completed_epoch, optimizer_updates):
        self.receipt = dict(
            status="phase_complete",
            next_phase=next_phase,
            completed_epoch=completed_epoch,
            optimizer_updates=optimizer_updates,
        )
        super().__init__(next_phase)


def remaining_dev_reserve(epochs, limits, seed, condition, case_ids, history, progress, default):
    """Reserve *all* uncompleted scheduled passes, without renewing begun cases."""
    completed = {row["epoch"] for row in history}
    reserve = 0.0
    for epoch in epochs:
        if epoch in completed:
            continue
        for case_id in case_ids:
            slot = canonical_hash(
                dict(seed=seed, supervision_condition=condition, epoch=epoch, case_id=case_id)
            )
            seconds = limits[slot] if limits is not None else default
            if progress.get("epoch") == epoch - 1:
                if case_id in progress.get("generated", {}):
                    continue
                if case_id in progress.get("case_deadlines_epoch", {}):
                    seconds = min(
                        seconds, max(0, progress["case_deadlines_epoch"][case_id] - time.time())
                    )
                else:
                    seconds -= progress.get("case_seconds_spent", {}).get(case_id, 0)
            reserve += max(0, seconds)
    return reserve


def phase_job(model_id, phase, seconds, *, oversized=False):
    if phase not in {"fit", "development"} or seconds <= 2:
        raise ValueError("Invalid owned phase")
    gpu = FIT_GPU if phase == "fit" or oversized else DEV_GPU
    stages = ["learning", "primary_training", model_id]
    if phase == "development":
        stages += ["development"]
        if gpu == DEV_GPU:
            stages += ["secondary_learning"]
    return dict(
        stage=phase,
        budget_stages=stages,
        gpu_devices=[gpu],
        cleanup_seconds=5,
        seconds=seconds,
        slice_seconds=seconds,
        resources=dict(
            cpus=4 if phase == "fit" else 5,
            gpus=1,
            memory_mb=24576 if phase == "fit" else 30720,
            gres="gpu:rtx5090:1" if gpu == FIT_GPU else "gpu:rtx2080ti:1",
        ),
    )


def owned_device(expected):
    import torch

    if not os.environ.get("SLURM_STEP_ID") or "EXACT_REPAIR_DEADLINE_EPOCH" not in os.environ:
        raise ValueError("Runtime requires a deadline-bound Slurm owner")
    if torch.cuda.device_count() != 1:
        raise ValueError("Runtime requires exactly one Slurm-visible GPU")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.removeprefix("GPU-").lower() != expected.removeprefix("GPU-").lower():
        raise ValueError("Visible GPU UUID differs from the admitted owner")
    return actual


def run_phase(manifest_path, phase, output):
    """Run one resolved segment; immutable contracts and one shared state lock."""
    import torch

    from exact.repair.protocol import load_protocol_v3, training_projection_v3
    from exact.repair.semantic_fidelity import read_fidelity_training_artifact
    from tools.repair import train
    from tools.repair.prepare import load_preparation

    manifest, output = read(manifest_path), Path(output)
    if manifest.get("schema") != "exact-repair/primary-phase-runtime/v1" or not manifest.get(
        "execution_authorized"
    ):
        raise ValueError("Primary phase contract is not admitted")
    admission = bound(manifest["admission"])
    if not all(
        admission.get(k) is True
        for k in (
            "teacher_qualified",
            "shared_weak_labels_qualified",
            "weak_scale_frozen",
            "measured_common_endpoint_admitted",
            "gpu_and_stage_projection_passed",
        )
    ):
        raise ValueError("Unresolved primary admission gate")
    protocol = training_projection_v3(
        load_protocol_v3(authenticate(manifest["protocol"]), for_execution=True)
    )
    prepared = authenticate(manifest["preparation"])
    cases, caches, preparation = load_preparation(prepared)
    from tools.repair.common_training import validate_closed_preparation

    validate_closed_preparation(cases, preparation, protocol, case_limit=None, retry_labels=False)
    config = train._protocol_arguments(protocol)
    expected = manifest["phase_gpu_uuids"][phase]
    if (phase == "fit" and expected != FIT_GPU) or expected not in {FIT_GPU, DEV_GPU}:
        raise ValueError("Invalid phase device route")
    if phase == "development" and expected == FIT_GPU:
        routing = bound(manifest["oversized_routing_receipt"])
        if routing.get("status") != "resource_limited" or routing.get("phase") != "development":
            raise ValueError("Oversized routing requires the prior resource-limited DEV receipt")
    actual = owned_device(expected)
    # A scheduler-assigned bound includes all stage/model reservations, even after
    # a replacement or oversized route. No private elapsed allowance is renewed.
    remaining = float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]) - time.time() - 5
    if remaining <= 0:
        raise TimeoutError("Owned phase deadline exhausted")
    state = Path(manifest["state_directory"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "phase.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        labels = read_fidelity_training_artifact(
            authenticate(manifest["shared_weak_labels"]), "train"
        )
        config.update(
            device="cuda",
            execution_phase=phase,
            checkpoint_path=state / "training-state.pt",
            deadline_seconds=remaining,
            total_training_seconds=21600,
            seed=manifest["seed"],
            profile=train._protocol_profile(protocol),
            fidelity_labels=labels if config["target_basis"] == "symbolic_plus_llm" else {},
        )
        fitting = [
            (c, caches[c.case_id]) for c in cases if c.split == "train" and c.case_id in caches
        ]
        dev = train.scheduled_development(
            cases, caches, expected_case_ids=config["development_case_ids"]
        )
        try:
            # Normal completion has the existing full selection report. Publish
            # selected weights only after both scheduled passes and common endpoint.
            artifact, report = train._train_payload(fitting, dev, config)
            from tools.repair.training_report import read_report

            metrics = read_report(report, state / "reports")
            endpoint_met = (
                metrics["status"] == "complete"
                and metrics["optimizer_updates"] == admission["common_optimizer_updates"]
            )
            result = dict(
                status="complete" if endpoint_met else "incomplete_endpoint",
                checkpoint=artifact,
                report=report,
                endpoint_met=endpoint_met,
                optimizer_updates=metrics["optimizer_updates"],
                retryable=False,
            )
        except PhaseBoundary as boundary:
            result = boundary.receipt
        except torch.cuda.OutOfMemoryError as error:
            result = dict(
                status="resource_limited",
                detail=str(error),
                retryable=False,
                route_at_owned_boundary=phase == "development" and expected == DEV_GPU,
            )
        except TimeoutError as error:
            result = dict(status="resource_limited", detail=str(error), retryable=False)
        except ValueError as error:
            if not str(error).startswith("No eligible development checkpoint"):
                raise
            result = dict(status="incomplete_selection", detail=str(error), retryable=False)
        result.update(
            manifest=binding(Path(manifest_path)),
            phase=phase,
            gpu_uuid=actual,
            state=(
                binding(state / "training-state.pt")
                if (state / "training-state.pt").exists()
                else None
            ),
            budgets_reset=False,
        )
        immutable(output / "report.json", result)
        return result


def qualify(prepared_path, output, seconds=600):
    """One disposable symbolic epoch on the actual frozen TRAIN inventories.

    No teacher selection, DEV outcomes, weak labels or selectable model. Incomplete
    qualification is a bounded engineering result and does not justify replay.
    """
    import torch

    from exact.repair.protocol import load_protocol_v3, training_projection_v3
    from tools.repair import train
    from tools.repair.prepare import load_preparation

    actual = owned_device(FIT_GPU)
    output = Path(output)
    prep = read(prepared_path)
    protocol = training_projection_v3(load_protocol_v3(authenticate(prep["protocol"])))
    cases, caches, _ = load_preparation(prepared_path)
    config = train._protocol_arguments(protocol)
    started = time.time()
    remaining = min(seconds, float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]) - started - 10)
    if remaining <= 0:
        raise TimeoutError("Qualification deadline exhausted")
    config.update(
        epochs=1,
        development_epochs=[1],
        development_case_seconds=None,
        final_development_reserve_seconds=0,
        execution_phase="fit",
        device="cuda",
        checkpoint_path=output / "engineering-state.pt",
        seed=13,
        deadline_seconds=remaining,
        total_training_seconds=remaining,
        # No DEV will execute in this diagnostic; reserve zero for it.
        decode_seconds=0.001,
        post_decode_annotation_manifest=None,
        profile=train._protocol_profile(protocol),
    )
    fitting = [(c, caches[c.case_id]) for c in cases if c.split == "train" and c.case_id in caches]
    dev = train.scheduled_development(
        cases, caches, expected_case_ids=config["development_case_ids"]
    )
    torch.cuda.reset_peak_memory_stats()
    try:
        train.train_cases(fitting, dev, **config)
        raise AssertionError("Engineering fit crossed DEV boundary")
    except PhaseBoundary as boundary:
        result = boundary.receipt
    except TimeoutError as error:
        result = dict(status="resource_limited", detail=str(error), retryable=False)
    saved = torch.load(output / "engineering-state.pt", weights_only=True, map_location="cpu")
    result.update(
        schema="exact-repair/current-inventory-throughput/v1",
        prepared=binding(Path(prepared_path)),
        gpu_uuid=actual,
        optimizer_updates=saved["optimizer_updates"],
        expected_train_cases=128,
        committed_caches=len(fitting),
        cases=[
            dict(
                case_id=c.case_id,
                family=c.family,
                control=c.control,
                cache_available=c.case_id in caches,
                optimizer_updates=saved.get("optimizer_case_counts", {}).get(c.case_id, 0),
                status=(
                    "measured"
                    if saved.get("optimizer_case_counts", {}).get(c.case_id, 0)
                    else "not_measured"
                ),
            )
            for c in cases
            if c.split == "train"
        ],
        update_seconds=saved.get("optimizer_update_seconds", []),
        full_epoch_completed=saved["optimizer_updates"] == len(fitting),
        elapsed_seconds=time.time() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        phase_state=binding(output / "engineering-state.pt"),
        trained_model=False,
        heldout_outcomes_opened=False,
        hosted_calls=0,
        scope="symbolic engineering only; combined-loss/concurrent-load admission still required",
    )
    immutable(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "fit", "development"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "qualify":
        qualify(args.manifest, args.output)
    else:
        run_phase(args.manifest, args.mode, args.output)


if __name__ == "__main__":
    main()
