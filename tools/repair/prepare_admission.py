"""Freeze a finite engineering slice without starting new scientific stage clocks."""

from __future__ import annotations

import argparse
import copy
import time
import uuid
from pathlib import Path

from tools.repair.batch import _stage_remaining, freeze, prepare_dispatch, read, sha
from tools.repair.evaluation_ownership import PROFILE, resources
from tools.repair.historical_regression import binding
from tools.repair.primary_runtime import DEV_GPU
from tools.repair.shared_release import bound, immutable, validate_completion


def prepare(campaign, output, repository):
    campaign, output, repository = map(lambda p: Path(p).resolve(), (campaign, output, repository))
    registry = read(campaign / "supervisor/registry.json")
    predecessor = registry["primary_evaluation_capacity_preparation"]["completion"]
    completed = bound(predecessor)
    terminal, outputs, _, _ = validate_completion(completed["audit_receipt"])
    for path, digest in outputs.items():
        if sha(Path(terminal["work"]) / path) != digest:
            raise ValueError("Predecessor output changed")
    campaign_value = read(campaign / "campaign.json")
    review_ref = registry["primary_runtime_preparation"]["throughput_review"]
    review = bound(review_ref)
    report_ref = review["report_bindings"]["qualify-current-inventory-5090-profile-recovery-001"]
    historical = bound(report_ref)
    if historical["status"] != "resource_limited" or historical["optimizer_updates"] != 99:
        raise ValueError("Prior terminal engineering observation changed")
    immutable(
        output / "historical-state-reference.json",
        dict(
            scope="terminal_train_engineering_state_io_only",
            checkpoint=historical["phase_state"],
            report=report_ref,
            throughput_review=review_ref,
            expected_optimizer_updates=99,
            no_optimization_or_scientific_replay=True,
        ),
    )
    nonces = {lane: uuid.uuid4().hex for lane in ("native", "inference")}
    manifest = dict(
        schema="exact-repair/admission-component-contract/v1",
        scope="authored_train_components",
        profile=PROFILE,
        seed=13,
        nonces=nonces,
        rows=dict(native=16, inference=128),
        case_seconds=20,
        barrier_seconds=120,
        barrier_directory=str(output / "barrier"),
        deadline_epoch=campaign_value["primary_freeze_epoch"],
        architecture=dict(
            node_types=13, relations=553, layers=3, width=128, heads=4, encoder="hgt", pairwise=True
        ),
        selected_weights=False,
        optimizer_updates=0,
        test_payloads_opened=False,
        hosted_calls=0,
        scientific_admission=False,
        predecessor=predecessor,
        limitations=[
            "authored fixture components, not representative actual cases",
            "no selected weights or combined loss",
            "generation and full E3/E4/E5 execution remain unqualified",
        ],
    )
    immutable(output / "fixture-manifest.json", manifest)
    audit_id = "audit-primary-admission-instrumentation-001"
    jobs = [
        dict(
            id=audit_id,
            seconds=180,
            slice_seconds=180,
            cleanup_seconds=5,
            budget_stages=["learning"],
            stage="learning-preparation",
            priority=176,
            gpu_devices=[],
            resources=dict(cpus=2, gpus=0, memory_mb=8192, gres="none"),
            commands=[
                [
                    "{python}",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/repair_admission_instrumentation_test.py",
                    "tests/repair_primary_runtime_test.py",
                    "tests/repair_training_report_transport_test.py",
                    "--basetemp={work}/fixtures",
                    "--junitxml={work}/fixture-tests.xml",
                    "--tb=short",
                ],
                [
                    "{python}",
                    "-m",
                    "tools.repair.checkpoint_diagnostic",
                    str(output / "historical-state-reference.json"),
                    "{work}/checkpoint-diagnostic",
                ],
            ],
        )
    ]
    for lane in ("native", "inference"):
        jobs.append(
            dict(
                id="qualify-admission-component-" + lane + "-001",
                seconds=360,
                slice_seconds=360,
                cleanup_seconds=10,
                budget_stages=["learning"]
                + (["secondary_learning"] if lane == "inference" else []),
                stage="learning-engineering",
                priority=175,
                resources=resources(lane),
                gpu_devices=[DEV_GPU] if lane == "inference" else [],
                commands=[
                    [
                        "{python}",
                        "-m",
                        "tools.repair.admission_fixture",
                        str(output / "fixture-manifest.json"),
                        lane,
                        "{work}",
                    ]
                ],
            )
        )
    ledger = read(campaign / "ledger.json")
    balances = {}
    for job in jobs:
        job.update(deadline_epoch=campaign_value["primary_freeze_epoch"], deadline_policy="defer")
        # This checks a copy: preparation must not start a stage clock.
        if any(name not in ledger["stages"] for name in job["budget_stages"]):
            raise ValueError("Engineering preparation cannot start a new stage")
        available = _stage_remaining(copy.deepcopy(ledger), job, time.time())
        if job["seconds"] > 0.7 * available:
            raise ValueError("Engineering slice exceeds 70 percent of remaining stage capacity")
        balances[job["id"]] = dict(remaining_seconds=available, reserved_seconds=job["seconds"])
    immutable(
        output / "admission.json",
        dict(
            status="finite_component_slice_only",
            observed_epoch=time.time(),
            stage_balances=balances,
            prior_primary_gpu_seconds=ledger["stages"]["primary_training"]["cumulative"][
                "gpu_seconds"
            ],
            old_qualification_seconds=660,
            old_measured_cached_train_cases=99,
            old_unmeasured_cached_train_cases=27,
            old_missing_train_caches=2,
            prior_qualification_replayed=False,
            scientific_descriptors_enabled=False,
            teacher_incident_id="4d926e051616ed6ca634e9d1",
            calibration_clock_reset=False,
        ),
    )
    spec = dict(
        repository=str(repository),
        campaign=str(campaign),
        allocation="14451",
        python="/home/pgcotovio/Exact-OM/.venv/bin/python",
        ledger=str(campaign / "ledger.json"),
        capacity=registry["capacity"],
        source_store=str(campaign / "sources"),
        protocol_source=str(
            campaign / "learning/common-training-contract-002/protocols/hgt-pair-symbolic-s13.json"
        ),
        input_files=[
            str(output / "fixture-manifest.json"),
            str(output / "admission.json"),
            predecessor["path"],
            str(output / "historical-state-reference.json"),
            historical["phase_state"]["path"],
            report_ref["path"],
            review_ref["path"],
        ],
        jobs=jobs,
        purpose="Checkpoint instrumentation and finite concurrent authored TRAIN component qualification; no TEST or primary fitting admission",
    )
    immutable(output / "batch-spec.json", spec)
    batch = freeze(
        output / "batch-spec.json", campaign / "batches/primary-admission-instrumentation-001"
    )
    descriptors = []
    for job in jobs:
        lane = next(
            (k for k in nonces if job["id"] == "qualify-admission-component-" + k + "-001"), None
        )
        descriptors.append(
            prepare_dispatch(
                batch,
                job["id"],
                campaign / "attempts" / job["id"] / "001",
                tmux_socket=campaign / "supervisor/tmux.sock",
                depends_on=[audit_id] if lane else ["audit-primary-evaluation-capacity-001"],
                nonce=nonces[lane] if lane else None,
            )
        )
    result = dict(
        schema="exact-repair/admission-preparation/v1",
        status="prepared_not_queued",
        batch=binding(batch),
        fixture_manifest=binding(output / "fixture-manifest.json"),
        admission=binding(output / "admission.json"),
        descriptors=descriptors,
        scientific_admission=False,
    )
    immutable(output / "prepared.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    prepare(args.campaign, args.output, args.repository)


if __name__ == "__main__":
    main()
