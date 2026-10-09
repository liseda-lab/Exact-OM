"""Prepare the first corrective queue and supervisor; never submit from this module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exact.experiments import dispatch
from exact.experiments.supervision import _registry, validate_admission
from exact.repair.api import write_artifact
from tools.repair import batch, supervisor_bootstrap
from tools.repair.corrective_conference import bound_bytes

CAMPAIGN = Path("/home/pgcotovio/Exact-OM/data/exact-repair-corrective-20261009")
REPOSITORY = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/pgcotovio/Exact-OM/.venv/bin/python")
PROFILES = {
    "gpu_qualification": dict(cpus=4, gpus=1, memory_mb=24576),
    "native_calibration": dict(cpus=3, gpus=0, memory_mb=20480),
    "hosted_calibration": dict(cpus=1, gpus=0, memory_mb=2048),
}


def plan(campaign=CAMPAIGN, repository=REPOSITORY, python=PYTHON):
    """Validate actual prepared inputs and compile a finite declarative first queue."""
    campaign, repository, python = (Path(v).resolve() for v in (campaign, repository, python))
    contract = batch.read(campaign / "campaign.json")
    authorization = batch.read(campaign / "authorization.json")
    ledger = batch.read(campaign / "ledger.json")
    if (
        authorization.get("launch_authorized") is not True
        or authorization.get("monetary_ceiling_usd") != 35
        or authorization.get("calibration_ceiling_usd") != 2
        or authorization.get("allocation") != "14451"
    ):
        raise ValueError("Frozen campaign authorization differs from this deployment")
    if not isinstance(ledger.get("attempts"), dict) or ledger["attempts"]:
        raise ValueError("First deployment requires an empty mapping of cumulative attempts")
    if ledger.get("limit_worker_seconds") is not None:
        raise ValueError("This campaign has stage limits, not a new global worker-hour limit")
    if ledger["stage_limits"]["calibration"]["elapsed_seconds"] != 14400:
        raise ValueError("Calibration must retain the four-hour cumulative stage")
    deadline = contract["primary_freeze_epoch"]
    annotation = campaign / "annotations/calibration/manifest.json"
    hosted = batch.read(annotation)
    if (
        hosted.get("phase") != "calibration"
        or hosted.get("authorized") is not True
        or hosted.get("cost_ceiling_usd") != 35
        or len(hosted.get("slots", [])) != 32
    ):
        raise ValueError("Expected the approved 32-slot/$35 campaign annotation calibration")
    conference_path = campaign / "inputs/conference/manifest.json"
    conference = batch.read(conference_path)
    if conference.get("scheduled") != 21 or conference.get("train_count") != 13:
        raise ValueError("Conference release must retain all 21 scheduled frozen pairs")
    training = [row for row in conference["rows"] if row["split"] == "train"]
    if len(training) != 13 or any("ekaw" in row["ontology_names"] for row in training):
        raise ValueError("Conference whole-ontology holdout crossed into TRAIN")
    inputs = {
        campaign / "campaign.json",
        campaign / "authorization.json",
        campaign / "planning-contract.json",
        campaign / "implementation-source.json",
        campaign / "inputs/calibration-preparation.json",
        annotation,
        conference_path,
        campaign / "protocols/hgt-pair-symbolic-s13.json",
    }
    for row in hosted["slots"]:
        bound_bytes(row["packet"])
        inputs.add(Path(row["packet"]["path"]))
    for row in training:
        if row["status"] != "ready_for_whole_source_qualification":
            continue
        for item in [row["matcher"], *row["ontology_bindings"]]:
            bound_bytes(item)
            inputs.add(Path(item["path"]))
    jobs = []

    def job(name, profile, commands, seconds, *, priority, stage=None, device=None):
        resources = dict(PROFILES[profile])
        resources["gres"] = contract["gpu_devices"][device]["gres"] if device else "none"
        value = dict(
            id=name,
            logical_id=name,
            resources=resources,
            resource_profile=profile,
            gpu_devices=[device] if device else [],
            commands=commands,
            seconds=seconds,
            cleanup_seconds=2,
            deadline_epoch=deadline,
            deadline_policy="defer",
            priority=priority,
            budget_stages=[stage] if stage else [],
            stage=stage or "engineering",
            preserve_completed_scientific_rows=True,
        )
        jobs.append(value)
        return value

    for device, entry in contract["gpu_devices"].items():
        name = entry["gres"].split(":")[1]
        job(
            "qualify-" + name,
            "gpu_qualification",
            [
                [
                    "{python}",
                    "-m",
                    "tools.repair.qualify_learning",
                    "--output",
                    "{work}",
                    "--prepared",
                    str(campaign / "inputs/calibration-preparation.json"),
                    "--device",
                    "cuda",
                    "--width",
                    "128",
                    "--layers",
                    "3",
                    "--heads",
                    "4",
                    "--seconds",
                    "900",
                ]
            ],
            900,
            priority=120,
            device=device,
        )
    for index in range(3):
        manifest = campaign / "calibration" / f"study-{index}.json"
        value = batch.read(manifest)
        if (
            value.get("heldout_outcomes_opened") is not False
            or value.get("cohort") != "exposed_development"
        ):
            raise ValueError("Initial study shards must use exposed development only")
        inputs.add(manifest)
        job(
            f"calibration-study-{index}",
            "native_calibration",
            [
                [
                    "{python}",
                    "-m",
                    "tools.repair.corrective_study",
                    "--manifest",
                    str(manifest),
                    "--output",
                    "{work}",
                ]
            ],
            7200,
            priority=100,
            stage="calibration",
        )
    hosted_job = job(
        "calibration-hosted",
        "hosted_calibration",
        [["{python}", "-m", "tools.repair.corrective_semantics", str(annotation), "{work}"]],
        3300,
        priority=115,
        stage="calibration",
    )
    hosted_job["deadline_epoch"] = min(deadline, hosted["deadline_epoch"])
    conference_commands = [
        [
            "{python}",
            "-m",
            "tools.repair.corrective_conference",
            "qualify",
            str(conference_path),
            row["id"],
            "{work}/" + "-".join(row["ontology_names"]),
            "--seconds",
            "300",
            "--check-seconds",
            "30",
        ]
        for row in training
        if row["status"] == "ready_for_whole_source_qualification"
    ]
    if conference_commands:
        job(
            "calibration-conference-train",
            "native_calibration",
            conference_commands,
            13 * 300,
            priority=110,
            stage="calibration",
        )
    if len(contract["gpu_devices"]) != 2 or {
        v["gres"] for v in contract["gpu_devices"].values()
    } != {"gpu:rtx5090:1", "gpu:rtx2080ti:1"}:
        raise ValueError("Frozen hardware declaration changed")
    for path in inputs:
        if not path.is_file():
            raise ValueError("Missing first-batch input: " + str(path))
    specification = dict(
        schema="exact-repair/frozen-batch/v1",
        repository=str(repository),
        python=str(python),
        campaign=str(campaign),
        allocation="14451",
        node="liseda-05",
        ledger=str(campaign / "ledger.json"),
        source_store=str(campaign / "sources"),
        protocol_source=str(campaign / "protocols/hgt-pair-symbolic-s13.json"),
        capacity=contract["capacity"],
        jobs=jobs,
        input_files=sorted(str(path) for path in inputs),
        primary_freeze_epoch=deadline,
        annotation_cost_ceiling_usd=35,
        heldout_outcomes_opened=False,
        existing_access_preserved=True,
    )
    return specification


def registry_for(specification, descriptors):
    campaign = Path(specification["campaign"])
    contract = batch.read(campaign / "campaign.json")
    registry = dict(
        schema_version=1,
        runs=[],
        pending_batches=list(descriptors),
        capacity=specification["capacity"],
        gpu_devices=contract["gpu_devices"],
        resource_profiles=PROFILES,
        remaining_work_status="pending",
        pause_paths=[str(campaign / "STOP"), str(campaign / "PAUSE")],
        campaign_context=dict(
            campaign=str(campaign),
            protocol="specs 15/16 corrective study",
            primary_freeze_epoch=contract["primary_freeze_epoch"],
            measurements_deadline_epoch=contract["measurements_deadline_epoch"],
            checked_snapshot="2026-10-17 Europe/London",
            abstract="2026-10-18",
            paper="2026-10-25",
            priority="six paired primary fits, independent semantic evaluation, Conference",
            monetary_ceiling_usd=35,
            calibration_ceiling_usd=2,
            production_matcher="deferred; publisher LogMap only",
            preserve_steps=["14451.0", "14451.extern"],
            reference_alignments="evaluator bindings only; not repair truth",
        ),
        plan=str(campaign / "planning-contract.json"),
        next_batch_actions="Read source-bound calibration receipts, freeze admitted common acquisition and later six-fit paired schedule; routine preparation is authorized.",
    )
    registry["pending_batches"].append(
        dict(
            id="prepare-common-acquisition",
            depends_on=[row["id"] for row in descriptors],
            resources=dict(cpus=0, gpus=0, memory_mb=0),
            gpu_devices=[],
            preparation_only=True,
            priority=200,
            stage="acquisition-preparation",
            deadline_epoch=contract["primary_freeze_epoch"],
            deadline_policy="defer",
            needs_user=False,
            instructions="Validate hardware/throughput/semantic calibration; preserve unsupported Conference denominators. Freeze common TRAIN acquisition and provider/scale/quota choices before any fit; prepare reviewed launch descriptors with preserved cumulative budgets. Continue the declared subsequent programme without routine preparation approval.",
        )
    )
    validate_registry(registry, specification["allocation"])
    return registry


def validate_registry(registry, allocation):
    _registry(registry["runs"])
    names = [row["id"] for row in registry["pending_batches"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate first-queue identities")
    allowed = set(names) | {row["id"] for row in registry["runs"]}
    for row in registry["pending_batches"]:
        validate_admission(registry, row)
        if any(
            parent not in allowed or parent == row["id"] for parent in row.get("depends_on", [])
        ):
            raise ValueError("Unknown or self-referential preparation prerequisite")
        if row.get("launch"):
            dispatch._validate(row, allocation)
            dispatch._validate_resource_binding(registry, row)
    return registry


def prepare(
    campaign=CAMPAIGN,
    repository=REPOSITORY,
    python=PYTHON,
    *,
    codex=Path("/home/pgcotovio/.local/bin/codex"),
    codex_config=Path("/home/pgcotovio/.codex/config.toml"),
):
    """Freeze only after commit, prepare every descriptor, and leave start explicit."""
    campaign, repository = Path(campaign).resolve(), Path(repository).resolve()
    specification = plan(campaign, repository, python)
    supervisor = campaign / "supervisor"
    if (supervisor / "registry.json").exists() or (supervisor / "policy.json").exists():
        raise ValueError("Existing deployment must be reconciled, never overwritten")
    if (campaign / "batches/first").exists():
        raise ValueError("First frozen batch already exists; reconcile partial preparation")
    write_artifact(campaign / "first-batch-specification.json", specification)
    frozen = batch.freeze(campaign / "first-batch-specification.json", campaign / "batches/first")
    content = batch.read(frozen)
    code = Path(content["code"])
    supervisor_bootstrap.prepare(
        supervisor,
        allocation="14451",
        node="liseda-05",
        code=code,
        repository=repository,
        python=python,
        codex=codex,
        codex_config=codex_config,
        instructions=code / "specs/exact-repair/runs/corrective-20261009/SUPERVISOR.md",
    )
    descriptors = []
    for job in specification["jobs"]:
        descriptor = batch.prepare_dispatch(
            frozen,
            job["id"],
            campaign / "attempts" / job["id"] / "001",
            tmux_socket=supervisor / "tmux.sock",
        )
        descriptor["launch"]["run"].update(
            max_repairs=2, followup_stages=["prepare-common-acquisition"]
        )
        if job["id"].startswith("calibration-study-"):
            descriptor["launch"]["run"]["science_report_relative"] = "report.json"
        write_artifact(
            Path(descriptor["budget_reservation"]["attempt"]) / "descriptor.json", descriptor
        )
        descriptors.append(descriptor)
    registry = registry_for(specification, descriptors)
    write_artifact(supervisor / "registry.json", registry)
    record = dict(
        status="prepared_not_started",
        commit=content["commit"],
        batch=str(frozen),
        supervisor=str(supervisor),
        prepared_jobs=[row["id"] for row in descriptors],
        budget_reserved=False,
        paid_calls_made=False,
        detached_processes_started=False,
        start_command=[
            str(python),
            "-m",
            "tools.repair.supervisor_bootstrap",
            "start",
            str(supervisor),
        ],
    )
    write_artifact(campaign / "deployment-preparation.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare"))
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--python", type=Path, default=PYTHON)
    args = parser.parse_args()
    if args.action == "prepare":
        value = prepare(args.campaign, args.repository, args.python)
    else:
        value = plan(args.campaign, args.repository, args.python)
        write_artifact(args.campaign / "first-batch-plan.json", value)
        value = {
            "status": "validated_not_frozen",
            "jobs": [
                {k: row[k] for k in ("id", "resources", "seconds", "budget_stages", "gpu_devices")}
                for row in value["jobs"]
            ],
        }
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
