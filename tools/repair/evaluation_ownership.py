"""Declaration-only lane planning and authenticated, terminal pool handoff.

CPU preparation and native controls never acquire a GPU. Every matched neural or
E3 row owns the same device for its entire case; there is no live device transfer.
"""

from __future__ import annotations

import os
from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.batch import read, sha
from tools.repair.primary_runtime import DEV_GPU, owned_device
from tools.repair.shared_release import authenticate, bound

PROFILE = "primary-evaluation-native3-inference3-20GB-each/v1"
LANES = ("pool", "native", "inference")


def lane_for(row):
    if row["adapter"]["operation"] == "pool":
        return "pool"
    # The two uniform E3 arms retain the learned arm's device/load profile.
    if row["adapter"]["model"] or row["experiment"] == "E3":
        return "inference"
    return "native"


def resources(lane):
    if lane not in LANES:
        raise ValueError("Unknown evaluation ownership lane")
    return dict(
        cpus=3,
        memory_mb=20000,
        gpus=int(lane == "inference"),
        gres="gpu:rtx2080ti:1" if lane == "inference" else "none",
    )


def partition(schedule, rows):
    """Whole cases per shard; pools first, then two serial lanes per cohort.

    Cohorts are barriers, Conference first. Within each lane, shards form a
    strict chain. No estimate assumes a second inference device. Every matched
    GPU comparison stays in one shard with one immutable resource profile.
    """
    shards = []
    previous_cohort = []
    for cohort in ("conference", "generated"):
        cases = [c for c in schedule["cases"] if c["cohort"] == cohort]
        groups = [cases[i : i + 4] for i in range(0, len(cases), 4)]
        pools = []
        for lane in LANES:
            previous = None
            for index, chunk in enumerate(groups):
                ids = {c["case_id"] for c in chunk}
                selected = [r for r in rows if r["case_id"] in ids and lane_for(r) == lane]
                if not selected:
                    continue
                key = f"{cohort}-{lane}-{index:02d}"
                dependencies = (
                    [previous] if previous else (previous_cohort if lane == "pool" else pools[-1:])
                )
                shards.append(
                    dict(
                        id=key,
                        cohort=cohort,
                        lane=lane,
                        rows=selected,
                        cases={c["case_id"]: c for c in chunk},
                        depends_on=dependencies,
                        resources=resources(lane),
                        gpu_devices=[DEV_GPU] if lane == "inference" else [],
                        seconds=sum(r["case_seconds"] for r in selected) + 30,
                        concurrent_load_profile=PROFILE,
                    )
                )
                previous = key
                if lane == "pool":
                    pools.append(key)
            if lane == "pool":
                ends = []
            elif previous:
                ends.append(previous)
        previous_cohort = ends
    by_row = {r["id"]: s for s in shards for r in s["rows"]}
    if len(by_row) != len(rows):
        raise ValueError("Duplicate or missing scheduled row")
    for comparison in schedule["comparisons"]:
        a, b = (by_row[comparison[k]] for k in ("left", "right"))
        if a["lane"] == b["lane"] == "inference" and a["id"] != b["id"]:
            raise ValueError("Matched inference comparison crossed ownership boundaries")
    return shards


def capacity(shards, schedule):
    """Timeout-inclusive upper reservation, including all process boundaries."""
    cohorts = []
    for cohort in ("conference", "generated"):
        seconds = {
            lane: sum(s["seconds"] for s in shards if s["cohort"] == cohort and s["lane"] == lane)
            for lane in LANES
        }
        cohorts.append(
            dict(
                cohort=cohort,
                lane_seconds=seconds,
                critical_path_seconds=seconds["pool"]
                + max(seconds["native"], seconds["inference"]),
                worker_seconds=sum(seconds.values()),
            )
        )
    wall = sum(c["critical_path_seconds"] for c in cohorts)
    # Bound all launch/registration gaps by one full supervisor check per
    # descriptor, even though the detached dispatcher normally polls every15s.
    handoff = 300 * len(shards)
    worker = sum(c["worker_seconds"] for c in cohorts)
    gpu = sum(c["lane_seconds"]["inference"] for c in cohorts)
    window = schedule["repair_launch_window_seconds"]
    # The original reserve is a floor on the worker envelope, not an invitation
    # to spend the rest on additional rows.
    conference_floor = schedule["conference_worker_fraction"] * 2 * window
    conference_reserved = max(conference_floor, cohorts[0]["worker_seconds"])
    generated = cohorts[1]["worker_seconds"]
    passed = (
        wall + handoff <= 0.7 * window
        and gpu <= 0.7 * window
        and worker <= 0.7 * 2 * window
        and cohorts[0]["worker_seconds"] <= 0.7 * conference_reserved
        and generated <= 0.7 * (2 * window - conference_reserved)
    )
    return dict(
        schema="exact-repair/evaluation-lane-capacity/v1",
        execution_authorized=False,
        projection_passed=passed,
        capacity_qualified=False,
        cohorts=cohorts,
        critical_path_seconds=wall,
        dispatch_handoff_allowance_seconds=handoff,
        elapsed_projection_with_handoffs_seconds=wall + handoff,
        worker_seconds=worker,
        gpu_seconds=gpu,
        repair_window_seconds=window,
        wall_capacity_at_70_percent_seconds=0.7 * window,
        worker_capacity_at_70_percent_seconds=0.7 * 2 * window,
        conference_reserved_worker_seconds=conference_reserved,
        maximum_concurrent_resources=dict(cpus=6, memory_mb=40000, gpus=1),
        concurrent_load_profile=PROFILE,
        gpu_uuid=DEV_GPU,
        last_eight_hours_reserved=True,
        qualification_required="CPU generation/native and the exact two-worker inference load profile; arithmetic is not a measured qualification",
        test_payloads_opened=False,
        hosted_calls=0,
    )


def validate_lane(manifest, *, check_visibility=True):
    lane = manifest["ownership_lane"]
    if lane not in LANES or any(lane_for(r) != lane for r in manifest["rows"]):
        raise ValueError("Evaluation row does not belong to its ownership lane")
    expected = resources(lane)
    if (
        manifest["cpus"] != expected["cpus"]
        or manifest["memory_mb"] != expected["memory_mb"]
        or manifest["device"] != ("cuda" if lane == "inference" else "cpu")
        or manifest.get("gpu_uuid") != (DEV_GPU if lane == "inference" else None)
        or manifest["concurrent_load_profile"] != PROFILE
    ):
        raise ValueError("Evaluation resource/device/load profile changed")
    if check_visibility:
        if lane == "inference":
            owned_device(DEV_GPU)
        else:
            import torch

            if torch.cuda.device_count() != 0:
                raise ValueError("CPU evaluation requires zero Slurm-visible GPUs")
    return expected


def validate_owner(manifest, stage):
    step_id = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    owners = [
        (key, entry)
        for key, entry in stage.get("attempts", {}).items()
        if entry["status"] != "settled"
        and (Path(key) / "step.json").exists()
        and read(Path(key) / "step.json").get("step_id") == step_id
    ]
    if len(owners) != 1:
        raise ValueError("Evaluation requires one charged stage reservation owner")
    key, owner = owners[0]
    if "ownership_lane" in manifest:
        expected = validate_lane(manifest, check_visibility=False)
        gpu = manifest["ownership_lane"] == "inference"
        required = {"evaluation", "secondary_evaluation"} if gpu else {"evaluation"}
        if (
            not required <= set(owner["budget_stages"])
            or owner["gpu_devices"] != ([DEV_GPU] if gpu else [])
            or any(owner["resources"][k] != expected[k] for k in expected)
            or read(Path(key) / "step.json").get("dispatch_nonce") != manifest["dispatch_nonce"]
        ):
            raise ValueError("Evaluation charged owner/device/nonce differs")
    elif not {"evaluation", "secondary_evaluation"} <= set(owner["budget_stages"]):
        raise ValueError("Evaluation requires one charged stage/GPU reservation owner")


def external_pools(manifest):
    """Read only completed immutable producers; never wait or regenerate a pool.

    A producer's future output digests cannot be frozen before it runs. Its
    manifest, dispatch nonce, job and output location can. The terminal receipt
    supplies digests, which are checked against every consumed row and artifact.
    """
    local = {r["id"] for r in manifest["rows"]}
    needed = {
        r["adapter"]["pool_dependency"]
        for r in manifest["rows"]
        if r["adapter"]["pool_dependency"] and r["adapter"]["pool_dependency"] not in local
    }
    contracts = manifest.get("pool_producers", {})
    if set(contracts) != needed:
        raise ValueError("External pool dependency contract is incomplete or extra")
    result = {}
    for row_id, contract in contracts.items():
        producer = bound(contract["manifest"])
        if producer["ownership_lane"] != "pool":
            raise ValueError("Shared pool producer is not a pool owner")
        candidates = [r for r in producer["rows"] if r["id"] == row_id]
        if len(candidates) != 1 or candidates[0]["adapter"]["operation"] != "pool":
            raise ValueError("Shared pool producer row differs")
        row = candidates[0]
        for consumer in manifest["rows"]:
            if consumer["adapter"]["pool_dependency"] == row_id and any(
                row[k] != consumer[k] for k in ("case_id", "input_identity", "protocol")
            ):
                raise ValueError("Shared pool dependency input/protocol differs")
        if any(producer.get(k) != manifest.get(k) for k in ("schedule", "source_commit")):
            raise ValueError("Shared pool source/schedule differs")
        case_id = row["case_id"]
        if producer["public_inputs"].get(case_id) != manifest["public_inputs"].get(
            case_id
        ) or producer["cases"].get(case_id) != manifest["cases"].get(case_id):
            raise ValueError("Shared pool bound public input differs")
        attempt = Path(contract["attempt"])
        terminal, step = read(attempt / "completion.json"), read(attempt / "step.json")
        if (
            terminal.get("status") != "complete"
            or terminal.get("exit_code") != 0
            or terminal.get("dispatch_nonce") != contract["dispatch_nonce"]
            or step.get("dispatch_nonce") != contract["dispatch_nonce"]
            or terminal.get("step_id") != step.get("step_id")
            or terminal.get("job_id") != contract["job_id"]
            or terminal.get("work") != contract["work"]
            or producer.get("dispatch_nonce") != contract["dispatch_nonce"]
        ):
            raise ValueError("Shared pool terminal owner/nonce differs")
        batch_path = Path(terminal["batch"])
        if sha(batch_path) != batch_path.with_name("batch.sha256").read_text().strip():
            raise ValueError("Shared pool producer batch changed")
        batch = read(batch_path)
        job = next(j for j in batch["jobs"] if j["id"] == contract["job_id"])
        if batch["frozen_files"].get(contract["manifest"]["path"]) != contract["manifest"][
            "sha256"
        ] or not any(contract["manifest"]["path"] in cmd for cmd in job["commands"]):
            raise ValueError("Shared pool manifest is not a frozen producer command")
        work, outputs = Path(contract["work"]), read(attempt / "outputs.json")
        report_path = work / "report.json"
        if "report.json" not in outputs:
            raise ValueError("Shared pool producer has no bound denominator report")
        authenticate(dict(path=str(report_path), sha256=outputs["report.json"]))
        report = read(report_path)
        if (
            report.get("manifest_identity") != canonical_hash(producer)
            or report.get("expected_rows") != len(producer["rows"])
            or [r["id"] for r in report["rows"]] != [r["id"] for r in producer["rows"]]
        ):
            raise ValueError("Shared pool report changed identity or denominator")
        reported = next(r for r in report["rows"] if r["id"] == row_id)
        relative = "rows/" + row_id + "/completion.json"
        path = work / relative
        if reported["status"] == "not_attempted":
            if (
                relative in outputs
                or "rows/" + row_id + "/started.json" in outputs
                or reported.get("result") is not None
            ):
                raise ValueError("Unattempted shared pool has conflicting attempt evidence")
            # A deadline tail is a missing scientific result, not a crashed
            # producer and never permission to regenerate it in a consumer.
            result[row_id] = dict(status="not_attempted", result=None)
            continue
        if relative not in outputs:
            raise ValueError("Shared pool row is not a terminal output")
        authenticate(dict(path=str(path), sha256=outputs[relative]))
        saved = read(path)
        if saved != {k: v for k, v in reported.items() if k != "id"}:
            raise ValueError("Shared pool row and terminal report differ")
        from tools.repair.primary_evaluation import validate_saved

        validate_saved(saved, canonical_hash((canonical_hash(producer), row)))
        for artifact in saved.get("artifacts", []):
            child = Path(artifact["path"]).resolve()
            if (
                not child.is_relative_to(work.resolve())
                or outputs.get(str(child.relative_to(work.resolve()))) != artifact["sha256"]
            ):
                raise ValueError("Shared pool artifact is not a terminal output")
        pool = (saved.get("result") or {}).get("pool")
        if pool and (saved["status"] != "complete" or pool not in saved.get("artifacts", [])):
            raise ValueError("Shared pool payload lacks a complete bound artifact")
        if saved["status"] in {"error", "interrupted_unknown"}:
            raise ValueError("Shared pool implementation failure needs reconciliation")
        result[row_id] = saved
    return result
