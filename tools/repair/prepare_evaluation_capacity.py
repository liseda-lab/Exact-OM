"""Save disabled CPU/native and exclusive-GPU evaluation dispatch contracts."""

from __future__ import annotations

import argparse
import hashlib
import uuid
from pathlib import Path

from tools.repair.batch import read, sha
from tools.repair.corrective_campaign import source_identity
from tools.repair.evaluation_ownership import PROFILE, capacity, partition
from tools.repair.historical_regression import binding
from tools.repair.primary_evaluation import SCHEMA, compile_rows
from tools.repair.primary_runtime import DEV_GPU
from tools.repair.shared_release import bound, immutable, validate_completion


def prepare(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    registry = read(campaign / "supervisor/registry.json")
    prior_ref = registry["primary_evaluation_preparation"]["preparation"]
    prior = bound(prior_ref)
    completed_ref = registry["primary_evaluation_preparation"]["completion"]
    completed = bound(completed_ref)
    receipt = completed["audit_receipt"]
    terminal, outputs, _, _ = validate_completion(receipt)
    for relative, digest in outputs.items():
        if sha(Path(terminal["work"]) / relative) != digest:
            raise ValueError("Predecessor audit output changed")
    schedule = bound(prior["schedule"])
    rows = compile_rows(schedule)
    shards = partition(schedule, rows)
    projection = capacity(shards, schedule)
    if not projection["projection_passed"]:
        raise ValueError("Lane reservation does not fit the unchanged evaluation envelope")
    source = source_identity()
    if source["dirty_hash"] != hashlib.sha256(b"").hexdigest():
        raise ValueError("Commit evaluation capacity source before preparation")
    immutable(output / "source.json", source)
    immutable(output / "execution-capacity.json", projection)
    ids = {s["id"]: "evaluation-capacity-" + s["id"] + "-001" for s in shards}
    nonces = {s["id"]: uuid.uuid4().hex for s in shards}
    producers = {}
    manifests, dispatch = [], []
    for shard in shards:
        key, job = shard["id"], ids[shard["id"]]
        lane = shard["lane"]
        needed = {
            r["adapter"]["pool_dependency"]
            for r in shard["rows"]
            if r["adapter"]["pool_dependency"]
        }
        manifest = dict(
            schema=SCHEMA,
            scope="primary_test",
            execution_authorized=False,
            status="disabled_pending_qualified_capacity_models_and_inputs",
            source_commit=source["revision"],
            schedule=prior["schedule"],
            rows=shard["rows"],
            cases=shard["cases"],
            public_inputs={},
            models={},
            model_freeze_receipt=None,
            capacity_admission=None,
            ownership_lane=lane,
            concurrent_load_profile=PROFILE,
            dispatch_nonce=nonces[key],
            pool_producers={k: producers[k] for k in sorted(needed)},
            gpu_uuid=DEV_GPU if lane == "inference" else None,
            device="cuda" if lane == "inference" else "cpu",
            cpus=3,
            memory_mb=20000,
            ledger_path=str(campaign / "ledger.json"),
            not_before_epoch=schedule["models_freeze_epoch"],
            deadline_epoch=schedule["measurements_deadline_epoch"],
            final_annotation_aggregation_reserve_seconds=28800,
            no_evaluator_access=True,
            hosted_calls=0,
            pending_gates=[
                "measured CPU/native and matched two-worker inference profile",
                "six frozen selected models and matched protocols",
                "post-freeze bound public inputs and observables",
                "remaining evaluation/stage capacity",
                "recreate enabled successor descriptors; never edit this snapshot",
            ],
        )
        path = output / "execution" / (key + ".json")
        immutable(path, manifest)
        reference = binding(path)
        attempt = campaign / "attempts" / job / "001"
        if lane == "pool":
            for row in shard["rows"]:
                producers[row["id"]] = dict(
                    manifest=reference,
                    attempt=str(attempt),
                    job_id=job,
                    dispatch_nonce=nonces[key],
                    work=str(campaign / "work" / job),
                )
        manifests.append(reference)
        dispatch.append(
            dict(
                id=job,
                manifest=reference,
                attempt=str(attempt),
                nonce=nonces[key],
                depends_on=[ids[d] for d in shard["depends_on"]],
                resources=shard["resources"],
                gpu_devices=shard["gpu_devices"],
                seconds=shard["seconds"],
                ownership_lane=lane,
                budget_stages=(
                    ["evaluation", "secondary_evaluation"]
                    if lane == "inference"
                    else ["evaluation"]
                ),
            )
        )
    value = dict(
        schema="exact-repair/primary-evaluation-capacity-preparation/v1",
        status="prepared_not_queued",
        execution_authorized=False,
        predecessor=prior_ref,
        predecessor_completion=completed_ref,
        audit_receipt=receipt,
        source=binding(output / "source.json"),
        source_commit=source["revision"],
        schedule=prior["schedule"],
        execution_capacity=binding(output / "execution-capacity.json"),
        execution_manifests=manifests,
        dispatch=dispatch,
        expected_rows=len(rows),
        comparisons=len(schedule["comparisons"]),
        semantic_slots=128,
        independent_swaps=26,
        annotation_attempts=154,
        conference_all_pairs=21,
        conference_test_pairs=7,
        conference_semantic_slots=43,
        endpoint_projection_unchanged=prior["endpoint_projection"],
        teacher_incident_id=prior["teacher_incident_id"],
        test_payloads_opened=False,
        hosted_calls=0,
        primary_optimizer_updates=0,
        steps=[],
    )
    immutable(output / "prepared.json", value)
    return value


def audit(prepared, output):
    value = read(prepared)
    prior = bound(value["predecessor"])
    validate_completion(value["audit_receipt"])
    schedule = bound(value["schedule"])
    rows = compile_rows(schedule)
    shards = partition(schedule, rows)
    projection = capacity(shards, schedule)
    if projection != bound(value["execution_capacity"]):
        raise ValueError("Capacity projection differs from frozen declarations")
    manifests = [bound(ref) for ref in value["execution_manifests"]]
    if len(manifests) != len(shards) or len(value["dispatch"]) != len(shards):
        raise ValueError("Missing ownership shards")
    by_id = {d["id"]: d for d in value["dispatch"]}
    pools = {
        r["id"]: (ref, m, d)
        for ref, m, d in zip(value["execution_manifests"], manifests, value["dispatch"])
        for r in m["rows"]
        if r["adapter"]["operation"] == "pool"
    }
    for shard, manifest, dispatch in zip(shards, manifests, value["dispatch"]):
        if (
            manifest["execution_authorized"]
            or manifest["public_inputs"]
            or manifest["models"]
            or manifest["rows"] != shard["rows"]
            or manifest["cases"] != shard["cases"]
            or manifest["ownership_lane"] != shard["lane"]
            or dispatch["resources"] != shard["resources"]
            or dispatch["seconds"] != shard["seconds"]
            or dispatch["nonce"] != manifest["dispatch_nonce"]
            or dispatch["depends_on"]
            != ["evaluation-capacity-" + s + "-001" for s in shard["depends_on"]]
        ):
            raise ValueError("Ownership shard changed")
        from tools.repair.evaluation_ownership import validate_lane

        validate_lane(manifest, check_visibility=False)
        needed = {
            r["adapter"]["pool_dependency"]
            for r in manifest["rows"]
            if r["adapter"]["pool_dependency"]
        }
        if set(manifest["pool_producers"]) != needed:
            raise ValueError("Pool dependencies missing")
        for key, contract in manifest["pool_producers"].items():
            ref, producer, job = pools[key]
            if (
                contract["manifest"] != ref
                or contract["dispatch_nonce"] != producer["dispatch_nonce"]
                or contract["job_id"] != job["id"]
                or contract["attempt"] != job["attempt"]
                or contract["work"] != str(Path(job["attempt"]).parents[2] / "work" / job["id"])
                or contract["job_id"] not in by_id
            ):
                raise ValueError("Pool producer lineage differs")
    if prior["endpoint_projection"] != value["endpoint_projection_unchanged"]:
        raise ValueError("Capacity preparation must not alter fitting endpoints")
    result = dict(
        schema="exact-repair/primary-evaluation-capacity-audit/v1",
        status="complete",
        preparation=binding(Path(prepared)),
        expected_rows=len(rows),
        comparisons=len(schedule["comparisons"]),
        ownership_shards=len(shards),
        lane_rows={
            lane: sum(len(m["rows"]) for m in manifests if m["ownership_lane"] == lane)
            for lane in ("pool", "native", "inference")
        },
        capacity_projection=projection,
        test_execution_admitted=False,
        test_payloads_opened=False,
        hosted_calls=0,
        primary_optimizer_updates=0,
    )
    immutable(Path(output) / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else audit)(args.input, args.output)


if __name__ == "__main__":
    main()
