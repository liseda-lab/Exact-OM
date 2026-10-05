"""Recover common-inventory manifest failure without replaying spent rows.

Run this frozen adapter against the original --code scientific export. Atomic
write debris is retained as metadata; published artifacts still require hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import sys
import time


def slice_bounds(plan):
    start, stop = plan["original_slice"]
    failed = plan["failed_index"]
    if (any(type(v) is not int for v in (start, stop, failed))
            or not 0 <= start <= failed < stop - 1
            or plan["only_unstarted_slice"] != [failed + 1, stop]
            or plan["original_row_count"] != stop - start
            or plan["prior_costs_reset"] is not False):
        raise ValueError("Retain the full slice and continue only untouched rows")
    return start, stop, failed


def validate_original(plan, evidence, identity):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair.scaling_payload_recovery import payload_manifest
    from tools.repair import common_inventory as science

    start, stop, failed = slice_bounds(plan)
    completion, owner, command = (bound(evidence[k]) for k in ("completion", "ownership", "command"))
    frozen = bound(plan["frozen_batch"])
    work = Path(completion["work"]) / "inventory"
    argv = [frozen["python"], "-m", "tools.repair.common_inventory",
        plan["schedule"]["path"], str(work), "--start", str(start), "--stop", str(stop)]
    jobs = [j for j in frozen["jobs"] if j["id"] == completion["job_id"]]
    error = "command 0 exited 1: PermissionError: [Errno 13] Permission denied: " + repr(evidence["temporary"]["path"])
    if (completion["status"] != "failed" or completion["exit_code"] != 1
            or completion["error"]["message"] != error
            or completion["step_id"] != plan["original_step"]
            or owner != dict(step_id=completion["step_id"], dispatch_nonce=completion["dispatch_nonce"])
            or completion["batch"] != plan["frozen_batch"]["path"]
            or Path(evidence["command"]["path"]) != Path(evidence["completion"]["path"]).parent / "command-0.json"
            or command != dict(argv=argv, cwd=plan["source_root"])
            or frozen["code"] != plan["source_root"] or len(jobs) != 1
            or [[arg.replace("{python}", frozen["python"]).replace("{work}", completion["work"])
                 for arg in cmd] for cmd in jobs[0]["commands"]] != [argv]):
        raise ValueError("Original execution, nonce, error, command or slice differs")
    schedule = bound(plan["schedule"])
    if stop > len(schedule["rows"]):
        raise ValueError("Original slice exceeds schedule")
    prior = {}
    for item in evidence["rows"]:
        index, ref = item["index"], item["receipt"]
        if index not in range(start, failed) or index in prior:
            raise ValueError("Retained prefix differs")
        row = schedule["rows"][index]
        path = work / "rows" / (row["id"] + ".json")
        saved = bound(ref)
        if (Path(ref["path"]) != path
                or saved != checked_checkpoint(path, canonical_hash((identity, row)))
                or saved["row"] != row or not saved["cleanup_complete"]
                or (work / "inflight" / path.name).exists()):
            raise ValueError("Original finished row identity or cleanup differs")
        science.fresh.validate_payloads(saved)
        science.fresh.raise_on_software_failure(saved)
        prior[index] = (ref, saved)
    if set(prior) != set(range(start, failed)):
        raise ValueError("Every spent prefix row must be retained")
    row = schedule["rows"][failed]
    guard = work / "inflight" / (row["id"] + ".json")
    result = bound(evidence["result"])
    if (str(guard) != evidence["guard"]["path"]
            or checked_checkpoint(guard, canonical_hash((identity, row))) != bound(evidence["guard"])
            or bound(evidence["guard"])["row"] != row
            or (work / "rows" / guard.name).exists()
            or Path(evidence["result"]["path"]) != work / "payloads" / row["id"] / "result.json"
            or result["row_id"] != row["id"] or result["schedule_hash"] != canonical_hash(schedule)
            or result["status"] != "verification_timeout" or result["logical_status"] != "UNKNOWN"
            or result["semantic_benefit"] is not None or result["semantic_status"] != "not_evaluated"
            or result["study_kind"] != "common_inventory_diagnostic"):
        raise ValueError("Only the original published UNKNOWN timeout can be retained")
    payloads, debris = payload_manifest(work / "payloads" / row["id"])
    if (payloads != evidence["payloads"] or debris != [evidence["temporary"]]
            or evidence["temporary"]["mode"] != 0 or evidence["temporary"]["size"] != 0):
        raise ValueError("Original published payloads or temporary metadata changed")
    for row in schedule["rows"][failed + 1:stop]:
        if any(p.exists() for p in (work / "rows" / (row["id"] + ".json"),
                work / "inflight" / (row["id"] + ".json"), work / "payloads" / row["id"])):
            raise ValueError("Continuation row already spent work")
    return prior


def one_row(science, schedule, row, output, identity):
    """Original evaluator and limits, with call telemetry saved before manifesting."""
    from exact.repair.records import canonical_hash
    from exact.repair.workers import bounded_call
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint
    from tools.repair.scaling_payload_recovery import payload_manifest

    path = output / "rows" / (row["id"] + ".json")
    guard = output / "inflight" / path.name
    payload = output / "payloads" / row["id"]
    row_identity = canonical_hash((identity, row))
    saved = checked_checkpoint(path, row_identity)
    if saved is None:
        if guard.exists() or payload.exists():
            raise RuntimeError("Interrupted row requires ownership and budget reconciliation")
        item = schedule["cases"][row["case_index"]]
        arm = next(a for a in schedule["arms"] if a["id"] == row["arm_id"])
        raw = dict(row=row, cleanup_complete=True, elapsed_seconds=0, resources={}, result=None)
        if item["status"] != "materialized" or arm["status"] != "available":
            saved = checkpoint(path, row_identity, **raw, payloads=[], status="unavailable",
                reason=item["status"] if item["status"] != "materialized" else arm["status"])
        else:
            checkpoint(guard, row_identity, row=row, started_epoch=time.time())
            started = time.monotonic()
            outcome = bounded_call(science.evaluate_row, schedule, row, str(payload),
                timeout=row["seconds"], cpu_seconds=row["cpu_seconds"], memory_mb=row["memory_mb"])
            raw.update(status=outcome.status, detail=outcome.detail,
                cleanup_complete=outcome.cleanup_complete, elapsed_seconds=time.monotonic()-started,
                resources=dict(outcome.resource_usage),
                result=outcome.value if outcome.status == "complete" else None)
            checkpoint(output / "calls" / path.name, row_identity, **raw)
            published, debris = payload_manifest(payload)
            saved = checkpoint(path, row_identity, **raw, payloads=published, unpublished_payloads=debris)
            science.fresh.validate_payloads(saved)
            if not outcome.cleanup_complete or "cleanup incomplete" in outcome.detail:
                raise RuntimeError("Worker cleanup incomplete; retain inflight ownership guard")
            guard.unlink()
    science.fresh.validate_payloads(saved)
    if not saved["cleanup_complete"] or guard.exists():
        raise RuntimeError("Saved row still requires cleanup reconciliation")
    science.fresh.raise_on_software_failure(saved)
    return saved


def run(plan_path, output, *, owner_check=None):
    from exact.repair.records import canonical_hash
    from exact.repair.study import runtime_manifest
    from tools.repair.batch import read, sha, checked_batch
    from tools.repair.expanded_corpus import binding, bound
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint
    from tools.repair.schema_cleanup_recovery import confirm_owner_gone
    from tools.repair import common_inventory as science

    plan, output = read(plan_path), Path(output).resolve()
    start, stop, failed = slice_bounds(plan)
    if (binding(__file__) != plan["runner"] or str(output) != plan["output"]
            or Path(science.__file__).resolve().parents[2] != Path(plan["source_root"]).resolve()
            or runtime_manifest() != plan["runtime"]):
        raise ValueError("Frozen adapter, scientific source, runtime or output changed")
    identity = canonical_hash((plan["schedule"]["sha256"], sha(science.__file__),
                              sha(science.fresh.__file__), plan["runtime"]))
    if identity != plan["scientific_identity"]:
        raise ValueError("Original scientific identity changed")
    frozen = checked_batch(plan["frozen_batch"]["path"])
    if frozen != bound(plan["frozen_batch"]) or frozen["code"] != plan["source_root"]:
        raise ValueError("Original frozen batch differs")
    schedule = bound(plan["schedule"])
    science.validate_schedule(schedule)
    evidence = bound(plan["evidence"])
    original_work = Path(bound(evidence["completion"])["work"]).resolve()
    if output.parent != original_work or output == original_work / "inventory":
        raise ValueError("Separate recovery output in the original work root required")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = validate_original(plan, evidence, identity)
        owner = (owner_check or confirm_owner_gone)(plan["original_step"])
        if (owner["step_id"] != plan["original_step"] or owner["surviving_processes"] != 0
                or not owner["cgroup_absent"] or owner["signals_sent"] != 0):
            raise ValueError("Original owner cleanup is unresolved")
        recovery_identity = canonical_hash((binding(plan_path), identity))
        refs = []
        for index in range(start, stop):
            row = schedule["rows"][index]
            if index < failed:
                ref, saved = prior[index]
            elif index == failed:
                path = output / "retained-unknown.json"
                expected = dict(row=row, status="recovered_result_after_manifest_failure",
                    detail="Original published UNKNOWN timeout retained without replay; outer call telemetry was not persisted.",
                    result=evidence["result"], cleanup_complete=True, resources={},
                    elapsed_seconds=bound(evidence["result"])["elapsed_seconds"],
                    payloads=evidence["payloads"], unpublished_payloads=[evidence["temporary"]],
                    original_guard=evidence["guard"], recovery_evidence=plan["evidence"],
                    ownership=owner, additional_scientific_seconds=0,
                    outer_telemetry_available=False, prior_costs_reset=False)
                saved = checked_checkpoint(path, recovery_identity)
                if saved is None:
                    saved = checkpoint(path, recovery_identity, **expected)
                else:
                    expected["ownership"] = saved["ownership"]
                    if {k:v for k,v in saved.items() if k not in ("identity", "content_hash")} != expected:
                        raise ValueError("Retained unknown provenance changed")
                ref = binding(path)
            else:
                saved = one_row(science, schedule, row, output / "continuation", recovery_identity)
                ref = binding(output / "continuation" / "rows" / (row["id"] + ".json"))
            science.fresh.validate_payloads(saved)
            science.fresh.raise_on_software_failure(saved)
            status = bound(saved["result"])["status"] if saved.get("result") else saved["status"]
            refs.append(dict(ref, row_id=row["id"], status=status))
        if [bound(ref)["row"] for ref in refs] != schedule["rows"][start:stop]:
            raise ValueError("Recovered row order or identities changed")
        report = checkpoint(output / "report.json", recovery_identity,
            schema=science.SCHEMA, status="complete", schedule=plan["schedule"],
            slice=[start, stop], scheduled=stop-start, recorded=len(refs), rows=refs,
            outcomes=dict(Counter(r["status"] for r in refs)), runtime=plan["runtime"],
            study_complete=False, gates_passed=False, model_fitting=False, generation=False,
            supervision_admitted=False, common_inventory_regret=None,
            regret_status="unavailable_no_complete_external_teacher",
            semantic_scope="Selected-only original query basis; intended-parent qualification remains outstanding",
            followup=schedule["followup"], api_spend_usd=0, prior_costs_reset=False,
            scientific_rows_replayed=0, manifest_recovery=dict(plan=binding(plan_path),
                evidence=plan["evidence"], retained_finished_rows=failed-start,
                retained_unknown_rows_without_replay=1, continued_rows=stop-failed-1,
                original_error_retained=True, outer_failed_row_telemetry_available=False))
        checkpoint(output / "science-health.json", recovery_identity,
            schema=science.fresh.SCHEMA, status="complete", rows=refs,
            scheduled=stop-start, recorded=len(refs), diagnostic_report=binding(output / "report.json"),
            scope="Common-inventory monitoring projection; no primary quality or semantic-regret claim",
            gates_passed=False, study_complete=False)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    run(args.plan, args.output)
