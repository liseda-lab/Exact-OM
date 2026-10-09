"""Resume frozen scaling work without replaying rows lost during payload hashing.

Run this committed adapter with --code pointing at the original scientific export.
Only the orchestration/manifest changes; generation, selection and budgets do not.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import re
import sys
import time


def payload_manifest(directory):
    """Bind published files; inventory atomic-write debris without opening it."""
    from tools.repair.expanded_corpus import binding

    published, unpublished = [], []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file() or path.suffix == ".lock":
            continue
        # write_artifact uses tempfile.mkstemp('.<destination>.', ...).
        # Do not suppress permission/hash errors on any published artifact.
        if re.fullmatch(r"\..+\.json\.[a-z0-9_]{8}", path.name):
            stat = path.stat()
            unpublished.append(
                dict(
                    path=str(path),
                    size=stat.st_size,
                    mode=stat.st_mode & 0o777,
                    uid=stat.st_uid,
                    reason="unpublished_atomic_write_temporary",
                )
            )
        else:
            published.append(binding(path))
    return published, unpublished


def validate_prefix(plan, science, identity):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checked_checkpoint

    schedule = bound(plan["schedule"])
    start, stop = plan["slice"]
    recovery = plan.get("recovery")
    if not recovery:
        return {}
    evidence = bound(recovery)
    completion = bound(evidence["completion"])
    owner = bound(evidence["owner"])
    command = bound(evidence["command"])
    original = Path(completion["work"]) / "scaling-revision-002"
    failed = evidence["failed_index"]
    if (
        completion["status"] != "failed"
        or completion["exit_code"] != 1
        or completion["batch"] != plan["scientific_batch"]["path"]
        or completion["step_id"] != evidence["step_id"]
        or owner != dict(step_id=completion["step_id"], dispatch_nonce=completion["dispatch_nonce"])
        or not completion["error"]["message"].startswith("command 0 exited 1: PermissionError:")
        or evidence["temporary"]["path"] not in completion["error"]["message"]
        or command["cwd"] != plan["source_root"]
        or command["argv"][1:]
        != [
            "-m",
            "tools.repair.scaling",
            plan["schedule"]["path"],
            str(original),
            "--start",
            str(start),
            "--stop",
            str(stop),
        ]
        or not start <= failed < stop
        or failed % 2
    ):
        raise ValueError("Original scaling execution/error/slice differs")
    prefix = {}
    for item in evidence["rows"]:
        i, ref = item["index"], item["receipt"]
        if i in prefix or i not in range(start, failed):
            raise ValueError("Retained prefix differs")
        row = schedule["rows"][i]
        path = original / "rows" / (row["id"] + ".json")
        saved = checked_checkpoint(path, canonical_hash((identity, row)))
        if (
            str(path) != ref["path"]
            or saved != bound(ref)
            or saved["row"] != row
            or not saved["cleanup_complete"]
            or (original / "inflight" / path.name).exists()
        ):
            raise ValueError("Original finished row differs")
        science.validate_payloads(saved)
        prefix[i] = ref
    if set(prefix) != set(range(start, failed)):
        raise ValueError("Missing retained prefix")
    row = schedule["rows"][failed]
    guard = original / "inflight" / (row["id"] + ".json")
    result = bound(evidence["result"])
    if (
        str(guard) != evidence["guard"]["path"]
        or checked_checkpoint(guard, canonical_hash((identity, row))) != bound(evidence["guard"])
        or result["row"] != row
        or row["cache_mode"] != "cold"
        or result["logical_status"] != "UNKNOWN"
        or result["semantic_benefit"] is not None
        or result["generation_status"] != "timeout"
        or result["selection_status"] != "timeout"
        or (original / "rows" / guard.name).exists()
        or Path(evidence["result"]["path"]) != original / "payloads" / row["id"] / "result.json"
    ):
        raise ValueError("Only the retained cold UNKNOWN timeout result can be recovered")
    payloads, debris = payload_manifest(original / "payloads" / row["id"])
    if payloads != evidence["payloads"] or debris != [evidence["temporary"]]:
        raise ValueError("Original payload inventory changed")
    if result["pool"]:
        bound(result["pool"])
    for i in range(failed + 1, stop):
        key = schedule["rows"][i]["id"]
        if any(
            p.exists()
            for p in (
                original / "rows" / (key + ".json"),
                original / "inflight" / (key + ".json"),
                original / "payloads" / key,
            )
        ):
            raise ValueError("Continuation row already spent work")
    return prefix


def run(plan_path, output, *, owner_check=None):
    from exact.repair.records import canonical_hash
    from exact.repair.study import runtime_manifest
    from exact.repair.workers import bounded_call
    from tools.repair.batch import read, sha, checked_batch
    from tools.repair.expanded_corpus import binding, bound
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint
    from tools.repair import scaling as science

    plan, output = read(plan_path), Path(output).resolve()
    schedule = bound(plan["schedule"])
    start, stop = plan["slice"]
    if (
        not 0 <= start < stop <= len(schedule["rows"])
        or start % 2
        or stop % 2
        or str(output) != plan["output"]
        or binding(__file__) != plan["adapter"]
        or str(Path(science.__file__).resolve().parents[2]) != plan["source_root"]
        or runtime_manifest() != plan["runtime"]
    ):
        raise ValueError("Frozen adapter/science/runtime/output or slice changed")
    batch = checked_batch(plan["scientific_batch"]["path"])
    if bound(plan["scientific_batch"]) != batch or batch["code"] != plan["source_root"]:
        raise ValueError("Original scientific batch changed")
    old_identity = canonical_hash((plan["schedule"], sha(science.__file__), plan["runtime"]))
    if old_identity != plan["scientific_identity"]:
        raise ValueError("Scientific dependency identity changed")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prefix = validate_prefix(plan, science, old_identity)
        evidence = bound(plan["recovery"]) if plan.get("recovery") else None
        ownership = owner_check(evidence["step_id"]) if evidence else None
        if evidence and (
            ownership["step_id"] != evidence["step_id"]
            or ownership["surviving_processes"] != 0
            or not ownership["cgroup_absent"]
            or ownership["signals_sent"] != 0
            or output.parent != Path(bound(evidence["completion"])["work"]).resolve()
        ):
            raise ValueError("Original owner cleanup or recovery work root differs")
        identity = canonical_hash((binding(plan_path), old_identity))
        refs, receipts = [], {}
        for i in range(start, stop):
            row = schedule["rows"][i]
            key = row["id"]
            if i in prefix:
                ref = prefix[i]
                saved = bound(ref)
                cache = Path(ref["path"]).parents[1] / "payloads" / key / "compiler-cache"
            else:
                path, guard = output / "rows" / (key + ".json"), output / "inflight" / (
                    key + ".json"
                )
                row_identity = canonical_hash((identity, row))
                saved = checked_checkpoint(path, row_identity)
                cache = output / "payloads" / key / "compiler-cache"
                if evidence and i == evidence["failed_index"]:
                    cache = Path(evidence["result"]["path"]).parent / "compiler-cache"
                    if saved is None:
                        saved = checkpoint(
                            path,
                            row_identity,
                            row=row,
                            status="recovered_result_after_manifest_failure",
                            detail="Original published UNKNOWN result retained without replay; outer call telemetry was not persisted.",
                            result=evidence["result"],
                            cleanup_complete=True,
                            resources={},
                            elapsed_seconds=bound(evidence["result"])["elapsed_seconds"],
                            payloads=evidence["payloads"],
                            unpublished_payloads=[evidence["temporary"]],
                            original_guard=evidence["guard"],
                            recovery_evidence=plan["recovery"],
                            ownership=ownership,
                            additional_scientific_seconds=0,
                            outer_telemetry_available=False,
                            prior_costs_reset=False,
                        )
                elif saved is None:
                    if guard.exists() or (output / "payloads" / key).exists():
                        raise RuntimeError(
                            "Interrupted row requires owner/ledger/cleanup reconciliation"
                        )
                    cache_source = None
                    if row["cache_mode"] == "warm":
                        cold = schedule["rows"][i - 1]
                        prior, source = receipts[i - 1]
                        if cold["pair_id"] != row["pair_id"] or not prior["cleanup_complete"]:
                            raise ValueError("Warm row requires charged cold predecessor")
                        science.validate_payloads(prior)
                        cache_source = str(source)
                    checkpoint(guard, row_identity, row=row, started_epoch=time.time())
                    started = time.monotonic()
                    outcome = bounded_call(
                        science.evaluate,
                        schedule,
                        row,
                        str(output / "payloads" / key),
                        cache_source,
                        timeout=row["seconds"],
                        cpu_seconds=row["cpu_seconds"],
                        memory_mb=row["memory_mb"],
                    )
                    # Save call outcome before inventorying files, so manifest errors
                    # never erase cleanup or resource evidence again.
                    raw = dict(
                        row=row,
                        status=outcome.status,
                        detail=outcome.detail,
                        result=outcome.value if outcome.status == "complete" else None,
                        cleanup_complete=outcome.cleanup_complete,
                        resources=dict(outcome.resource_usage),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    checkpoint(output / "calls" / (key + ".json"), row_identity, **raw)
                    payloads, debris = payload_manifest(output / "payloads" / key)
                    saved = checkpoint(
                        path, row_identity, **raw, payloads=payloads, unpublished_payloads=debris
                    )
                    if not outcome.cleanup_complete or "cleanup incomplete" in outcome.detail:
                        raise RuntimeError("Worker cleanup incomplete; keep inflight guard")
                    guard.unlink()
                if guard.exists() or not saved["cleanup_complete"]:
                    raise RuntimeError("Saved row requires cleanup reconciliation")
                ref = binding(path)
            science.validate_payloads(saved)
            receipts[i] = saved, cache
            refs.append(dict(ref, status=saved["status"]))
        return checkpoint(
            output / "report.json",
            identity,
            schema="exact-repair/scaling-shard/v1",
            status="complete",
            schedule=plan["schedule"],
            runtime=plan["runtime"],
            rows=refs,
            scheduled_rows=stop - start,
            recorded_rows=len(refs),
            counts=dict(Counter(r["status"] for r in refs)),
            recovery_plan=binding(plan_path),
            scientific_source=plan["scientific_batch"],
            study_complete=False,
            gates_passed=False,
            original_costs_preserved=True,
            scientific_rows_replayed=0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    # Import the owner check from this adapter's committed export first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.repair.schema_cleanup_recovery import confirm_owner_gone

    # tools.repair is already imported; clear only package routing, before any
    # scientific module is loaded, so all scientific dependencies remain frozen.
    for name in ("tools.repair.schema_cleanup_recovery", "tools.repair", "tools"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(args.code.resolve()))
    run(args.plan, args.output, owner_check=confirm_owner_gone)
