"""Continue endpoint qualification after an operational slice, without replay.

The frozen scientific runner remains unchanged. Completed rows are copied byte
for byte into a revision, the interrupted probe stays unknown, and only untouched
probes run. Original receipts, payloads and the ownership guard remain intact.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
from pathlib import Path
import sys


SCHEMA = "exact-repair/scaling-endpoint-continuation/v1"
CAUSE = {"type": "TimeoutError", "message": "worker time/memory budget exhausted"}


def load_study(plan):
    from tools.repair.expanded_corpus import bound
    from tools.repair.batch import sha

    ref = plan["study_runner"]
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Frozen qualification runner changed")
    spec = importlib.util.spec_from_file_location("frozen_endpoint_study", ref["path"])
    study = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = study
    spec.loader.exec_module(study)
    bound(plan["schedule"])
    return study


def validate(plan, study):
    """Verify the whole spent prefix, source identity, and one interrupted probe."""
    from exact.repair.records import canonical_hash
    from tools.repair.batch import checked_batch, read
    from tools.repair.expanded_corpus import bound, binding
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair.scaling_payload_recovery import payload_manifest

    if (plan["schema"] != SCHEMA or plan["recovery_kind"] != "operational_continuation"
            or plan["scientific_budgets_changed"] is not False
            or plan["prior_costs_reset"] is not False):
        raise ValueError("Continuation cannot change scientific budgets or clear costs")
    batch = checked_batch(plan["scientific_batch"]["path"])
    if batch != bound(plan["scientific_batch"]) or batch["code"] != plan["source_root"]:
        raise ValueError("Original scientific export differs")
    key, runtime = study.identity(plan["schedule"]["path"])
    schedule = study.load_schedule(plan["schedule"]["path"])
    if key != plan["scientific_identity"] or runtime != plan["runtime"]:
        raise ValueError("Scientific checkpoint dependencies changed")
    completion, owner, command = (bound(plan[k]) for k in ("completion", "owner", "command"))
    jobs = [j for j in batch["jobs"] if j["id"] == completion["job_id"]]
    original, output = Path(plan["original"]), Path(plan["output"])
    if (completion["status"] != "failed" or completion["exit_code"] != 1
            or completion["error"] != CAUSE or completion["step_id"] != plan["original_step"]
            or completion["batch"] != plan["scientific_batch"]["path"]
            or owner != dict(step_id=completion["step_id"], dispatch_nonce=completion["dispatch_nonce"])
            or len(jobs) != 1 or completion["elapsed_seconds"] < jobs[0]["seconds"]
            or completion["peak_rss_mb"] >= jobs[0]["resources"]["memory_mb"]
            or original != Path(completion["work"]) / "qualification"
            or output.parent != original.parent or output == original
            or plan["publish_report"] != str(original / "report.json")):
        raise ValueError("Original operational timeout, ownership or work identity differs")
    expected = [a.format(code=batch["code"], python=batch["python"], work=completion["work"])
                for a in jobs[0]["commands"][1]]
    if (command != dict(argv=expected, cwd=batch["code"])
            or Path(plan["command"]["path"]) != Path(plan["completion"]["path"]).with_name("command-1.json")):
        raise ValueError("Original qualification command differs")
    # Test completion is evidence for reuse only, never a scientific gate.
    import xml.etree.ElementTree as ET
    if binding(plan["validation"]["path"]) != plan["validation"]:
        raise ValueError("Original test receipt changed")
    suites = list(ET.parse(plan["validation"]["path"]).getroot().iter("testsuite"))
    if (sum(int(s.get("tests", 0)) for s in suites) != 22
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))):
        raise ValueError("Completed original qualification tests required")
    ledger = bound(plan["ledger_snapshot"])
    charge = ledger["attempts"].get(str(Path(plan["completion"]["path"]).parent), {})
    if (charge.get("status") != "settled" or charge.get("logical_id") != completion["job_id"]
            or charge.get("elapsed_seconds", -1) < completion["elapsed_seconds"]
            or (charge.get("cpu_seconds") or 0) < completion["cpu_seconds"]):
        raise ValueError("Original attempt must remain fully charged")
    stages = [(i, stage) for i in range(len(schedule["cases"]))
              for stage in ("native", *study.scaling.METHODS)]
    interrupted = plan["interrupted_index"]
    if not 0 < interrupted < len(stages) - 1 or stages[interrupted][1] == "native":
        raise ValueError("Expected interrupted generation and untouched continuation")
    if len(plan["rows"]) != interrupted:
        raise ValueError("Every spent prefix row is required")
    for (index, stage), ref in zip(stages[:interrupted], plan["rows"], strict=True):
        path = original / "rows" / (canonical_hash((index, stage)) + ".json")
        saved = checked_checkpoint(path, canonical_hash((key, index, stage)))
        if (binding(path) != ref or saved != bound(ref)
                or (saved["case_index"], saved["stage"]) != (index, stage)
                or (original / "inflight" / path.name).exists()):
            raise ValueError("Spent prefix identity or ownership differs")
        study.check_saved(saved)
        if stage != "native":
            study.generation_contract(saved, schedule["cases"][index])
    index, stage = stages[interrupted]
    guard = original / "inflight" / (canonical_hash((index, stage)) + ".json")
    saved_guard = checked_checkpoint(guard, canonical_hash((key, index, stage)))
    if (binding(guard) != plan["guard"] or saved_guard != bound(plan["guard"])
            or (saved_guard["case_index"], saved_guard["stage"]) != (index, stage)
            or not completion["started_epoch"] <= saved_guard["started_epoch"] <= completion["finished_epoch"]
            or (original / "rows" / guard.name).exists()):
        raise ValueError("Interrupted probe identity differs")
    published, debris = payload_manifest(original / "payloads" / guard.stem)
    if published != plan["interrupted_payloads"] or debris != plan["unpublished_payloads"]:
        raise ValueError("Interrupted payload inventory changed")
    for index, stage in stages[interrupted + 1:]:
        name = canonical_hash((index, stage))
        if any(p.exists() for p in (original / "rows" / (name + ".json"),
                original / "inflight" / (name + ".json"), original / "payloads" / name)):
            raise ValueError("Continuation probe already spent scientific work")
    expected_rows = {Path(ref["path"]).name for ref in plan["rows"]}
    if ({p.name for p in (original / "rows").glob("*.json")} != expected_rows
            or {p.name for p in (original / "inflight").glob("*.json")} != {guard.name}):
        raise ValueError("Unexpected original checkpoint or guard")
    return schedule, key, completion, saved_guard


def seed_revision(plan, key, completion, guard, owner):
    """Materialize reusable evidence, never a renewed allowance for a spent probe."""
    from tools.repair.batch import sha
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checkpoint, checked_checkpoint

    if (owner["step_id"] != plan["original_step"] or owner["surviving_processes"] != 0
            or not owner["cgroup_absent"] or owner["signals_sent"] != 0):
        raise ValueError("Original owner cleanup is unresolved")
    output = Path(plan["output"])
    for ref in plan["rows"]:
        path = output / "rows" / Path(ref["path"]).name
        bound(ref)
        if path.exists():
            if sha(path) != ref["sha256"]:
                raise ValueError("Reused prefix changed in revision")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(ref["path"]).read_bytes())
    path = output / "rows" / Path(plan["guard"]["path"]).name
    expected = dict(case_index=guard["case_index"], case_id=plan["interrupted_case_id"],
        stage=guard["stage"], method=guard["stage"], status="unknown_after_operational_timeout",
        detail="Outer worker slice ended during this probe; no completed call receipt. Retained unknown without replay.",
        cleanup_complete=True, resources={},
        elapsed_seconds=completion["finished_epoch"] - guard["started_epoch"],
        elapsed_scope="Upper bound from guard to outer completion; row CPU/RSS telemetry unavailable. Full attempt charged separately.",
        result=None, pool=None, payloads=plan["interrupted_payloads"],
        unpublished_payloads=plan["unpublished_payloads"], original_guard=plan["guard"],
        original_completion=plan["completion"], ownership=owner,
        additional_scientific_seconds=0, prior_costs_reset=False)
    saved = checked_checkpoint(path, guard["identity"])
    if saved is not None:
        expected["ownership"] = saved["ownership"]
        if {k: v for k, v in saved.items() if k not in ("identity", "content_hash")} != expected:
            raise ValueError("Reconciled unknown provenance changed")
    else:
        checkpoint(path, guard["identity"], **expected)


def run(plan_path, *, owner_check):
    from tools.repair.batch import read
    from tools.repair.expanded_corpus import binding, immutable

    plan = read(plan_path)
    if binding(__file__)["sha256"] != plan["adapter_sha256"]:
        raise ValueError("Frozen continuation adapter changed")
    study = load_study(plan)
    original, output = Path(plan["original"]), Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (original / "worker.lock").open("a") as lock, (output / "continuation.lock").open("a") as revision_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(revision_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule, key, completion, guard = validate(plan, study)
        if schedule["cases"][guard["case_index"]]["case_id"] != plan["interrupted_case_id"]:
            raise ValueError("Interrupted case identity differs")
        owner = owner_check(plan["original_step"])
        seed_revision(plan, key, completion, guard, owner)
        # The original runner checks every identity and skips all 15 spent rows.
        report = study.qualify(plan["schedule"]["path"], output)
        study.admission(plan["schedule"]["path"], output / "report.json")
        immutable(output / "continuation.json", dict(schema=SCHEMA,
            plan=binding(plan_path), report=binding(output / "report.json"),
            retained_rows=len(plan["rows"]), interrupted_unknown_rows=1,
            continued_rows=report["scheduled"] - len(plan["rows"]) - 1,
            scientific_rows_replayed=0, prior_costs_reset=False,
            software_repair_attempt=False, original_guard_preserved=True))
        # This stable report did not exist in the failed attempt. No original
        # result is overwritten; downstream frozen commands can keep their path.
        immutable(plan["publish_report"], report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    from tools.repair.schema_cleanup_recovery import confirm_owner_gone
    run(args.plan, owner_check=confirm_owner_gone)
