"""Preserve a stopped development row and continue only its untouched suffix.

Run this separately frozen adapter with --code pointing to the original export.
The failed row stays unknown, with its original guard, artifacts and costs intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import sys

CAUSE = "RuntimeError: Nested worker cleanup incomplete; reconcile descendants before continuation"
COMMAND_ERROR = "command 0 exited 1: RuntimeError: Worker cleanup incomplete; retain inflight ownership guard"


def slice_bounds(plan):
    start, stop = plan["original_slice"]
    failed = plan["failed_index"]
    if (any(type(v) is not int for v in (start, stop, failed))
            or not 0 <= start <= failed < stop - 1
            or plan["only_unstarted_slice"] != [failed + 1, stop]
            or plan["original_row_count"] != stop - start
            or plan["prior_costs_reset"] is not False):
        raise ValueError("Retain the entire slice and continue only untouched rows")
    return start, stop, failed


def reconciled_unknown(previous, guard, original_ref, evidence_ref, owner, step):
    if (previous["status"] != "error" or previous["detail"] != CAUSE
            or not previous["cleanup_complete"] or previous["result"] is not None
            or previous["identity"] != guard["identity"] or previous["row"] != guard["row"]):
        raise ValueError("Only the exact retained nested cleanup error admits reconciliation")
    if (owner["step_id"] != step or owner["surviving_processes"] != 0
            or not owner["cgroup_absent"] or owner["signals_sent"] != 0):
        raise ValueError("Original owner cleanup is unresolved")
    result = {k: v for k, v in previous.items() if k not in ("identity", "content_hash")}
    result.update(status="unknown_after_cleanup_reconciliation",
        detail="No final scientific result; original nested cleanup error retained. Owner gone; spent row not replayed.",
        original_status=previous["status"], original_detail=previous["detail"],
        original_receipt=original_ref, cleanup_reconciliation=evidence_ref,
        recovery_action="retain_unknown_without_replay", ownership=owner,
        additional_scientific_seconds=0, prior_costs_reset=False)
    return result


def validate_original(plan, evidence, identity):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair import development_decode as science

    start, stop, failed = slice_bounds(plan)
    completion, owner, command = (bound(evidence[k]) for k in ("completion", "ownership", "command"))
    frozen = bound(plan["frozen_batch"])
    work = Path(completion["work"]) / "decode"
    expected_argv = [frozen["python"], "-m", "tools.repair.development_decode",
        plan["schedule"]["path"], str(work), "--start", str(start), "--stop", str(stop)]
    jobs = [j for j in frozen["jobs"] if j["id"] == completion["job_id"]]
    if (completion["status"] != "failed" or completion["exit_code"] != 1
            or completion["error"]["message"] != COMMAND_ERROR
            or completion["step_id"] != plan["original_step"]
            or owner != dict(step_id=completion["step_id"], dispatch_nonce=completion["dispatch_nonce"])
            or completion["batch"] != plan["frozen_batch"]["path"]
            or Path(evidence["command"]["path"]) != Path(evidence["completion"]["path"]).parent / "command-0.json"
            or command != dict(argv=expected_argv, cwd=plan["source_root"])
            or frozen["code"] != plan["source_root"] or len(jobs) != 1
            or [[arg.replace("{python}", frozen["python"]).replace("{work}", completion["work"])
                 for arg in cmd] for cmd in jobs[0]["commands"]] != [expected_argv]):
        raise ValueError("Original execution, nonce, command, frozen job or slice differs")
    schedule = bound(plan["schedule"])
    if stop > len(schedule["rows"]):
        raise ValueError("Original slice exceeds the schedule")
    prior = {}
    for item in evidence["rows"]:
        index, ref = item["index"], item["receipt"]
        if index not in range(start, failed + 1) or index in prior:
            raise ValueError("Retained prefix differs")
        row = schedule["rows"][index]
        expected = canonical_hash((identity, row))
        path = work / "rows" / (row["id"] + ".json")
        saved = bound(ref)
        if (Path(ref["path"]) != path or saved != checked_checkpoint(path, expected)
                or saved["row"] != row or not saved["cleanup_complete"]):
            raise ValueError("Original row identity or outer cleanup differs")
        science.fresh.validate_payloads(saved)
        guard = work / "inflight" / path.name
        if index != failed:
            if guard.exists():
                raise ValueError("Finished row has an unresolved guard")
            science.fresh.raise_on_software_failure(saved)
        elif (str(guard) != evidence["guard"]["path"]
              or checked_checkpoint(guard, expected) != bound(evidence["guard"])
              or saved["status"] != "error" or saved["detail"] != CAUSE
              or saved["result"] is not None):
            raise ValueError("Failed row must retain the exact cleanup error and guard")
        prior[index] = (ref, saved)
    if set(prior) != set(range(start, failed + 1)):
        raise ValueError("Every spent prefix row must be retained")
    for row in schedule["rows"][failed + 1:stop]:
        if any(p.exists() for p in (work / "rows" / (row["id"] + ".json"),
                work / "inflight" / (row["id"] + ".json"), work / "payloads" / row["id"])):
            raise ValueError("Continuation row already spent work")
    return prior


def run(plan_path, output, *, owner_check=None):
    from exact.repair.records import canonical_hash
    from exact.repair.study import runtime_manifest
    from tools.repair.batch import read, sha, checked_batch
    from tools.repair.expanded_corpus import binding, bound
    from tools.repair.expanded_profile import checked_checkpoint, checkpoint
    from tools.repair.schema_cleanup_recovery import confirm_owner_gone
    from tools.repair import development_decode as science

    plan, output = read(plan_path), Path(output).resolve()
    start, stop, failed = slice_bounds(plan)
    if (binding(__file__) != plan["runner"] or str(output) != plan["output"]
            or Path(science.__file__).resolve().parents[2] != Path(plan["source_root"]).resolve()
            or runtime_manifest() != plan["runtime"]):
        raise ValueError("Frozen runner, scientific source, runtime or output changed")
    identity = canonical_hash((plan["schedule"]["sha256"], sha(science.__file__),
                              sha(science.fresh.__file__), plan["runtime"]))
    if identity != plan["scientific_identity"]:
        raise ValueError("Original scientific dependency identity changed")
    frozen = checked_batch(plan["frozen_batch"]["path"])
    if frozen != bound(plan["frozen_batch"]) or frozen["code"] != plan["source_root"]:
        raise ValueError("Original frozen batch differs")
    schedule = bound(plan["schedule"])
    science.validate_schedule(schedule)
    evidence = bound(plan["evidence"])
    original_work = Path(bound(evidence["completion"])["work"]).resolve()
    if output.parent != original_work or output == original_work / "decode":
        raise ValueError("Recovery requires a separate output in the original logical work root")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = validate_original(plan, evidence, identity)
        owner = (owner_check or confirm_owner_gone)(plan["original_step"])
        recovery_identity = canonical_hash((binding(plan_path), identity))
        revised = output / "reconciled-unknown.json"
        expected = reconciled_unknown(prior[failed][1], bound(evidence["guard"]),
            prior[failed][0], plan["evidence"], owner, plan["original_step"])
        saved = checked_checkpoint(revised, recovery_identity)
        if saved:
            expected["ownership"] = saved["ownership"]
            if {k: v for k, v in saved.items() if k not in ("identity", "content_hash")} != expected:
                raise ValueError("Reconciled unknown provenance changed")
        else:
            saved = checkpoint(revised, recovery_identity, **expected)
        continuation = science.run(Path(plan["schedule"]["path"]), output / "continuation", failed + 1, stop)
        if continuation["scheduled"] != stop - failed - 1 or continuation["recorded"] != stop - failed - 1:
            raise ValueError("Continuation denominator changed")
        refs = []
        for i in range(start, failed):
            ref, original = prior[i]
            status = bound(original["result"])["status"] if original.get("result") else original["status"]
            refs.append(dict(ref, row_id=original["row"]["id"], status=status))
        refs.append(dict(binding(revised), row_id=saved["row"]["id"], status=saved["status"]))
        refs.extend(continuation["rows"])
        if [bound(ref)["row"] for ref in refs] != schedule["rows"][start:stop]:
            raise ValueError("Recovered row order or identities changed")
        report = checkpoint(output / "report.json", recovery_identity,
            schema=science.SCHEMA, status="complete", schedule=plan["schedule"],
            slice=[start, stop], scheduled=stop-start, recorded=len(refs), rows=refs,
            outcomes=dict(Counter(r["status"] for r in refs)), runtime=plan["runtime"],
            study_complete=False, gates_passed=False, model_fitting=False,
            heldout_cases_opened=False, supervision_admitted=False,
            checkpoint_selection="unavailable_untrained_diagnostic_models",
            intended_parent_qualification="original acquisition limitations retained",
            semantic_scope="selected-only original query basis; native receipts retained; not admitted labels",
            followup="xr21-expanded-training-001", api_spend_usd=0,
            prior_costs_reset=False, scientific_rows_replayed=0,
            cleanup_recovery=dict(plan=binding(plan_path), evidence=plan["evidence"],
                retained_finished_rows=failed-start, retained_unknown_rows_without_replay=1,
                continued_rows=stop-failed-1, original_error_retained=True))
        # The deployed scanner accepts the shared fresh-evaluation row envelope.
        # Keep the development report and publish a bound monitoring projection;
        # no controller deployment or change to scientific row contents is needed.
        checkpoint(output / "science-health.json", recovery_identity,
            schema=science.fresh.SCHEMA, status="complete", rows=refs,
            scheduled=stop-start, recorded=len(refs),
            diagnostic_report=binding(output / "report.json"),
            scope="Development diagnostic monitoring projection; no held-out or fitted-model claim",
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
