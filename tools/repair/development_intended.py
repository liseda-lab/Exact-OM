"""Separately budgeted development intended-parent and typed-query qualification.

Original acquisition receipts and labels are immutable. This resource experiment
measures native eligibility, never trains or manufactures assignment supervision.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.learning import OwlTeacherOracle, evaluate_teacher
from exact.repair.owl import snapshot_from_axioms
from exact.repair.records import canonical_hash
from exact.repair.workers import bounded_call
from tools.repair.acquisition import RecordingVerifier, raise_on_software_failure
from tools.repair.acquisition import validate_schedule as acquisition_schedule
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checked_checkpoint, checkpoint
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import case_from_dict

SCHEMA = "exact-repair/development-intended-resource/v1"
BUDGET = dict(wall_seconds=600, cpu_seconds=1200, memory_mb=8192)


def query_scope(case):
    return [
        dict(
            probe_id=p.probe_id,
            family=p.family,
            desired=p.desired,
            axiom_type=type(p.axiom).__name__,
            axiom_hash=canonical_hash(p.axiom),
            nonvacuity_mode="explicit" if p.nonvacuity is not None else "typed_derived",
            conditions=[canonical_hash(c) for c in p.conditions()] if p.desired else [],
        )
        for p in case.probes
    ]


def validate_schedule(plan):
    if (
        plan.get("schema") != SCHEMA
        or plan.get("split") != "development"
        or plan.get("expected_cases") != 32
        or plan.get("expected_parents") != 16
        or plan.get("heldout_use") is not False
        or plan.get("model_fitting") is not False
        or plan.get("supervision_admitted") is not False
        or plan.get("budget") != BUDGET
    ):
        raise ValueError("Development resource boundary or budget differs")
    source = verify_binding(plan["original_acquisition_plan"])
    if source.get("split") != "development":
        raise ValueError("Original acquisition must be development")
    cases = acquisition_schedule(source)
    if (
        len({c.structural_parent for _, _, c in cases}) != 16
        or plan["cases"] != [r for r, _, _ in cases]
        or plan["query_scope"]
        != [dict(case_id=c.case_id, probes=query_scope(c)) for _, _, c in cases]
    ):
        raise ValueError("Frozen development identities, parents or queries differ")
    for parent in {c.structural_parent for _, _, c in cases}:
        group = [c for _, _, c in cases if c.structural_parent == parent]
        if len(group) != 2 or {c.control for c in group} != {"corrupted", "coherent"}:
            raise ValueError("Development parent variants differ")
    return cases


class ProgressVerifier(RecordingVerifier):
    """Fsync completed obligations before the next potentially blocking query."""

    def check_theory(self, *args, **kwargs):
        sequence = self.sequence
        path = self.directory / f"check-{sequence}.obligations.jsonl"
        if path.exists():
            raise ValueError("Native progress already exists; refusing replay")

        def completed(obligation, support):
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            epoch=time.time(),
                            obligation=asdict(obligation),
                            support=asdict(support),
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())

        kwargs["on_complete"] = completed
        return super().check_theory(*args, **kwargs)


def case_worker(record, directory):
    case = case_from_dict(record)
    if case.split != "development" or case.schema_revision != "v3":
        raise ValueError("Intended resource worker only admits development v3")
    return check_intended(case, directory)


def check_intended(case, directory):
    """Check a caller-qualified case; persist policy and typed-query evidence."""
    directory = Path(directory)
    verifier = ProgressVerifier(directory / "native")
    snapshot = snapshot_from_axioms(case.intended_theory)
    write_artifact(directory / "phase.json", dict(phase="intended_policy", epoch=time.time()))
    report = verifier.check_theory(
        snapshot,
        case.problem.policy.monitored_classes,
        activated=case.intended_active,
        required=case.problem.policy.required,
        prohibited=case.problem.policy.prohibited,
    )
    write_artifact(directory / "intended-policy.json", report.to_dict())
    result = dict(
        logical_status=report.logical_status,
        query_complete=False,
        original_guard_passed=False,
        intended_target_satisfied=None,
        semantic=None,
        query_denominator=len(case.probes),
        query_scope=query_scope(case),
    )
    if report.logical_status == "VERIFIED_FEASIBLE":
        write_artifact(directory / "phase.json", dict(phase="typed_queries", epoch=time.time()))
        semantic = evaluate_teacher(OwlTeacherOracle(verifier, snapshot, case.probes), case.probes)
        result.update(
            semantic=asdict(semantic),
            query_complete=semantic.complete,
            original_guard_passed=semantic.complete,
            intended_target_satisfied=(
                all(o.credit if o.desired else not o.entailed for o in semantic.outcomes)
                if semantic.complete
                else None
            ),
        )
        write_artifact(directory / "intended-queries.json", result)
    return result


def one_case(row, record, case, output, identity, budget, *, worker=case_worker, schema=SCHEMA):
    directory = Path(output) / "cases" / canonical_hash((case.case_id, row["case_hash"]))
    receipt, guard = directory / "completion.json", directory / "inflight.json"
    expected = canonical_hash((identity, row))
    saved = checked_checkpoint(receipt, expected)
    if saved:
        for ref in saved["payloads"]:
            if sha(ref["path"]) != ref["sha256"]:
                raise ValueError("Native resource payload changed")
        if not saved["cleanup_complete"]:
            raise RuntimeError("Native resource cleanup needs ownership reconciliation")
        if guard.exists():
            if read(guard)["identity"] != expected:
                raise ValueError("Native resource guard changed")
            guard.unlink()
        raise_on_software_failure(saved)
        return binding(receipt)
    if guard.exists():
        raise RuntimeError("Interrupted native resource row needs owner/budget reconciliation")
    write_artifact(guard, dict(identity=expected, case_id=case.case_id, started_epoch=time.time()))
    began = time.monotonic()
    call = bounded_call(
        worker,
        record,
        str(directory),
        timeout=budget["wall_seconds"],
        cpu_seconds=budget["cpu_seconds"],
        memory_mb=budget["memory_mb"],
    )
    # Persist outer errors before hashing or deciding whether recovery is safe.
    write_artifact(
        directory / "call.json",
        dict(
            status=call.status,
            detail=call.detail,
            cleanup_complete=call.cleanup_complete,
            resources=dict(call.resource_usage),
            budget=budget,
        ),
    )
    payloads = [
        binding(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and p.name not in {"completion.json", "inflight.json"}
        and not p.name.startswith(".")
        and not p.name.endswith(".tmp")
    ]
    value = call.value if call.status == "complete" else None
    status = (
        call.status
        if value is None
        else (
            "qualified"
            if value["original_guard_passed"] and value["intended_target_satisfied"]
            else (
                "intended_target_mismatch"
                if value["original_guard_passed"]
                else "native_unknown_or_infeasible"
            )
        )
    )
    saved = checkpoint(
        receipt,
        expected,
        schema=schema,
        case_id=case.case_id,
        parent=case.structural_parent,
        family=case.family,
        split=case.split,
        control=case.control,
        status=status,
        outer_status=call.status,
        detail=call.detail,
        cleanup_complete=call.cleanup_complete,
        resources=dict(call.resource_usage),
        elapsed_seconds=time.monotonic() - began,
        budget=budget,
        result=value,
        payloads=payloads,
        query_denominator=len(case.probes),
        query_scope=query_scope(case),
        supervision_admitted=False,
        original_results_replaced=False,
    )
    if not call.cleanup_complete:
        raise RuntimeError("Native resource cleanup incomplete; retain inflight ownership guard")
    guard.unlink()
    raise_on_software_failure(saved)
    return binding(receipt)


def run(plan_path, output, start, stop):
    from exact.repair.study import runtime_manifest

    plan = read(plan_path)
    cases = validate_schedule(plan)
    if not 0 <= start < stop <= len(cases):
        raise ValueError("Invalid development resource slice")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = canonical_hash(
        (
            binding(plan_path),
            sha(__file__),
            sha(Path(__file__).with_name("acquisition.py")),
            runtime_manifest(),
        )
    )
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for row, record, case in cases[start:stop]:
            write_artifact(
                output / "progress.json", dict(completed_rows=len(rows), case_id=case.case_id)
            )
            rows.append(one_case(row, record, case, output, identity, plan["budget"]))
        saved = [verify_binding(r) for r in rows]
        return checkpoint(
            output / "report.json",
            identity,
            schema=SCHEMA,
            status="complete",
            plan=binding(plan_path),
            scheduled_rows=stop - start,
            recorded_rows=len(rows),
            rows=rows,
            study_scheduled_cases=32,
            study_parent_groups=16,
            counts=dict(Counter(r["status"] for r in saved)),
            query_denominator=sum(r["query_denominator"] for r in saved),
            heldout_cases_opened=False,
            supervision_admitted=False,
            gates_passed=False,
            original_results_replaced=False,
            fitting_eligible=False,
            next_stage="xr21-expanded-training-001",
            limitations=[
                "Separately named resource experiment; original30second guards and all costs preserved.",
                "Typed nonvacuity is derived where not explicit; no unwanted probes were added.",
                "Only intended parents are checked; no generated assignment labels or model fitting.",
                "Qualification is scoped to original development input/policy/query and native runtime.",
                "Actual supervision, optimizer capacity,18protocols,selection,heldout controls and report remain.",
            ],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    run(args.plan, args.output, args.start, args.stop)


if __name__ == "__main__":
    main()
