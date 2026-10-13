"""Materialize frozen public Conference inputs under a charged post-freeze worker.

No reference alignment, evaluator target or logical result is consumed. Full
ontology construction can be unsupported; such rows remain in the denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
import zipfile
from pathlib import Path

from exact.repair.records import canonical_hash, read_record
from exact.repair.workers import bounded_call
from tools.repair.batch import read
from tools.repair.corrective_conference import (
    bound_bytes, construct_row_native, immutable_bytes,
)
from tools.repair.corrective_campaign import observable_queries
from tools.repair.evaluation_ownership import validate_lane, validate_owner
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, immutable

SCHEMA = "exact-repair/frozen-conference-test-inputs/v1"


def validate(plan, now):
    if plan.get("schema") != SCHEMA or plan.get("execution_authorized") is not True:
        raise ValueError("Conference TEST input preparation is not admitted")
    freeze = bound(plan["model_freeze_receipt"])
    from tools.repair.fixed_endpoint_freeze import validate as validate_endpoint_freeze

    validate_endpoint_freeze(freeze)
    schedule = bound(plan["schedule"])
    expected_models = {f"hgt-pair-{c}-s{s}" for s in (13, 37, 73)
                       for c in ("symbolic", "symbolic_plus_llm")}
    if (freeze.get("test_outcomes_opened") is not False
            or freeze.get("final_freeze") is not True
            or set(freeze.get("models", {})) != expected_models
            or freeze.get("schedule") != plan["schedule"]
            or now < plan["not_before_epoch"]
            or plan["not_before_epoch"] != schedule["models_freeze_epoch"]):
        raise ValueError("Conference TEST remains closed until the frozen schedule opens")
    declaration = bound(schedule["conference_declaration"])
    rows = [r for r in declaration["rows"] if r["split"] == "test"]
    cases = [c for c in schedule["cases"] if c["cohort"] == "conference"]
    if (len(declaration["rows"]) != 21 or len(rows) != 7
            or {r["id"] for r in rows} != {c["case_id"] for c in cases}
            or plan["case_ids"] != [c["case_id"] for c in cases]
            or plan["case_seconds"] <= 0 or plan["case_seconds"] > 300):
        raise ValueError("Conference whole-ontology split or fixed denominator changed")
    return {r["id"]: r for r in rows}, cases


def materialize(row, case, directory):
    """Run in the bounded child; archive extraction is included in its budget."""
    directory = Path(directory)
    # Authenticate source archive and open only this exact publisher member.
    bound_bytes(row["matcher_archive"])
    with zipfile.ZipFile(row["matcher_archive"]["path"]) as archive:
        if archive.namelist().count(row["matcher_member"]) != 1:
            raise ValueError("Expected exactly one frozen publisher matcher member")
        raw = archive.read(row["matcher_member"])
    matcher = immutable_bytes(directory / "LogMap.rdf", raw)
    if matcher["path"] in {r["path"] for r in row["reference_bindings"]}:
        raise ValueError("Reference alignment cannot become matcher input")
    admitted = dict(row, matcher=matcher)
    try:
        result = construct_row_native(admitted)
    except ValueError as error:
        if not str(error).startswith("Unsupported or absent matcher endpoint at row "):
            raise
        return dict(status="unavailable_source_namespace_mismatch", detail=str(error),
                    matcher=matcher, public_input=None, reference_rows_opened=False)
    if result["status"] != "constructed":
        return dict(result, matcher=matcher, public_input=None, reference_rows_opened=False)
    problem = read_record(result["problem"])
    metadata = {k: case[k] for k in ("case_id", "structural_parent", "split")}
    public = dict(schema="exact-repair/public-study-input/v1", metadata=metadata,
                  input=result["problem"], observable_queries=observable_queries(problem),
                  observable_mapping_scores={k: v["score"] for k, v in problem.evidence})
    immutable(directory / "public-input.json", public)
    return dict(status="public_input_constructed", public_input=binding(directory / "public-input.json"),
                matcher=matcher, source_routes=result["source_routes"],
                mapping_count=result["mapping_count"],
                complete_source_axiom_counts=result["complete_source_axiom_counts"],
                reference_rows_opened=False, logical_qualification_claimed=False)


def one_case(row, case, output, identity, seconds, deadline):
    directory = Path(output) / hashlib.sha256(case["case_id"].encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    completed, guard = directory / "completion.json", directory / "started.json"
    row_identity = canonical_hash((identity, row, case))
    if completed.exists():
        result = read(completed)
        if result["identity"] != row_identity:
            raise ValueError("Conference input resume identity changed")
        if result.get("result", {}).get("public_input"):
            authenticate(result["result"]["public_input"])
        if result["status"] == "error" or not result["cleanup_complete"]:
            raise RuntimeError("Prior Conference construction failure requires diagnosis")
        return binding(completed)
    if guard.exists():
        raise RuntimeError("Interrupted Conference construction requires owner reconciliation; no replay")
    available = min(seconds, deadline - time.time())
    if available <= 2:
        result = dict(identity=row_identity, status="unavailable_deadline", result={},
                      cleanup_complete=True, charged_seconds=0)
    else:
        started = time.time()
        immutable(guard, dict(identity=row_identity, started_epoch=started,
                              reserved_seconds=available))
        outcome = bounded_call(materialize, row, case, str(directory / "payload"),
                               timeout=available - 2, memory_mb=20000,
                               cpu_seconds=3 * (available - 2))
        result = dict(identity=row_identity, status=outcome.status, result=outcome.value or {},
                      cleanup_complete=outcome.cleanup_complete, detail=outcome.detail,
                      resources=dict(outcome.resource_usage), charged_seconds=time.time()-started)
    immutable(completed, result)
    if result["status"] == "error" or not result["cleanup_complete"]:
        raise RuntimeError("Conference input construction implementation/cleanup failure")
    return binding(completed)


def run(plan, output):
    rows, cases = validate(plan, time.time())
    validate_lane(plan)
    ledger = read(plan["ledger_path"])
    validate_owner(plan, ledger)
    first = ledger["stages"]["evaluation"]["started_epoch"]
    deadline = min(plan["deadline_epoch"], first + 40*3600,
                   float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    identity = canonical_hash(plan)
    immutable(output / "identity.json", dict(identity=identity))
    results = []
    for case in cases:
        ref = one_case(rows[case["case_id"]], case, output / "cases", identity,
                       plan["case_seconds"], deadline)
        results.append(dict(case_id=case["case_id"], completion=ref, **bound(ref)))
    report = dict(schema=SCHEMA, manifest_identity=identity, expected_cases=7,
                  planned_conference_pairs=21, rows=results, schedule=plan["schedule"],
                  model_freeze_receipt=plan["model_freeze_receipt"], reference_rows_opened=False,
                  logical_qualification_claimed=False, hosted_calls=0)
    immutable(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(read(args.manifest), args.output)


if __name__ == "__main__":
    main()
