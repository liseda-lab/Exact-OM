"""Recover only previously unavailable HGT rows, preserving original evaluation evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair import evaluate_campaign as evaluation
from tools.repair.report_campaign import check_receipt, output_binding

EPOCHS = {"xr21-t2-hgt-pairwise-s13": 5, "xr21-t2-hgt-unary-s13": 10}


def checked(value):
    return json.loads(evaluation.check_binding(value).read_text())


def checked_reuse(schedule):
    """Prove row compatibility and completion before admitting any reused evidence."""
    reuse = schedule["reuse"]
    original = checked(reuse["schedule"])
    report = checked(reuse["report"])
    completion = checked(reuse["completion"])
    outputs = checked(reuse["outputs"])
    authorization = checked(reuse["authorization"])
    if authorization["original_evaluation"] != reuse["report"]:
        raise ValueError("Recovery authorization binds another evaluation")
    if completion["status"] != "complete" or completion["exit_code"] != 0:
        raise ValueError("Original evaluation is incomplete")
    if report["schedule"] != reuse["schedule"] or report["process_status"] != "complete":
        raise ValueError("Original evaluation schedule/completion differs")
    work = Path(completion["work"])
    if outputs.get("evaluation-report.json") != reuse["report"]["sha256"]:
        raise ValueError("Original evaluation output receipt differs")
    # Every original proof, label, cache, cost record and receipt stays bound.
    for relative, digest in outputs.items():
        evaluation.check_binding({"path": str(work / relative), "sha256": digest})
    if Path(reuse["report"]["path"]) != work / "evaluation-report.json":
        raise ValueError("Original evaluation work directory differs")
    for key, value in original.items():
        if key not in {"arms", "inner_seconds"} and schedule.get(key) != value:
            raise ValueError("Original evaluation dependency changed: " + key)
    old_arms = {a["id"]: a for a in original["arms"]}
    if [a["id"] for a in original["arms"]] != [a["id"] for a in schedule["arms"]]:
        raise ValueError("Model denominator/order changed")
    recovered = set()
    for arm in schedule["arms"]:
        old = old_arms[arm["id"]]
        if old == arm:
            continue
        if (
            arm["id"] not in EPOCHS
            or old["status"] != "unavailable"
            or arm["status"] != "available"
            or arm["protocol"] != old["protocol"]
            or arm["selected_epoch"] != EPOCHS[arm["id"]]
        ):
            raise ValueError("Only original selected unavailable HGT models may change")
        recovered.add(arm["id"])
    if not recovered:
        raise ValueError("No recovered model rows")
    prior_budget = checked(reuse["stage_budget"])
    if (
        reuse["stage_budget"]["sha256"] != outputs.get("evaluation-budget.json")
        or Path(reuse["stage_budget"]["path"]) != work / "evaluation-budget.json"
        or not math.isfinite(prior_budget["spent_seconds"])
        or prior_budget["spent_seconds"] < 0
        or prior_budget["identity"] != canonical_hash(original)
        or prior_budget["limit_seconds"] != original["inner_seconds"]
        or prior_budget.get("active") is not None
        or not math.isclose(
            schedule["inner_seconds"],
            original["inner_seconds"] - prior_budget["spent_seconds"],
        )
        or schedule["inner_seconds"] <= 0
    ):
        raise ValueError("Cumulative evaluation stage allowance changed")
    planned = {r["id"]: r for r in original["rows"]}
    rows = {r["id"]: r for r in report["rows"]}
    if (
        len(planned) != len(original["rows"])
        or len(rows) != len(report["rows"])
        or set(planned) != set(rows)
        or report["recorded"] != len(planned)
        or report["scheduled"] != len(planned)
    ):
        raise ValueError("Original evaluation denominator differs")
    result = {}
    for row_id, row in rows.items():
        if (
            any(row.get(k) != v for k, v in planned[row_id].items())
            or row["schedule_hash"] != canonical_hash(original)
            or row["row_id"] != row_id
        ):
            raise ValueError("Original row schedule identity changed")
        receipt = work / "rows" / row_id / "receipt.json"
        if (
            outputs.get(str(receipt.relative_to(work))) != evaluation.file_hash(receipt)
            or json.loads(receipt.read_text()) != row
        ):
            raise ValueError("Original row differs from its receipt")
        if row.get("artifact"):
            payload = checked(row["artifact"])
            if payload["schedule_hash"] != row["schedule_hash"] or payload["row_id"] != row_id:
                raise ValueError("Original row artifact identity changed")
        if row.get("arm_id") in recovered:
            if row["status"] != "unavailable" or row.get("artifact"):
                raise ValueError("Cannot repeat previously evaluated model rows")
            if (receipt.parent / "budget.json").exists():
                raise ValueError("Recovered row has prior execution costs; reconcile before reuse")
        else:
            result[row_id] = row
    if sorted(result) != reuse["row_ids"]:
        raise ValueError("Reused row denominator changed")
    return result


def prepare(campaign, original_schedule, output, *, previous_run="xr21-t2-evaluation-002"):
    """Freeze selected models first; never use prior test outcomes to choose settings."""
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing addendum schedule")
    candidate = output.with_name(output.stem + "-model-selection.json")
    evaluation.prepare(campaign, candidate)
    # Compare the serialized records: in-memory case records contain tuples.
    schedule = json.loads(candidate.read_text())
    original = json.loads(Path(original_schedule).read_text())
    registry = json.loads((campaign / "supervisor/registry.json").read_text())
    run = next(r for r in registry["runs"] if r["id"] == previous_run)
    completion = check_receipt(run)
    report_ref = output_binding(run, "evaluation-report.json")
    budget_ref = output_binding(run, "evaluation-budget.json")
    budget = checked(budget_ref)
    recovered = {a["id"] for a, old in zip(schedule["arms"], original["arms"]) if a != old}
    schedule["inner_seconds"] = original["inner_seconds"] - budget["spent_seconds"]
    schedule["reuse"] = {
        "schema": "exact-repair/evaluation-row-reuse/v1",
        "schedule": evaluation.binding(original_schedule),
        "report": report_ref,
        "completion": evaluation.binding(run["completion_path"]),
        "outputs": evaluation.binding(Path(run["completion_path"]).with_name("outputs.json")),
        "stage_budget": budget_ref,
        "authorization": registry["user_recovery_authorization"],
        "row_ids": sorted(r["id"] for r in original["rows"] if r.get("arm_id") not in recovered),
        "policy": "Original row/schedule hashes and costs retained; only unavailable recovered HGT rows execute",
    }
    checked_reuse(schedule)
    write_artifact(output, schedule)
    files = {str(output), str(candidate)}
    for value in schedule["reuse"].values():
        if isinstance(value, dict) and "sha256" in value:
            files.add(value["path"])
    for arm in [*schedule["arms"], *original["arms"]]:
        for value in arm.values():
            if isinstance(value, dict) and "sha256" in value:
                evaluation.check_binding(value)
                files.add(value["path"])
    for value in (schedule["source_preparation"], schedule["plan"]):
        files.add(value["path"])
    files.update(str(Path(completion["work"]) / p) for p in checked(schedule["reuse"]["outputs"]))
    write_artifact(output.with_name("input-files.json"), sorted(files))
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("original_schedule", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    schedule = prepare(args.campaign, args.original_schedule, args.output)
    print(
        json.dumps(
            {"scheduled": schedule["scheduled_rows"], "reused": len(schedule["reuse"]["row_ids"])}
        )
    )


if __name__ == "__main__":
    main()
