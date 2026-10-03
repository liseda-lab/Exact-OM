"""Receipt-bound cumulative XR-2.1 smoke reporting; never run or select a model."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.batch import locked, read, sha

SCHEMA = "exact-repair/campaign-report/v3"
DEFERRED = [
    "B07 reserved test controls (20 rows remain explicitly deferred)",
    "larger cohorts and seeds 37/73",
    "real-ontology experiments and production matcher input",
    "expanded action grammar",
    "LLM semantic annotations (disabled; API spend zero)",
    "G0-G2 qualification and XR-E00-XR-E09 complete studies",
]


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def checked(item):
    if sha(item["path"]) != item["sha256"]:
        raise ValueError("Report dependency changed: " + item["path"])
    return read(item["path"])


def check_receipt(run, *, success=True):
    receipt = read(run["completion_path"])
    if receipt.get("step_id") != run["step_id"] or receipt.get("dispatch_nonce") != run.get(
        "dispatch_nonce"
    ):
        raise ValueError("Completion identity differs from registered launch")
    if success and (receipt["status"] != "complete" or receipt["exit_code"] != 0):
        raise ValueError("Required process is not complete")
    return receipt


def output_binding(run, relative):
    receipt = check_receipt(run)
    path = Path(receipt["work"]) / relative
    outputs = read(Path(run["completion_path"]).with_name("outputs.json"))
    if outputs.get(relative) != sha(path):
        raise ValueError("Report input differs from completed worker output")
    return binding(path)


def totals(attempts):
    """Separate paid work from reservations; keep allocated and measured CPU distinct."""
    settled = [r for r in attempts.values() if r["status"] == "settled"]
    reserved = [r for r in attempts.values() if r["status"] != "settled"]
    return {
        "attempts": len(attempts),
        "settled_attempts": len(settled),
        "worker_seconds": sum(r["elapsed_seconds"] for r in settled),
        "reserved_worker_seconds": sum(r["reserved_seconds"] for r in reserved),
        "allocated_cpu_seconds": sum(
            r["elapsed_seconds"] * r["resources"]["cpus"] for r in settled
        ),
        "allocated_gpu_seconds": sum(
            r["elapsed_seconds"] * r["resources"]["gpus"] for r in settled
        ),
        "allocated_memory_mb_seconds": sum(
            r["elapsed_seconds"] * r["resources"]["memory_mb"] for r in settled
        ),
        "measured_cpu_seconds": sum(r.get("cpu_seconds") or 0 for r in settled),
        "peak_worker_tree_rss_mb": max((r.get("peak_rss_mb") or 0 for r in settled), default=0),
        "external_api_cost_usd": 0,
    }


def costs(campaign):
    campaign = Path(campaign).resolve()
    lineage = read(campaign / "budget-lineage.json")
    pilot = checked(
        {
            "path": lineage["inherited"]["pilot-ledger.json"]["snapshot"],
            "sha256": lineage["inherited"]["pilot-ledger.json"]["sha256"],
        }
    )
    smoke = checked(
        {
            "path": lineage["inherited"]["smoke-ledger.json"]["snapshot"],
            "sha256": lineage["inherited"]["smoke-ledger.json"]["sha256"],
        }
    )
    with locked(campaign / "resource-ledger.lock"):
        ledger = read(campaign / "resource-ledger.json")
        ledger_binding = binding(campaign / "resource-ledger.json")
    inherited = pilot["attempts"]
    if any(ledger["attempts"].get(k) != v for k, v in inherited.items()):
        raise ValueError("Inherited pilot accounting changed")
    new = {k: v for k, v in ledger["attempts"].items() if k not in inherited}
    if any(not Path(k).is_relative_to(campaign / "attempts") for k in new):
        raise ValueError("Unexpected campaign cost ownership")
    if set(smoke["attempts"]) & set(ledger["attempts"]):
        raise ValueError("Historical smoke would be double-counted")
    current = totals(new)
    if any(
        value.get("cumulative", {}).get("external_api_cost_usd", 0) != 0
        for value in (ledger, pilot, smoke)
    ):
        raise ValueError("Unexpected external API expenditure")
    pilot_totals, smoke_totals = totals(inherited), totals(smoke["attempts"])
    expected = (
        lineage["historical_pilot_worker_seconds"] + lineage["incremental_limit_worker_seconds"]
    )
    amendment = ledger.get("time_limit_amendment")
    unlimited = ledger["limit_worker_seconds"] is None
    if unlimited:
        decision = checked(amendment) if amendment else {}
        if not (
            decision.get("schema") == "exact-repair/time-limit-amendment/v1"
            and decision.get("campaign") == str(campaign)
            and decision.get("previous_limit_worker_seconds") == expected
            and "limit_worker_seconds" in decision
            and decision["limit_worker_seconds"] is None
            and decision.get("costs_reset") is False
            and decision.get("user_authorization")
        ):
            raise ValueError("Unlimited campaign time requires a bound user amendment")
    elif not math.isclose(ledger["limit_worker_seconds"], expected):
        raise ValueError("Campaign budget ceiling changed")
    accounted = totals(ledger["attempts"])
    if not math.isclose(
        accounted["worker_seconds"] + accounted["reserved_worker_seconds"],
        ledger["cumulative"]["worker_seconds"],
    ):
        raise ValueError("Cumulative worker accounting does not reconcile")
    remaining = (
        None
        if unlimited
        else (
            lineage["incremental_limit_worker_seconds"]
            - current["worker_seconds"]
            - current["reserved_worker_seconds"]
        )
    )
    if remaining is not None and remaining < 0:
        raise ValueError("Authorized incremental budget exhausted")
    jobs = sorted({v["logical_id"] for v in new.values()})
    return {
        "schema": "exact-repair/campaign-costs/v3",
        "ledger": ledger_binding,
        "lineage": binding(campaign / "budget-lineage.json"),
        "historical_pilot": pilot_totals,
        "historical_smoke": smoke_totals,
        "xr21_incremental": current,
        "combined": totals({**ledger["attempts"], **smoke["attempts"]}),
        "incremental_limit_worker_seconds": (
            None if unlimited else lineage["incremental_limit_worker_seconds"]
        ),
        "time_limit_amendment": amendment,
        "unreserved_worker_seconds": remaining,
        "jobs": {
            job: totals({k: v for k, v in new.items() if v["logical_id"] == job}) for job in jobs
        },
        "attempts": new,
        "historical_attempts": {"pilot": inherited, "smoke": smoke["attempts"]},
        "costs_reset": False,
        "external_api_cost_usd": 0,
        "accounting_scope": "batch workers including failed/replaced attempts; controller/editor overhead is outside worker ledger",
        "label_cost_policy": "shared labels charged once at source; provenance times are descriptive, not added again",
    }


def prepare(campaign, output, run_id="xr21-t2-report-001"):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing report manifest")
    registry = read(campaign / "supervisor/registry.json")
    plan = read(campaign / "plan.json")
    runs = {r["id"]: r for r in registry["runs"]}
    pending = next(p for p in registry["pending_batches"] if p["id"] == run_id)
    if pending["logical_id"] != "xr21-t2-report":
        raise ValueError("Expected the existing logical report job")
    dependencies = [runs[name] for name in pending["depends_on"]]
    evaluation_run = next(r for r in dependencies if r["logical_id"] == "xr21-t2-evaluation")
    controls_run = next(r for r in dependencies if r["logical_id"] == "xr21-t2-controls")
    preparation_run = next(
        r
        for r in runs.values()
        if r["logical_id"] == "xr21-t2-prepare"
        and not r.get("superseded_by")
        and r.get("enabled", True)
    )
    evaluation = output_binding(
        evaluation_run, evaluation_run.get("evaluation_report_relative", "evaluation-report.json")
    )
    controls = output_binding(controls_run, "control-report.json")
    evaluated = checked(evaluation)
    schedule = evaluated["schedule"]
    schedule_value = checked(schedule)
    checked_evaluation_rows(schedule_value, evaluated)
    snapshot = output.with_name("registry-input.json")
    write_artifact(snapshot, registry)
    manifest = {
        "schema": SCHEMA,
        "run_id": run_id,
        "campaign": str(campaign),
        "plan": binding(campaign / "plan.json"),
        "registry": binding(snapshot),
        "evaluation": evaluation,
        "evaluation_schedule": schedule,
        "controls": controls,
        "controls_schedule": binding(campaign / "artifacts/controls-001/schedule.json"),
        "preparation_gate": output_binding(preparation_run, "preparation-gate.json"),
        "budget_lineage": binding(campaign / "budget-lineage.json"),
        "dependencies": dependencies,
        "planned_arm_ids": [a["id"] for a in plan["arms"]],
        "worker_seconds": plan["budgets"]["report"],
        "deferred_scope": DEFERRED,
        "historical_scope_completions": registry.get("historical_scope_completions", []),
    }
    files = {
        value["path"]
        for value in manifest.values()
        if isinstance(value, dict) and "sha256" in value
    }
    files.update(
        item["path"]
        for item in schedule_value.get("reuse", {}).values()
        if isinstance(item, dict) and "sha256" in item
    )
    for run in runs.values():
        check_receipt(run, success=False)
        files.add(run["completion_path"])
    for run in dependencies:
        outputs_path = Path(run["completion_path"]).with_name("outputs.json")
        files.add(str(outputs_path))
        outputs = read(outputs_path)
        # Bind all actual row/proof/label/cache evidence through the worker's manifest.
        for relative, digest in outputs.items():
            path = Path(read(run["completion_path"])["work"]) / relative
            if sha(path) != digest:
                raise ValueError("Completed output changed: " + str(path))
            files.add(str(path))
    for arm in checked(schedule)["arms"]:
        for value in arm.values():
            if isinstance(value, dict) and "sha256" in value:
                if sha(value["path"]) != value["sha256"]:
                    raise ValueError("Frozen model/terminal evidence changed")
                files.add(value["path"])
    for item in read(campaign / "budget-lineage.json")["inherited"].values():
        if sha(item["snapshot"]) != item["sha256"]:
            raise ValueError("Historical snapshot changed")
        files.add(item["snapshot"])
    for previous in manifest["historical_scope_completions"]:
        completion = checked(previous["completion"])
        checked(previous["costs"])
        checked(completion["report"])
        files.update(
            item["path"]
            for item in (previous["completion"], previous["costs"], completion["report"])
        )
    manifest["inputs"] = [binding(p) for p in sorted(files)]
    costs(campaign)
    write_artifact(output, manifest)
    return manifest


def matched_rows(schedule_rows, rows, *, key):
    planned = [key(row) for row in schedule_rows]
    actual = [key(row) for row in rows]
    if (
        len(set(planned)) != len(planned)
        or len(set(actual)) != len(actual)
        or set(planned) != set(actual)
    ):
        raise ValueError("Missing, duplicate or unscheduled result rows")
    by_id = {key(row): row for row in rows}
    return [by_id[value] for value in planned]


def checked_evaluation_rows(schedule, evaluation):
    rows = matched_rows(schedule["rows"], evaluation["rows"], key=lambda r: r["id"])
    if schedule.get("reuse"):
        from tools.repair.evaluation_addendum import checked_reuse

        reused = checked_reuse(schedule)
        if (
            evaluation.get("reuse") != schedule["reuse"]
            or evaluation.get("reused_rows") != len(reused)
            or evaluation.get("new_rows") != len(rows) - len(reused)
        ):
            raise ValueError("Merged evaluation reuse accounting changed")
        identity = canonical_hash(schedule)
        for row in rows:
            if row["id"] in reused:
                if row != reused[row["id"]]:
                    raise ValueError("Merged evaluation changed an original reused row")
            elif row["schedule_hash"] != identity:
                raise ValueError("New evaluation row has another schedule identity")
    return rows


def summary(rows):
    known = [r.get("measurement", r).get("semantic_benefit") for r in rows]
    observed = [v for v in known if v is not None]
    return {
        "scheduled": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "logical_statuses": dict(Counter(r.get("logical_status", "UNKNOWN") for r in rows)),
        "known_semantic_count": len(observed),
        "semantic_mean_all_scheduled": (
            sum(observed) / len(rows) if rows and len(observed) == len(rows) else None
        ),
        "semantic_mean_known_subset": sum(observed) / len(observed) if observed else None,
    }


def report(manifest_path, output):
    if (Path(output) / "report.json").exists():
        raise FileExistsError("Preserve existing report output; use a new revision directory")
    manifest = read(manifest_path)
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported report manifest")
    for item in manifest["inputs"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Frozen reporting input changed: " + item["path"])
    registry, plan = checked(manifest["registry"]), checked(manifest["plan"])
    schedule, evaluation = checked(manifest["evaluation_schedule"]), checked(manifest["evaluation"])
    controls, control_schedule = checked(manifest["controls"]), checked(
        manifest["controls_schedule"]
    )
    for run in manifest["dependencies"]:
        check_receipt(run)
    if controls["schedule_sha256"] != manifest["controls_schedule"]["sha256"]:
        raise ValueError("Control schedule identity mismatch")
    rows = checked_evaluation_rows(schedule, evaluation)
    detailed = []
    for planned, row in zip(schedule["rows"], rows):
        if any(row.get(k) != v for k, v in planned.items()):
            raise ValueError("Evaluation row differs from frozen schedule")
        value = dict(row)
        if row.get("artifact"):
            measurement = checked(row["artifact"])
            if (
                measurement["row_id"] != row["id"]
                or measurement["schedule_hash"] != row["schedule_hash"]
            ):
                raise ValueError("Evaluation artifact identity mismatch")
            if measurement["status"] != row["status"] or measurement.get(
                "logical_status", "UNKNOWN"
            ) != row.get("logical_status", "UNKNOWN"):
                raise ValueError("Evaluation artifact status mismatch")
            value["measurement"] = measurement
        detailed.append(value)
    planned_controls = [
        {"case_id": c["case_id"], "arm_id": a["arm_id"]}
        for c in control_schedule["cases"]
        for a in control_schedule["arms"]
    ]
    control_rows = matched_rows(
        planned_controls, controls["rows"], key=lambda r: (r["case_id"], r["arm_id"])
    )
    arm_ids = [a["id"] for a in schedule["arms"]]
    if arm_ids != manifest["planned_arm_ids"] or len(arm_ids) != 6:
        raise ValueError("Planned six-arm denominator changed")
    primary = [r for r in detailed if r["kind"] == "generated_pool"]
    cases = {c["case"]["case_id"]: c["case"] for c in schedule["cases"]}
    arms = []
    for arm in schedule["arms"]:
        run = next(r for r in registry["runs"] if r["id"] == arm["run_id"])
        receipt = check_receipt(run, success=arm["status"] == "available")
        selected = [r for r in primary if r["arm_id"] == arm["id"]]
        if len(selected) != plan["case_counts"]["test"] or {r["case_id"] for r in selected} != set(
            cases
        ):
            raise ValueError("Model/test denominator changed")
        training = checked(arm["training_report"]) if arm["status"] == "available" else None
        arms.append(
            {
                "id": arm["id"],
                "run_id": arm["run_id"],
                "availability": arm["status"],
                "process_status": receipt["status"],
                "training_status": training["status"] if training else "unavailable",
                "epochs": training["epochs"] if training else None,
                "selected_epoch": arm.get("selected_epoch"),
                "verification": summary(selected),
                "scientific_result_status": "not_established",
                "provenance": arm,
                "recovery_lineage": [r for r in registry["runs"] if r["logical_id"] == arm["id"]],
            }
        )
    circuits = [r for r in detailed if r["kind"] == "circuit"]
    report_value = {
        "schema": SCHEMA,
        "manifest": binding(manifest_path),
        "historical_scope_completions": manifest.get("historical_scope_completions", []),
        "evaluation_provenance": {
            "report": manifest["evaluation"],
            "schedule": manifest["evaluation_schedule"],
            "reuse": evaluation.get("reuse"),
            "reused_rows": evaluation.get("reused_rows", 0),
            "new_rows": evaluation.get("new_rows", len(rows)),
        },
        "process_status": "report_materialized_pending_worker_receipt",
        "finite_scope_status": "accounted_pending_report_receipt_and_final_costs",
        "study_status": "incomplete_with_unavailable_and_deferred_rows",
        "scientific_result_status": "exploratory_not_established",
        "gates": {k: "not_established" for k in ("G0", "G1", "G2")},
        "xr_studies": {f"XR-E{i:02}": "not_complete" for i in range(10)},
        "arms": arms,
        "planned_model_arm_count": len(arms),
        "primary_generated_pool": {
            **summary(primary),
            "rows": primary,
            "by_parent": {
                parent: summary(
                    [r for r in primary if cases[r["case_id"]]["structural_parent"] == parent]
                )
                for parent in sorted({c["structural_parent"] for c in cases.values()})
            },
        },
        "circuit_diagnostic": {
            **summary(circuits),
            "rows": circuits,
            "scope": schedule["circuit_scope"],
            "decoder": schedule["decoder"],
        },
        "common_inventory_diagnostic": {
            **summary(control_rows),
            "rows": control_rows,
            "scope": "diagnostic only; not primary generated-pool measure; no strongest-symbolic claim",
        },
        "preparation_gate": checked(manifest["preparation_gate"]),
        "source_preparation": schedule["source_preparation"],
        "original_cache_hashes": schedule["original_cache_hashes"],
        "original_label_seconds": schedule["original_label_seconds"],
        "original_label_cpu_seconds": schedule["original_label_cpu_seconds"],
        "operational_attempts": [
            {"run": r, "completion": read(r["completion_path"])} for r in registry["runs"]
        ],
        "costs_at_report": costs(manifest["campaign"]),
        "deferred_scope": manifest["deferred_scope"],
        "claims": schedule["claims"],
        "production_matcher": "deferred",
        "llm_annotations": "disabled",
        "external_api_cost_usd": 0,
        "model_selection": "frozen before held-out evaluation; no refitting or selection by this report",
        "measurement_interpretation": "first_verified_seconds is selector-local; end-to-end row resources include startup, generation and independent labels; no matched-quality or learned-efficiency claim",
        "statistical_inference": "not performed; exploratory smoke has two held-out parents and one seed",
    }
    write_artifact(Path(output) / "report.json", report_value)
    return report_value


def close(campaign, output, run_id="xr21-t2-report-001"):
    """Publish final costs only after the runner settled its own report attempt."""
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    with locked(campaign / "supervisor/registry.json.lock"):
        if any((output / name).exists() for name in ("completion.json", "costs.json")):
            raise FileExistsError("Preserve existing scope closure; use a new revision directory")
        registry = read(campaign / "supervisor/registry.json")
        run = next(r for r in registry["runs"] if r["id"] == run_id)
        receipt = check_receipt(run)
        report_ref = output_binding(run, run.get("report_relative", "report.json"))
        payload = checked(report_ref)
        if payload["planned_model_arm_count"] != 6 or len(payload["arms"]) != 6:
            raise ValueError("Incomplete report arm accounting")
        if registry["pending_batches"]:
            raise ValueError("Cannot close with pending batches")
        for other in registry["runs"]:
            if other.get("enabled", True) and not other.get("superseded_by"):
                check_receipt(other)
        final_costs = costs(campaign)
        if final_costs["combined"]["reserved_worker_seconds"]:
            raise ValueError("Cannot close with unsettled reservations")
        attempt = str(Path(run["completion_path"]).parent)
        charged = final_costs["attempts"].get(attempt)
        if not charged or not math.isclose(charged["elapsed_seconds"], receipt["elapsed_seconds"]):
            raise ValueError("Report worker receipt not settled in cumulative ledger")
        write_artifact(output / "costs.json", final_costs)
        completion = {
            "schema": "exact-repair/finite-scope-completion/v3",
            "finite_scope_status": "accounted_with_unavailable_and_deferred",
            "process_status": "complete",
            "study_status": payload["study_status"],
            "scientific_result_status": payload["scientific_result_status"],
            "source_commit": read(receipt["batch"])["commit"] if receipt.get("batch") else None,
            "report": report_ref,
            "historical_scope_completions": payload.get("historical_scope_completions", []),
            "costs": binding(output / "costs.json"),
            "completion_receipt": binding(run["completion_path"]),
            "planned_model_arm_count": 6,
            "available_model_arms": sum(a["availability"] == "available" for a in payload["arms"]),
            "unavailable_model_arms": sum(
                a["availability"] == "unavailable" for a in payload["arms"]
            ),
            "denominators": {
                k: payload[k]["scheduled"]
                for k in [
                    "primary_generated_pool",
                    "circuit_diagnostic",
                    "common_inventory_diagnostic",
                ]
            },
            "deferred_scope": payload["deferred_scope"],
            "monitoring": "idle_no_eligible_work",
            "remaining_work_status": "terminal",
        }
        write_artifact(output / "completion.json", completion)
        registry.update(
            remaining_work_status="terminal",
            scope_completion=binding(output / "completion.json"),
            cumulative_costs=binding(output / "costs.json"),
            deferred_scope=payload["deferred_scope"],
            monitoring_status="idle_no_eligible_work",
        )
        write_artifact(campaign / "supervisor/registry.json", registry)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "close"])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", default="xr21-t2-report-001")
    args = parser.parse_args()
    kwargs = {} if args.command == "run" else {"run_id": args.run_id}
    result = {"prepare": prepare, "run": report, "close": close}[args.command](
        args.source, args.output, **kwargs
    )
    print(json.dumps({"schema": result["schema"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
