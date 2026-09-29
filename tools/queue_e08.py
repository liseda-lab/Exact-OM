#!/usr/bin/env python3
"""Measure every frozen E08 treatment before admitting the complete comparison."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import subprocess
import tempfile
from pathlib import Path

from tools.measured_once import (
    promote_measured_cells,
    require_reuse_only,
    reused_comparison_forecast,
)
from tools.qualify_cached_family import (
    binding,
    memory_limit_bytes,
    run_probe,
    run_worker,
    usage,
)
from tools.resume_cached_batch import (
    FAILED_SETUP_SECONDS,
    RECONCILIATION_ID,
    account_inventory,
    campaign_limits,
    check_admission,
    comparison_forecast,
    load_helpers,
    read,
    verify_nested_bindings,
)

FAMILY = "E08"
ARMS = ("current", "unified_bank", "provenance_dedup")
PROJECT = Path("/home/pgcotovio/Exact-OM")
DATA = PROJECT / "data/experiments-v2"
PARENT = DATA / "e01-cardinality-02/E01"
BUDGET = PARENT / "runtime/exact-om-focused-v2/budget.json"
CONTROL = DATA / "d1-production-probe-01/d0-control.json"


def no_previous_owner():
    """Allow only the retained interactive shell, supervisor, and this numeric step."""
    own = "14372." + os.environ.get("SLURM_STEP_ID", "")
    steps = subprocess.check_output(
        ["squeue", "--steps", "-h", "-j", "14372", "-o", "%i"], text=True
    ).splitlines()
    unexpected = set(map(str.strip, steps)) - {"14372.0", "14372.16", "14372.extern", own}
    if unexpected:
        raise ValueError("An earlier numeric worker remains live: " + repr(sorted(unexpected)))
    import psutil

    prior_roots = ("/e01-cardinality-02/", "/e10-finalization-01/", "/next-batch-recovery-04/")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info["name"].startswith(("python", "srun")) and any(
            any(root in arg for root in prior_roots) for arg in (proc.info["cmdline"] or [])
        ):
            raise ValueError("A previous owner or descendant remains: " + str(proc.pid))


def validate_budget(state, lock):
    if state["limits"] != campaign_limits(lock):
        raise ValueError("Cumulative caps or protected final allowance changed")
    if any(k != "historical/G0" and v["status"] == "reserved" for k, v in state["work"].items()):
        raise ValueError("Previous owner has unclosed accounting")


def ready_lock(saved, estimate, commit):
    """Change operational readiness only, preserving the frozen E08 scientific design."""
    lock = copy.deepcopy(saved)
    step = next(row for row in lock["steps"] if row["id"] == FAMILY)
    if [a["id"] for a in step["arms"]] != list(ARMS) or step["inherits"] != ["E05_initial"]:
        raise ValueError("E08 comparison or inherited policy changed")
    step["estimate"] = estimate
    for states in step["readiness"].values():
        states["screen"].update(
            status="screen_ready",
            inspected_commit=commit,
            missing=[],
            implemented_paths=[
                "exact/experiments/core_recipes.py",
                "exact/impl/models/pair_adaptive_channels.py",
            ],
            tests=[
                "tests/campaign_preparation_test.py",
                "tests/cached_family_qualification_test.py",
                "tests/e08_queue_test.py",
            ],
            reason="All three matched frozen E08 treatments require measured full-arm costs before scientific admission; configuration previews do not admit work.",
        )
    return lock


def cells_for(campaign, directory, selection):
    from exact.experiments import harness
    from exact.experiments.campaign import (
        campaign_plan,
        materialize_campaign,
        validate_comparison_cells,
    )

    plan = campaign_plan(campaign, stage="screen")
    rows = [row for row in plan["rows"] if row["step"] == FAMILY]
    if plan["budget_errors"] or len(rows) != 3 or any(row["issues"] for row in rows):
        raise ValueError("Complete E08 plan is not valid: " + repr(rows))
    ready = {r["step"] for r in plan["rows"] if r["status"] == "screen_ready" and not r["issues"]}
    if ready != {FAMILY}:
        raise ValueError("Only E08 may be dispatched: " + repr(ready))
    suite = materialize_campaign(campaign, directory / "declarations", stage="screen")
    source = next(row for row in suite.sources if row.config.experiment_id == FAMILY)
    cells = harness.build_cells(
        suite,
        source,
        stage="screen",
        output_root=directory / "runtime",
        inherited_overlay=harness.inherited_selection_overlay(selection, source.config.depends_on),
    )
    validate_comparison_cells(cells, source)
    validate_cells(cells)
    return cells


def validate_cells(cells):
    if len(cells) != 3 or {c.arm_id for c in cells} != set(ARMS):
        raise ValueError("All three E08 arms are mandatory")
    for cell in cells:
        c = cell.resolved_config
        if (
            cell.generate_rationales
            or cell.split_role != "development"
            or cell.seed != 17
            or cell.source_cap != 300
            or set(c["data"]["refs"]) - {"train", "valid"}
            or c["llm"]["experiment"]["gate"]["mode"] != "off"
            or c["matching"]["extraction"]["mode"] != "greedy"
            or c["supervision"]["mode"] != "label_free"
            or any(x != "label_free" for x in c["supervision"]["components"].values())
        ):
            raise ValueError("E08 reference, supervision, extraction or hosted policy changed")


def matched_measurements(rows):
    """Require complete treatment coverage and equal unlabeled candidate populations."""
    if len(rows) != 3 or {r["name"] for r in rows} != set(ARMS):
        raise ValueError("Matched measurements must cover every E08 treatment")
    pools = []
    for row in rows:
        if (
            row["status"] != "passed"
            or row["execution_status"] != "complete"
            or row["prefix"]
            or any(row["new_usage"].values())
            or row["processed_pairs"] != row["dataset_rows"]
        ):
            raise ValueError("Every matched measurement must be complete and cached-only")
        pool = Path(row["output_dir"]) / "dataset/candidate_pool_sample_manifest.json"
        value = read(pool)
        pools.append(
            {"gold_free_summary": value["gold_free_summary"], "per_kind": value["per_kind"]}
        )
    if any(p != pools[0] for p in pools[1:]):
        raise ValueError("E08 treatments changed frozen candidate populations")
    return pools[0]


def preflight(args, h):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import (
        CampaignLock,
        external_acceptance_selection,
        external_selection_result,
    )

    h.check_pause()
    no_previous_owner()
    identity = h.validate_source(args.code_root, allow_working_tree=args.allow_working_tree)
    lock = copy.deepcopy(load_yaml_mapping(PARENT / "campaign.lock.yaml"))
    complete = read(DATA / "e01-cardinality-02/completion.json")
    if complete.get("status") != "complete" or complete.get("exit_code") != 0:
        raise ValueError("Latest E01 owner is incomplete")
    for name in ("selection", "cumulative_budget"):
        h.verify(complete[name])
    h.history_record(
        lock, "E01", PARENT / "campaign.lock.yaml", BUDGET.parent, args.root / "history"
    )
    h.history_record(
        lock,
        "E10-analytic",
        DATA / "next-batch-recovery-04/E10-analytic/campaign.lock.yaml",
        DATA / "next-batch-recovery-04/E10-analytic/runtime/exact-om-focused-v2",
        args.root / "history",
    )
    parsed = CampaignLock.model_validate(lock)
    imports = {}
    for step in parsed.steps:
        if step.external_selection:
            imports[step.id] = external_selection_result(parsed, step, args.root)["status"]
        elif step.external_acceptance:
            imports[step.id] = external_acceptance_selection(parsed, step, args.root)["status"]
    state = read(BUDGET)
    validate_budget(state, lock)
    inventory = state["work"].get(RECONCILIATION_ID, {})
    if inventory.get("status") != "failed" or inventory.get("seconds") != FAILED_SETUP_SECONDS:
        raise ValueError("Retained standalone native inventory cost evidence is missing")
    checks = read(args.root / "checks.json")
    if checks.get("status") != "passed" or checks.get("pytest", {}).get("exit_code") != 0:
        raise ValueError("Successful focused source checks are required")
    for relative, item in checks["source_files"].items():
        if binding(args.code_root / relative)["sha256"] != item["sha256"]:
            raise ValueError("Checked source bytes changed: " + relative)
    sources = read(args.root / "cache-sources.json")
    for item in sources["bindings"]:
        h.verify(item)
    if binding(args.root / "cache-source/budget.json") != binding(BUDGET):
        raise ValueError("Cache source view must retain the latest E01 budget")
    old = read(CONTROL)
    if old["status"] != "passed":
        raise ValueError("Retained full D0 pilot allowance evidence missing")
    oldrow = dict(old["measurement"], prefix=False)
    verify_nested_bindings(oldrow, h)
    allowance = 1.5 * oldrow["wall_seconds"]
    with tempfile.TemporaryDirectory(prefix="e08-preflight-") as tmp:
        directory = Path(tmp)
        check_admission(
            state, "reserve", 1800 + 3 * allowance, directory, "planned/E08-qualification"
        )
        trial = ready_lock(lock, h.estimate([oldrow], 3, CONTROL), identity["commit"])
        path = directory / "campaign.yaml"
        h.freeze(path, trial)
        cells = cells_for(path, directory, read(h.verify(complete["selection"])))
        for cell in cells:
            h.freeze(
                args.root / "qualification" / (cell.arm_id + ".config.yaml"), cell.resolved_config
            )
    h.freeze(args.root / "prospective-campaign.yaml", lock)
    records = [
        args.root / name
        for name in ("reuse-plan.json", "repair.json", "checks.json", "cache-sources.json")
    ]
    operational = [
        args.code_root / "tools/queue_e08.py",
        args.code_root / "tools/qualify_cached_family.py",
        args.code_root / "tools/resume_cached_batch.py",
        args.code_root / "tools/run_experiment_validation.py",
        DATA / "d1-production-probe-01/launch.json",
    ]
    operational += [
        args.root / n for n in ("continue.py", "submit.sh", "run-step.sh", "worker-entry.sh")
    ]
    operational += [
        DATA / "next-batch-recovery-02" / n for n in ("continue.py", "qualification.py")
    ]
    h.freeze(
        args.root / "preflight.json",
        {
            "status": (
                "preview_only" if args.allow_working_tree else "preflight_passed_not_admitted"
            ),
            "source": identity,
            "budget_import": binding(BUDGET),
            "parent_completion": binding(DATA / "e01-cardinality-02/completion.json"),
            "prospective_campaign": binding(args.root / "prospective-campaign.yaml"),
            "source_selection": complete["selection"],
            "historical_decisions_verified": imports,
            "pilot_allowance_seconds_per_arm": allowance,
            "pilot_scope": "Qualification only; no scientific comparison admission from baseline cost. Forecast is not a timeout.",
            "pilot_cost_basis": binding(CONTROL),
            "inventory_cost_basis": {
                "budget": binding(BUDGET),
                "work_id": RECONCILIATION_ID,
                "seconds": FAILED_SETUP_SECONDS,
                "safety_factor": 1.5,
            },
            "queue": list(ARMS),
            "configs": {
                a: binding(args.root / "qualification" / (a + ".config.yaml")) for a in ARMS
            },
            "records": [binding(p) for p in records],
            "operational_files": [binding(p) for p in operational],
            "budget_limits_changed": False,
            "generate_rationales": False,
        },
    )
    print("E08 matched qualification preflight passed; comparison not yet admitted", flush=True)


def run(args, h):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import (
        CampaignLock,
        execute_campaign,
        external_selection_result,
    )

    h.check_pause()
    if (
        os.environ.get("SLURM_JOB_ID") != "14372"
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise ValueError("Use a detached numeric step in allocation 14372")
    no_previous_owner()
    receipt = read(args.root / "preflight.json")
    if (
        receipt["status"] != "preflight_passed_not_admitted"
        or h.validate_source(args.code_root) != receipt["source"]
    ):
        raise ValueError("Clean frozen source preflight required")
    for key in (
        "budget_import",
        "parent_completion",
        "prospective_campaign",
        "source_selection",
        "pilot_cost_basis",
    ):
        h.verify(receipt[key])
    for item in [*receipt["records"], *receipt["operational_files"], *receipt["configs"].values()]:
        h.verify(item)
    if Path(receipt["budget_import"]["path"]) != BUDGET:
        raise ValueError("Stale cumulative budget import")
    lock = load_yaml_mapping(h.verify(receipt["prospective_campaign"]))
    validate_budget(read(BUDGET), lock)
    current = args.root / "qualification/runtime/exact-om-focused-v2"
    if current.exists() or (args.root / "E08/runtime").exists():
        raise FileExistsError(
            "Existing runtime requires recorded checkpoint recovery, not top-level replay"
        )
    retained = read(DATA / "d1-production-probe-01/launch.json")
    os.environ.update(retained["environment"])
    os.environ["PYTHONPATH"] = str(args.code_root)
    # The established launcher credential is consumed internally and never logged.
    os.environ["OPENROUTER_API_KEY"] = (PROJECT / "api_key").read_text().strip()
    h.status("preparing_qualification", cumulative_budget=str(current / "budget.json"))
    sources = read(args.root / "cache-sources.json")
    for item in sources["bindings"]:
        h.verify(item)
    if binding(args.root / "cache-source/budget.json") != receipt["budget_import"]:
        raise ValueError("Cache view imported a stale budget")
    h.copy_state(args.root / "cache-source", current)
    cached = usage(current / "openrouter")
    rows = []
    for arm in ARMS:
        h.check_pause()
        h.status("measuring_E08", arm=arm, cumulative_budget=str(current / "budget.json"))
        rows.append(
            run_probe(
                code_root=args.code_root,
                script=args.root / "continue.py",
                config=h.verify(receipt["configs"][arm]),
                name=arm,
                directory=args.root / "qualification" / arm,
                shared=current,
                campaign=CampaignLock.model_validate(lock),
                forecast_seconds=receipt["pilot_allowance_seconds_per_arm"],
            )
        )
    pool = matched_measurements(rows)
    if usage(current / "openrouter") != cached:
        raise ValueError("Qualification incurred incremental hosted usage")
    wave = args.root / FAMILY
    measurement = wave / "measured-cost.json"
    h.freeze(
        measurement,
        {
            "status": "measured_forecast",
            "rows": rows,
            "candidate_population": pool,
            "method": "Maximum complete matched treatment duration for each of three arms, setup once per arm, factor 1.5; standalone native inventory once. No cache-speedup discount.",
        },
    )
    proof = comparison_forecast(rows, 3, measurement, h)
    proof["inventory_cost_basis"] = receipt["inventory_cost_basis"]
    proof = reused_comparison_forecast(proof)
    h.freeze(wave / "forecast.json", proof)
    with tempfile.TemporaryDirectory(prefix="e08-admission-") as tmp:
        from exact.experiments.budget import BudgetLedger

        check_path = check_admission(
            read(current / "budget.json"),
            "channels",
            proof["whole_family_seconds"],
            Path(tmp),
            "planned/E08",
        )
        BudgetLedger(check_path, read(check_path)["limits"]).admit(
            "planned/E08-cache-copy", group="reserve", seconds=1800
        )
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    if (
        len(free) != 1
        or proof["estimate"]["peak_ram_gb"] > memory_limit_bytes() / 1024**3
        or proof["estimate"]["peak_vram_gb"] > float(free[0]) / 1024
    ):
        raise ValueError("Measured family memory does not fit this node")
    campaign = wave / "campaign.lock.yaml"
    h.freeze(campaign, ready_lock(lock, proof["estimate"], receipt["source"]["commit"]))
    with tempfile.TemporaryDirectory(prefix="e08-cells-") as tmp:
        cells = cells_for(campaign, Path(tmp), read(h.verify(receipt["source_selection"])))
        if any(
            binding(h.verify(receipt["configs"][c.arm_id]))["sha256"]
            != binding_config(c.resolved_config)
            for c in cells
        ):
            raise ValueError("Measured configuration differs from scientific comparison")
    h.check_pause()
    destination = wave / "runtime/exact-om-focused-v2"
    h.copy_state(current, destination)
    current = destination
    promote_measured_cells(
        campaign, current, rows, FAMILY, read(h.verify(receipt["source_selection"])), args.code_root
    )
    cached = usage(current / "openrouter")
    os.environ.update(
        EXACT_OPENROUTER_REQUEST_CAP=str(cached["attempts"]),
        EXACT_OPENROUTER_TOKEN_CAP=str(cached["billable_tokens"]),
        EXACT_OPENROUTER_RETRY_UNKNOWN="0",
    )
    h.freeze(
        wave / "launch.json",
        {
            "campaign": binding(campaign),
            "source": receipt["source"],
            "budget_path": str(current / "budget.json"),
            "step_id": "14372." + os.environ["SLURM_STEP_ID"],
        },
    )
    h.status(
        "running",
        step=FAMILY,
        cumulative_budget=str(current / "budget.json"),
        campaign=str(campaign),
    )
    execute_campaign(
        campaign,
        stage="screen",
        output_root=wave / "runtime",
        workdir=args.code_root,
        jobs=1,
        reuse_plan_only=True,
    )
    h.check_pause()
    with account_inventory(h, current, FAMILY, "channels", proof["inventory_seconds"]):
        with require_reuse_only():
            h.guarded_execute(campaign, wave, args.code_root)
    if usage(current / "openrouter") != cached:
        raise ValueError("Comparison incurred incremental hosted usage")
    h.history_record(lock, FAMILY, campaign, current, args.root / "history")
    parsed = CampaignLock.model_validate(lock)
    decision = external_selection_result(
        parsed, next(s for s in parsed.steps if s.id == FAMILY), args.root
    )
    h.freeze(
        args.root / "completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "cells": 3,
            "selection": binding(current / "screen/selection.json"),
            "decision": decision["status"],
            "cumulative_budget": binding(current / "budget.json"),
            "source": receipt["source"],
            "cached_usage_unchanged": True,
            "generate_rationales": False,
        },
    )
    h.status("complete", cells=3, cumulative_budget=str(current / "budget.json"))
    return 0


def binding_config(config):
    import hashlib

    from exact.core.entities.configs.yaml_io import dump_yaml_document

    return hashlib.sha256(dump_yaml_document(config).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--allow-working-tree", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        return run_worker(args)
    h = load_helpers(args.root)
    if args.preflight:
        preflight(args, h)
        return 0
    with (args.root / "continuation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(args, h)
        except BaseException as exc:
            latest = next(
                (
                    p
                    for p in [
                        args.root / "E08/runtime/exact-om-focused-v2/budget.json",
                        args.root / "qualification/runtime/exact-om-focused-v2/budget.json",
                    ]
                    if p.exists()
                ),
                BUDGET,
            )
            h.status(
                "blocked",
                cumulative_budget=str(latest),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
