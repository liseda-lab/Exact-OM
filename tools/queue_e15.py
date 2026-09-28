#!/usr/bin/env python3
"""Measure every frozen E15 treatment before admitting the complete comparison."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import subprocess
import tempfile
from pathlib import Path

from tools.qualify_cached_family import binding, run_probe, run_worker, usage
from tools.resume_cached_batch import (
    account_inventory,
    campaign_limits,
    check_admission,
    load_helpers,
    read,
    verify_nested_bindings,
)

FAMILY = "E15"
ARMS = ("analytic_fixed", "distribution_margin", "reciprocal_consensus", "current_supervised")
MODES = dict(
    zip(ARMS, ("current_fallback", "score_partition", "reciprocal_consensus", "current_fallback"))
)
CASES = ("D0_E03",)
KEYS = tuple(case + "--" + arm for case in CASES for arm in ARMS)
PROJECT = Path("/home/pgcotovio/Exact-OM")
DATA = PROJECT / "data/experiments-v2"
PARENT = DATA / "e03-qualification-01/E03"
BUDGET = PARENT / "runtime/exact-om-focused-v2/budget.json"
CONTROL = DATA / "d1-production-probe-01/d0-control.json"


def no_previous_owner(root, supervisor_step):
    """Allow only the retained interactive shell, supervisor, and this numeric step."""
    if supervisor_step != "14372.23":
        raise ValueError("This preparation binds the observed supervisor step 14372.23")
    own = "14372." + os.environ.get("SLURM_STEP_ID", "")
    steps = subprocess.check_output(
        ["squeue", "--steps", "-h", "-j", "14372", "-o", "%i"], text=True
    ).splitlines()
    unexpected = set(map(str.strip, steps)) - {"14372.0", supervisor_step, "14372.extern", own}
    if unexpected:
        raise ValueError("An earlier numeric worker remains live: " + repr(sorted(unexpected)))
    import psutil

    registry = read(DATA / "hourly-supervisor-01/registry.json")
    prior_roots = {str(Path(r["status_path"]).parent) + "/" for r in registry["runs"]}
    prior_roots.discard(str(root) + "/")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if (proc.info["name"] or "").startswith(("python", "srun")) and any(
            any(path in arg for path in prior_roots) for arg in (proc.info["cmdline"] or [])
        ):
            raise ValueError("A previous owner or descendant remains: " + str(proc.pid))


def validate_budget(state, lock):
    if state["limits"] != campaign_limits(lock):
        raise ValueError("Cumulative caps or protected final allowance changed")
    if any(k != "historical/G0" and v["status"] == "reserved" for k, v in state["work"].items()):
        raise ValueError("Previous owner has unclosed accounting")


def prepare_steps(saved, training):
    """Bind the existing bounded D0 alias without changing any E15 treatment."""
    lock = copy.deepcopy(saved)
    step = next(row for row in lock["steps"] if row["id"] == FAMILY)
    if [a["id"] for a in step["arms"]] != list(ARMS):
        raise ValueError("E15 requires all three label-free arms and the supervised control")
    if step["case"] != "D0" or step["additional_cases"] or step["inherits"] != ["E05_initial"]:
        raise ValueError("E15 case or inheritance changed")
    if "E03" not in step["requires"]:
        raise ValueError("E15 must follow completed E03 selection")
    case = lock["cases"]["D0_E03"]
    for key, field in (("candidates", "pool"), ("references", "reference")):
        if case[key]["train"] != {"path": training[field], "sha256": training[field + "_sha256"]}:
            raise ValueError("Previously bounded training binding changed")
    step["case"] = "D0_E03"
    return lock


def ready_lock(saved, estimate, commit):
    lock = copy.deepcopy(saved)
    step = next(row for row in lock["steps"] if row["id"] == FAMILY)
    if [a["id"] for a in step["arms"]] != list(ARMS):
        raise ValueError("All four E15 arms are required")
    step["estimate"] = estimate
    for states in step["readiness"].values():
        states["screen"].update(
            status="screen_ready",
            inspected_commit=commit,
            missing=[],
            implemented_paths=[
                "exact/impl/trainer/fitting.py",
                "exact/impl/models/selector/fitting.py",
            ],
            tests=["tests/grouped_fitting_test.py", "tests/e15_queue_test.py"],
            reason="Four full matched selector probes including bounded disjoint training precede whole-comparison admission; no fitted artifacts are invented.",
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
    if plan["budget_errors"] or len(rows) != len(ARMS) or any(row["issues"] for row in rows):
        raise ValueError("Complete E15 plan is not valid: " + repr(rows))
    ready = {r["step"] for r in plan["rows"] if r["status"] == "screen_ready" and not r["issues"]}
    if ready != {FAMILY}:
        raise ValueError("Only E15 may be dispatched: " + repr(ready))
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
    if len(cells) != 4 or {(c.task_id, c.arm_id) for c in cells} != {
        (case + "-global_alignment", arm) for case in CASES for arm in ARMS
    }:
        raise ValueError("All four E15 case/arm cells are mandatory")
    components = {
        "retrieval",
        "fusion",
        "rerank",
        "llm",
        "accept",
        "calibration",
        "structure",
        "relation",
    }
    for cell in cells:
        c = cell.resolved_config
        expected = {
            key: (
                "supervised"
                if cell.arm_id == "current_supervised" and key in {"rerank", "accept"}
                else "label_free"
            )
            for key in components
        }
        selector = c["selector"]
        if (
            cell.generate_rationales
            or cell.split_role != "development"
            or cell.seed != 17
            or cell.source_cap != 300
            or set(c["data"]["refs"]) != {"train", "valid"}
            or c["llm"]["experiment"]["gate"]["mode"] != "off"
            or c["matching"]["extraction"]["mode"] != "greedy"
            or c["matching"]["threshold"] != 0.7
            or c["matching"]["calibration"]["mode"] != "none"
            or c["matching"]["calibration"]["threshold_mode"] != "fixed"
            or c["supervision"]["mode"] != "label_free"
            or c["supervision"]["components"] != expected
            or c["supervision"]["negative_label_policy"] != "confirmed_negatives"
            or not c["data"]["train_candidates"]
            or not selector["enabled"]
            or not selector["runtime_enabled"]
            or selector["runtime_global_only"]
            or selector["label_free_mode"] != MODES[cell.arm_id]
        ):
            raise ValueError("E15 scientific, supervision, or hosted policy changed")


def matched_measurements(rows):
    """Require complete case/treatment coverage and equal pools within each case."""
    if len(rows) != 4 or {r["name"] for r in rows} != set(KEYS):
        raise ValueError("Matched measurements must cover every E15 case and treatment")
    pools = {}
    for row in rows:
        case = row["case_id"]
        if (
            case not in CASES
            or not row["name"].startswith(case + "--")
            or row["status"] != "passed"
            or row["execution_status"] != "complete"
            or row["prefix"]
            or any(row["new_usage"].values())
            or not 0 < row["processed_pairs"] == row["dataset_rows"] <= 6000
        ):
            raise ValueError("Every matched measurement must be complete and cached-only")
        value = read(Path(row["output_dir"]) / "dataset/candidate_pool_sample_manifest.json")
        pool = {"gold_free_summary": value["gold_free_summary"], "per_kind": value["per_kind"]}
        if case in pools and pool != pools[case]:
            raise ValueError("E15 treatments changed frozen candidate populations")
        pools[case] = pool
    return pools


def fitted_evidence(rows, training):
    """Require real disjoint grouped fits only for the declared supervised control."""
    evidence = []
    for row in rows:
        arm = row["name"].split("--")[1]
        directory = Path(row["output_dir"]) / "fitting"
        paths = list(directory.rglob("selector.json"))
        if arm != "current_supervised":
            if paths or list(directory.rglob("score_calibrator.json")):
                raise ValueError("Label-free E15 arms must not fit target labels")
            continue
        if len(paths) != 1:
            raise ValueError("One genuine fitted selector is required")
        path = paths[0]
        value = read(path)
        provenance = value["fit_provenance"]
        sources = set(provenance["training_sources"])
        folds = value["folds"]
        heldout = [s for fold in folds for s in fold["heldout_sources"]]
        if (
            value["kind"] != "fitted_selector"
            or not value["rank_model"]
            or not value["accept_model"]
            or sources != set(training["sources"])
            or len(sources) != 2000
            or provenance["negative_label_policy"] != "confirmed_negatives"
            or provenance["score_calibration"] != "none"
            or sources & set(provenance["application"]["source_ids"])
            or {fold["fold"] for fold in folds} != set(range(5))
            or len(heldout) != 2000
            or set(heldout) != sources
            or any(set(f["train_sources"]) != sources - set(f["heldout_sources"]) for f in folds)
        ):
            raise ValueError("E15 requires a genuine disjoint bounded grouped-OOF fit")
        evidence.append(binding(path))
    if len(evidence) != 1:
        raise ValueError("The supervised control fit is required")
    return evidence


def family_forecast(rows, measurement, h):
    """Setup once per arm; native inventory has one conservative allowance per case."""
    from exact.experiments.campaign import WorkEstimate

    pools = matched_measurements(rows)
    maxima = {case: max(r["wall_seconds"] for r in rows if r["case_id"] == case) for case in CASES}
    estimate = h.estimate(rows, 4, measurement)
    estimate["cold_seconds"] = sum(row["wall_seconds"] for row in rows)
    comparison = WorkEstimate.model_validate(estimate).seconds()
    # A whole matched run includes its native load; this upper bound includes no
    # per-pair setup extrapolation and claims no warm-cache saving.
    inventory = 1.5 * read(CONTROL)["measurement"]["wall_seconds"]
    return {
        "estimate": estimate,
        "comparison_seconds": comparison,
        "inventory_seconds": inventory,
        "whole_family_seconds": comparison + inventory,
        "hours": (comparison + inventory) / 3600,
        "case_maxima_seconds": maxima,
        "candidate_populations": pools,
    }


def preflight(args, h):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import (
        CampaignLock,
        external_acceptance_selection,
        external_selection_result,
    )

    h.check_pause()
    no_previous_owner(args.root, args.supervisor_step)
    identity = h.validate_source(args.code_root, allow_working_tree=args.allow_working_tree)
    lock = copy.deepcopy(load_yaml_mapping(PARENT / "campaign.lock.yaml"))
    complete = read(DATA / "e03-qualification-01/completion.json")
    if complete.get("status") != "complete" or complete.get("exit_code") != 0:
        raise ValueError("Latest E03 owner is incomplete")
    for name in ("selection", "cumulative_budget"):
        h.verify(complete[name])
    if len(complete["manifests"]) != 4:
        raise ValueError("Upstream E03 requires all four completed cells")
    for item in complete["manifests"]:
        if read(h.verify(item)).get("status") != "complete":
            raise ValueError("Incomplete upstream E03 cell")
    training = read(DATA / "e03-qualification-01/training.json")
    for field in ("pool", "reference"):
        h.verify({"path": training[field], "sha256": training[field + "_sha256"]})
    if training["source_groups"] != 2000 or len(set(training["sources"])) != 2000:
        raise ValueError("Exactly the original bounded 2000 training groups are required")
    h.freeze(args.root / "training.json", training)
    lock = prepare_steps(lock, training)
    h.history_record(
        lock, "E03", PARENT / "campaign.lock.yaml", BUDGET.parent, args.root / "history"
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
        raise ValueError("Cache source view must retain the latest E03 budget")
    oldrows = {"D0": dict(read(CONTROL)["measurement"], prefix=False)}
    for row in oldrows.values():
        if row["status"] != "passed" or row["execution_status"] != "complete":
            raise ValueError("Retained complete pilot evidence missing")
        verify_nested_bindings(row, h)
    # Qualification forecast only: bound training extraction by the authenticated
    # same-case scoring rate, with one complete D0 duration for fit/OOF overhead.
    basis = read(DATA / "prepared-campaign-13/forecast.json")["phase_breakdown"][
        "D0-current-control"
    ]
    for key in ("manifest_binding", "timing_binding", "worker_binding"):
        h.verify(basis[key])
    baseline = oldrows["D0"]["wall_seconds"]
    allowances = {
        key: 1.5
        * (
            baseline
            + (
                training["pairs"] * basis["seconds_per_pair"] + baseline
                if key.split("--")[1] == "current_supervised"
                else 0
            )
        )
        for key in KEYS
    }
    with tempfile.TemporaryDirectory(prefix="e15-preflight-") as tmp:
        directory = Path(tmp)
        check_admission(
            state,
            "reserve",
            1800 + sum(allowances.values()),
            directory,
            "planned/E15-qualification",
        )
        trial = ready_lock(lock, h.estimate(list(oldrows.values()), 4, CONTROL), identity["commit"])
        path = directory / "campaign.yaml"
        h.freeze(path, trial)
        cells = cells_for(path, directory, read(h.verify(complete["selection"])))
        for cell in cells:
            key = cell.task_id.split("-", 1)[0] + "--" + cell.arm_id
            h.freeze(args.root / "qualification" / (key + ".config.yaml"), cell.resolved_config)
    h.freeze(args.root / "prospective-campaign.yaml", lock)
    records = [
        args.root / name
        for name in (
            "reuse-plan.json",
            "repair.json",
            "checks.json",
            "cache-sources.json",
            "training.json",
        )
    ]
    operational = [
        args.code_root / "tools/queue_e15.py",
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
            "parent_completion": binding(DATA / "e03-qualification-01/completion.json"),
            "prospective_campaign": binding(args.root / "prospective-campaign.yaml"),
            "source_selection": complete["selection"],
            "historical_decisions_verified": imports,
            "pilot_allowance_seconds_per_arm": allowances,
            "pilot_scope": "Qualification only; no scientific comparison admission from baseline cost. Forecast is not a timeout.",
            "pilot_cost_basis": [
                binding(CONTROL),
                binding(DATA / "prepared-campaign-13/forecast.json"),
                binding(training["pool"]),
                binding(training["reference"]),
            ],
            "queue": list(KEYS),
            "training_pairs": training["pairs"],
            "training_groups": training["source_groups"],
            "configs": {
                a: binding(args.root / "qualification" / (a + ".config.yaml")) for a in KEYS
            },
            "records": [binding(p) for p in records],
            "operational_files": [binding(p) for p in operational],
            "budget_limits_changed": False,
            "generate_rationales": False,
        },
    )
    print("E15 matched qualification preflight passed; comparison not yet admitted", flush=True)


def run(args, h):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import CampaignLock, execute_campaign

    h.check_pause()
    if (
        os.environ.get("SLURM_JOB_ID") != "14372"
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise ValueError("Use a detached numeric step in allocation 14372")
    no_previous_owner(args.root, args.supervisor_step)
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
    ):
        h.verify(receipt[key])
    for item in [
        *receipt["pilot_cost_basis"],
        *receipt["records"],
        *receipt["operational_files"],
        *receipt["configs"].values(),
    ]:
        h.verify(item)
    if Path(receipt["budget_import"]["path"]) != BUDGET:
        raise ValueError("Stale cumulative budget import")
    lock = load_yaml_mapping(h.verify(receipt["prospective_campaign"]))
    validate_budget(read(BUDGET), lock)
    current = args.root / "qualification/runtime/exact-om-focused-v2"
    if current.exists() or (args.root / "E15/runtime").exists():
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
    for arm in KEYS:
        h.check_pause()
        h.status("measuring_E15", arm=arm, cumulative_budget=str(current / "budget.json"))
        rows.append(
            run_probe(
                code_root=args.code_root,
                script=args.root / "continue.py",
                config=h.verify(receipt["configs"][arm]),
                name=arm,
                directory=args.root / "qualification" / arm,
                shared=current,
                campaign=CampaignLock.model_validate(lock),
                forecast_seconds=receipt["pilot_allowance_seconds_per_arm"][arm],
                case_id=arm.split("--")[0],
            )
        )
    h.freeze(
        args.root / "fitting-evidence.json",
        fitted_evidence(rows, read(args.root / "training.json")),
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
            "method": "Sum of all four matched complete-arm durations times 1.5; each full training/OOF fit and setup counted once. Native inventory has a separate whole-run upper bound. No cache-speedup discount.",
        },
    )
    proof = family_forecast(rows, measurement, h)
    h.freeze(wave / "forecast.json", proof)
    with tempfile.TemporaryDirectory(prefix="e15-admission-") as tmp:
        from exact.experiments.budget import BudgetLedger

        check_path = check_admission(
            read(current / "budget.json"),
            "decisions",
            proof["whole_family_seconds"],
            Path(tmp),
            "planned/E15",
        )
        BudgetLedger(check_path, read(check_path)["limits"]).admit(
            "planned/E15-cache-copy", group="reserve", seconds=1800
        )
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    if (
        len(free) != 1
        or proof["estimate"]["peak_ram_gb"] > 56
        or proof["estimate"]["peak_vram_gb"] > float(free[0]) / 1024
    ):
        raise ValueError("Measured family memory does not fit this node")
    campaign = wave / "campaign.lock.yaml"
    h.freeze(campaign, ready_lock(lock, proof["estimate"], receipt["source"]["commit"]))
    with tempfile.TemporaryDirectory(prefix="e15-cells-") as tmp:
        cells = cells_for(campaign, Path(tmp), read(h.verify(receipt["source_selection"])))
        if any(
            binding(h.verify(receipt["configs"][c.task_id.split("-", 1)[0] + "--" + c.arm_id]))[
                "sha256"
            ]
            != binding_config(c.resolved_config)
            for c in cells
        ):
            raise ValueError("Measured configuration differs from scientific comparison")
    h.check_pause()
    destination = wave / "runtime/exact-om-focused-v2"
    h.copy_state(current, destination)
    current = destination
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
    with account_inventory(h, current, FAMILY, "decisions", proof["inventory_seconds"]):
        h.guarded_execute(campaign, wave, args.code_root)
    if usage(current / "openrouter") != cached:
        raise ValueError("Comparison incurred incremental hosted usage")
    manifests = sorted(
        (current / "screen/runs/E15").glob("*/D*-global_alignment/seed-17/experiment_manifest.json")
    )
    if len(manifests) != 4 or any(read(p).get("status") != "complete" for p in manifests):
        raise ValueError("E15 completion requires all four complete primary cells")
    h.freeze(
        args.root / "completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "cells": 4,
            "selection": binding(current / "screen/selection.json"),
            "manifests": [binding(p) for p in manifests],
            "cumulative_budget": binding(current / "budget.json"),
            "source": receipt["source"],
            "cached_usage_unchanged": True,
            "generate_rationales": False,
        },
    )
    h.status("complete", cells=4, cumulative_budget=str(current / "budget.json"))
    return 0


def binding_config(config):
    import hashlib

    from exact.core.entities.configs.yaml_io import dump_yaml_document

    return hashlib.sha256(dump_yaml_document(config).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--supervisor-step", default="14372.23")
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
                        args.root / "E15/runtime/exact-om-focused-v2/budget.json",
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
