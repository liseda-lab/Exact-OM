#!/usr/bin/env python3
"""Measure every frozen E03 treatment before admitting the complete comparison."""
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

FAMILY = "E03"
ARMS = ("none", "platt", "isotonic", "distribution_threshold")
CASES = ("D0_E03",)
KEYS = tuple(case + "--" + arm for case in CASES for arm in ARMS)
PROJECT = Path("/home/pgcotovio/Exact-OM")
DATA = PROJECT / "data/experiments-v2"
PARENT = DATA / "e24-qualification-01/E24"
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
    """Sequence the optional loss diagnostic after its declared calibration selection."""
    lock = copy.deepcopy(saved)
    step = next(row for row in lock["steps"] if row["id"] == FAMILY)
    if [a["id"] for a in step["arms"]] != [*ARMS, "fp_only"]:
        raise ValueError("E03 primary comparison or diagnostic changed")
    if step["case"] != "D0" or step["additional_cases"] or step["inherits"] != ["E05_initial"]:
        raise ValueError("E03 case or inheritance changed")
    diagnostic = copy.deepcopy(step)
    diagnostic.update(
        id="E03-fp-only",
        requires=[*step["requires"], "E03"],
        inherits=[*step["inherits"], "E03"],
        arms=[step["arms"].pop()],
        readiness={"fp_only": step["readiness"].pop("fp_only")},
        selection={"decisions": []},
        policy_paths=[],
    )
    # E03 specifies the selected calibrator; a fixed `none` overlay would erase it.
    diagnostic["arms"][0]["overlay"].pop("matching")
    for state in diagnostic["readiness"]["fp_only"].values():
        state.update(
            status="blocked_input_resolution",
            missing=["selected E03 calibration recipe"],
            reason="Optional changed-loss diagnostic follows the complete primary calibration selection; not admitted by this queue.",
        )
    lock["steps"].insert(lock["steps"].index(step) + 1, diagnostic)
    # A distinct binding keeps the original D0 historical imports immutable.
    # Its ontology, reporting universe and valid labels are byte-identical to D0.
    case = copy.deepcopy(lock["cases"]["D0"])
    case["candidates"]["train"] = {"path": training["pool"], "sha256": training["pool_sha256"]}
    case["references"]["train"] = {
        "path": training["reference"],
        "sha256": training["reference_sha256"],
    }
    if "train" in case.get("local_references", {}):
        case["local_references"]["train"] = copy.deepcopy(case["references"]["train"])
    lock["cases"]["D0_E03"] = case
    step["case"] = diagnostic["case"] = "D0_E03"
    # E24 is completed in its original immutable root. The current historical
    # importer supports one case only, so never pretend its two-case import passed.
    completed = next(row for row in lock["steps"] if row["id"] == "E24")
    completed["estimate"] = None
    for state in completed["readiness"].values():
        state["screen"].update(
            status="blocked_input_resolution",
            missing=["multi-case historical import"],
            reason="Completed E24 receipt is preserved; no rerun. Multi-case historical import is required only by later E24 consumers, not E03.",
        )
    return lock


def ready_lock(saved, estimate, commit):
    lock = copy.deepcopy(saved)
    step = next(row for row in lock["steps"] if row["id"] == FAMILY)
    if [a["id"] for a in step["arms"]] != list(ARMS):
        raise ValueError("All four primary E03 arms are required")
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
            tests=["tests/grouped_fitting_test.py", "tests/e03_queue_test.py"],
            reason="Four full matched calibration probes including bounded disjoint training precede whole-comparison admission; no fitted artifacts are invented.",
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
        raise ValueError("Complete E03 plan is not valid: " + repr(rows))
    ready = {r["step"] for r in plan["rows"] if r["status"] == "screen_ready" and not r["issues"]}
    if ready != {FAMILY}:
        raise ValueError("Only E03 may be dispatched: " + repr(ready))
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
        raise ValueError("All four E03 case/arm cells are mandatory")
    for cell in cells:
        c = cell.resolved_config
        supervised = cell.arm_id in {"platt", "isotonic"}
        expected = {
            key: "supervised" if key == "calibration" and supervised else "label_free"
            for key in c["supervision"]["components"]
        }
        if (
            cell.generate_rationales
            or cell.split_role != "development"
            or cell.seed != 17
            or cell.source_cap != 300
            or set(c["data"]["refs"]) - {"train", "valid"}
            or c["llm"]["experiment"]["gate"]["mode"] != "off"
            or c["matching"]["extraction"]["mode"] != "greedy"
            or c["matching"]["threshold"] != 0.7
            or c["matching"]["calibration"]["mode"] != (cell.arm_id if supervised else "none")
            or c["matching"]["calibration"]["threshold_mode"]
            != ("otsu" if cell.arm_id == "distribution_threshold" else "fixed")
            or c["supervision"]["components"] != expected
            or c["supervision"]["negative_label_policy"] != "confirmed_negatives"
            or not c["data"]["train_candidates"]
        ):
            raise ValueError("E03 scientific, training, or hosted policy changed")


def matched_measurements(rows):
    """Require complete case/treatment coverage and equal pools within each case."""
    if len(rows) != 4 or {r["name"] for r in rows} != set(KEYS):
        raise ValueError("Matched measurements must cover every E03 case and treatment")
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
            raise ValueError("E03 treatments changed frozen candidate populations")
        pools[case] = pool
    return pools


def fitted_evidence(rows, training):
    """Require real OOF artifacts from the declared disjoint training population."""
    evidence = []
    for row in rows:
        arm = row["name"].split("--")[1]
        if arm not in {"platt", "isotonic"}:
            continue
        path = Path(row["output_dir"]) / "fitting/score_calibrator.json"
        value = read(path)
        provenance = value["fit_provenance"]
        sources = set(provenance["training_sources"])
        oof = value["oof_predictions"]
        if (
            value["kind"] != "score_calibrator"
            or value["calibrator"]["mode"] != arm
            or sources != set(training["sources"])
            or len(sources) != 2000
            or provenance["negative_label_policy"] != "confirmed_negatives"
            or sources & set(provenance["application"]["source_ids"])
            or {r["source"] for r in oof} != sources
            or {r["fold"] for r in oof} != set(range(5))
        ):
            raise ValueError("E03 requires genuine disjoint bounded grouped-OOF fits")
        evidence.append(binding(path))
    if len(evidence) != 2:
        raise ValueError("Both fitted calibration arms are required")
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
    complete = read(DATA / "e24-qualification-01/completion.json")
    if complete.get("status") != "complete" or complete.get("exit_code") != 0:
        raise ValueError("Latest E24 owner is incomplete")
    for name in ("selection", "cumulative_budget"):
        h.verify(complete[name])
    from tools.run_experiment_validation import bounded_training

    training = bounded_training(
        CampaignLock.model_validate(lock).cases["D0"], PARENT, args.root, limit=2000
    )
    h.freeze(args.root / "training.json", training)
    lock = prepare_steps(lock, training)
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
        raise ValueError("Cache source view must retain the latest E24 budget")
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
                if key.split("--")[1] in {"platt", "isotonic"}
                else 0
            )
        )
        for key in KEYS
    }
    with tempfile.TemporaryDirectory(prefix="e03-preflight-") as tmp:
        directory = Path(tmp)
        check_admission(
            state,
            "reserve",
            1800 + sum(allowances.values()),
            directory,
            "planned/E03-qualification",
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
        args.code_root / "tools/queue_e03.py",
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
            "parent_completion": binding(DATA / "e24-qualification-01/completion.json"),
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
    print("E03 matched qualification preflight passed; comparison not yet admitted", flush=True)


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
    if current.exists() or (args.root / "E03/runtime").exists():
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
        h.status("measuring_E03", arm=arm, cumulative_budget=str(current / "budget.json"))
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
    with tempfile.TemporaryDirectory(prefix="e03-admission-") as tmp:
        from exact.experiments.budget import BudgetLedger

        check_path = check_admission(
            read(current / "budget.json"),
            "decisions",
            proof["whole_family_seconds"],
            Path(tmp),
            "planned/E03",
        )
        BudgetLedger(check_path, read(check_path)["limits"]).admit(
            "planned/E03-cache-copy", group="reserve", seconds=1800
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
    with tempfile.TemporaryDirectory(prefix="e03-cells-") as tmp:
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
        (current / "screen/runs/E03").glob("*/D*-global_alignment/seed-17/experiment_manifest.json")
    )
    if len(manifests) != 4 or any(read(p).get("status") != "complete" for p in manifests):
        raise ValueError("E03 completion requires all four complete primary cells")
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
                        args.root / "E03/runtime/exact-om-focused-v2/budget.json",
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
