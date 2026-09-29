#!/usr/bin/env python3
"""Measure every frozen E19 fusion treatment before admitting the complete comparison."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import subprocess
import tempfile
from pathlib import Path

from tools.experiment_resources import guarded_execute
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
    account_inventory,
    campaign_limits,
    check_admission,
    load_helpers,
    read,
    verify_nested_bindings,
)

FAMILY = "E19"
ARMS = ("analytic_shipped", "analytic_fitted", "learned_global")
CASES = ("D0_E03",)
KEYS = tuple(case + "--" + arm for case in CASES for arm in ARMS)
PROJECT = Path("/home/pgcotovio/Exact-OM")
DATA = PROJECT / "data/experiments-v2"
PARENT = DATA / "e10-acceptance-recovery-02/E10"
MEASURED_FIT = DATA / "e10-acceptance-01/qualification/D0_E03--winner_only/measurement.json"
BUDGET = PARENT / "runtime/exact-om-focused-v2/budget.json"
CONTROL = DATA / "d1-production-probe-01/d0-control.json"


def no_previous_owner(root, supervisor_step):
    """Allow only the retained interactive shell, supervisor, and this numeric step."""
    if supervisor_step != "14372.28":
        raise ValueError("This preparation binds the observed supervisor step 14372.28")
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
    """Bind the existing bounded D0 alias without changing any E19 treatment."""
    lock = copy.deepcopy(saved)
    step = next(row for row in lock["steps"] if row["id"] == FAMILY)
    if [a["id"] for a in step["arms"]] != list(ARMS):
        raise ValueError("E19 requires all three declared fusion recipes")
    if (
        step["case"] != "D0"
        or step["additional_cases"]
        or step["inherits"] != ["selected_E10_analytic_setting", "E26", "E05_initial"]
    ):
        raise ValueError("E19 case or inheritance changed")
    if not {"E10", "E26", "selected_E10_analytic_setting"}.issubset(step["requires"]):
        raise ValueError("E19 must follow the frozen analytic selection")
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
        raise ValueError("All three E19 arms are required")
    step["estimate"] = estimate
    for states in step["readiness"].values():
        states["screen"].update(
            status="screen_ready",
            inspected_commit=commit,
            missing=[],
            implemented_paths=[
                "exact/impl/trainer/fitting.py",
                "exact/impl/models/selector/fusion_fitting.py",
            ],
            tests=["tests/grouped_fitting_test.py", "tests/e19_queue_test.py"],
            reason="Three full matched fusion probes including bounded disjoint training precede whole-comparison admission; no fitted artifacts are invented.",
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
        raise ValueError("Complete E19 plan is not valid: " + repr(rows))
    ready = {r["step"] for r in plan["rows"] if r["status"] == "screen_ready" and not r["issues"]}
    if ready != {FAMILY}:
        raise ValueError("Only E19 may be dispatched: " + repr(ready))
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
    """Keep every scientific field fixed except the declared fusion treatment."""
    if len(cells) != len(ARMS) or {(c.task_id, c.arm_id) for c in cells} != {
        (case + "-global_alignment", arm) for case in CASES for arm in ARMS
    }:
        raise ValueError("All three E19 case/arm cells are mandatory")
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
    common = None
    for cell in cells:
        c = cell.resolved_config
        expected = {
            key: (
                "supervised"
                if key == "fusion" and cell.arm_id != "analytic_shipped"
                else "label_free"
            )
            for key in components
        }
        fusion = c["matching"]["fusion"]
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
            or c["selector"]["enabled"]
            or c["selector"]["runtime_enabled"] is not None
            or c["selector"]["runtime_global_only"] is not None
            or not fusion["enabled"]
            or fusion["mode"] != cell.arm_id
            or fusion["gamma"] != 2
            or fusion["tau"] != 0.5
            or fusion["beta"] != 0.8
            or fusion.get("artifact")
        ):
            raise ValueError("E19 scientific, supervision, or hosted policy changed")
        shared = copy.deepcopy(c)
        shared["matching"]["fusion"]["mode"] = "analytic_shipped"
        shared["supervision"]["components"]["fusion"] = "label_free"
        if common is not None and shared != common:
            raise ValueError("E19 arms may differ only in fusion and its supervision role")
        common = shared


def matched_measurements(rows):
    """Require complete case/treatment coverage and equal pools within each case."""
    if len(rows) != len(ARMS) or {r["name"] for r in rows} != set(KEYS):
        raise ValueError("Matched measurements must cover every E19 case and treatment")
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
            raise ValueError("E19 treatments changed frozen candidate populations")
        pools[case] = pool
    return pools


def fitted_evidence(rows, training):
    """Verify actual disjoint grouped fusion fits and the unfitted control."""
    import math

    if len(rows) != len(ARMS) or {r["name"] for r in rows} != set(KEYS):
        raise ValueError("All E19 qualification cells are required")
    evidence = []
    sources = set(training["sources"])
    for row in rows:
        arm = row["name"].split("--")[1]
        paths = list((Path(row["output_dir"]) / "fitting").rglob("fusion.json"))
        if arm == "analytic_shipped":
            if paths:
                raise ValueError("Analytic control cannot consume a fitted fusion artifact")
            continue
        if len(paths) != 1:
            raise ValueError("One actual fitted fusion artifact is required per learned arm")
        value = read(paths[0])
        provenance = value["fit_provenance"]
        folds = provenance["folds"]
        heldout = [s for fold in folds for s in fold["heldout_sources"]]
        if (
            value["mode"] != arm
            or value["seed"] != 17
            or value["negative_label_policy"] != "confirmed_negatives"
            or len(sources) != 2000
            or len(heldout) != 2000
            or set(heldout) != sources
            or sources & set(provenance["application"]["source_ids"])
            or {f["fold"] for f in folds} != set(range(5))
            or any(set(f["train_sources"]) != sources - set(f["heldout_sources"]) for f in folds)
            or provenance["regularization"] not in {0.001, 0.01}
            or len(value["oof_predictions"]) != training["pairs"]
            or {p["Src"] for p in value["oof_predictions"]} != sources
        ):
            raise ValueError("E19 requires a disjoint bounded grouped-OOF fusion fit")
        if arm == "analytic_fitted":
            params = value["parameters"]
            if not 0.4 <= params["tau"] <= 0.6 or not 0.5 <= params["gamma"] <= 3:
                raise ValueError("Fitted constants exceed declared bounds")
            weights = params["multipliers"]
        else:
            weights = value["weights"]
        if (
            set(weights) != set(value["feature_schema"])
            or not weights
            or any(not math.isfinite(w) or w < 0 for w in weights.values())
            or abs(sum(weights.values()) / len(weights) - 1) > 1e-6
        ):
            raise ValueError("Fusion weights must be finite nonnegative mean-one values")
        evidence.append(binding(paths[0]))
    return evidence


def family_forecast(rows, measurement, h):
    """Setup once per arm; native inventory has one conservative allowance per case."""
    from exact.experiments.campaign import WorkEstimate

    pools = matched_measurements(rows)
    maxima = {case: max(r["wall_seconds"] for r in rows if r["case_id"] == case) for case in CASES}
    estimate = h.estimate(rows, len(ARMS), measurement)
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


def qualification_allowances(fit):
    """A full same-case grouped fit bounds each new fusion qualification arm."""
    if (
        fit["name"] != "D0_E03--winner_only"
        or fit["case_id"] != "D0_E03"
        or fit["status"] != "passed"
        or fit["execution_status"] != "complete"
        or fit["prefix"]
        or any(fit["new_usage"].values())
        or not 0 < fit["processed_pairs"] == fit["dataset_rows"] <= 6000
        or fit["wall_seconds"] <= 0
    ):
        raise ValueError("Complete cached same-case supervised measurement required")
    control = read(CONTROL)["measurement"]
    return {
        key: 1.5
        * (control["wall_seconds"] if key.endswith("--analytic_shipped") else fit["wall_seconds"])
        for key in KEYS
    }


def validate_qualification_basis(cells, config):
    """Match the measured evidence workload, with own newly measured fusion fits."""
    for cell in cells:
        expected = copy.deepcopy(config)
        expected["selector"]["enabled"] = False
        expected["selector"]["runtime_enabled"] = None
        expected["selector"]["runtime_global_only"] = None
        expected["supervision"]["components"]["rerank"] = "label_free"
        expected["supervision"]["components"]["accept"] = "label_free"
        expected["supervision"]["components"]["fusion"] = (
            "label_free" if cell.arm_id == "analytic_shipped" else "supervised"
        )
        expected["matching"]["fusion"]["mode"] = cell.arm_id
        if cell.resolved_config != expected:
            raise ValueError("Qualification workload differs beyond the declared E19 treatments")


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
    complete = read(DATA / "e10-acceptance-recovery-02/completion.json")
    if complete.get("status") != "complete" or complete.get("exit_code") != 0:
        raise ValueError("Latest E10 owner is incomplete")
    for name in ("selection", "cumulative_budget"):
        h.verify(complete[name])
    if len(complete["manifests"]) != 2:
        raise ValueError("Upstream E10 requires both completed cells")
    for item in complete["manifests"]:
        if read(h.verify(item)).get("status") != "complete":
            raise ValueError("Incomplete upstream E10 cell")
    training = read(DATA / "e03-qualification-01/training.json")
    for field in ("pool", "reference"):
        h.verify({"path": training[field], "sha256": training[field + "_sha256"]})
    if training["source_groups"] != 2000 or len(set(training["sources"])) != 2000:
        raise ValueError("Exactly the original bounded 2000 training groups are required")
    h.freeze(args.root / "training.json", training)
    lock = prepare_steps(lock, training)
    h.history_record(
        lock, "E10", PARENT / "campaign.lock.yaml", BUDGET.parent, args.root / "history"
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
        raise ValueError("Cache source view must retain the latest E10 budget")
    oldrows = {"D0": dict(read(CONTROL)["measurement"], prefix=False)}
    for row in oldrows.values():
        if row["status"] != "passed" or row["execution_status"] != "complete":
            raise ValueError("Retained complete pilot evidence missing")
        verify_nested_bindings(row, h)
    # This latest measured fit uses the same bounded source groups and scoring
    # stack. It sizes qualification only; all three E19 recipes must then be measured.
    fit = read(MEASURED_FIT)
    verify_nested_bindings(fit, h)
    for field in ("extraction_code", "native_code_sha256"):
        if identity[field] != complete["source"][field]:
            raise ValueError("Measured fit source/native identity changed: " + field)
    if state["work"][fit["budget_work_id"]]["status"] != "complete":
        raise ValueError("Measured fit lacks completed cumulative accounting")
    from tools.queue_e10_acceptance import fitted_evidence as prior_fitted_evidence

    paired = read(DATA / "e10-acceptance-01/qualification/D0_E03--winner_runnerup/measurement.json")
    verify_nested_bindings(paired, h)
    prior_fits = prior_fitted_evidence([fit, paired], training)
    allowances = qualification_allowances(fit)
    with tempfile.TemporaryDirectory(prefix="e19-preflight-") as tmp:
        directory = Path(tmp)
        check_admission(
            state,
            "reserve",
            1800 + sum(allowances.values()),
            directory,
            "planned/E19-qualification",
        )
        trial = ready_lock(lock, h.estimate([fit], len(ARMS), MEASURED_FIT), identity["commit"])
        path = directory / "campaign.yaml"
        h.freeze(path, trial)
        cells = cells_for(path, directory, read(h.verify(complete["selection"])))
        validate_qualification_basis(cells, load_yaml_mapping(h.verify(fit["config"])))
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
        args.code_root / "tools/queue_e19.py",
        args.code_root / "tools/queue_e10_acceptance.py",
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
            "parent_completion": binding(DATA / "e10-acceptance-recovery-02/completion.json"),
            "prospective_campaign": binding(args.root / "prospective-campaign.yaml"),
            "source_selection": complete["selection"],
            "historical_decisions_verified": imports,
            "pilot_allowance_seconds_per_arm": allowances,
            "pilot_scope": "Qualification only; no scientific comparison admission from baseline cost. Forecast is not a timeout.",
            "pilot_cost_basis": [
                binding(CONTROL),
                binding(MEASURED_FIT),
                *prior_fits,
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
    print("E19 matched qualification preflight passed; comparison not yet admitted", flush=True)


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
    if current.exists() or (args.root / "E19/runtime").exists():
        raise FileExistsError(
            "Existing runtime requires recorded checkpoint recovery, not top-level replay"
        )
    fit = read(MEASURED_FIT)
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    measured = fit["worker_measurement"]
    if (
        len(free) != 1
        or 1.5 * measured["peak_rss_bytes"] / 1024**3 > memory_limit_bytes() / 1024**3
        or 1.5 * measured["peak_cuda_reserved_bytes"] / 1024**2 > float(free[0])
    ):
        raise ValueError("Measured qualification footprint does not fit this GPU step")
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
        h.status("measuring_E19", arm=arm, cumulative_budget=str(current / "budget.json"))
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
            "method": "Sum of all three matched complete-arm durations times 1.5; each full training/OOF fit and setup counted once. Native inventory has a separate whole-run upper bound. No cache-speedup discount.",
        },
    )
    proof = family_forecast(rows, measurement, h)
    proof = reused_comparison_forecast(proof)
    h.freeze(wave / "forecast.json", proof)
    with tempfile.TemporaryDirectory(prefix="e19-admission-") as tmp:
        from exact.experiments.budget import BudgetLedger

        check_path = check_admission(
            read(current / "budget.json"),
            "channels",
            proof["whole_family_seconds"],
            Path(tmp),
            "planned/E19",
        )
        BudgetLedger(check_path, read(check_path)["limits"]).admit(
            "planned/E19-cache-copy", group="reserve", seconds=1800
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
    with tempfile.TemporaryDirectory(prefix="e19-cells-") as tmp:
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
            guarded_execute(campaign, wave, args.code_root, check_pause=h.check_pause)
    if usage(current / "openrouter") != cached:
        raise ValueError("Comparison incurred incremental hosted usage")
    manifests = sorted(
        (current / "screen/runs/E19").glob("*/D*-global_alignment/seed-17/experiment_manifest.json")
    )
    if len(manifests) != len(ARMS) or any(read(p).get("status") != "complete" for p in manifests):
        raise ValueError("E19 completion requires all three complete primary cells")
    h.freeze(
        args.root / "completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "cells": len(ARMS),
            "selection": binding(current / "screen/selection.json"),
            "manifests": [binding(p) for p in manifests],
            "cumulative_budget": binding(current / "budget.json"),
            "source": receipt["source"],
            "cached_usage_unchanged": True,
            "generate_rationales": False,
        },
    )
    h.status("complete", cells=len(ARMS), cumulative_budget=str(current / "budget.json"))
    return 0


def binding_config(config):
    import hashlib

    from exact.core.entities.configs.yaml_io import dump_yaml_document

    return hashlib.sha256(dump_yaml_document(config).encode()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--supervisor-step", default="14372.28")
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
                        args.root / "E19/runtime/exact-om-focused-v2/budget.json",
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
