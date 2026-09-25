#!/usr/bin/env python3
"""Continue the declared cached E10 family while retaining E01's explicit blocker."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.resume_cached_batch import (  # noqa: E402
    OLD_COMMIT,
    RECONCILIATION_ID,
    account_inventory,
    campaign_limits,
    check_admission,
    comparison_forecast,
    load_helpers,
    read,
    set_family_ready,
    verify_nested_bindings,
)

BASE_COMMIT = "8edb236a2b79a8881c4a3f445b31988bafe541d2"
FAMILY = "E10-analytic"
E01_REASON = (
    "E01 retains its failed hard-anchor comparison: conflicting protected exact matches "
    "make declared one-to-one constraints infeasible. A scientific anchor-policy decision "
    "is required; no E01 completion or selection is imported."
)
EXPECTED_ARMS = {
    "analytic_g1_t04",
    "analytic_g1_t05",
    "analytic_g2_t04",
    "analytic_g2_t05",
    "analytic_g3_t04",
    "analytic_g3_t05",
}


def latest_budget(root):
    return root.parent / "next-batch-recovery-03/E01/runtime/exact-om-focused-v2/budget.json"


def validate_budget(state, limits):
    if state["limits"] != limits:
        raise ValueError("Cumulative limits differ; caps and protected allowance cannot change")
    active = [
        key
        for key, row in state["work"].items()
        if row["status"] == "reserved" and key != "historical/G0"
    ]
    if active:
        raise ValueError("Previous accounting is still reserved: " + repr(active))
    row = state["work"].get(RECONCILIATION_ID, {})
    if row.get("status") != "failed" or row.get("seconds", 0) <= 0:
        raise ValueError("Latest ledger must retain already reconciled failed setup")


def independent_lock(saved, estimate, commit):
    """Change readiness only; keep the complete E10 scientific declaration unchanged."""
    lock = copy.deepcopy(saved)
    e01 = next(step for step in lock["steps"] if step["id"] == "E01")
    if e01.get("external_selection"):
        raise ValueError("Failed E01 must not acquire a completed selection")
    e01["estimate"] = None
    for states in e01["readiness"].values():
        states["screen"].update(
            status="blocked_input_resolution", reason=E01_REASON, missing=[E01_REASON]
        )
    step = next(step for step in lock["steps"] if step["id"] == FAMILY)
    if (
        {arm["id"] for arm in step["arms"]} != EXPECTED_ARMS
        or len(step["arms"]) != 6
        or step["requires"] != ["E00", "pool_freeze", "E26", "E05_initial"]
        or step["inherits"] != ["E26", "E05_initial"]
        or step["policy_paths"] != ["matching.fusion.gamma", "matching.fusion.tau"]
    ):
        raise ValueError("Independent complete E10 comparison or dependencies changed")
    set_family_ready(lock, FAMILY, estimate, commit)
    return lock


def require_only_e10(plan):
    ready = {
        row["step"] for row in plan["rows"] if row["status"] == "screen_ready" and not row["issues"]
    }
    selected = [row for row in plan["rows"] if row["step"] == FAMILY]
    if (
        plan["budget_errors"]
        or ready != {FAMILY}
        or not selected
        or any(row["issues"] for row in selected)
    ):
        raise ValueError("Exactly the complete independent E10 family must be runnable")


def require_numeric_step():
    if (
        os.environ.get("SLURM_JOB_ID") != "14372"
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise ValueError("Use a detached numeric step inside allocation 14372")


def check_previous_owner(root):
    """Refuse stale-wrapper recovery while the previous numeric worker still exists."""
    import psutil

    steps = subprocess.check_output(
        ["squeue", "--steps", "-h", "-j", "14372", "-o", "%i"], text=True
    ).splitlines()
    if "14372.11" in {step.strip() for step in steps}:
        raise ValueError("Previous Slurm worker 14372.11 is still alive")
    old_root = str(root.parent / "next-batch-recovery-03")
    old_code = str(root.parent / "runtime/next-batch-repair-code-03")
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.info["pid"] == os.getpid():
            continue
        command = process.info.get("cmdline") or []
        if process.info.get("name", "").startswith(("python", "srun", "bash")) and any(
            old_root in part or old_code in part for part in command
        ):
            raise ValueError(
                "Previous launcher or worker remains alive, PID " + str(process.info["pid"])
            )


def controls(qualification, helpers):
    old = read(helpers.verify(qualification["old_control"]))
    new = read(helpers.verify(qualification["new_measurement"]))
    rows = [dict(old["measurement"], prefix=False), new]
    verify_nested_bindings(rows, helpers)
    if any(
        row.get("status") != "passed"
        or row.get("execution_status") != "complete"
        or row.get("prefix")
        or any(row.get("new_usage", {}).values())
        for row in rows
    ):
        raise ValueError("Full passed cached controls without incremental usage are required")
    return rows


def preflight(args, helpers):
    from qualification import binding

    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.campaign import (
        CampaignLock,
        campaign_plan,
        external_acceptance_selection,
        external_selection_result,
        materialize_campaign,
        validate_comparison_cells,
    )

    helpers.check_pause()
    check_previous_owner(args.root)
    identity = helpers.validate_source(args.code_root)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", "exact"],
        cwd=args.code_root,
        text=True,
    ).strip()
    if changed:
        raise ValueError("Numerical source changed; retained qualification cannot be reused")
    qualification_path = (
        args.root.parent / "next-batch-recovery-02/qualification/source-qualification.json"
    )
    qualification = read(qualification_path)
    if qualification.get("status") != "passed" or qualification["source"]["commit"] != OLD_COMMIT:
        raise ValueError("Retained qualification is not passed under the declared source")
    for field in ("extraction_code", "native_code_sha256"):
        if identity[field] != qualification["source"][field]:
            raise ValueError("Retained source/native identity differs: " + field)
    verify_nested_bindings(qualification, helpers)
    rows = controls(qualification, helpers)
    parent = args.root.parent / "next-batch-recovery-03"
    saved_path = parent / "E01/campaign.lock.yaml"
    saved = load_yaml_mapping(saved_path)
    budget = latest_budget(args.root)
    state = read(budget)
    validate_budget(state, campaign_limits(saved))
    if state["work"].get(rows[1]["budget_work_id"], {}).get("status") != "complete":
        raise ValueError("Qualification's measured accounting is missing")
    completed_path = helpers.DATA / "mechanism-screen-03/completion.json"
    completed = read(completed_path)
    if completed.get("status") != "complete" or completed.get("exit_code") != 0:
        raise ValueError("E09 prerequisite is incomplete")
    for item in [completed["selection"], *completed["completed_cells"]]:
        helpers.verify(item)
    parsed = CampaignLock.model_validate(saved)
    imported = {}
    for step in parsed.steps:
        if step.external_selection:
            imported[step.id] = external_selection_result(parsed, step, parent)["status"]
        elif step.external_acceptance:
            imported[step.id] = external_acceptance_selection(parsed, step, parent)["status"]
    if "E01" in imported:
        raise ValueError("E01 is blocked, not a historical completed comparison")
    wave = args.root / FAMILY
    if (wave / "runtime").exists():
        raise FileExistsError("Existing runtime requires saved campaign recovery")
    measurement = wave / "measured-cost.json"
    helpers.freeze(
        measurement,
        {
            "status": "measured_forecast",
            "step": FAMILY,
            "rows": rows,
            "qualification": binding(qualification_path),
            "method": "Maximum complete D0 arm measurement, setup once per arm, factor 1.5; standalone native inventory once per family. No cache speedup assumed.",
        },
    )
    proof = comparison_forecast(rows, 6, measurement, helpers)
    helpers.freeze(wave / "forecast.json", proof)
    lock = independent_lock(saved, proof["estimate"], identity["commit"])
    campaign = wave / "campaign.lock.yaml"
    helpers.freeze(campaign, lock)
    require_only_e10(campaign_plan(campaign, stage="screen"))
    with tempfile.TemporaryDirectory(prefix="exact-independent-e10-preflight-") as temporary:
        temp = Path(temporary)
        check_admission(state, "channels", proof["whole_family_seconds"], temp, "planned/" + FAMILY)
        suite = materialize_campaign(campaign, temp / "declarations", stage="screen")
        source = next(row for row in suite.sources if row.config.experiment_id == FAMILY)
        selection = read(helpers.verify(completed["selection"]))
        cells = harness.build_cells(
            suite,
            source,
            stage="screen",
            output_root=temp / "runtime",
            inherited_overlay=harness.inherited_selection_overlay(
                selection, source.config.depends_on
            ),
        )
        validate_comparison_cells(cells, source)
        if len(cells) != 6 or {cell.arm_id for cell in cells} != EXPECTED_ARMS:
            raise ValueError("Incomplete six-arm comparison")
        for cell in cells:
            config = cell.resolved_config
            if (
                cell.generate_rationales
                or cell.split_role != "development"
                or config["llm"]["experiment"]["gate"]["mode"] != "off"
                or set(config["data"]["refs"]) - {"train", "valid"}
                or config["matching"]["extraction"]["mode"] != "greedy"
            ):
                raise ValueError("Scientific role, rationale, hosted gate or extraction changed")
    operational = [Path(__file__), args.code_root / "tools/resume_cached_batch.py"]
    operational += [args.root / name for name in ("run-step.sh", "worker-entry.sh", "submit.sh")]
    operational += [
        args.root.parent / "next-batch-recovery-02" / name
        for name in ("continue.py", "qualification.py")
    ]
    records = [args.root / name for name in ("reuse-plan.json", "repair.json")]
    if (args.root / "diagnosis.json").exists():
        records.append(args.root / "diagnosis.json")
    receipt = {
        "status": "preflight_passed_not_admitted",
        "source": identity,
        "parent_attempt": str(parent),
        "queue": [FAMILY],
        "budget_import": binding(budget),
        "saved_campaign": binding(saved_path),
        "qualification": binding(qualification_path),
        "e09_completion": binding(completed_path),
        "campaign": binding(campaign),
        "measurement": binding(measurement),
        "forecast": binding(wave / "forecast.json"),
        "records": [binding(path) for path in records],
        "operational_files": [binding(path) for path in operational],
        "historical_decisions_verified": imported,
        "cell_preflight": {
            FAMILY: {"cells": 6, "config_hashes": {cell.arm_id: cell.config_hash for cell in cells}}
        },
        "e01_disposition": {"status": "blocked_input_resolution", "reason": E01_REASON},
        "budget_limits_changed": False,
        "generate_rationales": False,
    }
    helpers.freeze(args.root / "preflight.json", receipt)
    print(
        {"status": receipt["status"], "queue": [FAMILY], "forecast_hours": proof["hours"]},
        flush=True,
    )
    return receipt


def run(args, helpers):
    from qualification import binding, usage

    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import (
        CampaignLock,
        campaign_plan,
        execute_campaign,
        external_selection_result,
    )

    helpers.check_pause()
    require_numeric_step()
    check_previous_owner(args.root)
    receipt = read(args.root / "preflight.json")
    if receipt["status"] != "preflight_passed_not_admitted" or receipt["queue"] != [FAMILY]:
        raise ValueError("Passed E10-only preflight is required")
    for key in (
        "budget_import",
        "saved_campaign",
        "qualification",
        "e09_completion",
        "campaign",
        "measurement",
        "forecast",
    ):
        helpers.verify(receipt[key])
    for item in [*receipt["records"], *receipt["operational_files"]]:
        helpers.verify(item)
    if helpers.validate_source(args.code_root) != receipt["source"]:
        raise ValueError("Frozen execution source changed")
    if Path(receipt["budget_import"]["path"]) != latest_budget(args.root):
        raise ValueError("Cannot import an earlier budget")
    wave = args.root / FAMILY
    destination = wave / "runtime/exact-om-focused-v2"
    if (wave / "runtime").exists():
        raise FileExistsError("Never rerun this top-level launcher over an existing runtime")
    campaign = helpers.verify(receipt["campaign"])
    lock = load_yaml_mapping(campaign)
    proof = read(helpers.verify(receipt["forecast"]))
    current = latest_budget(args.root).parent
    state = read(current / "budget.json")
    validate_budget(state, campaign_limits(lock))
    require_only_e10(campaign_plan(campaign, stage="screen"))
    with tempfile.TemporaryDirectory(prefix="exact-independent-e10-admission-") as temporary:
        path = check_admission(
            state, "channels", proof["whole_family_seconds"], Path(temporary), "planned/" + FAMILY
        )
        helpers.freeze(wave / "budget-admission.json", read(path))
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    if (
        len(free) != 1
        or proof["estimate"]["peak_ram_gb"] > min(56, physical - 6)
        or proof["estimate"]["peak_vram_gb"] > float(free[0]) / 1024
    ):
        raise ValueError("Measured complete-family memory forecast does not fit")
    retained_launch = read(helpers.DATA / "d1-production-probe-01/launch.json")
    os.environ.update(retained_launch["environment"])
    os.environ["PYTHONPATH"] = str(args.code_root)
    os.environ["OPENROUTER_API_KEY"] = (helpers.PROJECT / "api_key").read_text().strip()
    helpers.status(
        "preparing_recovery", step=FAMILY, cumulative_budget=str(current / "budget.json")
    )
    helpers.check_pause()
    helpers.copy_state(current, destination)
    inherited = read(destination / "budget.json")
    if inherited["limits"] != state["limits"] or any(
        inherited["work"].get(key) != row for key, row in state["work"].items()
    ):
        raise ValueError("Cache copy changed inherited cumulative accounting")
    current = destination
    with tempfile.TemporaryDirectory(prefix="exact-independent-e10-final-admission-") as temporary:
        check_admission(
            inherited,
            "channels",
            proof["whole_family_seconds"],
            Path(temporary),
            "planned/" + FAMILY,
        )
    cached = usage(current / "openrouter")
    os.environ["EXACT_OPENROUTER_REQUEST_CAP"] = str(cached["attempts"])
    os.environ["EXACT_OPENROUTER_TOKEN_CAP"] = str(cached["billable_tokens"])
    os.environ["EXACT_OPENROUTER_RETRY_UNKNOWN"] = "0"
    helpers.freeze(
        wave / "launch.json",
        {
            "campaign": binding(campaign),
            "source": receipt["source"],
            "qualification": receipt["qualification"],
            "budget_path": str(current / "budget.json"),
            "slurm_job_id": "14372",
            "slurm_step_id": os.environ["SLURM_STEP_ID"],
            "forecast": receipt["forecast"],
            "preflight": binding(args.root / "preflight.json"),
        },
    )
    dispositions = {
        name: {"status": "blocked_input_resolution", "reason": reason}
        for name, reason in helpers.DEFERRED.items()
    }
    dispositions["E01"] = receipt["e01_disposition"]
    helpers.write(args.root / "dispositions.json", dispositions)
    helpers.status(
        "running",
        step=FAMILY,
        campaign=str(campaign),
        cumulative_budget=str(current / "budget.json"),
    )
    helpers.check_pause()
    execute_campaign(
        campaign,
        stage="screen",
        output_root=wave / "runtime",
        workdir=args.code_root,
        jobs=1,
        reuse_plan_only=True,
    )
    helpers.check_pause()
    with account_inventory(helpers, current, FAMILY, "channels", proof["inventory_seconds"]):
        helpers.guarded_execute(campaign, wave, args.code_root)
    if usage(current / "openrouter") != cached:
        raise ValueError("Cached-only comparison produced incremental hosted usage")
    helpers.history_record(lock, FAMILY, campaign, current, args.root / "history")
    parsed = CampaignLock.model_validate(lock)
    decision = external_selection_result(
        parsed, next(row for row in parsed.steps if row.id == FAMILY), args.root
    )
    helpers.freeze(
        wave / "completion.json",
        {
            "status": "complete",
            "selection": binding(current / "screen/selection.json"),
            "decision": decision["status"],
            "cells": 6,
            "cached_usage_unchanged": True,
        },
    )
    dispositions[FAMILY] = {"status": "complete", "decision": decision["status"], "cells": 6}
    helpers.write(args.root / "dispositions.json", dispositions)
    helpers.freeze(
        args.root / "completion.json",
        {
            "status": "complete",
            "dispositions": dispositions,
            "cumulative_budget": binding(current / "budget.json"),
            "source": receipt["source"],
            "budget_limits_changed": False,
            "generate_rationales": False,
        },
    )
    helpers.status(
        "complete", cumulative_budget=str(current / "budget.json"), dispositions=dispositions
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    args.root, args.code_root = args.root.resolve(), args.code_root.resolve()
    helpers = load_helpers(args.root)
    helpers.PARENT = args.root.parent / "next-batch-recovery-03"
    helpers.ORDER = [FAMILY]
    helpers.check_pause()
    if args.preflight:
        preflight(args, helpers)
        return 0
    with (args.root / "continuation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(args, helpers)
        except BaseException as exc:
            budget = args.root / FAMILY / "runtime/exact-om-focused-v2/budget.json"
            helpers.status(
                "blocked",
                cumulative_budget=str(budget if budget.exists() else latest_budget(args.root)),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
