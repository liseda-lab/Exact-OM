#!/usr/bin/env python3
"""Recover the predeclared cached E01/E10 queue without changing numerical code."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

OLD_COMMIT = "9329389656a982149ce888bf134eff5f45b9cd29"
FAILED_SETUP_START = 1790320214.0098376
FAILED_SETUP_END = 1790322580.335175
FAILED_SETUP_SECONDS = FAILED_SETUP_END - FAILED_SETUP_START
RECONCILIATION_ID = "recovery/next-batch-recovery-02/E01/pre-admission-setup"


def read(path):
    return json.loads(Path(path).read_text())


def load_helpers(root):
    """Import frozen helpers without running their non-resumable main function."""
    old = root.parent / "next-batch-recovery-02"
    sys.path.insert(0, str(old))
    path = old / "continue.py"
    spec = importlib.util.spec_from_file_location("retained_cached_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.HERE, module.PARENT = root, old
    return module


def corrected_lock(parent, helpers):
    """Carry forward the already amended blueprint; preserve every scientific field."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping

    lock = copy.deepcopy(load_yaml_mapping(helpers.PARENT / "E01/campaign.lock.yaml"))
    prior = load_yaml_mapping(parent)
    inherited = copy.deepcopy(prior["blueprint"])
    inherited["path"] = str((Path(parent).parent / inherited["path"]).resolve())
    before = copy.deepcopy(load_yaml_mapping(helpers.verify(lock["blueprint"])))
    after = copy.deepcopy(load_yaml_mapping(helpers.verify(inherited)))
    if before["profiles"]["core_14d"]["llm_request_planning_cap"] != 20000:
        raise ValueError("Unexpected stale blueprint; recovery needs its declared parent")
    before["profiles"]["core_14d"]["llm_request_planning_cap"] = 50000
    if before != after:
        raise ValueError("Blueprint differs beyond the recorded request-cap amendment")
    lock["blueprint"] = inherited
    return lock


def campaign_limits(lock):
    """Construct exactly the limits used by the unchanged campaign runner."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import InputBinding

    blueprint = load_yaml_mapping(InputBinding.model_validate(lock["blueprint"]).verify(Path(".")))
    profile = blueprint["profiles"][lock["profile"]]
    envelopes = dict(blueprint["budget_envelopes_core_hours"])
    if lock["profile"] == "extended_21d":
        for name, extra in blueprint["budget_envelopes_extended_extra_hours"].items():
            envelopes[name] += extra
    return {
        "envelopes_hours": envelopes,
        "node_hours_cap": profile["node_hours_cap"],
        "requests_cap": profile["llm_request_planning_cap"],
        "tokens_cap": profile["llm_token_planning_cap"],
        "final_requests_reserved": lock["final_requests_reserved"],
        "final_tokens_reserved": lock["final_tokens_reserved"],
    }


def reconcile_budget(state, start, end):
    """Retain all prior rows and add the failed pre-admission setup upper bound."""
    from exact.experiments.budget import interval_seconds

    result = copy.deepcopy(state)
    seconds = interval_seconds([[start, end]])
    if RECONCILIATION_ID in result["work"]:
        raise ValueError("Failed setup has already been reconciled")
    active = [
        key
        for key, item in result["work"].items()
        if item["status"] == "reserved" and key != "historical/G0"
    ]
    if active:
        raise ValueError("Previous work has unclosed reservations: " + str(active))
    result["work"][RECONCILIATION_ID] = {
        "group": "decisions",
        "status": "failed",
        "start": start,
        "end": end,
        "seconds": seconds,
        "requests": 0,
        "tokens": 0,
        "projected_usd": 0,
        "actual_usd": 0,
        "accounting": "Conservative elapsed setup upper bound after retained copy accounting ended and before budget-mismatch failure; no comparison was admitted.",
    }
    result["intervals"].append([start, end])
    result["node_seconds"] = interval_seconds(result["intervals"])
    result["elapsed_seconds"] = max(row[1] for row in result["intervals"]) - min(
        row[0] for row in result["intervals"]
    )
    return result


def copy_reconciled_state(helpers, source, destination, interval=None):
    """Publish inherited missing costs before a cache-copy failure can occur."""
    original = helpers.freeze

    def freeze_accounted(path, value):
        if path == destination / "budget.json" and interval is not None:
            value = reconcile_budget(value, *interval)
        return original(path, value)

    helpers.freeze = freeze_accounted
    try:
        helpers.copy_state(source, destination)
    finally:
        helpers.freeze = original


def comparison_forecast(rows, arms, measurement, helpers):
    """Keep arm setup once; account standalone native inventory once per family."""
    from exact.experiments.campaign import WorkEstimate

    estimate = helpers.estimate(rows, arms, measurement)
    comparison = WorkEstimate.model_validate(estimate).seconds()
    inventory = FAILED_SETUP_SECONDS * 1.5
    return {
        "estimate": estimate,
        "comparison_seconds": comparison,
        "inventory_seconds": inventory,
        "whole_family_seconds": comparison + inventory,
        "hours": (comparison + inventory) / 3600,
    }


def verify_nested_bindings(value, helpers):
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            helpers.verify({key: value[key] for key in ("path", "sha256")})
        else:
            for child in value.values():
                verify_nested_bindings(child, helpers)
    elif isinstance(value, list):
        for child in value:
            verify_nested_bindings(child, helpers)


def set_family_ready(lock, name, estimate, commit):
    step = next(row for row in lock["steps"] if row["id"] == name)
    step["estimate"] = estimate
    for states in step["readiness"].values():
        states["screen"].update(
            status="screen_ready",
            inspected_commit=commit,
            missing=[],
            reason="Retained full D0 source qualification; unchanged numerical bytes; measured complete-family admission includes standalone native inventory once.",
        )
    return step


def check_admission(state, group, seconds, directory, work_id):
    from exact.experiments.budget import BudgetLedger

    path = directory / "budget-admission.json"
    path.write_text(json.dumps(state, sort_keys=True))
    BudgetLedger(path, state["limits"]).admit(work_id, group=group, seconds=seconds)
    return path


def preflight(args, helpers):
    from qualification import binding

    from exact.experiments import harness
    from exact.experiments.campaign import (
        campaign_plan,
        materialize_campaign,
        validate_comparison_cells,
    )

    helpers.check_pause()
    plan_path = args.root / "reuse-plan.json"
    recovery = read(plan_path)
    budget = helpers.verify(recovery["accounting"]["latest_budget"])
    expected = helpers.PARENT / "E01/runtime/exact-om-focused-v2/budget.json"
    if budget != expected or recovery["old_source_commit"] != OLD_COMMIT:
        raise ValueError("Recovery does not bind the latest failed family and exact source")
    identity = helpers.validate_source(args.code_root)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", OLD_COMMIT, "--", "exact"], cwd=args.code_root, text=True
    ).strip()
    if changed:
        raise ValueError("Numerical implementation changed; retained qualification is invalid")
    qualification_path = helpers.PARENT / "qualification/source-qualification.json"
    qualification = read(qualification_path)
    if qualification["status"] != "passed" or qualification["source"]["commit"] != OLD_COMMIT:
        raise ValueError("Retained source qualification is not passed")
    for field in ("extraction_code", "native_code_sha256"):
        if identity[field] != qualification["source"][field]:
            raise ValueError("Retained qualification source/native identity changed: " + field)
    verify_nested_bindings(qualification, helpers)
    old_control = read(helpers.verify(qualification["old_control"]))
    new = read(helpers.verify(qualification["new_measurement"]))
    verify_nested_bindings(new, helpers)
    rows = [dict(old_control["measurement"], prefix=False), new]
    if any(
        row.get("status") != "passed"
        or row.get("execution_status") != "complete"
        or row.get("prefix")
        or any(row.get("new_usage", {}).values())
        for row in rows
    ):
        raise ValueError("Full cached controls and zero-incremental usage are required")
    verify_nested_bindings(rows, helpers)
    state = read(budget)
    accounting = state["work"].get(new["budget_work_id"], {})
    if accounting.get("status") != "complete":
        raise ValueError("Qualification has no retained completed accounting")
    reconciled = reconcile_budget(state, FAILED_SETUP_START, FAILED_SETUP_END)
    e09 = helpers.DATA / "prepared-campaign-13/campaign.lock.yaml"
    lock = corrected_lock(e09, helpers)
    if campaign_limits(lock) != state["limits"]:
        raise ValueError("Corrected campaign and cumulative budget limits differ")
    completed = read(helpers.DATA / "mechanism-screen-03/completion.json")
    if completed.get("status") != "complete" or completed.get("exit_code") != 0:
        raise ValueError("E09 prerequisite is not complete")
    for item in [completed["selection"], *completed["completed_cells"]]:
        helpers.verify(item)
    selection = read(
        helpers.DATA / "mechanism-screen-03/runtime/exact-om-focused-v2/screen/selection.json"
    )
    checks = {}
    for name in helpers.ORDER:
        trial = copy.deepcopy(lock)
        # The failed E01 declaration is retained as the base, but only this trial runs.
        for step in trial["steps"]:
            if step["id"] in helpers.ORDER:
                step["estimate"] = None
                for states in step["readiness"].values():
                    states["screen"].update(
                        status="blocked_input_resolution",
                        reason="Other queued family",
                        missing=["Other queued family"],
                    )
        step = next(row for row in trial["steps"] if row["id"] == name)
        proof = comparison_forecast(
            rows, len(step["arms"]), helpers.PARENT / "E01/measured-cost.json", helpers
        )
        set_family_ready(trial, name, proof["estimate"], identity["commit"])
        with tempfile.TemporaryDirectory(prefix="exact-cached-recovery-preflight-") as temporary:
            temp = Path(temporary)
            check_admission(
                reconciled,
                step["budget_group"],
                proof["whole_family_seconds"],
                temp,
                "planned/" + name,
            )
            path = temp / "campaign.yaml"
            helpers.freeze(path, trial)
            plan = campaign_plan(path, stage="screen")
            selected = [row for row in plan["rows"] if row["step"] == name]
            if plan["budget_errors"] or any(row["issues"] for row in selected):
                raise ValueError("Configuration preflight failed: " + str(selected))
            suite = materialize_campaign(path, temp / "declarations", stage="screen")
            source = next(row for row in suite.sources if row.config.experiment_id == name)
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
            if len(cells) != len(step["arms"]):
                raise ValueError("Incomplete comparison")
            for cell in cells:
                if (
                    cell.generate_rationales
                    or cell.split_role != "development"
                    or cell.resolved_config["llm"]["experiment"]["gate"]["mode"] != "off"
                    or set(cell.resolved_config["data"]["refs"]) - {"train", "valid"}
                ):
                    raise ValueError("Scientific role/rationale/decision policy changed")
            checks[name] = {
                "cells": len(cells),
                "config_hashes": {cell.arm_id: cell.config_hash for cell in cells},
                "forecast": proof,
            }
    receipt = {
        "status": "preflight_passed_not_admitted",
        "source": identity,
        "parent_attempt": str(helpers.PARENT),
        "queue": helpers.ORDER,
        "budget_import": binding(budget),
        "qualification": binding(qualification_path),
        "saved_campaign": binding(helpers.PARENT / "E01/campaign.lock.yaml"),
        "e09_campaign": binding(e09),
        "blueprint": lock["blueprint"],
        "e09_completion": binding(helpers.DATA / "mechanism-screen-03/completion.json"),
        "reuse_plan": binding(plan_path),
        "cell_preflight": checks,
        "operational_files": [
            binding(Path(__file__)),
            binding(helpers.PARENT / "continue.py"),
            binding(helpers.PARENT / "qualification.py"),
        ],
        "failed_setup_interval": [FAILED_SETUP_START, FAILED_SETUP_END],
        "budget_limits_changed": False,
        "generate_rationales": False,
        "private_reference_contents_accessed": False,
    }
    helpers.freeze(args.root / "preflight.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "queue": receipt["queue"],
                "forecast_hours": {k: v["forecast"]["hours"] for k, v in checks.items()},
            }
        ),
        flush=True,
    )
    return receipt


@contextmanager
def account_inventory(helpers, runtime, family, group, forecast):
    from qualification import usage

    from exact.experiments import harness
    from exact.experiments.budget import BudgetLedger

    original = harness.build_dataset_inventory
    called = False

    def measured(*args, **kwargs):
        nonlocal called
        if called:
            raise ValueError("Standalone inventory must run once per family")
        called = True
        helpers.check_pause()
        path = runtime / "budget.json"
        account = BudgetLedger(path, read(path)["limits"])
        work_id = "preparation/" + helpers.HERE.name + "/" + family + "/inventory"
        account.admit(work_id, group=group, seconds=forecast)
        before, start, status = usage(runtime / "openrouter"), time.time(), "failed"
        try:
            result = original(*args, **kwargs)
            status = "complete"
            return result
        finally:
            after = usage(runtime / "openrouter")
            if before != after:
                status = "failed"
            account.finish(
                work_id,
                start=start,
                end=time.time(),
                status=status,
                requests=after["attempts"] - before["attempts"],
                tokens=after["billable_tokens"] - before["billable_tokens"],
                actual_usd=(
                    None
                    if after["unpriced_attempts"] != before["unpriced_attempts"]
                    else after["reported_cost_usd"] - before["reported_cost_usd"]
                ),
            )
            if before != after:
                raise ValueError("Cached-only inventory produced incremental hosted usage")

    harness.build_dataset_inventory = measured
    try:
        yield
    finally:
        harness.build_dataset_inventory = original


def run(args, helpers):
    from qualification import binding, usage

    from exact.experiments.campaign import (
        CampaignLock,
        campaign_plan,
        execute_campaign,
        external_selection_result,
    )

    helpers.check_pause()
    if (
        os.environ.get("SLURM_JOB_ID") != "14372"
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise ValueError("Use a detached numeric Slurm step in existing allocation 14372")
    receipt = read(args.root / "preflight.json")
    if receipt["status"] != "preflight_passed_not_admitted":
        raise ValueError("A passed frozen preflight is required")
    for item in [
        receipt["budget_import"],
        receipt["qualification"],
        receipt["saved_campaign"],
        receipt["e09_campaign"],
        receipt["e09_completion"],
        receipt["reuse_plan"],
        *receipt["operational_files"],
    ]:
        helpers.verify(item)
    if helpers.validate_source(args.code_root) != receipt["source"]:
        raise ValueError("Preflight source identity changed")
    if any((args.root / name / "runtime").exists() for name in helpers.ORDER):
        raise FileExistsError("Recovery runtime exists; use its saved campaign continuation")
    current = Path(receipt["budget_import"]["path"]).parent
    lock = corrected_lock(Path(receipt["e09_campaign"]["path"]), helpers)
    if campaign_limits(lock) != read(current / "budget.json")["limits"]:
        raise ValueError("Campaign limits changed after preflight")
    qualification = read(Path(receipt["qualification"]["path"]))
    rows = [
        dict(read(helpers.verify(qualification["old_control"]))["measurement"], prefix=False),
        read(helpers.verify(qualification["new_measurement"])),
    ]
    # The retained launch owns these settings and reads its credential internally.
    retained_launch = read(helpers.DATA / "d1-production-probe-01/launch.json")
    os.environ.update(retained_launch["environment"])
    os.environ["PYTHONPATH"] = str(args.code_root)
    os.environ["OPENROUTER_API_KEY"] = (helpers.PROJECT / "api_key").read_text().strip()
    dispositions = {
        name: {"status": "blocked_input_resolution", "reason": reason}
        for name, reason in helpers.DEFERRED.items()
    }
    for name in helpers.ORDER:
        helpers.check_pause()
        wave = args.root / name
        step = next(row for row in lock["steps"] if row["id"] == name)
        setup_started = False
        try:
            measurement = wave / "measured-cost.json"
            helpers.freeze(
                measurement,
                {
                    "status": "measured_forecast",
                    "step": name,
                    "rows": rows,
                    "qualification": receipt["qualification"],
                    "inventory_interval": receipt["failed_setup_interval"],
                    "method": "Maximum measured complete D0 duration per arm with setup once; factor 1.5. Standalone native dataset inventory elapsed upper bound separately forecast once per family with factor 1.5.",
                },
            )
            proof = comparison_forecast(rows, len(step["arms"]), measurement, helpers)
            helpers.freeze(wave / "forecast.json", proof)
            state = read(current / "budget.json")
            candidate = (
                reconcile_budget(state, *receipt["failed_setup_interval"])
                if name == "E01"
                else state
            )
            with tempfile.TemporaryDirectory(prefix="exact-cached-admission-") as temporary:
                check = check_admission(
                    candidate,
                    step["budget_group"],
                    proof["whole_family_seconds"],
                    Path(temporary),
                    "planned/" + name,
                )
                helpers.freeze(wave / "budget-admission.json", read(check))
            free = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
            ).splitlines()
            physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
            if (
                len(free) != 1
                or proof["estimate"]["peak_ram_gb"] > min(56, physical - 6)
                or proof["estimate"]["peak_vram_gb"] > float(free[0]) / 1024
            ):
                raise ValueError("Measured complete-family memory forecast does not fit")
            set_family_ready(lock, name, proof["estimate"], receipt["source"]["commit"])
            campaign = wave / "campaign.lock.yaml"
            helpers.freeze(campaign, lock)
            plan = campaign_plan(campaign, stage="screen")
            ready = {
                row["step"]
                for row in plan["rows"]
                if row["status"] == "screen_ready" and not row["issues"]
            }
            if plan["budget_errors"] or ready != {name}:
                raise ValueError("Exactly one complete family must be runnable: " + repr(ready))
            destination = wave / "runtime/exact-om-focused-v2"
            setup_started = True
            helpers.status(
                "preparing_recovery", step=name, cumulative_budget=str(current / "budget.json")
            )
            copy_reconciled_state(
                helpers,
                current,
                destination,
                receipt["failed_setup_interval"] if name == "E01" else None,
            )
            current = destination
            if campaign_limits(lock) != read(current / "budget.json")["limits"]:
                raise ValueError("Copied cumulative limits differ")
            with tempfile.TemporaryDirectory(prefix="exact-cached-final-admission-") as temporary:
                check_admission(
                    read(current / "budget.json"),
                    step["budget_group"],
                    proof["whole_family_seconds"],
                    Path(temporary),
                    "planned/" + name,
                )
            helpers.freeze(
                wave / "launch.json",
                {
                    "campaign": binding(campaign),
                    "source": receipt["source"],
                    "qualification": receipt["qualification"],
                    "budget_path": str(current / "budget.json"),
                    "slurm_job_id": "14372",
                    "slurm_step_id": os.environ["SLURM_STEP_ID"],
                    "reuse_plan": receipt["reuse_plan"],
                    "forecast": binding(wave / "forecast.json"),
                },
            )
            helpers.status(
                "running",
                step=name,
                campaign=str(campaign),
                cumulative_budget=str(current / "budget.json"),
            )
            cached = usage(current / "openrouter")
            os.environ["EXACT_OPENROUTER_REQUEST_CAP"] = str(cached["attempts"])
            os.environ["EXACT_OPENROUTER_TOKEN_CAP"] = str(cached["billable_tokens"])
            os.environ["EXACT_OPENROUTER_RETRY_UNKNOWN"] = "0"
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
            with account_inventory(
                helpers, current, name, step["budget_group"], proof["inventory_seconds"]
            ):
                helpers.guarded_execute(campaign, wave, args.code_root)
            if usage(current / "openrouter") != cached:
                raise ValueError("Cached-only comparison produced incremental hosted usage")
            helpers.history_record(lock, name, campaign, current, args.root / "history")
            parsed = CampaignLock.model_validate(lock)
            decision = external_selection_result(
                parsed, next(row for row in parsed.steps if row.id == name), args.root
            )
            helpers.freeze(
                wave / "completion.json",
                {
                    "status": "complete",
                    "selection": binding(current / "screen/selection.json"),
                    "decision": decision["status"],
                    "cells": len(step["arms"]),
                    "cached_usage_unchanged": True,
                },
            )
            dispositions[name] = {
                "status": "complete",
                "decision": decision["status"],
                "cells": len(step["arms"]),
            }
        except Exception as exc:
            dispositions[name] = {
                "status": (
                    "deferred_budget"
                    if "deferred_budget" in str(exc)
                    else "blocked_input_resolution"
                ),
                "reason": str(exc),
            }
            helpers.write(args.root / "dispositions.json", dispositions)
            if name == "E01" or setup_started:
                raise
            step["estimate"] = None
            for states in step["readiness"].values():
                states["screen"].update(
                    status=dispositions[name]["status"], reason=str(exc), missing=[str(exc)]
                )
        helpers.write(args.root / "dispositions.json", dispositions)
    helpers.freeze(
        args.root / "completion.json",
        {
            "status": "complete",
            "dispositions": dispositions,
            "cumulative_budget": binding(current / "budget.json"),
            "generate_rationales": False,
            "budget_limits_changed": False,
            "source": receipt["source"],
        },
    )
    helpers.status(
        "complete", dispositions=dispositions, cumulative_budget=str(current / "budget.json")
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
    sys.path.insert(0, str(args.code_root))
    helpers = load_helpers(args.root)
    helpers.check_pause()
    if args.preflight:
        preflight(args, helpers)
        return 0
    with (args.root / "continuation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(args, helpers)
        except BaseException as exc:
            candidates = [
                args.root / name / "runtime/exact-om-focused-v2/budget.json"
                for name in reversed(helpers.ORDER)
            ]
            latest = next(
                (path for path in candidates if path.exists()),
                helpers.PARENT / "E01/runtime/exact-om-focused-v2/budget.json",
            )
            helpers.status(
                "blocked",
                cumulative_budget=str(latest),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
