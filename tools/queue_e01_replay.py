#!/usr/bin/env python3
"""Queue the approved six-arm E01 extraction replay after the active E10 owner."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.resume_cached_batch import (  # noqa: E402
    campaign_limits,
    check_admission,
    load_helpers,
    read,
)

BASE_COMMIT = "e9c7fd11bc6769b0912451a55702dd0897caf593"
UPSTREAM = "B1-E10-analytic-recovery-04"
ARMS = (
    "threshold_unrestricted",
    "greedy",
    "mutual_best",
    "stable_marriage",
    "assignment_accepted_utility",
    "assignment_legacy",
)
AMENDMENT = (
    "2026-09-25 user-approved E01 amendment: unrestricted threshold baseline versus "
    "one-to-one extractors; shared lexical-anchor conflict competition, unchanged "
    "pair scores, threshold, source sample, eligible entities and reference roles. "
    "Legacy assignment remains diagnostic. No private test optimization."
)


def amend_lock(saved, blueprint, estimate, commit):
    """Change only the authorized E01 declaration; preserve other scientific settings."""
    lock = copy.deepcopy(saved)
    lock["blueprint"] = blueprint
    step = next(row for row in lock["steps"] if row["id"] == "E01")
    old = {arm["id"]: arm for arm in step["arms"]}
    if len(old) != len(step["arms"]) or set(old) != set(ARMS[1:]) or step.get("external_selection"):
        raise ValueError("Expected the uncompleted original five-arm E01 comparison")
    baseline = copy.deepcopy(old["greedy"])
    baseline["id"] = ARMS[0]
    step["arms"] = [baseline, *[old[name] for name in ARMS[1:]]]
    step["readiness"][ARMS[0]] = copy.deepcopy(step["readiness"]["greedy"])
    for arm in step["arms"]:
        name = arm["id"]
        arm["role"] = (
            "baseline"
            if name == ARMS[0]
            else (
                "control"
                if name == "greedy"
                else "diagnostic" if name == "assignment_legacy" else "candidate"
            )
        )
        arm["required_control"] = name in ARMS[:2]
        arm["deployable"] = name != "assignment_legacy"
        matching = arm["overlay"].setdefault("matching", {})
        matching.update(
            cardinality=None if name == ARMS[0] else 1,
            target_cardinality=None if name == ARMS[0] else 1,
        )
        matching.setdefault("extraction", {}).update(
            mode="threshold" if name == ARMS[0] else name,
            anchor_conflict_policy="compete",
        )
        step["readiness"][name]["screen"].update(
            status="screen_ready",
            inspected_commit=commit,
            missing=[],
            reason="Bound saved scores and unchanged eligibility; tested extraction-only amendment.",
        )
    decision = step["selection"]["decisions"][0]
    decision.update(baseline=ARMS[0], candidates=list(ARMS[1:-1]), required_controls=list(ARMS[:2]))
    for tie in decision["tie_breaks"]:
        if tie["kind"] == "arm_order":
            tie["order"] = list(ARMS[1:-1])
    step["policy_paths"] = [
        "matching.extraction",
        "matching.cardinality",
        "matching.target_cardinality",
    ]
    step["estimate"] = estimate
    step["design"]["assumptions"].append(AMENDMENT)
    step["design"][
        "primary_comparison"
    ] = "E01 unrestricted threshold versus constrained extraction"
    for other in lock["steps"]:
        if other["id"] == "E10-analytic":
            other["estimate"] = None
            for states in other["readiness"].values():
                states["screen"].update(
                    status="blocked_input_resolution",
                    missing=["Owned by independent E10 queue"],
                    reason="Executed separately; E01 waits for its final cumulative ledger.",
                )
    return lock


def successor(registry, name):
    by_id = {row["id"]: row for row in registry["runs"]}
    seen = set()
    while True:
        if name in seen or name not in by_id:
            raise ValueError("Invalid upstream replacement chain")
        seen.add(name)
        row = by_id[name]
        if not row.get("superseded_by"):
            if not row.get("enabled", True):
                raise ValueError("Upstream disabled without a replacement")
            return row
        name = row["superseded_by"]


def closed_budget(state, limits):
    if state["limits"] != limits:
        raise ValueError("Cumulative limits or protected allowance changed")
    active = [
        key
        for key, row in state["work"].items()
        if row["status"] == "reserved" and key != "historical/G0"
    ]
    if active:
        raise ValueError("Previous budget still has active owners: " + repr(active))


def live_steps():
    return set(
        subprocess.check_output(
            ["squeue", "--steps", "-h", "-j", "14372", "-o", "%i"],
            text=True,
        ).split()
    )


def wait_for_upstream(helpers):
    """Follow registered recoveries, never copy a running owner's mutable ledger."""
    last = None
    while True:
        helpers.check_pause()
        upstream = successor(read(helpers.DATA / "hourly-supervisor-01/registry.json"), UPSTREAM)
        completion = Path(upstream["completion_path"])
        state = read(upstream["status_path"]) if Path(upstream["status_path"]).exists() else {}
        if completion.exists() and upstream["step_id"] not in live_steps():
            result = read(completion)
            if result.get("status") == "complete" and result.get("exit_code", 0) == 0:
                exit_path = Path(upstream["exit_path"])
                if not exit_path.exists():
                    time.sleep(1)
                    continue
                if int(exit_path.read_text()) != 0:
                    raise ValueError("Upstream completion lacks successful launcher exit")
                import psutil

                old_root = str(completion.parent)
                for process in psutil.process_iter(["pid", "name", "cmdline"]):
                    if (
                        process.info["pid"] != os.getpid()
                        and any(
                            (part == old_root or part.startswith(old_root + "/"))
                            for part in (process.info.get("cmdline") or [])
                        )
                        and (process.info.get("name") or "").startswith(("python", "bash", "srun"))
                    ):
                        raise ValueError("Upstream launcher/descendant remains alive")
                return upstream, result
        current = (upstream["id"], state.get("status"))
        if current != last:
            helpers.status(
                "waiting_dependency",
                upstream=upstream["id"],
                upstream_step=upstream["step_id"],
                upstream_status=state.get("status"),
            )
            last = current
        time.sleep(30)


def source_identity(code_root, helpers):
    identity = helpers.validate_source(code_root)
    allowed = {
        "exact/impl/extraction.py",
        "exact/core/entities/configs/experimental.py",
        "exact/impl/trainer/runner.py",
        "exact/impl/trainer/checkpointing.py",
        "exact/impl/trainer/audit_io.py",
        "exact/core/contracts/trainer.py",
        "exact/experiments/extraction_replay.py",
        "exact/experiments/campaign.py",
        "exact/experiments/preparation.py",
        "exact/experiments/core_recipes.py",
        "exact/experiments/difference_replay.py",
        # Reviewed selection-schema binding repair; predictions are unchanged.
        "exact/experiments/harness.py",
    }
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", BASE_COMMIT, "--", "exact"],
            cwd=code_root,
            text=True,
        ).splitlines()
    )
    if changed - allowed:
        raise ValueError("Unreviewed upstream numerical changes: " + repr(changed - allowed))
    prior = read(helpers.DATA / "next-batch-recovery-04/preflight.json")["source"]
    if identity["native_code_sha256"] != prior["native_code_sha256"]:
        raise ValueError("Native ontology implementation changed")
    return identity


def preflight(args, helpers):
    from qualification import binding

    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.campaign import (
        campaign_plan,
        materialize_campaign,
        validate_comparison_cells,
    )
    from exact.experiments.extraction_replay import build_packet, validate_packet

    helpers.check_pause()
    identity = source_identity(args.code_root, helpers)
    parent = helpers.DATA / "next-batch-recovery-03/E01"
    saved_path = parent / "campaign.lock.yaml"
    saved = load_yaml_mapping(saved_path)
    blueprint = load_yaml_mapping(helpers.verify(saved["blueprint"]))
    declared = load_yaml_mapping(args.code_root / "specs/experiments/campaign-v2.yaml")
    old_family = next(row for row in blueprint["experiments"] if row["id"] == "E01")
    new_family = next(row for row in declared["experiments"] if row["id"] == "E01")
    for key in ("initial_treatment_cap", "treatments"):
        old_family[key] = new_family[key]
    helpers.freeze(args.root / "blueprint.yaml", blueprint)
    measurement = parent / "measured-cost.json"
    estimate = copy.deepcopy(read(parent / "forecast.json")["estimate"])
    estimate["cold_seconds"] *= 6 / 5
    estimate["measurement_artifact"] = binding(measurement)
    helpers.verify(estimate["measurement_artifact"])
    campaign = args.root / "E01/campaign.lock.yaml"
    lock = amend_lock(saved, binding(args.root / "blueprint.yaml"), estimate, identity["commit"])
    helpers.freeze(campaign, lock)
    plan = campaign_plan(campaign, stage="screen")
    ready = {
        row["step"] for row in plan["rows"] if row["status"] == "screen_ready" and not row["issues"]
    }
    if plan["budget_errors"] or ready != {"E01"}:
        raise ValueError("Exactly the complete amended E01 must be runnable: " + repr(ready))
    old_runs = parent / "runtime/exact-om-focused-v2/screen/runs/E01"
    suffix = Path("D0-global_alignment/seed-17")
    packet = args.root / "score-packet.json"
    if not packet.exists():
        build_packet(
            old_runs / "assignment_accepted_utility" / suffix,
            packet,
            evidence_run=old_runs / "greedy" / suffix,
        )
    scores = [
        binding(
            old_runs
            / name
            / suffix
            / "checkpoints/inference_additional_models_8588305bdb2a.jsonl.zst"
        )
        for name in ARMS[1:]
    ]
    if len({row["sha256"] for row in scores}) != 1:
        raise ValueError("Original E01 arms do not share bit-identical scored pairs")
    inventory = parent / "runtime/exact-om-focused-v2/screen/dataset_inventory.json"
    with tempfile.TemporaryDirectory(prefix="exact-e01-preflight-") as temporary:
        suite = materialize_campaign(campaign, Path(temporary) / "declarations", stage="screen")
        source = next(row for row in suite.sources if row.config.experiment_id == "E01")
        selected = read(
            helpers.verify(read(helpers.DATA / "mechanism-screen-03/completion.json")["selection"])
        )
        cells = harness.build_cells(
            suite,
            source,
            stage="screen",
            output_root=Path(temporary),
            inherited_overlay=harness.inherited_selection_overlay(
                selected, source.config.depends_on
            ),
        )
        validate_comparison_cells(cells, source)
        validate_packet(read(packet), cells[0].resolved_config)
        if len(cells) != 6 or {cell.arm_id for cell in cells} != set(ARMS):
            raise ValueError("Incomplete amended comparison")
        for cell in cells:
            if cell.generate_rationales or cell.split_role != "development":
                raise ValueError("Rationale/role changed")
    receipt = {
        "status": "preflight_passed_waiting_for_latest_budget",
        "source": identity,
        "amendment": AMENDMENT,
        "campaign": binding(campaign),
        "packet": binding(packet),
        "inventory": binding(inventory),
        "parent_campaign": binding(saved_path),
        "operational_files": [
            binding(args.root / name) for name in ("run-step.sh", "worker-entry.sh", "submit.sh")
        ]
        + [
            binding(helpers.DATA / "next-batch-recovery-02" / name)
            for name in ("continue.py", "qualification.py")
        ],
        "shared_scores": scores,
        "upstream_id": UPSTREAM,
        "forecast": {
            "seconds": estimate["cold_seconds"] * estimate["safety_factor"],
            "method": "Six retained full-pipeline D0 wall bounds, factor 1.5; no replay speedup assumed.",
        },
        "budget_limits": campaign_limits(lock),
        "generate_rationales": False,
        "reused": ["candidate_pool", "eligible_entities", "pair_scores", "dataset_inventory"],
        "recomputed": ["decisions", "extraction", "evaluation", "reports"],
    }
    helpers.freeze(args.root / "preflight.json", receipt)
    return receipt


@contextmanager
def replay_execution(receipt, helpers):
    """Keep the standard harness; substitute only verified scoring and inventory inputs."""
    from exact.experiments import harness

    packet = helpers.verify(receipt["packet"])
    inventory = read(helpers.verify(receipt["inventory"]))
    original_run, original_inventory = harness._run_subprocess, harness.build_dataset_inventory

    def run(command, **kwargs):
        if (
            len(command) != 4
            or Path(command[1]).name != "run_exact_job.py"
            or command[2] != "--run-config"
        ):
            raise ValueError("Extraction replay received an unexpected worker command")
        command = [
            command[0],
            "-m",
            "exact.experiments.extraction_replay",
            "--run-config",
            command[3],
            "--packet",
            str(packet),
        ]
        return original_run(command, **kwargs)

    def reuse_inventory(suite, *, stage, output_root, selection_record=None):
        if stage != "screen":
            raise ValueError("Saved development inventory cannot serve final reporting")
        rows = inventory["rows"]
        verified = set()
        for row in rows:
            for item in row.get("inputs", {}).values():
                if isinstance(item, dict) and item.get("sha256"):
                    key = (item["path"], item["sha256"])
                    if key not in verified:
                        helpers.verify({name: item[name] for name in ("path", "sha256")})
                        verified.add(key)
        harness._write_inventory(suite, stage=stage, output_root=output_root, rows=rows)
        return rows

    harness._run_subprocess, harness.build_dataset_inventory = run, reuse_inventory
    try:
        yield
    finally:
        harness._run_subprocess, harness.build_dataset_inventory = original_run, original_inventory


def run(args, helpers):
    from qualification import binding, usage

    from exact.experiments.budget import BudgetLedger
    from exact.experiments.campaign import execute_campaign

    if (
        os.environ.get("SLURM_JOB_ID") != "14372"
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise ValueError("Use a numeric Slurm step within allocation 14372")
    receipt = read(args.root / "preflight.json")
    if source_identity(args.code_root, helpers) != receipt["source"]:
        raise ValueError("Frozen execution source changed")
    for key in ("campaign", "packet", "inventory", "parent_campaign"):
        helpers.verify(receipt[key])
    for item in receipt["operational_files"]:
        helpers.verify(item)
    wave = args.root / "E01"
    current = wave / "runtime/exact-om-focused-v2"
    if not current.exists():
        upstream, completion = wait_for_upstream(helpers)
        source_budget = helpers.verify(completion["cumulative_budget"])
        state = read(source_budget)
        closed_budget(state, receipt["budget_limits"])
        with tempfile.TemporaryDirectory(prefix="exact-e01-admission-") as temporary:
            check_admission(
                state, "decisions", receipt["forecast"]["seconds"], Path(temporary), "planned/E01"
            )
        helpers.check_pause()
        current.mkdir(parents=True)
        helpers.freeze(current / "budget.json", state)
        ledger = BudgetLedger(current / "budget.json", state["limits"])
        work_id = "preparation/" + args.root.name
        ledger.admit(work_id, group="reserve", seconds=300)
        started, outcome = time.time(), "failed"
        try:
            source_db = source_budget.parent / "openrouter/requests.sqlite3"
            destination = current / "openrouter/requests.sqlite3"
            destination.parent.mkdir()
            with sqlite3.connect(
                source_db.as_uri() + "?mode=ro", uri=True
            ) as before, sqlite3.connect(destination) as after:
                before.backup(after)
                if after.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise ValueError("Imported request ledger failed integrity check")
            helpers.freeze(
                args.root / "budget-import.json",
                {
                    "source_budget": binding(source_budget),
                    "upstream": upstream,
                    "request_cache": binding(source_db),
                    "limits_changed": False,
                },
            )
            outcome = "complete"
        finally:
            ledger.finish(
                work_id,
                start=started,
                end=time.time(),
                status=outcome,
                requests=0,
                tokens=0,
                actual_usd=0,
            )
    elif not (args.root / "budget-import.json").exists():
        raise ValueError("Incomplete budget import needs explicit recovery")
    state = read(current / "budget.json")
    closed_budget(state, receipt["budget_limits"])
    cached = usage(current / "openrouter")
    os.environ.update(
        EXACT_OPENROUTER_REQUEST_CAP=str(cached["attempts"]),
        EXACT_OPENROUTER_TOKEN_CAP=str(cached["billable_tokens"]),
        EXACT_OPENROUTER_RETRY_UNKNOWN="0",
        CUDA_VISIBLE_DEVICES="",
    )
    helpers.status("running", step="E01", cumulative_budget=str(current / "budget.json"))
    campaign = helpers.verify(receipt["campaign"])
    execute_campaign(
        campaign,
        stage="screen",
        output_root=wave / "runtime",
        workdir=args.code_root,
        jobs=1,
        reuse_plan_only=True,
    )
    with replay_execution(receipt, helpers):
        helpers.guarded_execute(campaign, wave, args.code_root)
    if usage(current / "openrouter") != cached:
        raise ValueError("Extraction-only replay changed hosted usage")
    helpers.freeze(
        args.root / "completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "cells": 6,
            "selection": binding(current / "screen/selection.json"),
            "cumulative_budget": binding(current / "budget.json"),
            "source": receipt["source"],
            "amendment": AMENDMENT,
            "cached_usage_unchanged": True,
            "generate_rationales": False,
        },
    )
    helpers.status("complete", cells=6, cumulative_budget=str(current / "budget.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.root, args.code_root = args.root.resolve(), args.code_root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers(args.root)
    helpers.PARENT = args.root.parent / "next-batch-recovery-03"
    helpers.check_pause()
    if args.preflight:
        preflight(args, helpers)
        return
    with (args.root / "continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args, helpers)
        except BaseException as exc:
            helpers.status(
                "blocked",
                error={"type": type(exc).__name__, "message": str(exc)},
                cumulative_budget=str(args.root / "E01/runtime/exact-om-focused-v2/budget.json"),
            )
            raise


if __name__ == "__main__":
    main()
