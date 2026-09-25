#!/usr/bin/env python3
"""Recover E08 admission from complete probes after a metadata-only repair."""
from __future__ import annotations

import argparse
import ast
import fcntl
import os
import subprocess
import tempfile
from pathlib import Path

from tools import queue_e08 as queue
from tools.qualify_cached_family import (
    binding,
    check_controls,
    usage,
    validate_full_measurement,
)
from tools.resume_cached_batch import (
    FAILED_SETUP_SECONDS,
    RECONCILIATION_ID,
    account_inventory,
    check_admission,
    comparison_forecast,
    load_helpers,
    read,
    verify_nested_bindings,
)

OLD_ROOT = queue.DATA / "e08-qualification-01"
BUDGET = OLD_ROOT / "qualification/runtime/exact-om-focused-v2/budget.json"
METADATA_SOURCE = "exact/impl/datasets/base.py"
AUDIT_SOURCE = "exact/experiments/population_audit.py"
UNCHANGED_STAGES = [
    "native_projection",
    "evidence",
    "pair_scores",
    "decisions",
    "extraction",
    "evaluation",
]
CHANGED_STAGES = ["candidate_population_metadata", "admission_verification"]
# Reviewed metadata-only block: used solely to prove the precise source delta.
METADATA_BRANCH = """if self._df is not None and "cand_sim" in self._df.columns:
    pool_frame = self._df[self._df["cand_sim"].notna()].reset_index(drop=True)
    if self._df_save_path.is_file():
        # CSV parsing can change the last bit of a float. Use the durable
        # retrieval metadata for this manifest only; never change _df.
        metadata_columns = {
            "Src",
            "Tgt",
            "SrcKind",
            "TgtKind",
            "cand_sim",
            "cand_sim_semantic",
            "cand_sim_lexical",
            "cand_sim_retrieval",
            "cand_sim_cross_encoder",
            "cand_channels",
        }
        saved = pd.read_csv(
            self._df_save_path,
            usecols=lambda column: column in metadata_columns,
        )
        saved = restrict(saved)
        saved = saved[saved["cand_sim"].notna()].reset_index(drop=True)
        keys = [key for key in ("Src", "Tgt", "SrcKind", "TgtKind") if key in pool_frame]
        if not set(keys).issubset(saved.columns) or not saved[keys].equals(
            pool_frame[keys]
        ):
            raise ValueError(
                "Saved candidate metadata changed effective pair identities or order"
            )
        pool_frame = saved
"""


def check_owner(args, h):
    h.check_pause()
    check_controls(args.root / "continue.py", args.root)
    queue.no_previous_owner()
    import psutil

    own = psutil.Process()
    ancestors = {own.pid, *(p.pid for p in own.parents())}
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.pid in ancestors:
            continue
        command = proc.info["cmdline"] or []
        if proc.info["name"].startswith(("python", "srun", "bash")) and any(
            str(OLD_ROOT) + "/" in arg for arg in command
        ):
            raise ValueError("Previous E08 owner or descendant remains: " + str(proc.pid))


def validate_metadata_patch(before, after):
    """Allow exactly the sampled-manifest frame choice; preserve every other AST node."""
    original = ast.parse(before)
    classes = [
        n for n in original.body if isinstance(n, ast.ClassDef) and n.name == "BaseAlignmentDataset"
    ]
    if len(classes) != 1:
        raise ValueError("Unknown dataset class for metadata-only recovery")
    method = next(
        n
        for n in classes[0].body
        if isinstance(n, ast.FunctionDef) and n.name == "restrict_sources"
    )
    expected = ast.parse(
        'if pool_frame is None and self._df is not None and "cand_sim" in self._df.columns:\n'
        '    pool_frame = self._df[self._df["cand_sim"].notna()].reset_index(drop=True)'
    ).body[0]
    matches = [
        index for index, node in enumerate(method.body) if ast.dump(node) == ast.dump(expected)
    ]
    if len(matches) != 1:
        raise ValueError("Unknown historical candidate metadata branch")
    method.body[matches[0]] = ast.parse(METADATA_BRANCH).body[0]
    if ast.dump(original) != ast.dump(ast.parse(after)):
        raise ValueError("Source changed beyond the metadata-only candidate frame correction")


def verify_source_impact(code_root, old, current, plan):
    if plan["old_source_commit"] != old["commit"]:
        raise ValueError("Recovery source parent differs from measured source")
    if old["native_code_sha256"] != current["native_code_sha256"]:
        raise ValueError("Native implementation changed; retained measurements are incompatible")
    before_files, after_files = old["extraction_code"]["files"], current["extraction_code"]["files"]
    changed = {
        p for p in set(before_files) | set(after_files) if before_files.get(p) != after_files.get(p)
    }
    if changed != {METADATA_SOURCE, AUDIT_SOURCE}:
        raise ValueError("Unexpected source dependency impact: " + repr(sorted(changed)))
    # Prediction fingerprints intentionally omit evaluator modules; also inspect
    # every tracked/untracked exact source so evaluation cannot silently change.
    source_diff = subprocess.check_output(
        ["git", "diff", "--name-only", old["commit"], "--", "exact"],
        cwd=code_root,
        text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "exact"],
        cwd=code_root,
        text=True,
    ).splitlines()
    if set(source_diff + untracked) - {METADATA_SOURCE, AUDIT_SOURCE}:
        raise ValueError("Other numerical or evaluation source changed")
    before = subprocess.check_output(
        ["git", "show", old["commit"] + ":" + METADATA_SOURCE], cwd=code_root, text=True
    )
    validate_metadata_patch(before, (code_root / METADATA_SOURCE).read_text())
    if plan.get("stage_impact") != {"unchanged": UNCHANGED_STAGES, "changed": CHANGED_STAGES}:
        raise ValueError("Recovery must bind the exact dependency-based stage impact")
    for key, expected in {
        "measurements_reused": True,
        "numerical_reexecution": False,
        "scientific_design_changed": False,
        "new_hosted_requests": 0,
        "generate_rationales": False,
    }.items():
        if plan.get(key) != expected:
            raise ValueError("Recovery scope changed: " + key)
    return {
        "changed_source": sorted(changed),
        "stage_impact": plan["stage_impact"],
        "metadata_only_ast_verified": True,
    }


def validate_latest_budget(item, lock, h):
    if Path(item["path"]) != BUDGET:
        raise ValueError("Stale cumulative budget import")
    status = read(OLD_ROOT / "status.json")
    if status.get("cumulative_budget") != str(BUDGET):
        raise ValueError("Historical owner identifies a different latest budget")
    state = read(h.verify(item))
    queue.validate_budget(state, lock)
    inventory = state["work"].get(RECONCILIATION_ID, {})
    if inventory.get("status") != "failed" or inventory.get("seconds") != FAILED_SETUP_SECONDS:
        raise ValueError("Standalone inventory cost evidence missing")
    return state


def validate_receipts(rows, state, configs, h):
    if len(rows) != 3 or {r["name"] for r in rows} != set(queue.ARMS):
        raise ValueError("Retained measurements must cover all three E08 treatments")
    for row in rows:
        required_usage = {"attempts", "billable_tokens", "unknown", "unpriced_attempts"}
        if (
            row.get("status") != "passed"
            or row.get("execution_status") != "complete"
            or row.get("prefix") is not False
            or not required_usage <= row.get("new_usage", {}).keys()
            or any(row["new_usage"].values())
            or row.get("seed") != 17
            or row.get("source_cap") != 300
            or row.get("generate_rationales") is not False
            or row.get("no_private_test_references") is not True
            or row.get("config") != configs[row["name"]]
        ):
            raise ValueError(
                "Retained qualification changed scientific scope or cached-only policy"
            )
        verify_nested_bindings(row, h)
        validate_full_measurement(
            {"cursor": {"next_pair": row["processed_pairs"], "dataset_rows": row["dataset_rows"]}},
            row["worker_measurement"],
            row["worker_calls"],
            prefix=False,
        )
        account = state["work"].get(row["budget_work_id"], {})
        if (
            account.get("status") != "complete"
            or account.get("group") != "reserve"
            or account.get("requests") != 0
            or account.get("tokens") != 0
            or not 0 < row["wall_seconds"] <= account.get("seconds", 0)
        ):
            raise ValueError("Retained qualification lacks complete cumulative accounting")


def cached_caps(cached):
    return {
        "EXACT_OPENROUTER_REQUEST_CAP": str(cached["attempts"]),
        "EXACT_OPENROUTER_TOKEN_CAP": str(cached["billable_tokens"]),
        "EXACT_OPENROUTER_RETRY_UNKNOWN": "0",
    }


def verify_configuration(campaign, selection, configs, h):
    with tempfile.TemporaryDirectory(prefix="e08-recovery-cells-") as tmp:
        cells = queue.cells_for(campaign, Path(tmp), read(h.verify(selection)))
        if any(
            binding(h.verify(configs[c.arm_id]))["sha256"]
            != queue.binding_config(c.resolved_config)
            for c in cells
        ):
            raise ValueError("Measured configuration differs from scientific comparison")


def preflight(args, h):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.population_audit import audit_matched_measurements

    check_owner(args, h)
    if (args.root / "E08/runtime").exists():
        raise FileExistsError("Existing runtime needs another recorded checkpoint recovery")
    original = read(OLD_ROOT / "preflight.json")
    verify_nested_bindings(original, h)
    failed = read(OLD_ROOT / "status.json")
    if (
        original.get("status") != "preflight_passed_not_admitted"
        or failed.get("status") != "blocked"
        or failed.get("error", {}).get("message")
        != "E08 treatments changed frozen candidate populations"
        or (OLD_ROOT / "launcher-exit-code").read_text().strip() != "1"
        or (OLD_ROOT / "completion.json").exists()
        or (OLD_ROOT / "E08/runtime").exists()
    ):
        raise ValueError("Historical E08 failure differs from the recorded pre-admission incident")
    identity = h.validate_source(args.code_root, allow_working_tree=args.allow_working_tree)
    plan = read(args.root / "reuse-plan.json")
    impact = verify_source_impact(args.code_root, original["source"], identity, plan)
    if not all(
        plan["accounting"].get(k) is True
        for k in ("preserve_all_rows", "protected_allowance_unchanged")
    ):
        raise ValueError("All cumulative charges and protected allowances must be retained")
    lock = load_yaml_mapping(h.verify(original["prospective_campaign"]))
    state = validate_latest_budget(plan["accounting"]["latest_budget"], lock, h)
    checks = read(args.root / "checks.json")
    if checks.get("status") != "passed" or checks.get("pytest", {}).get("exit_code") != 0:
        raise ValueError("Successful focused checks are required")
    required = {METADATA_SOURCE, AUDIT_SOURCE, "tools/recover_e08.py"}
    if not required <= checks.get("source_files", {}).keys():
        raise ValueError("Checks must bind every repaired execution source")
    for relative, item in checks["source_files"].items():
        if binding(args.code_root / relative)["sha256"] != item["sha256"]:
            raise ValueError("Checked source changed: " + relative)
    paths = [OLD_ROOT / "qualification" / arm / "measurement.json" for arm in queue.ARMS]
    rows = [read(path) for path in paths]
    validate_receipts(rows, state, original["configs"], h)
    audit = audit_matched_measurements(rows)
    h.freeze(args.root / "population-audit.json", audit)
    measurement = args.root / "E08/measured-cost.json"
    h.freeze(
        measurement,
        {
            "status": "measured_forecast",
            "rows": rows,
            "candidate_population": audit["candidate_population"],
            "population_audit": binding(args.root / "population-audit.json"),
            "method": "Retained complete matched treatment durations; maximum per arm, setup once per arm, factor 1.5, standalone inventory once. Metadata-only repair; no cache-speedup discount.",
        },
    )
    proof = comparison_forecast(rows, 3, measurement, h)
    proof["inventory_cost_basis"] = original["inventory_cost_basis"]
    h.freeze(args.root / "E08/forecast.json", proof)
    with tempfile.TemporaryDirectory(prefix="e08-recovery-admission-") as tmp:
        from exact.experiments.budget import BudgetLedger

        check_path = check_admission(
            state, "channels", proof["whole_family_seconds"], Path(tmp), "planned/E08"
        )
        BudgetLedger(check_path, state["limits"]).admit(
            "planned/E08-cache-copy", group="reserve", seconds=1800
        )
    campaign = args.root / "E08/campaign.lock.yaml"
    h.freeze(campaign, queue.ready_lock(lock, proof["estimate"], identity["commit"]))
    verify_configuration(campaign, original["source_selection"], original["configs"], h)
    h.freeze(args.root / "prospective-campaign.yaml", lock)
    operational = [
        args.code_root / p
        for p in (
            "tools/recover_e08.py",
            "tools/queue_e08.py",
            "tools/qualify_cached_family.py",
            "tools/resume_cached_batch.py",
            "tools/run_experiment_validation.py",
        )
    ]
    operational += [
        args.root / p for p in ("continue.py", "submit.sh", "run-step.sh", "worker-entry.sh")
    ]
    operational += [
        queue.DATA / "next-batch-recovery-02" / p for p in ("continue.py", "qualification.py")
    ]
    operational += [queue.DATA / "d1-production-probe-01/launch.json"]
    h.freeze(
        args.root / "preflight.json",
        {
            "status": (
                "preview_only" if args.allow_working_tree else "preflight_passed_not_admitted"
            ),
            "source": identity,
            "source_impact": impact,
            "historical_preflight": binding(OLD_ROOT / "preflight.json"),
            "historical_status": binding(OLD_ROOT / "status.json"),
            "historical_exit": binding(OLD_ROOT / "launcher-exit-code"),
            "budget_import": binding(BUDGET),
            "source_selection": original["source_selection"],
            "configs": original["configs"],
            "measurements": [binding(p) for p in paths],
            "prospective_campaign": binding(args.root / "prospective-campaign.yaml"),
            "campaign": binding(campaign),
            "forecast": binding(args.root / "E08/forecast.json"),
            "measurement": binding(measurement),
            "population_audit": binding(args.root / "population-audit.json"),
            "records": [
                binding(args.root / p) for p in ("reuse-plan.json", "repair.json", "checks.json")
            ],
            "operational_files": [binding(p) for p in operational],
            "cache_usage": usage(BUDGET.parent / "openrouter"),
            "budget_limits_changed": False,
            "generate_rationales": False,
            "probes_rerun": False,
        },
    )
    print(
        "E08 retained-measurement recovery preflight passed; comparison not yet admitted",
        flush=True,
    )


def run(args, h):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import (
        CampaignLock,
        execute_campaign,
        external_selection_result,
    )
    from exact.experiments.population_audit import audit_matched_measurements

    check_owner(args, h)
    if (
        os.environ.get("SLURM_JOB_ID") != "14372"
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise ValueError("Use a detached numeric step in allocation 14372")
    receipt = read(args.root / "preflight.json")
    if (
        receipt["status"] != "preflight_passed_not_admitted"
        or h.validate_source(args.code_root) != receipt["source"]
    ):
        raise ValueError("Clean frozen recovery source preflight required")
    verify_nested_bindings(receipt, h)
    lock = load_yaml_mapping(h.verify(receipt["prospective_campaign"]))
    state = validate_latest_budget(receipt["budget_import"], lock, h)
    rows = [read(h.verify(p)) for p in receipt["measurements"]]
    validate_receipts(rows, state, receipt["configs"], h)
    audit = audit_matched_measurements(rows)
    if audit != read(h.verify(receipt["population_audit"])):
        raise ValueError("Retained candidate evidence changed after recovery preflight")
    proof = read(h.verify(receipt["forecast"]))
    with tempfile.TemporaryDirectory(prefix="e08-recovery-admission-") as tmp:
        from exact.experiments.budget import BudgetLedger

        check_path = check_admission(
            state, "channels", proof["whole_family_seconds"], Path(tmp), "planned/E08"
        )
        BudgetLedger(check_path, state["limits"]).admit(
            "planned/E08-cache-copy", group="reserve", seconds=1800
        )
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    if (
        len(free) != 1
        or proof["estimate"]["peak_ram_gb"] > 56
        or proof["estimate"]["peak_vram_gb"] > float(free[0]) / 1024
    ):
        raise ValueError("Measured complete E08 family memory does not fit this node")
    wave, campaign = args.root / "E08", h.verify(receipt["campaign"])
    destination = wave / "runtime/exact-om-focused-v2"
    if (wave / "runtime").exists():
        raise FileExistsError(
            "Existing runtime requires recorded checkpoint recovery, not top-level replay"
        )
    verify_configuration(campaign, receipt["source_selection"], receipt["configs"], h)
    retained = read(queue.DATA / "d1-production-probe-01/launch.json")
    os.environ.update(retained["environment"])
    os.environ["PYTHONPATH"] = str(args.code_root)
    # Consume the established launcher credential internally; never log it.
    os.environ["OPENROUTER_API_KEY"] = (queue.PROJECT / "api_key").read_text().strip()
    check_owner(args, h)
    h.status(
        "preparing_recovered_E08",
        cumulative_budget=str(destination / "budget.json"),
        budget_import=str(BUDGET),
    )
    if usage(BUDGET.parent / "openrouter") != receipt["cache_usage"]:
        raise ValueError("Historical cached usage changed since preflight")
    h.copy_state(BUDGET.parent, destination)
    cached = usage(destination / "openrouter")
    if cached != receipt["cache_usage"]:
        raise ValueError("Copied cached usage differs from retained ledger")
    os.environ.update(cached_caps(cached))
    h.freeze(
        wave / "launch.json",
        {
            "campaign": binding(campaign),
            "source": receipt["source"],
            "budget_path": str(destination / "budget.json"),
            "step_id": "14372." + os.environ["SLURM_STEP_ID"],
            "recovery_preflight": binding(args.root / "preflight.json"),
        },
    )
    h.status(
        "running",
        step=queue.FAMILY,
        cumulative_budget=str(destination / "budget.json"),
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
    with account_inventory(h, destination, queue.FAMILY, "channels", proof["inventory_seconds"]):
        h.guarded_execute(campaign, wave, args.code_root)
    if usage(destination / "openrouter") != cached:
        raise ValueError("Comparison incurred incremental hosted usage")
    h.history_record(lock, queue.FAMILY, campaign, destination, args.root / "history")
    parsed = CampaignLock.model_validate(lock)
    decision = external_selection_result(
        parsed, next(s for s in parsed.steps if s.id == queue.FAMILY), args.root
    )
    h.freeze(
        args.root / "completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "cells": 3,
            "selection": binding(destination / "screen/selection.json"),
            "decision": decision["status"],
            "cumulative_budget": binding(destination / "budget.json"),
            "source": receipt["source"],
            "cached_usage_unchanged": True,
            "generate_rationales": False,
        },
    )
    h.status("complete", cells=3, cumulative_budget=str(destination / "budget.json"))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--allow-working-tree", action="store_true")
    args = parser.parse_args(argv)
    h = load_helpers(args.root)
    if args.preflight:
        preflight(args, h)
        return 0
    with (args.root / "continuation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(args, h)
        except BaseException as exc:
            latest = args.root / "E08/runtime/exact-om-focused-v2/budget.json"
            h.status(
                "blocked",
                cumulative_budget=str(latest if latest.exists() else BUDGET),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
