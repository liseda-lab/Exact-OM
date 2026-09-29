#!/usr/bin/env python3
"""Recover qualified E10 after the verified historical-producer handoff repair."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from tools import queue_e10_acceptance as queue
from tools.qualify_cached_family import binding, usage
from tools.resume_cached_batch import (
    account_inventory,
    check_admission,
    load_helpers,
    read,
    verify_nested_bindings,
)

PARENT = queue.DATA / "e10-acceptance-01"
OLD_SOURCE = "f31eeeb3c91f6b4c37c09c374b498cdd25a2495f"
CHANGED_NUMERICAL_SCOPE = {
    "exact/experiments/campaign.py",
    "exact/experiments/harness.py",
    "exact/experiments/staged_selection.py",
}
BUDGET = PARENT / "E10/runtime/exact-om-focused-v2/budget.json"
FAILED_WORK = "recovery/e10-acceptance-01/failed-producer-planning"


def reconcile(state, terminal):
    """Retain all rows and account the failed post-copy planning interval once."""
    from exact.experiments.budget import interval_seconds

    if terminal.get("status") != "blocked" or terminal.get("error", {}).get("message") != (
        "E10: artifact-dependent plan requires completed producer outputs; "
        "inspect or resume those producers first"
    ):
        raise ValueError("Unexpected parent failure; inspect before recovery")
    result = copy.deepcopy(state)
    if FAILED_WORK in result["work"]:
        raise ValueError("Failed planning is already accounted")
    start = result["work"]["preparation/e10-acceptance-01/E10"]["end"]
    end = datetime.fromisoformat(terminal["recorded_at"]).timestamp()
    seconds = interval_seconds([[start, end]])
    if seconds > 300:
        raise ValueError("Unrecognized failed setup interval")
    result["work"][FAILED_WORK] = dict(
        group="channels",
        status="failed",
        start=start,
        end=end,
        seconds=seconds,
        requests=0,
        tokens=0,
        projected_usd=0,
        actual_usd=0,
        accounting="Conservative post-copy planning interval through terminal failure; no worker or requests started.",
    )
    result["intervals"].append([start, end])
    result["node_seconds"] = interval_seconds(result["intervals"])
    result["elapsed_seconds"] = max(x[1] for x in result["intervals"]) - min(
        x[0] for x in result["intervals"]
    )
    return result


def verify_source(source, prior):
    """Only the reviewed producer handoff changed; native and numerical bytes match."""
    if prior["commit"] != OLD_SOURCE or source["native_code_sha256"] != prior["native_code_sha256"]:
        raise ValueError("Parent source or native packages changed")
    old, new = prior["extraction_code"]["files"], source["extraction_code"]["files"]
    changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    if changed != CHANGED_NUMERICAL_SCOPE:
        raise ValueError("Unexpected implementation dependency changes: " + repr(sorted(changed)))


def producer_cells(campaign, directory):
    """Exercise actual producer binding and return resolved consumer cells without models."""
    from dataclasses import replace

    from exact.experiments import harness
    from exact.experiments.campaign import (
        external_selection_result,
        load_campaign,
        materialize_campaign,
    )
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import _verify_materialized

    lock, _ = load_campaign(campaign)
    suite = materialize_campaign(campaign, directory / "declarations", stage="screen")
    suite = replace(suite, campaign={"root": str(directory / "runtime")})
    selections, manifests = {}, []
    for step in lock.steps:
        if step.external_selection:
            source = next(s for s in suite.sources if s.config.experiment_id == step.id)
            historical = external_selection_result(
                lock, step, campaign.parent, producer_manifests=manifests
            )
            selections[step.id] = harness._bind_external_selection(source, suite, historical)
    # Verify every consumed analytic output against its original content store;
    # importing a policy does not claim current-code prediction compatibility.
    analytic = [m for m in manifests if m["experiment_id"] == "E10-analytic"]
    if len(analytic) != 6:
        raise ValueError("All six historical analytic outcomes are required")
    for manifest in analytic:
        output = Path(manifest["fingerprint_payload"]["output_dir"])
        root = next(p for p in output.parents if (p / "budget.json").is_file())
        store = ArtifactStore(root)
        for stage in ("extraction", "evaluation"):
            _verify_materialized(
                store, store.verify(manifest["recovery"]["artifacts"][stage]), output
            )
    source = next(s for s in suite.sources if s.config.experiment_id == "E10")
    inherited = harness.inherited_selection_overlay(
        {"experiments": selections}, source.config.depends_on
    )
    source, inherited = harness._materialize_campaign_evidence(
        source,
        suite,
        [],
        selections,
        inherited,
        plan_only=True,
        external_manifests=manifests,
    )
    cells = harness.build_cells(
        suite, source, stage="screen", output_root=directory, inherited_overlay=inherited
    )
    queue.validate_cells(cells)
    for cell in cells:
        original = PARENT / "qualification" / ("D0_E03--" + cell.arm_id + ".config.yaml")
        if queue.binding_config(cell.resolved_config) != binding(original)["sha256"]:
            raise ValueError("Producer materialization changed the qualified numerical recipe")
    return cells


def preflight(args, h):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.budget import BudgetLedger
    from exact.experiments.campaign import execute_campaign

    h.check_pause()
    queue.no_previous_owner(args.root, "14372.23")
    if (PARENT / "launcher-exit-code").read_text().strip() != "1":
        raise ValueError("Parent has no verified terminal failure")
    if (BUDGET.parent / "screen").exists():
        raise ValueError("Unexpected scientific checkpoints need a different reuse plan")
    source = h.validate_source(args.code_root, allow_working_tree=args.allow_working_tree)
    prior = read(PARENT / "preflight.json")
    verify_source(source, prior["source"])
    lock = load_yaml_mapping(PARENT / "E10/campaign.lock.yaml")
    state = read(BUDGET)
    queue.validate_budget(state, lock)
    reconciled = reconcile(state, read(PARENT / "status.json"))
    rows = [read(PARENT / "qualification" / arm / "measurement.json") for arm in queue.KEYS]
    verify_nested_bindings(rows, h)
    for row in rows:
        if state["work"][row["budget_work_id"]]["status"] != "complete":
            raise ValueError("Qualification accounting is not complete")
    fits = queue.fitted_evidence(rows, read(PARENT / "training.json"))
    if fits != read(PARENT / "fitting-evidence.json"):
        raise ValueError("Original fitted qualification evidence changed")
    proof = queue.family_forecast(rows, PARENT / "E10/measured-cost.json", h)
    if proof != read(PARENT / "E10/forecast.json"):
        raise ValueError("Whole-family measured forecast changed")
    with tempfile.TemporaryDirectory(prefix="e10-recovery-admission-") as tmp:
        ledger = check_admission(
            reconciled, "channels", proof["whole_family_seconds"], Path(tmp), "planned/E10"
        )
        BudgetLedger(ledger, reconciled["limits"]).admit(
            "planned/cache-copy", group="reserve", seconds=1800
        )
    campaign = args.root / "E10/campaign.lock.yaml"
    h.freeze(campaign, lock)
    with tempfile.TemporaryDirectory(prefix="e10-producers-") as tmp:
        producer_cells(campaign, Path(tmp))
        execute_campaign(
            campaign,
            stage="screen",
            output_root=Path(tmp) / "plan",
            workdir=args.code_root,
            jobs=1,
            reuse_plan_only=True,
        )
    h.freeze(args.root / "budget-import.json", reconciled)
    h.freeze(args.root / "forecast.json", proof)
    h.freeze(
        args.root / "preflight.json",
        dict(
            status="preflight_passed_not_admitted",
            source=source,
            parent_budget=binding(BUDGET),
            budget_import=binding(args.root / "budget-import.json"),
            campaign=binding(campaign),
            parent_status=binding(PARENT / "status.json"),
            forecast=binding(args.root / "forecast.json"),
            reuse_plan=binding(args.root / "reuse-plan.json"),
            measurements=[
                binding(PARENT / "qualification" / a / "measurement.json") for a in queue.KEYS
            ],
            operational_files=[
                binding(args.root / name)
                for name in ("continue.py", "worker-entry.sh", "run-step.sh", "submit.sh")
            ],
            checks=binding(args.root / "checks.json"),
            complete_arm_seconds=[r["wall_seconds"] for r in rows],
            producer_cells=6,
            consumer_cells=2,
        ),
    )


def run(args, h):
    from exact.experiments.campaign import execute_campaign

    h.check_pause()
    if (
        os.environ.get("SLURM_JOB_ID") != "14372"
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise ValueError("Use a detached numeric step inside allocation 14372")
    queue.no_previous_owner(args.root, "14372.23")
    receipt = read(args.root / "preflight.json")
    if (
        receipt["status"] != "preflight_passed_not_admitted"
        or h.validate_source(args.code_root) != receipt["source"]
    ):
        raise ValueError("Clean frozen-source preflight required")
    for key in (
        "parent_budget",
        "budget_import",
        "campaign",
        "parent_status",
        "forecast",
        "reuse_plan",
        "checks",
    ):
        h.verify(receipt[key])
    for item in [*receipt["measurements"], *receipt["operational_files"]]:
        h.verify(item)
    wave = args.root / "E10"
    runtime = wave / "runtime/exact-om-focused-v2"
    if runtime.exists():
        raise FileExistsError(
            "Runtime exists; use recorded saved-family recovery, never replay preparation"
        )
    retained = read(queue.DATA / "d1-production-probe-01/launch.json")
    os.environ.update(retained["environment"])
    os.environ["PYTHONPATH"] = str(args.code_root)
    proof = read(args.root / "forecast.json")
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    if (
        len(free) != 1
        or proof["estimate"]["peak_ram_gb"] > 56
        or proof["estimate"]["peak_vram_gb"] > float(free[0]) / 1024
    ):
        raise ValueError("Measured family memory does not fit the allocated GPU step")
    # Preserve the established internal launcher credential handling; never log it.
    os.environ["OPENROUTER_API_KEY"] = (queue.PROJECT / "api_key").read_text().strip()
    h.status("preparing_recovery", cumulative_budget=str(runtime / "budget.json"))
    original = h.freeze

    def freeze_reconciled(path, value):
        if path == runtime / "budget.json":
            value = read(h.verify(receipt["budget_import"]))
        return original(path, value)

    h.freeze = freeze_reconciled
    try:
        h.copy_state(BUDGET.parent, runtime)
    finally:
        h.freeze = original
    cached = usage(runtime / "openrouter")
    os.environ.update(
        EXACT_OPENROUTER_REQUEST_CAP=str(cached["attempts"]),
        EXACT_OPENROUTER_TOKEN_CAP=str(cached["billable_tokens"]),
        EXACT_OPENROUTER_RETRY_UNKNOWN="0",
    )
    h.status(
        "running",
        step="E10",
        cumulative_budget=str(runtime / "budget.json"),
        campaign=str(wave / "campaign.lock.yaml"),
    )
    execute_campaign(
        wave / "campaign.lock.yaml",
        stage="screen",
        output_root=wave / "runtime",
        workdir=args.code_root,
        jobs=1,
        reuse_plan_only=True,
    )
    h.check_pause()
    with account_inventory(
        h, runtime, "E10", "channels", read(args.root / "forecast.json")["inventory_seconds"]
    ):
        h.guarded_execute(wave / "campaign.lock.yaml", wave, args.code_root)
    if usage(runtime / "openrouter") != cached:
        raise ValueError("Incremental hosted usage is forbidden")
    manifests = sorted(
        (runtime / "screen/runs/E10").glob(
            "*/D0_E03-global_alignment/seed-17/experiment_manifest.json"
        )
    )
    if len(manifests) != 2 or any(read(p).get("status") != "complete" for p in manifests):
        raise ValueError("Both scientific arms must complete")
    selection = read(runtime / "screen/selection.json")
    if selection["experiments"]["E10"]["status"] not in {"selected", "screened_out"}:
        raise ValueError("E10 has no completed development selection")
    h.freeze(
        args.root / "completion.json",
        dict(
            status="complete",
            exit_code=0,
            cells=2,
            selection=binding(runtime / "screen/selection.json"),
            manifests=[binding(p) for p in manifests],
            cumulative_budget=binding(runtime / "budget.json"),
            source=receipt["source"],
            cached_usage_unchanged=True,
            generate_rationales=False,
            parent_attempt=str(PARENT),
        ),
    )
    h.status("complete", cells=2, cumulative_budget=str(runtime / "budget.json"))
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
            latest = args.root / "E10/runtime/exact-om-focused-v2/budget.json"
            h.status(
                "blocked",
                error={"type": type(exc).__name__, "message": str(exc)},
                cumulative_budget=str(
                    latest if latest.exists() else args.root / "budget-import.json"
                ),
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
