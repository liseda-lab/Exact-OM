#!/usr/bin/env python3
"""Finalize complete measured evidence in a new root without scientific reruns."""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from tools import measured_once as once
from tools import prepared_batch as batch


def validate_evidence(recipe, directory, state):
    """Verify every arm and its numerical identity before any selection."""
    from exact.experiments import harness
    from exact.experiments.campaign import materialize_campaign
    from exact.experiments.runtime import CellRecovery

    for item in recipe["evidence"]:
        batch.verified(item)
    campaign = batch.verified(recipe["campaign"])
    suite = materialize_campaign(campaign, directory, stage="screen")
    source = next(s for s in suite.sources if s.config.experiment_id == recipe["scientific_step"])
    selection = batch.read(batch.verified(recipe["source_selection"]))
    cells = harness.build_cells(
        suite,
        source,
        stage="screen",
        output_root=directory / "preview-runtime",
        inherited_overlay=harness.inherited_selection_overlay(selection, source.config.depends_on),
    )
    rows = [batch.read(batch.verified(item)) for item in recipe["measurements"]]
    keyed = {row["name"]: row for row in rows}
    expected = {
        cell.task_id.removesuffix("-global_alignment") + "--" + cell.arm_id for cell in cells
    }
    if len(keyed) != len(rows) or set(keyed) != expected or len(expected) != len(cells):
        raise ValueError("Recovery requires every declared arm exactly once")
    workdir = Path(recipe["numerical_code_root"])
    for cell in cells:
        row = keyed[cell.task_id.removesuffix("-global_alignment") + "--" + cell.arm_id]
        _, artifacts, _ = once.verify_measurement(row, cell, state)
        candidate = replace(
            cell, recovery={"root": str(directory / "preview-runtime"), "reuse_plan_only": True}
        )
        recovery = CellRecovery(
            candidate, harness._provenance_payload(candidate, suite, workdir=workdir), workdir
        )
        if {key: value["artifact_id"] for key, value in recovery.identities.items()} != artifacts:
            raise ValueError("Saved numerical identity differs: " + row["name"])
    return rows


def retained_request_history(recipe, state):
    """Accept an explicit older wire ledger only with closed, zero-call descendants."""
    origin = batch.read(batch.verified(recipe["request_budget"]))
    if state["limits"] != origin["limits"] or any(
        state["work"].get(k) != v for k, v in origin["work"].items()
    ):
        raise ValueError("Request history budget lineage changed")
    for key, value in state["work"].items():
        if key not in origin["work"] and (
            value["status"] not in {"complete", "failed", "interrupted"}
            or any(value.get(field, 0) for field in ("requests", "tokens", "actual_usd"))
        ):
            raise ValueError("Newer hosted charges require their own request history")
    path = batch.verified(recipe["request_ledger"])
    from tools.qualify_cached_family import usage

    if usage(path.parent) != recipe["request_usage"]:
        raise ValueError("Retained request usage changed")
    return path


def reconcile_failures(ledger, state, recipe):
    """Add only separately evidenced, uncharged closed metadata intervals."""
    for item in recipe["failed_intervals"]:
        for binding in item["evidence"]:
            batch.verified(binding)
        key = "failed-handoff/" + item["run_id"]
        if key in state["work"]:
            raise ValueError("Failure interval already accounted")
        ledger.admit(key, group="reserve", seconds=0, forecast_known=False)
        ledger.finish(
            key,
            start=item["start"],
            end=item["end"],
            status="failed",
            requests=0,
            tokens=0,
            actual_usd=0,
        )


def run(path):
    from exact.experiments.budget import BudgetLedger
    from tools.experiment_resources import guarded_execute
    from tools.finalize_prepared_selection import register_lineage
    from tools.qualify_cached_family import usage
    from tools.resume_e19_once import check_owner

    recipe = batch.read(path)
    root, supervisor = Path(recipe["root"]), Path(recipe["supervisor"])
    with (root / "worker.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = batch.controls(supervisor, root)
        if (
            os.environ.get("SLURM_JOB_ID") != recipe["allocation"]
            or not os.environ.get("SLURM_STEP_ID", "").isdigit()
        ):
            raise ValueError("Detached numeric allocation step required")
        check_owner(
            SimpleNamespace(
                root=root, supervisor_step=(supervisor / "supervisor-step-id").read_text().strip()
            )
        )
        for prefix in ("", "numerical_"):
            code = Path(recipe[prefix + "code_root"])
            if (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
                != recipe[prefix + "commit"]
                or subprocess.check_output(
                    ["git", "diff", "--name-only", "HEAD"], cwd=code, text=True
                ).strip()
            ):
                raise ValueError("Frozen recovery source changed")
        started = time.time()
        # Resolve accounting while the failed parent remains enabled.
        account, state = batch.latest_account(registry)
        retained_request_history(recipe, state)
        runtime = root / "runtime" / recipe["campaign_id"]
        if runtime.exists():
            raise ValueError("Existing recovery runtime requires explicit reconciliation")
        batch.copy_account(account, state, runtime, request_ledger=recipe["request_ledger"])
        batch.status(
            root, "verifying_saved_measurements", cumulative_budget=str(runtime / "budget.json")
        )
        register_lineage(recipe)
        ledger = BudgetLedger(runtime / "budget.json", state["limits"])
        reconcile_failures(ledger, state, recipe)
        work = "recovery-finalization/" + recipe["run_id"]
        ledger.admit(work, group="reserve", seconds=0, forecast_known=False)
        outcome = "failed"
        try:
            rows = validate_evidence(
                recipe, root / "verification", batch.read(runtime / "budget.json")
            )
            campaign = batch.verified(recipe["campaign"])
            selection = batch.read(batch.verified(recipe["source_selection"]))
            ledger.finish(
                work,
                start=started,
                end=time.time(),
                status="complete",
                requests=0,
                tokens=0,
                actual_usd=0,
            )
            work = None
            once.promote_measured_cells(
                campaign,
                runtime,
                rows,
                recipe["scientific_step"],
                selection,
                Path(recipe["numerical_code_root"]),
            )
            work = "recovery-inventory/" + recipe["run_id"]
            started = time.time()
            ledger.admit(work, group="reserve", seconds=0, forecast_known=False)
            initial = usage(runtime / "openrouter")
            os.environ.update(batch.read(batch.verified(recipe["environment"])))
            os.environ.update(
                EXACT_OPENROUTER_LEDGER_DIR=str(runtime / "openrouter"),
                EXACT_OPENROUTER_REQUEST_CAP=str(initial["attempts"]),
                EXACT_OPENROUTER_TOKEN_CAP=str(initial["billable_tokens"]),
                EXACT_OPENROUTER_RETRY_UNKNOWN="0",
                EXACT_EVIDENCE_PREFETCH="0",
            )
            batch.status(
                root, "finalizing_saved_comparison", cumulative_budget=str(runtime / "budget.json")
            )
            with once.require_reuse_only():
                guarded_execute(
                    campaign,
                    root,
                    Path(recipe["numerical_code_root"]),
                    check_pause=lambda: batch.controls(supervisor, root),
                )
            if usage(runtime / "openrouter") != initial:
                raise ValueError("Finalization incurred hosted usage")
            manifests = sorted(
                (runtime / "screen/runs" / recipe["scientific_step"]).glob(
                    "*/D*-global_alignment/seed-17/experiment_manifest.json"
                )
            )
            if len(manifests) != len(rows) or any(
                batch.read(p).get("status") != "complete" for p in manifests
            ):
                raise ValueError("All measured cells must be finalized")
            outcome = "complete"
        finally:
            if work is not None:
                ledger.finish(
                    work,
                    start=started,
                    end=time.time(),
                    status=outcome,
                    requests=0,
                    tokens=0,
                    actual_usd=0,
                )
            observation = state.get("allocations", {}).get(recipe["allocation"])
            if observation:
                ledger.record_allocation(
                    recipe["allocation"], start=observation["start"], end=time.time()
                )
        batch.write(
            root / "completion.json",
            {
                "status": "complete",
                "exit_code": 0,
                "cells": len(manifests),
                "selection": batch.binding(runtime / "screen/selection.json"),
                "manifests": [batch.binding(p) for p in manifests],
                "cumulative_budget": batch.binding(runtime / "budget.json"),
                "campaign": batch.binding(campaign),
                "measured_results": batch.binding(runtime / "measured-results.json"),
                "source": {"commit": recipe["numerical_commit"]},
                "new_scientific_executions": 0,
                "cached_usage_unchanged": True,
            },
        )
        batch.status(
            root, "complete", cumulative_budget=str(runtime / "budget.json"), cells=len(manifests)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    run(args.recipe)


if __name__ == "__main__":
    main()
