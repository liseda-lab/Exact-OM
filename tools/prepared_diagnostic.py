"""Use the existing supervised Slurm lane for non-selecting diagnostics."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

from tools.prepared_batch import (
    binding,
    completed_run,
    copy_account,
    latest_account,
    prepare_launch,
    read,
    resolve_run,
    status,
    verified,
    write,
)


def prepare(root, batch, *, diagnostic, code, supervisor, environment_path, checks, python=None):
    root, code, supervisor = map(Path, (root, code, supervisor))
    root.mkdir(parents=True, exist_ok=True)
    if batch.get("needs_user"):
        raise ValueError("Diagnostic still needs a scientific decision")
    protocol = read(diagnostic)
    if protocol.get("kind") not in {"e24_directional_known_pairs", "e14_native_known_pairs"}:
        raise ValueError("Unsupported diagnostic protocol")
    policy = read(supervisor / "policy.json")
    recipe = {
        "schema_version": 1,
        "root": str(root.resolve()),
        "scientific_step": batch["scientific_step"],
        "diagnostic": binding(diagnostic),
        "group": binding(diagnostic),
        "depends_on": batch["depends_on"],
        "supervisor": str(supervisor.resolve()),
        "code_root": str(code.resolve()),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code, text=True
        ).strip(),
        "checks": checks,
        "repository": policy["repository"],
        "environment": binding(environment_path),
        "allocation": policy["allocation"],
        "campaign_id": "diagnostic-" + batch["scientific_step"],
        "dispatch_nonce": uuid.uuid4().hex,
    }
    if python is not None:
        recipe["python"] = str(Path(python).absolute())
    return prepare_launch(recipe, root, code, supervisor, batch)


def run_prepared_diagnostic(recipe_path, registry):
    from exact.experiments.budget import BudgetLedger

    recipe = read(recipe_path)
    root = Path(recipe["root"])
    diagnostic_path = verified(recipe["diagnostic"])
    protocol = read(diagnostic_path)
    if protocol.get("kind") not in {"e24_directional_known_pairs", "e14_native_known_pairs"}:
        raise ValueError("Unsupported diagnostic protocol")
    if protocol["kind"] == "e24_directional_known_pairs":
        from tools.run_directional_diagnostic import run_diagnostic
    else:
        from tools.run_bridge_diagnostic import run_diagnostic
    for identifier in recipe["depends_on"]:
        completed_run(resolve_run(registry, identifier))
    runtime = root / "runtime" / recipe["campaign_id"]
    parent, state = latest_account(registry)
    launch_path = root / "launch.json"
    if launch_path.exists():
        if read(launch_path)["recipe"] != binding(recipe_path):
            raise ValueError("Diagnostic resume recipe changed")
        if parent != runtime / "budget.json":
            raise ValueError("Diagnostic resume account is no longer authoritative")
    else:
        copy_account(parent, state, runtime)
        write(
            launch_path,
            {"recipe": binding(recipe_path), "parent_budget": binding(parent)},
            immutable=True,
        )
    ledger = BudgetLedger(runtime / "budget.json", state["limits"])
    work = "diagnostic/" + recipe["scientific_step"] + "/" + os.environ["SLURM_STEP_ID"]
    ledger.admit(work, group="reserve", seconds=0, forecast_known=False)
    started, outcome = time.time(), "failed"
    status(
        root, "running", cumulative_budget=str(runtime / "budget.json"), selection_eligible=False
    )
    try:
        run_diagnostic(diagnostic_path)
        completion = Path(protocol["output"]) / "completion.json"
        if read(completion).get("status") != "complete":
            raise ValueError("Diagnostic did not complete")
        report = read(completion)
        if report.get("selection_eligible") is not False:
            raise ValueError("Diagnostic cannot select a production setting")
        verified(report["diagnostic"])
        outcome = "complete"
    finally:
        ledger.finish(
            work, start=started, end=time.time(), status=outcome, requests=0, tokens=0, actual_usd=0
        )
        for allocation, observation in state.get("allocations", {}).items():
            if allocation == recipe["allocation"]:
                ledger.record_allocation(allocation, start=observation["start"], end=time.time())
    write(
        root / "completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "diagnostic": binding(completion),
            "cumulative_budget": binding(runtime / "budget.json"),
            "step_id": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
            "dispatch_nonce": recipe["dispatch_nonce"],
            "generate_rationales": False,
            "selection_eligible": False,
        },
        immutable=True,
    )
    status(
        root, "complete", cumulative_budget=str(runtime / "budget.json"), selection_eligible=False
    )
