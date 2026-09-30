#!/usr/bin/env python3
"""Finalize verified saved comparison metadata; never execute scientific cells."""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from tools.prepared_batch import (
    binding,
    controls,
    copy_account,
    latest_account,
    read,
    status,
    verified,
    write,
)


def reconstruct(recipe, directory):
    """Use saved declarations and signed progress, rebinding only imported metadata."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import (
        external_selection_result,
        load_campaign,
        load_progress,
        materialize_campaign,
    )
    from exact.experiments.harness import (
        ExperimentConfig,
        ExperimentSource,
        _bind_external_selection,
        _validate_selection_experiments,
    )

    campaign = verified(recipe["campaign"])
    parent = Path(recipe["parent_runtime"])
    verified(recipe["progress"])
    lock, _ = load_campaign(campaign)
    suite = materialize_campaign(campaign, directory, stage="screen")
    progress = read(parent / "screen/progress.json")
    sources = []
    for original in suite.sources:
        declared = progress["declarations"][original.config.experiment_id]
        path = verified(declared)
        sources.append(
            ExperimentSource(ExperimentConfig.model_validate(load_yaml_mapping(path)), path)
        )
    suite = replace(suite, sources=tuple(sources), campaign={"root": str(parent)})
    experiments = load_progress(suite, "screen")
    for source in suite.sources:
        step = next(s for s in lock.steps if s.id == source.config.experiment_id)
        if step.external_selection is not None:
            historical = external_selection_result(lock, step, campaign.parent)
            experiments[step.id] = _bind_external_selection(source, suite, historical)
    name = recipe["scientific_step"]
    if experiments[name] != progress["experiments"][name]:
        raise ValueError("Recovery changed the completed scientific decision")
    _validate_selection_experiments(suite, experiments)
    return suite, experiments


def verify_cells(recipe, suite, experiments, *, checksums=True):
    from exact.experiments.harness import build_cells, inherited_selection_overlay
    from exact.experiments.recovery import ArtifactStore

    parent = Path(recipe["parent_runtime"])
    name = recipe["scientific_step"]
    source = next(s for s in suite.sources if s.config.experiment_id == name)
    cells = build_cells(
        suite,
        source,
        stage="screen",
        output_root=parent.parent,
        inherited_overlay=inherited_selection_overlay(
            {"experiments": experiments}, source.config.depends_on
        ),
    )
    manifests = [verified(item) for item in recipe["manifests"]]
    if len(cells) != len(manifests):
        raise ValueError("Saved comparison is incomplete")
    expected = {(c.arm_id, c.task_id, c.seed): c for c in cells}
    result_set = read(verified(recipe["result_set"]))
    rows = {row["cell_id"]: row for row in result_set["cells"]}
    if len(rows) != len(result_set["cells"]):
        raise ValueError("Duplicate saved cells")
    seen = set()
    store = ArtifactStore(parent)
    for path in manifests:
        report = read(path)
        key = (report["arm_id"], report["task_id"], report["seed"])
        cell = expected.get(key)
        if cell is None or key in seen:
            raise ValueError("Unexpected or duplicate saved arm/task/seed")
        seen.add(key)
        required = {
            "experiment_id": name,
            "stage": "screen",
            "status": "complete",
            "return_code": 0,
            "extraction_complete": True,
            "generate_rationales": False,
            "split_role": "development",
            "source_cap": cell.source_cap,
            "execution_mode": cell.resolved_config.get("data", {}).get("execution_mode"),
            "resolved_config_hash": cell.config_hash,
            "experiment_config_hash": source.raw_hash(),
        }
        if any(report.get(k) != v for k, v in required.items()):
            raise ValueError("Saved cell configuration/role/population changed: " + str(key))
        identifier = f"{name}/screen/{key[0]}/{key[1]}/seed-{key[2]}"
        artifacts = report["recovery"]["artifacts"]
        if rows[identifier]["artifacts"] != artifacts or set(artifacts) != {
            "inputs",
            "extraction",
            "evaluation",
        }:
            raise ValueError("Saved artifact lineage differs")
        if checksums:
            for stage, artifact in artifacts.items():
                checked = store.verify(artifact)
                # Downstream consumers open the original run paths. Verify those
                # exposed bytes as well as the immutable content-addressed store.
                if stage != "inputs":
                    for name, output in checked["outputs"].items():
                        if binding(path.parent / name)["sha256"] != output["sha256"]:
                            raise ValueError("Saved working output differs: " + name)
    return manifests


def register_lineage(recipe):
    """Link a replacement only after its nonce-bound numeric step is registered."""
    supervisor = Path(recipe["supervisor"])
    own = recipe["allocation"] + "." + os.environ["SLURM_STEP_ID"]
    for _ in range(45):
        with (supervisor / "registry.json.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            registry = controls(supervisor, Path(recipe["root"]))
            current = next((r for r in registry["runs"] if r["id"] == recipe["run_id"]), None)
            if current is not None:
                if (
                    current["step_id"] != own
                    or current.get("dispatch_nonce") != recipe["dispatch_nonce"]
                ):
                    raise ValueError("Registered recovery owner differs")
                parent = next(r for r in registry["runs"] if r["id"] == recipe["parent_run_id"])
                if parent.get("superseded_by") not in (None, recipe["run_id"]):
                    raise ValueError("Original run already has another replacement")
                parent.update(
                    enabled=False,
                    superseded_by=recipe["run_id"],
                    reason=recipe.get(
                        "lineage_reason",
                        "Selection metadata repair; all three completed E20 arms and cumulative costs retained without numerical reruns.",
                    ),
                )
                parent.pop("pending_recovery", None)
                registry["remaining_work_status"] = "pending"
                write(supervisor / "registry.json", registry)
                return registry
        time.sleep(1)
    raise ValueError("Dispatcher has not registered the live recovery; retain reservation")


def run(recipe_path):
    from exact.experiments.budget import BudgetLedger
    from exact.experiments.campaign import write_progress
    from exact.experiments.harness import write_selection_record
    from tools.resume_e19_once import check_owner

    recipe = read(recipe_path)
    root, supervisor, code = map(Path, (recipe["root"], recipe["supervisor"], recipe["code_root"]))
    with (root / "worker.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = controls(supervisor, root)
        if (
            os.environ.get("SLURM_JOB_ID") != recipe["allocation"]
            or not os.environ.get("SLURM_STEP_ID", "").isdigit()
        ):
            raise ValueError("Retained numeric Slurm step required")
        check_owner(
            SimpleNamespace(
                root=root, supervisor_step=(supervisor / "supervisor-step-id").read_text().strip()
            )
        )
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
            != recipe["commit"]
            or subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD"], cwd=code, text=True
            ).strip()
        ):
            raise ValueError("Frozen finalizer source changed")
        registry = register_lineage(recipe)
        for item in recipe["evidence"]:
            verified(item)
        runtime = root / "runtime" / recipe["campaign_id"]
        if runtime.exists():
            raise ValueError("Existing recovery runtime requires explicit reconciliation")
        account, state = latest_account(registry)
        old = read(verified(recipe["parent_budget"]))
        if state["limits"] != old["limits"] or any(
            state["work"].get(k) != v
            for k, v in old["work"].items()
            if v["status"] != "reserved" or k == "historical/G0"
        ):
            raise ValueError("Latest ledger loses completed/failed parent charges")
        copy_account(account, state, runtime)
        ledger = BudgetLedger(runtime / "budget.json", state["limits"])
        tail = recipe["failed_finalization_interval"]
        tail_key = "failed-finalization/" + recipe["parent_run_id"]
        if tail_key not in state["work"]:
            ledger.admit(tail_key, group="reserve", seconds=0, forecast_known=False)
            ledger.finish(
                tail_key,
                start=tail["start"],
                end=tail["end"],
                status="failed",
                requests=0,
                tokens=0,
                actual_usd=0,
            )
        started = time.time()
        work = "selection-finalization/" + recipe["run_id"]
        ledger.admit(work, group="reserve", seconds=0, forecast_known=False)
        status(root, "verifying_saved_results", cumulative_budget=str(runtime / "budget.json"))
        outcome = "failed"
        try:
            suite, experiments = reconstruct(recipe, root / "declarations")
            manifests = verify_cells(recipe, suite, experiments)
            controls(supervisor, root)
            parent = Path(recipe["parent_runtime"])
            stage = runtime / "screen"
            stage.mkdir(parents=True)
            (stage / "runs").symlink_to(parent / "screen/runs", target_is_directory=True)
            (runtime / "artifacts").symlink_to(parent / "artifacts", target_is_directory=True)
            for item in recipe["evidence"]:
                path = verified(item)
                if path.parent == parent / "screen" and path.name not in (
                    "progress.json",
                    "design.json",
                    "selection.json",
                ):
                    shutil.copyfile(path, stage / path.name)
            selection = write_selection_record(suite, experiments, output_root=root / "runtime")
            suite = replace(suite, campaign={"root": str(runtime)})
            write_progress(
                suite, "screen", experiments, [read(p) for p in manifests], status="complete"
            )
            outcome = "complete"
        finally:
            ledger.finish(
                work,
                start=started,
                end=time.time(),
                status=outcome,
                requests=0,
                tokens=0,
                actual_usd=0,
            )
            for allocation, observation in state.get("allocations", {}).items():
                if allocation == recipe["allocation"]:
                    ledger.record_allocation(
                        allocation, start=observation["start"], end=time.time()
                    )
        write(
            root / "completion.json",
            {
                "status": "complete",
                "exit_code": 0,
                "campaign": recipe["campaign"],
                "selection": binding(selection),
                "manifests": [binding(p) for p in manifests],
                "cumulative_budget": binding(runtime / "budget.json"),
                "step_id": recipe["allocation"] + "." + os.environ["SLURM_STEP_ID"],
                "dispatch_nonce": recipe["dispatch_nonce"],
                "parent_run_id": recipe["parent_run_id"],
                "numerical_cells_executed": 0,
                "generate_rationales": False,
            },
            immutable=True,
        )
        status(root, "complete", cumulative_budget=str(runtime / "budget.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    run(args.recipe)
