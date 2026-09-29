"""Publish completed measured cells without repeating their numerical execution."""

from __future__ import annotations

import copy
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from tools.qualify_cached_family import binding


def reused_comparison_forecast(proof):
    """Reserve only remaining work; preserve original scientific execution costs."""
    result = copy.deepcopy(proof)
    result["reused_execution_forecast_seconds"] = result["comparison_seconds"]
    estimate = result["estimate"]
    for field in (
        "cold_seconds",
        "units",
        "seconds_per_unit",
        "preparation_seconds",
        "fitting_seconds",
        "evaluation_seconds",
        "requests",
        "tokens",
        "projected_usd",
    ):
        estimate[field] = 0
    result["comparison_seconds"] = 0
    result["whole_family_seconds"] = result["inventory_seconds"]
    result["hours"] = result["whole_family_seconds"] / 3600
    result["execution_policy"] = "Measured outputs are scientific evidence; no repeated worker."
    return result


def _read(path):
    return json.loads(Path(path).read_text())


def _verified(item):
    if binding(item["path"]) != item:
        raise ValueError("Measured artifact binding changed: " + str(item["path"]))
    return Path(item["path"])


def verify_measurement(row, cell, budget):
    """Bind a full execution to its intended cell before inspecting reuse."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.recovery import ArtifactStore

    if (
        row.get("status") != "passed"
        or row.get("execution_status") != "complete"
        or row.get("prefix") is not False
        or not 0 < row.get("processed_pairs", 0) == row.get("dataset_rows")
        or row.get("seed") != cell.seed
        or row.get("source_cap") != cell.source_cap
        or row.get("generate_rationales") is not False
        or cell.generate_rationales
        or cell.split_role != "development"
        or any(row.get("new_usage", {}).values())
    ):
        raise ValueError("Only complete matching development executions may be promoted")
    if load_yaml_mapping(_verified(row["config"])) != cell.resolved_config:
        raise ValueError("Measured configuration differs from its scientific cell")
    verified = {_verified(item).resolve() for item in row["bindings"]}
    output = Path(row["output_dir"]).resolve()
    required = {
        output / "experiment_manifest.json",
        output / "recovery-runtime.json",
        output / "dataset/candidate_pool_sample_manifest.json",
        output / "alignment/paper.maps_global.tsv",
        output / "source_decisions.json",
    }
    if not required <= verified:
        raise ValueError("Measurement lacks bound predictions, population or provenance")
    manifest = _read(output / "experiment_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Measured execution is incomplete")
    runtime = _read(output / "recovery-runtime.json")
    artifacts = manifest.get("recovery", {}).get("artifacts", {})
    if set(artifacts) != {"inputs", "extraction", "evaluation"}:
        raise ValueError("Measured execution lacks complete numerical artifacts")
    if artifacts["extraction"] != runtime["identity"]["artifact_id"]:
        raise ValueError("Measured runtime and extraction identity disagree")
    work_ids = row.get("budget_work_ids", [row["budget_work_id"]])
    if not work_ids or work_ids[-1] != row["budget_work_id"]:
        raise ValueError("Measurement accounting lineage is incomplete")
    for position, work_id in enumerate(work_ids):
        status = budget["work"].get(work_id, {}).get("status")
        allowed = {"complete"} if position == len(work_ids) - 1 else {"interrupted", "failed"}
        if status not in allowed:
            raise ValueError("Measured execution lacks closed original accounting")
    store = ArtifactStore(Path(runtime["root"]))
    for artifact_id in artifacts.values():
        store.verify(artifact_id)
    return store, artifacts, manifest


def promote_measured_cells(campaign, runtime, rows, family, selection, workdir):
    """Account only actual verification/import work, retaining original charges."""
    from exact.experiments.budget import BudgetLedger

    path = Path(runtime) / "budget.json"
    account = BudgetLedger(path, _read(path)["limits"])
    work_id = f"preparation/{family}/measured-results/{uuid.uuid4().hex}"
    account.admit(work_id, group="reserve", seconds=0)
    start, status = time.time(), "failed"
    try:
        receipt = _promote_measured_cells(campaign, runtime, rows, family, selection, workdir)
        status = "complete"
        return receipt
    finally:
        account.finish(
            work_id, start=start, end=time.time(), status=status, requests=0, tokens=0, actual_usd=0
        )


def _promote_measured_cells(campaign, runtime, rows, family, selection, workdir):
    """Import only exact numerical identities, leaving source manifests immutable."""
    from exact.experiments import harness
    from exact.experiments.campaign import materialize_campaign
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _hash

    runtime, workdir = Path(runtime), Path(workdir)
    suite = materialize_campaign(Path(campaign), runtime / "declarations/screen", stage="screen")
    source = next(s for s in suite.sources if s.config.experiment_id == family)
    cells = harness.build_cells(
        suite,
        source,
        stage="screen",
        output_root=runtime.parent,
        inherited_overlay=harness.inherited_selection_overlay(selection, source.config.depends_on),
    )
    keyed = {
        (row["name"] if "--" in row["name"] else row.get("case_id", "D0") + "--" + row["name"]): row
        for row in rows
    }
    keys = {c.task_id.removesuffix("-global_alignment") + "--" + c.arm_id for c in cells}
    if len(keyed) != len(rows) or set(keyed) != keys or len(keys) != len(cells):
        raise ValueError("Promotion requires every declared cell exactly once")
    budget = _read(runtime / "budget.json")
    store = ArtifactStore(runtime)
    imports = []
    # Validate every arm before importing any outputs or making a selection.
    for cell in cells:
        key = cell.task_id.removesuffix("-global_alignment") + "--" + cell.arm_id
        row = keyed[key]
        previous, artifacts, manifest = verify_measurement(row, cell, budget)
        expected = replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True})
        recovery = CellRecovery(
            expected, harness._provenance_payload(expected, suite, workdir=workdir), workdir
        )
        if {stage: item["artifact_id"] for stage, item in recovery.identities.items()} != artifacts:
            raise ValueError("Measured numerical identity differs: " + key)
        imports.append((cell, previous, artifacts, manifest, row))
    records = []
    for cell, previous, artifacts, manifest, row in imports:
        for artifact_id in artifacts.values():
            store.import_artifact(previous.root, artifact_id)
        # This is an explicit alias to a verified origin, not a fabricated result.
        # The normal runner creates its own current attempt, preserving this parent
        # and relocating evaluation references to the new output directory.
        index = runtime / "recovery/cells" / (_hash([cell.suite_id, cell.cell_id]) + ".json")
        if index.exists():
            if _read(index).get("artifacts") != artifacts:
                raise ValueError("Existing scientific cell has different numerical artifacts")
        else:
            harness._atomic_json(
                index,
                {
                    "artifacts": artifacts,
                    "attempt_id": manifest["recovery"]["attempt_id"],
                    "output_dir": row["output_dir"],
                    "imported_measurement": binding(
                        Path(row["output_dir"]) / "experiment_manifest.json"
                    ),
                },
            )
        records.append(
            {
                "cell": cell.cell_id,
                "source_manifest": binding(Path(row["output_dir"]) / "experiment_manifest.json"),
                "source_attempt": manifest["recovery"]["attempt_id"],
                "source_attempt_manifest": binding(
                    previous.root / "attempts" / manifest["recovery"]["attempt_id"] / "attempt.json"
                ),
                "source_root": str(previous.root),
                "artifacts": artifacts,
                "budget_work_ids": row.get("budget_work_ids", [row["budget_work_id"]]),
                "execution_measurement": manifest.get("execution_measurement"),
            }
        )
    receipt = {
        "schema_version": 1,
        "family": family,
        "cells": records,
        "original_costs_retained": True,
        "new_scientific_executions": 0,
    }
    path = runtime / "measured-results.json"
    if path.exists() and _read(path) != receipt:
        raise ValueError("A different measured-result import already exists")
    harness._atomic_json(path, receipt)
    return receipt


@contextmanager
def require_reuse_only():
    """Fail closed if artifact finalization would start another scientific worker."""
    from exact.experiments import harness

    original = harness._run_subprocess

    def forbidden(*args, **kwargs):
        raise RuntimeError("Measured-result finalization must not repeat a scientific worker")

    harness._run_subprocess = forbidden
    try:
        yield
    finally:
        harness._run_subprocess = original
