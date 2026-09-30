"""Saved-result finalization rejects missing arms and corrupted durable evidence."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments import harness
from exact.experiments.recovery import ArtifactStore, stage_identity
from tools import finalize_prepared_selection as finalize
from tools.prepared_batch import binding, read, write


@pytest.mark.parametrize(
    "damage",
    [None, "bytes", "role", "config", "seed", "duplicate", "missing", "working_bytes", "multiple"],
)
def test_saved_cells_require_complete_identical_verified_outputs(tmp_path, monkeypatch, damage):
    cell = SimpleNamespace(
        arm_id="control",
        task_id="D0-global_alignment",
        seed=17,
        source_cap=300,
        resolved_config={"data": {"execution_mode": "global_alignment"}},
        config_hash="config",
    )
    source = SimpleNamespace(
        config=SimpleNamespace(experiment_id="E20", depends_on=[]), raw_hash=lambda: "source"
    )
    suite = SimpleNamespace(sources=[source])
    monkeypatch.setattr(harness, "build_cells", lambda *a, **kw: [cell])
    monkeypatch.setattr(harness, "inherited_selection_overlay", lambda *a: {})
    store = ArtifactStore(tmp_path)
    identities = {}
    for stage in ("inputs", "extraction", "evaluation"):
        identity = stage_identity(
            stage,
            parameters={},
            inputs={},
            role="development",
            entity_kind="class",
            implementation={"fixture": "v1"},
            dependencies={},
            parents=list(identities.values())[-1:],
        )
        store.publish(identity, {stage + ".txt": b"verified"})
        (tmp_path / (stage + ".txt")).write_bytes(b"verified")
        identities[stage] = identity["artifact_id"]
    report = dict(
        experiment_id="E20",
        arm_id=cell.arm_id,
        task_id=cell.task_id,
        seed=17,
        stage="screen",
        status="complete",
        return_code=0,
        extraction_complete=True,
        generate_rationales=False,
        split_role="development",
        source_cap=300,
        execution_mode="global_alignment",
        resolved_config_hash="config",
        experiment_config_hash="source",
        recovery={"artifacts": identities},
    )
    if damage in ("role", "config", "seed"):
        report[{"role": "split_role", "config": "resolved_config_hash", "seed": "seed"}[damage]] = (
            "changed"
        )
    path = tmp_path / "manifest.json"
    write(path, report)
    result_set = tmp_path / "result-set.json"
    write(
        result_set,
        {
            "cells": [
                {
                    "cell_id": "E20/screen/control/D0-global_alignment/seed-17",
                    "artifacts": identities,
                }
            ]
        },
    )
    recipe = dict(
        parent_runtime=str(tmp_path),
        scientific_step="E20",
        manifests=[binding(path)],
        result_set=binding(result_set),
    )
    if damage == "multiple":
        cells = [cell]
        rows = [
            {"cell_id": "E20/screen/control/D0-global_alignment/seed-17", "artifacts": identities}
        ]
        for arm in ("contrastive", "cross_encoder"):
            other = SimpleNamespace(**{**vars(cell), "arm_id": arm})
            cells.append(other)
            other_path = tmp_path / (arm + ".json")
            write(other_path, {**report, "arm_id": arm})
            recipe["manifests"].append(binding(other_path))
            rows.append(
                {
                    "cell_id": f"E20/screen/{arm}/D0-global_alignment/seed-17",
                    "artifacts": identities,
                }
            )
        write(result_set, {"cells": rows})
        recipe["result_set"] = binding(result_set)
        monkeypatch.setattr(harness, "build_cells", lambda *a, **kw: cells)
    if damage == "bytes":
        blob = next((tmp_path / "artifacts/blobs").iterdir())
        blob.write_bytes(b"corrupt")
    if damage == "working_bytes":
        (tmp_path / "extraction.txt").write_bytes(b"corrupt")
    if damage == "duplicate":
        recipe["manifests"] *= 2
    if damage == "missing":
        recipe["manifests"] = []
    if damage and damage != "multiple":
        with pytest.raises(ValueError):
            finalize.verify_cells(recipe, suite, {})
    else:
        assert finalize.verify_cells(recipe, suite, {}) == [
            Path(item["path"]) for item in recipe["manifests"]
        ]


@pytest.mark.parametrize("changed", [False, True])
def test_failed_interval_keeps_original_owner_across_recovery(tmp_path, changed):
    from exact.experiments.budget import BudgetLedger

    ledger = BudgetLedger(
        tmp_path / "budget.json",
        {
            "node_hours_cap": 336,
            "envelopes_hours": {"reserve": 60},
            "requests_cap": 50000,
            "tokens_cap": 32000000,
        },
    )
    recipe = {
        "parent_run_id": "recovery-01",
        "failed_finalization_interval": {"run_id": "original", "start": 10, "end": 20},
    }
    finalize.charge_failed_finalization(recipe, ledger, {"work": {}})
    state = read(ledger.path)
    recipe["parent_run_id"] = "recovery-02"
    if changed:
        recipe["failed_finalization_interval"]["end"] = 21
        with pytest.raises(ValueError, match="charge differs"):
            finalize.charge_failed_finalization(recipe, ledger, state)
    else:
        finalize.charge_failed_finalization(recipe, ledger, state)
        assert read(ledger.path) == state
        assert list(state["work"]) == ["failed-finalization/original"]
