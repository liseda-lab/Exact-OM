"""Saved-result finalization rejects missing arms and corrupted durable evidence."""

from types import SimpleNamespace

import pytest

from exact.experiments import harness
from exact.experiments.recovery import ArtifactStore, stage_identity
from tools import finalize_prepared_selection as finalize
from tools.prepared_batch import binding, write


@pytest.mark.parametrize(
    "damage", [None, "bytes", "role", "config", "seed", "duplicate", "missing"]
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
    if damage == "bytes":
        blob = next((tmp_path / "artifacts/blobs").iterdir())
        blob.write_bytes(b"corrupt")
    if damage == "duplicate":
        recipe["manifests"] *= 2
    if damage == "missing":
        recipe["manifests"] = []
    if damage:
        with pytest.raises(ValueError):
            finalize.verify_cells(recipe, suite, {})
    else:
        assert finalize.verify_cells(recipe, suite, {}) == [path]
