"""Completed treatment reuse rejects numerical changes and damaged evidence."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments.recovery import ArtifactStore, stage_identity
from tools import recover_e22_measurements as repair
from tools.prepared_batch import binding, read, write


@dataclass
class Cell:
    arm_id: str
    output_dir: Path
    cell_id: str = "cell"
    task_id: str = "D0_E03-global_alignment"
    seed: int = 17
    split_role: str = "development"
    reference_role: str = "valid"
    source_cap: int = 300
    arm_role: str = "treatment"
    config_hash: str = "config"
    suite_id: str = "suite"
    recovery: dict = field(default_factory=dict)
    resolved_config: dict = field(default_factory=lambda: {"supervision": {"label_budget": 100}})


def source_cell(root, arm="budget_100"):
    cell = Cell(
        arm, root / "screen/runs/E22" / arm / "D0_E03-global_alignment/seed-17", cell_id=arm
    )
    cell.output_dir.mkdir(parents=True)
    write(cell.output_dir / "_inputs/resolved.config.yaml", cell.resolved_config)
    pool = cell.output_dir / "dataset/candidate_pool_sample_manifest.json"
    write(pool, {"fingerprint": "pool"})
    identities = {}
    store = ArtifactStore(root)
    for stage in repair.STAGES:
        identity = stage_identity(
            stage,
            parameters={"arm": arm},
            inputs={"source": "a" * 64},
            implementation={"files": "same"},
            role="development",
            entity_kind="class",
            dependencies={},
            seed=17,
        )
        identities[stage] = identity
        outputs = {"_locked_inputs/source": b"ontology"} if stage == "inputs" else {}
        if stage == "extraction":
            outputs = {
                "predictions.csv": b"source,target\ns,t\n",
                "dataset/candidate_pool_sample_manifest.json": pool,
                "stats/execution_measurement.json": json.dumps(
                    {
                        "artifact_id": identity["artifact_id"],
                        "wall_seconds": 123,
                    }
                ).encode(),
            }
        if stage == "evaluation":
            outputs = {"metrics.json": b'{"f1": 0.75}'}
        store.publish(identity, outputs)
        if stage != "inputs":
            store.restore(identity["artifact_id"], cell.output_dir)
    report = dict(
        experiment_id="E22",
        arm_id=arm,
        task_id=cell.task_id,
        seed=17,
        stage="screen",
        status="complete",
        return_code=0,
        extraction_complete=True,
        generate_rationales=False,
        split_role="development",
        reference_role="valid",
        source_cap=300,
        arm_role="treatment",
        resolved_config_hash="config",
        candidate_pool_fingerprint="pool",
        candidate_pool_manifest_provenance={
            **binding(pool),
            "bytes": pool.stat().st_size,
            "rows": None,
        },
        recovery={
            "reused_stages": [],
            "attempt_id": arm,
            "artifacts": {s: i["artifact_id"] for s, i in identities.items()},
        },
    )
    path = cell.output_dir / "experiment_manifest.json"
    write(path, report)
    return cell, SimpleNamespace(identities=identities), path, store


def test_completed_treatment_keeps_identity_and_original_measurement(tmp_path):
    cell, recovery, path, store = source_cell(tmp_path)
    report, saved = repair.verify_completed_cell(
        cell, recovery, path, store, pool_fingerprint="pool"
    )
    assert report["recovery"]["artifacts"] == {
        s: i["artifact_id"] for s, i in recovery.identities.items()
    }
    raw = store._blob(saved["extraction"]["outputs"]["stats/execution_measurement.json"]["sha256"])
    assert read(raw)["wall_seconds"] == 123


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "seed",
        "role",
        "rationale",
        "pool",
        "config",
        "identity",
        "reused",
        "exposed",
        "blob",
        "stored_identity",
        "measurement",
    ],
)
def test_completed_treatment_rejects_changed_or_damaged_evidence(tmp_path, damage):
    cell, recovery, path, store = source_cell(tmp_path)
    report = read(path)
    if damage in {"status", "seed", "role", "rationale", "pool"}:
        key, value = {
            "status": ("status", "failed"),
            "seed": ("seed", 29),
            "role": ("reference_role", "test"),
            "rationale": ("generate_rationales", True),
            "pool": ("candidate_pool_fingerprint", "different"),
        }[damage]
        report[key] = value
    elif damage == "config":
        write(path.parent / "_inputs/resolved.config.yaml", {"supervision": {"label_budget": 400}})
    elif damage == "identity":
        report["recovery"]["artifacts"]["extraction"] = "f" * 64
    elif damage == "reused":
        report["recovery"]["reused_stages"] = ["extraction"]
    elif damage == "exposed":
        (path.parent / "predictions.csv").write_bytes(b"changed")
    elif damage == "blob":
        payload = store.verify(recovery.identities["extraction"]["artifact_id"])
        store._blob(payload["outputs"]["predictions.csv"]["sha256"]).write_bytes(b"damaged")
    elif damage == "stored_identity":
        recovery.identities["extraction"]["seed"] = 29
    else:
        measurement = path.parent / "stats/execution_measurement.json"
        write(measurement, {"artifact_id": "wrong"})
    write(path, report)
    with pytest.raises((ValueError, FileNotFoundError)):
        repair.verify_completed_cell(cell, recovery, path, store, pool_fingerprint="pool")


@pytest.mark.parametrize("state,exit_code", [("running", 1), ("failed", 0), ("complete", 0)])
def test_live_or_successful_source_is_not_failed_recovery(tmp_path, state, exit_code):
    write(tmp_path / "completion.json", {"status": state, "exit_code": exit_code})
    (tmp_path / "exit-code").write_text(str(exit_code))
    write(tmp_path / "campaign.lock.yaml", {})
    with pytest.raises(ValueError, match="terminal failure"):
        repair._terminal_source(
            {
                "source_runtime": str(tmp_path / "runtime/campaign"),
                "source_campaign": binding(tmp_path / "campaign.lock.yaml"),
            }
        )


@pytest.mark.parametrize("damage", [None, "last_treatment", "control", "implementation"])
def test_all_five_arms_verified_before_treatment_publication(tmp_path, monkeypatch, damage):
    from exact.experiments import campaign, harness, runtime

    old = tmp_path / "old/runtime/campaign"
    target = tmp_path / "new/runtime/campaign"
    cells, identities, control_rows = [], {}, []
    for arm in ("budget_25", "label_free", "budget_100", "budget_400", "active_100"):
        cell, recovery, path, store = source_cell(old, arm)
        identities[arm] = recovery.identities
        if arm in {"budget_25", "label_free"}:
            for stage in repair.STAGES:
                artifact = store.verify(recovery.identities[stage]["artifact_id"])
                ArtifactStore(target).publish(
                    artifact["identity"],
                    {n: store._blob(o["sha256"]) for n, o in artifact["outputs"].items()},
                )
            control_rows.append({"arm": arm, "source_manifest": binding(path)})
        cell.output_dir = target / cell.output_dir.relative_to(old)
        cells.append(cell)
    write(
        target / "control-imports.json",
        {
            "controls": control_rows,
            "extraction_identities": {
                arm: v["extraction"]["artifact_id"] for arm, v in identities.items()
            },
        },
    )
    root = old.parent.parent
    write(root / "completion.json", {"status": "failed", "exit_code": 1})
    (root / "exit-code").write_text("1")
    write(root / "campaign.lock.yaml", {})
    settings = dict(
        source_runtime=str(old),
        source_code=str(tmp_path / "code"),
        source_campaign=binding(root / "campaign.lock.yaml"),
        parent_run_id="old",
    )
    recipe = dict(scientific_step="E22", parent_run_id="old", e22_completed_treatments=settings)
    source = SimpleNamespace(config=SimpleNamespace(experiment_id="E22", depends_on=[]))
    monkeypatch.setattr(campaign, "load_campaign", lambda p: (SimpleNamespace(steps=[]), None))
    monkeypatch.setattr(
        campaign, "materialize_campaign", lambda *a, **k: SimpleNamespace(sources=[source])
    )
    monkeypatch.setattr(harness, "build_cells", lambda *a, **k: cells)
    monkeypatch.setattr(harness, "inherited_selection_overlay", lambda *a: {})
    monkeypatch.setattr(harness, "_provenance_payload", lambda *a, **k: {})
    monkeypatch.setattr(
        runtime, "CellRecovery", lambda c, *a: SimpleNamespace(identities=identities[c.arm_id])
    )
    monkeypatch.setattr(
        runtime, "_code_identity", lambda p, **k: str(p) if damage == "implementation" else {}
    )
    if damage == "last_treatment":
        path = old / cells[-1].output_dir.relative_to(target) / "experiment_manifest.json"
        report = read(path)
        report["status"] = "running"
        write(path, report)
    elif damage == "control":
        ArtifactStore(target)._manifest_path(
            identities["budget_25"]["extraction"]["artifact_id"]
        ).unlink()
    if damage:
        with pytest.raises((ValueError, FileNotFoundError)):
            repair.import_completed_treatments(
                recipe, root / "campaign.lock.yaml", target, tmp_path / "new-code"
            )
        for arm in repair.TREATMENTS:
            assert (
                not ArtifactStore(target)
                ._manifest_path(identities[arm]["extraction"]["artifact_id"])
                .exists()
            )
    else:
        result = repair.import_completed_treatments(
            recipe, root / "campaign.lock.yaml", target, tmp_path / "code"
        )
        assert result["new_scientific_executions"] == 0
        assert len(result["treatments"]) == 3
        for arm in identities:
            ArtifactStore(target).verify(identities[arm]["extraction"]["artifact_id"])
        assert len(list((target / "recovery/cells").glob("*.json"))) == 3
