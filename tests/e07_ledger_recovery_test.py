"""Synchronization migration never relaxes scientific artifact identities."""

import pytest

from tools.recover_e07_ledger import verify_code, verify_identity


def identity():
    return {
        "artifact_id": "new",
        "implementation": {"sha256": "new-code"},
        "parameters": {"graph": "inductive"},
        "inputs": {"train": "hash"},
        "role": "development",
        "entity_kind": "class",
        "seed": 17,
        "parents": ["inputs"],
        "dependencies": {"torch": "pinned"},
    }


def test_only_reviewed_implementation_difference_migrates():
    expected = identity()
    saved = {**expected, "artifact_id": "old", "implementation": {"sha256": "old-code"}}
    verify_identity(saved, expected, {"sha256": "old-code"}, stage="extraction")


@pytest.mark.parametrize(
    "field,value",
    [
        ("parameters", {"graph": "off"}),
        ("inputs", {"train": "other"}),
        ("role", "test"),
        ("seed", 18),
        ("entity_kind", "individual"),
        ("parents", ["other"]),
        ("dependencies", {"torch": "changed"}),
    ],
)
def test_reject_changed_scientific_parent(field, value):
    expected = identity()
    saved = {
        **expected,
        "artifact_id": "old",
        "implementation": {"sha256": "old-code"},
        field: value,
    }
    with pytest.raises(ValueError, match="identity differs"):
        verify_identity(saved, expected, {"sha256": "old-code"}, stage="extraction")


def test_evaluation_requires_original_extraction_parent():
    expected = identity()
    saved = {**expected, "artifact_id": "old", "parents": ["old-extraction"]}
    verify_identity(saved, expected, {}, stage="evaluation", old_extraction="old-extraction")
    with pytest.raises(ValueError, match="identity differs"):
        verify_identity(saved, expected, {}, stage="evaluation", old_extraction="wrong")


def test_source_migration_rejects_unrelated_code_change(monkeypatch, tmp_path):
    old = {"files": {"exact/llm/ledger.py": "a", "exact/llm/routing.py": "b", "numerical": "c"}}
    new = {"files": {"exact/llm/ledger.py": "d", "exact/llm/routing.py": "e", "numerical": "c"}}
    repair = {
        "migration": "e07-ledger-retry-v1",
        "scientific_choices_unchanged": True,
        "reporting_labels_exposed": False,
        "changes": {
            "exact/llm/ledger.py": {"before": "a", "after": "d"},
            "exact/llm/routing.py": {"before": "b", "after": "e"},
        },
    }
    monkeypatch.setattr(
        "exact.experiments.runtime._code_identity",
        lambda root, evaluation: {} if evaluation else (old if root.name == "old" else new),
    )
    assert verify_code(tmp_path / "old", tmp_path / "new", repair) == old
    new["files"]["numerical"] = "changed"
    with pytest.raises(ValueError, match="outside reviewed"):
        verify_code(tmp_path / "old", tmp_path / "new", repair)


def test_all_six_arms_migrate_by_verified_artifact_without_family_progress(tmp_path, monkeypatch):
    import json
    from dataclasses import dataclass, field
    from types import SimpleNamespace

    from exact.experiments import runtime
    from exact.experiments.recovery import ArtifactStore, stage_identity
    from tools import recover_e07_ledger as module
    from tools.prepared_batch import binding, read, write

    @dataclass
    class Cell:
        arm_id: str
        config_hash: str
        resolved_config: dict
        resolved_supervision: dict = field(
            default_factory=lambda: {"rerank": {"resolved": "label_free"}}
        )
        suite_id: str = "suite"
        experiment_id: str = "E07"
        task_id: str = "D0_E03-global_alignment"
        seed: int = 17
        stage: str = "screen"
        split_role: str = "development"
        reference_role: str = "valid"
        source_cap: int = 200
        arm_role: str = "treatment"
        recovery: dict = field(default_factory=dict)

        @property
        def cell_id(self):
            return self.arm_id + "/" + self.task_id + "/seed-17"

    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_store = ArtifactStore(old_root)
    old_impl = {"sha256": "old-predictor"}
    checkpoints = {"brief_binary": 3, "facts_binary": 3, "retrieved_listwise": 2}
    cells, targets, reports, original_measurements = [], {}, {}, {}
    for arm in sorted(module.ARMS):
        cell = Cell(arm_id=arm, config_hash=arm, resolved_config={"arm": arm})
        cells.append(cell)
        source = old_root / "screen/runs/E07" / arm / cell.task_id / "seed-17"
        source.mkdir(parents=True)
        args = dict(
            parameters={"arm": arm},
            inputs={"ontology": "a" * 64},
            role="development",
            entity_kind="class",
            seed=17,
            dependencies={"numpy": "fixed"},
        )
        inputs = stage_identity("inputs", implementation={"schema": "locked"}, **args)
        before = stage_identity(
            "extraction", parents=[inputs["artifact_id"]], implementation=old_impl, **args
        )
        after = stage_identity(
            "extraction",
            parents=[inputs["artifact_id"]],
            implementation={"sha256": "new-predictor"},
            **args,
        )
        eval_before = stage_identity(
            "evaluation",
            parents=[before["artifact_id"]],
            implementation={"metric": "unchanged"},
            **args,
        )
        eval_after = stage_identity(
            "evaluation",
            parents=[after["artifact_id"]],
            implementation={"metric": "unchanged"},
            **args,
        )
        targets[arm] = {"inputs": inputs, "extraction": after, "evaluation": eval_after}
        old_store.publish(
            inputs, {"_inputs/resolved.config.yaml": json.dumps(cell.resolved_config).encode()}
        )
        old_store.restore(inputs["artifact_id"], source)
        artifacts = {"inputs": inputs["artifact_id"]}
        complete = arm == "brief_256_binary"
        if complete:
            measurement = {
                "schema_version": 1,
                "artifact_id": before["artifact_id"],
                "requests": 7,
                "tokens": 31,
                "seconds": 19.0,
            }
            original_measurements[arm] = measurement
            old_store.publish(
                before,
                {
                    "stats/execution_measurement.json": json.dumps(measurement).encode(),
                    "alignment.tsv": b"original numerical alignment",
                },
            )
            old_store.publish(eval_before, {"evaluation/metrics.json": b'{"f1":0.5}'})
            old_store.restore(before["artifact_id"], source)
            old_store.restore(eval_before["artifact_id"], source)
            artifacts.update(
                extraction=before["artifact_id"], evaluation=eval_before["artifact_id"]
            )
        elif arm in checkpoints:
            count = checkpoints[arm]
            old_store.checkpoint(
                before,
                completed_ids=[json.dumps([f"s{i}", "class", "t", "class"]) for i in range(count)],
                cursor={"next_pair": count, "dataset_rows": 4},
                outputs={
                    "checkpoints/values.json": b'{"values":[0.25]}',
                    "dataset/dataset.csv": (
                        "Src,SrcKind,Tgt,TgtKind,inference\n"
                        + "\n".join(f"s{i},class,t,class,True" for i in range(4))
                    ).encode(),
                    "checkpoints/inference_1.json": json.dumps(
                        dict(
                            kind="inference",
                            total_examples=4,
                            processed_examples=count,
                            mappings_count=count,
                            results_json_count=count,
                            explanation_records_count=count,
                            explanation_index_path="../explanations/index.json",
                        )
                    ).encode(),
                },
                state={"rng": [17, 23]},
            )
        write(source / "recovery-runtime.json", {"identity": before})
        report = {
            key: getattr(cell, key)
            for key in [
                "experiment_id",
                "arm_id",
                "task_id",
                "stage",
                "seed",
                "split_role",
                "reference_role",
                "source_cap",
                "arm_role",
                "resolved_supervision",
            ]
        }
        report.update(
            resolved_config_hash=cell.config_hash,
            generate_rationales=False,
            status="complete" if complete else "failed",
            return_code=0 if complete else 1,
            extraction_complete=complete,
            recovery={"artifacts": artifacts, "attempt_id": "old-attempt"},
        )
        manifest = source / "experiment_manifest.json"
        write(manifest, report)
        reports[arm] = binding(manifest)
    write(old_root / "screen/progress.json", {"design": "must not import family progress"})
    repair = tmp_path / "repair.json"
    write(repair, {"fixture": True})
    recipe = {
        "scientific_step": "E07",
        "e07_ledger_repair": {
            "source_runtime": str(old_root),
            "source_code": str(tmp_path),
            "repair_record": binding(repair),
            "manifests": reports,
        },
    }
    monkeypatch.setattr(module, "verify_code", lambda *_: old_impl)
    monkeypatch.setattr(module, "cells_for", lambda *_: (None, cells))
    monkeypatch.setattr("exact.experiments.harness._provenance_payload", lambda *a, **kw: {})
    monkeypatch.setattr(
        runtime, "CellRecovery", lambda cell, *_: SimpleNamespace(identities=targets[cell.arm_id])
    )
    result = module.import_saved(recipe, tmp_path / "campaign.yaml", new_root, tmp_path)
    assert len(result["rows"]) == 6
    assert not (new_root / "screen/progress.json").exists()
    new_store = ArtifactStore(new_root)
    for arm, identities in targets.items():
        new_store.verify(identities["inputs"]["artifact_id"])
        checkpoint = new_store.latest_checkpoint(identities["extraction"]["artifact_id"])
        if arm == "brief_256_binary":
            saved = new_store.verify(identities["extraction"]["artifact_id"])
            measurement = read(
                new_store._blob(saved["outputs"]["stats/execution_measurement.json"]["sha256"])
            )
            assert {key: measurement[key] for key in ("requests", "tokens", "seconds")} == {
                key: original_measurements[arm][key] for key in ("requests", "tokens", "seconds")
            }
            assert measurement["artifact_id"] == identities["extraction"]["artifact_id"]
            new_store.verify(identities["evaluation"]["artifact_id"])
        elif arm in checkpoints:
            assert checkpoint["cursor"]["next_pair"] == checkpoints[arm]
            assert checkpoint["state"]["rng"] == [17, 23]
            assert (
                new_store._blob(
                    checkpoint["outputs"]["checkpoints/values.json"]["sha256"]
                ).read_bytes()
                == b'{"values":[0.25]}'
            )
        else:
            assert checkpoint is None
    assert (
        module.import_saved(recipe, tmp_path / "campaign.yaml", new_root, tmp_path)["status"]
        == "pass"
    )


@pytest.mark.parametrize("failure", [None, "authorization", "migration"])
def test_account_authorization_migration_and_science_order(tmp_path, monkeypatch, failure):
    import os

    from exact.llm.ledger import RequestLedger
    from tests.prepared_batch_worker_test import _outputs, _worker
    from tools import (
        authorize_hosted_retries,
        experiment_resources,
        prepared_batch,
        recover_e07_ledger,
    )

    worker = _worker(tmp_path, monkeypatch)
    recipe = prepared_batch.read(worker.path)
    recipe.update(
        hosted_retry_authorizations={"fixture": True}, e07_ledger_repair={"fixture": True}
    )
    prepared_batch.write(worker.path, recipe)
    calls = []

    def authorize(actual, runtime):
        assert runtime == worker.runtime and (worker.root / "launch.json").exists()
        assert (
            RequestLedger(runtime / "openrouter").cached(worker.request) == b'{"result":"fixture"}'
        )
        assert os.environ["EXACT_OPENROUTER_RETRY_UNKNOWN"] == "0"
        calls.append("authorization")
        if failure == "authorization":
            raise ValueError("Authorization binding mismatch")

    def migrate(*args):
        calls.append("migration")
        if failure == "migration":
            raise ValueError("Migration identity mismatch")

    def execute(*args, **kwargs):
        calls.append("science")
        _outputs(worker)

    monkeypatch.setattr(authorize_hosted_retries, "apply_authorizations", authorize)
    monkeypatch.setattr(recover_e07_ledger, "import_saved", migrate)
    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    if failure:
        with pytest.raises(ValueError, match="mismatch"):
            prepared_batch.run_recipe(worker.path)
    else:
        prepared_batch.run_recipe(worker.path)
    expected = ["authorization", "migration", "science"]
    assert calls == expected[: 1 if failure == "authorization" else 2 if failure else 3]
    account = prepared_batch.read(worker.runtime / "budget.json")
    assert account["work"]["historical/closed"] == worker.parent_state["work"]["historical/closed"]
    if failure != "authorization":
        assert account["work"]["preparation/E07/ledger-repair/40"]["status"] == (
            "failed" if failure else "complete"
        )
