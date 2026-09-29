"""Dependency-proved E19 migration preserves scientific identities and costs."""

import pytest

from exact.experiments.recovery import ArtifactStore, stage_identity
from tools.recover_e19_precision import changed_identity, migrate_prefit, write


def identity(stage="inputs", parents=(), implementation=None):
    return stage_identity(
        stage,
        parameters={"tau": 0.5, "source_ids": ["train"]},
        inputs={"pool": "a" * 64},
        role="development",
        entity_kind="class",
        implementation=implementation or {"schema": "locked-bytes-v2"},
        dependencies={"torch": "pinned"},
        seed=17,
        parents=parents,
    )


def test_migration_changes_only_approved_implementation_and_descendant_parent():
    old_code, new_code = {"files": {"fit.py": "old"}}, {"files": {"fit.py": "new"}}
    original = identity("extraction", ["b" * 64], old_code)
    migrated = changed_identity(original, old_code, new_code)
    original_without = {
        k: v for k, v in original.items() if k not in {"artifact_id", "implementation"}
    }
    new_without = {k: v for k, v in migrated.items() if k not in {"artifact_id", "implementation"}}
    assert original_without == new_without
    assert migrated["artifact_id"] != original["artifact_id"]
    assert original["implementation"] == old_code
    with pytest.raises(ValueError, match="not the reviewed"):
        changed_identity(original, {"files": {"fit.py": "other"}}, new_code)
    evaluation = identity("evaluation", [original["artifact_id"]])
    updated = changed_identity(evaluation, old_code, new_code, [migrated["artifact_id"]])
    assert updated["implementation"] == evaluation["implementation"]
    assert updated["parents"] == [migrated["artifact_id"]]


def preparation(tmp_path):
    parent, root = tmp_path / "old", tmp_path / "new"
    name = "D0_E03--analytic_fitted"
    origin = parent / "qualification" / name
    out = origin / "run"
    fitting = out / "fitting/key"
    fitting.mkdir(parents=True)
    (out / "dataset").mkdir()
    root.mkdir()
    write(root / "repair.json", {"scope": "validation_only"})
    write(fitting / "training_scores.json", {"identity": "key", "rows": [{"S_base": 0.7}]})
    write(fitting / "training_units.json", {"source_groups": 1})
    write(
        out / "dataset/candidate_pool_sample_manifest.json",
        {"gold_free_summary": {"candidate_pairs": 3}},
    )
    old_code, new_code = {"files": {"fit.py": "old"}}, {"files": {"fit.py": "new"}}
    store = ArtifactStore(origin)
    inputs = identity()
    store.publish(inputs, {"_locked_inputs/pool": b"original"})
    extraction = identity("extraction", [inputs["artifact_id"]], old_code)
    write(out / "recovery-runtime.json", {"root": str(origin), "identity": extraction})
    return parent, root, origin, fitting, extraction, old_code, new_code


def test_raw_evidence_migration_is_checkpoint_not_complete_prediction(tmp_path):
    parent, root, origin, fitting, extraction, old_code, new_code = preparation(tmp_path)
    before = {str(p): p.read_bytes() for p in origin.rglob("*") if p.is_file()}
    migrate_prefit(parent, root, old_code, new_code)
    target = root / "qualification/D0_E03--analytic_fitted"
    store = ArtifactStore(target)
    updated = changed_identity(extraction, old_code, new_code)
    checkpoint = store.latest_checkpoint(updated["artifact_id"])
    assert checkpoint["cursor"] == {"next_pair": 0, "dataset_rows": 3}
    assert checkpoint["state"]["inference_pairs_complete"] == 0
    assert set(checkpoint["outputs"]) == {
        "fitting/key/training_scores.json",
        "fitting/key/training_units.json",
        "dataset/candidate_pool_sample_manifest.json",
    }
    with pytest.raises(FileNotFoundError):
        store.verify(updated["artifact_id"])
    assert before == {str(p): p.read_bytes() for p in origin.rglob("*") if p.is_file()}
    assert not (target / "run").exists()
    assert not (target / "recovery").exists()


@pytest.mark.parametrize(
    "name", ["fusion.json", "selector.json", "graph.json", "fusion.json.folds/0.json"]
)
def test_fitted_outputs_are_never_imported_as_raw_evidence(tmp_path, name):
    parent, root, origin, fitting, extraction, old_code, new_code = preparation(tmp_path)
    write(fitting / name, {"invalid": "fitted output"})
    with pytest.raises(ValueError, match="Unexpected fitted artifact"):
        migrate_prefit(parent, root, old_code, new_code)


def test_immutable_recovery_records_refuse_changed_content(tmp_path):
    p = tmp_path / "record.json"
    write(p, {"source": 1})
    write(p, {"source": 1})
    with pytest.raises(ValueError, match="Immutable"):
        write(p, {"source": 2})


def test_completed_control_migration_retains_original_manifest_and_verified_outputs(tmp_path):
    import json

    from tools import measured_once as once
    from tools.recover_e19_precision import bind, migrate_control

    parent, root = tmp_path / "old", tmp_path / "new"
    name = "D0_E03--analytic_shipped"
    origin = parent / "qualification" / name
    out = origin / "run"
    old_code, new_code = {"files": {"fit.py": "old"}}, {"files": {"fit.py": "new"}}
    store = ArtifactStore(origin)
    inputs = identity()
    extraction = identity("extraction", [inputs["artifact_id"]], old_code)
    evaluation = identity("evaluation", [extraction["artifact_id"]])
    store.publish(inputs, {"_locked_inputs/pool": b"original"})
    store.publish(
        extraction,
        {
            "alignment/paper.maps_global.tsv": b"s\tt\n",
            "stats/execution_measurement.json": json.dumps(
                {
                    "schema_version": 1,
                    "artifact_id": extraction["artifact_id"],
                    "wall_seconds": 100,
                    "origin_attempt": "original-attempt",
                }
            ).encode(),
            "source_decisions.json": b'{"s":"t"}',
            "dataset/candidate_pool_sample_manifest.json": b'{"gold_free_summary":{"candidate_pairs":2}}',
        },
    )
    store.publish(evaluation, {"evaluation/metrics.json": b'{"f1":0.5}'})
    ids = {x["stage"]: x["artifact_id"] for x in (inputs, extraction, evaluation)}
    for artifact in ids.values():
        store.restore(artifact, out)
    write(
        out / "experiment_manifest.json",
        {"status": "complete", "recovery": {"artifacts": ids, "attempt_id": "original-attempt"}},
    )
    write(origin / "attempts/original-attempt/attempt.json", {"source": "original"})
    write(out / "recovery-runtime.json", {"root": str(origin), "identity": extraction})
    config = origin / "config.yaml"
    config.write_text("unchanged: true\n")
    row = {
        "status": "passed",
        "execution_status": "complete",
        "prefix": False,
        "processed_pairs": 2,
        "dataset_rows": 2,
        "seed": 17,
        "source_cap": 300,
        "generate_rationales": False,
        "new_usage": {"attempts": 0},
        "config": bind(config),
        "output_dir": str(out),
        "budget_work_id": "original",
        "bindings": [
            bind(out / n)
            for n in (
                "experiment_manifest.json",
                "recovery-runtime.json",
                "source_decisions.json",
                "alignment/paper.maps_global.tsv",
                "dataset/candidate_pool_sample_manifest.json",
            )
        ],
    }
    write(origin / "measurement.json", row)
    write(root / "repair.json", {"scope": "validation_only"})
    original_manifest = (out / "experiment_manifest.json").read_bytes()
    account = {"work": {"original": {"status": "complete", "seconds": 100}}}
    migrate_control(
        parent, root, {"configs": {name: bind(config)}}, old_code, new_code, once, account
    )
    migrated = json.loads((root / "qualification" / name / "measurement.json").read_text())
    assert migrated["budget_work_id"] == "original"
    assert migrated["compatibility_migration"]["origin_artifacts"] == ids
    assert migrated["compatibility_migration"]["new_numerical_execution"] is False
    assert (out / "experiment_manifest.json").read_bytes() == original_manifest
    assert (
        root / "qualification" / name / "run/alignment/paper.maps_global.tsv"
    ).read_bytes() == b"s\tt\n"
    assert account == {"work": {"original": {"status": "complete", "seconds": 100}}}


def test_migrated_attempt_costs_are_complete_unique_and_closed():
    from tools.qualify_cached_family import continuation_work

    old = "qualification/old/arm/1"
    new = "qualification/new/arm/2"
    state = {"work": {old: {"status": "failed"}, new: {"status": "interrupted"}}}
    assert continuation_work(state, "/new/continue.py", "arm", [old, old]) == [old, new]
    for status in ("reserved", "complete"):
        state["work"][old]["status"] = status
        with pytest.raises(ValueError, match="closed original accounting"):
            continuation_work(state, "/new/continue.py", "arm", [old])
    with pytest.raises(ValueError, match="closed original accounting"):
        continuation_work(state, "/new/continue.py", "different_arm", [old])


@pytest.mark.parametrize("step", ["14372.39", "14372.40"])
def test_recovery_resolves_current_supervisor_after_redeployment(tmp_path, step):
    from tools.recover_e19_precision import current_supervisor_step

    (tmp_path / "supervisor-step-id").write_text(step + "\n")
    assert current_supervisor_step({"supervisor": str(tmp_path), "supervisor_step": "14372.37"}) == step


@pytest.mark.parametrize("step", ["14373.39", "14372.extern", "14372.0.1", ""])
def test_recovery_rejects_invalid_supervisor_receipt(tmp_path, step):
    from tools.recover_e19_precision import current_supervisor_step

    (tmp_path / "supervisor-step-id").write_text(step)
    with pytest.raises(ValueError, match="numeric retained-allocation"):
        current_supervisor_step({"supervisor": str(tmp_path)})
