"""Operational recovery checks without model calls or benchmark execution."""

import errno
import json
import shutil
import socket

import pytest

from exact.experiments import recovery
from exact.experiments.recovery import (
    ArtifactStore,
    build_reuse_plan,
    create_attempt,
    finish_attempt,
    implementation_identity,
    legacy_import_record,
    stage_identity,
    validate_current_result_set,
)


def identity(stage="evidence", *, parents=(), **overrides):
    options = dict(
        parameters={"pool": "frozen"},
        inputs={"source": "a" * 64},
        role="development",
        entity_kind="class",
        implementation={"code": "v1"},
        dependencies={"numpy": "locked"},
        seed=42,
    )
    options.update(overrides)
    return stage_identity(stage, parents=parents, **options)


@pytest.mark.parametrize(
    "stage",
    [
        "ontology",
        "evidence",
        "pair_scores",
        "fitted_heads",
        "llm_responses",
        "extraction",
        "evaluation",
    ],
)
def test_checkpoint_exact_ids_resume_and_relocation(tmp_path, stage):
    store = ArtifactStore(tmp_path / "original")
    key = identity(stage)
    completed = store.checkpoint(
        key,
        completed_ids=["s:2", "s:8"],
        cursor={"next": "s:9"},
        outputs={"state.bin": b"saved numerical state"},
        state={"rng": [42, 7], "optimizer_step": 2},
    )
    moved = ArtifactStore(tmp_path / "new")
    moved.import_checkpoint(store.root, key["artifact_id"])
    shutil.rmtree(store.root)
    resumed = moved.latest_checkpoint(key["artifact_id"])
    assert resumed == completed
    assert resumed["completed_ids"] == ["s:2", "s:8"]
    assert resumed["cursor"] == {"next": "s:9"}
    assert (
        moved.restore_checkpoint(resumed, tmp_path / "work")["state.bin"].read_bytes()
        == b"saved numerical state"
    )


def test_crash_disk_full_and_corruption_preserve_previous_checkpoint(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    key = identity()
    first = store.checkpoint(key, completed_ids=["s:1"], cursor="s:2", outputs={"scores": b"one"})
    publish = recovery._publish_json

    def disk_full(path, payload):
        if path.parent.name == key["artifact_id"]:
            raise OSError(errno.ENOSPC, "disk full before manifest publication")
        publish(path, payload)

    monkeypatch.setattr(recovery, "_publish_json", disk_full)
    with pytest.raises(OSError, match="disk full"):
        store.checkpoint(
            key, completed_ids=["s:1", "s:2"], cursor="s:3", outputs={"scores": b"two"}
        )
    assert store.latest_checkpoint(key["artifact_id"]) == first
    monkeypatch.setattr(recovery, "_publish_json", publish)
    second = store.checkpoint(
        key, completed_ids=["s:1", "s:2"], cursor="s:3", outputs={"scores": b"two"}
    )
    (tmp_path / second["outputs"]["scores"]["path"]).write_bytes(b"corrupt")
    assert store.latest_checkpoint(key["artifact_id"]) == first


def test_completed_artifact_relocates_all_parents_and_checksums(tmp_path):
    original = ArtifactStore(tmp_path / "original")
    parent = identity("evidence")
    child = identity("pair_scores", parents=[parent["artifact_id"]])
    original.publish(parent, {"evidence.json": b"raw similarities"})
    original.publish(child, {"scores.json": b"fusion scores"})
    relocated = ArtifactStore(tmp_path / "relocated")
    relocated.import_artifact(original.root, child["artifact_id"])
    before = (original.root / "artifacts" / "stages" / f'{child["artifact_id"]}.json').read_bytes()
    restored = relocated.restore(child["artifact_id"], tmp_path / "consumer")
    restored["scores.json"].write_bytes(b"consumer edits are independent")
    assert relocated.verify(child["artifact_id"])["status"] == "complete"
    assert (
        original.root / "artifacts" / "stages" / f'{child["artifact_id"]}.json'
    ).read_bytes() == before
    shutil.rmtree(original.root)
    relocated.verify(child["artifact_id"])
    blob = relocated.verify(parent["artifact_id"])["outputs"]["evidence.json"]["path"]
    (relocated.root / blob).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        relocated.verify(child["artifact_id"])


def test_immutable_artifact_and_attempt_revisions(tmp_path):
    store = ArtifactStore(tmp_path)
    key = identity()
    first = store.publish(key, {"facts": b"facts"})
    assert store.publish(key, {"facts": b"facts"}) == first
    with pytest.raises(ValueError, match="immutable"):
        store.publish(key, {"facts": b"changed"})
    attempt = create_attempt(tmp_path, design_id="frozen", provenance={"commit": "abc"})
    finish_attempt(
        tmp_path,
        attempt["attempt_id"],
        status="interrupted",
        continuation="exact-experiments --resume",
        summary={"last": "s:7"},
    )
    with pytest.raises(FileExistsError):
        finish_attempt(
            tmp_path, attempt["attempt_id"], status="complete", continuation="", summary={}
        )
    following = create_attempt(
        tmp_path,
        design_id="frozen",
        provenance={"commit": "def"},
        parent_attempt=attempt["attempt_id"],
    )
    assert following["parent_attempt"] == attempt["attempt_id"]


def test_duplicate_writer_and_explicit_stale_lock_recovery(tmp_path):
    store = ArtifactStore(tmp_path)
    key = identity()["artifact_id"]
    with store.lock(key):
        with pytest.raises(RuntimeError, match="writer"):
            with store.lock(key):
                pass
        with pytest.raises(ValueError, match="alive"):
            store.recover_stale_lock(key)
    path = store.directory / "locks" / f"{key}.json"
    path.write_text(json.dumps({"host": socket.gethostname(), "pid": 99999999}))
    store.recover_stale_lock(key)
    assert not path.exists()
    path.write_text(json.dumps({"host": "another-host", "pid": 99999999}))
    with pytest.raises(ValueError, match="another host"):
        store.recover_stale_lock(key)


def test_transitive_implementation_hash_ignores_docs_and_checkout_location(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "score.py").write_text("from normalizer import normalize\n")
    (root / "normalizer.py").write_text("def normalize(x): return x\n")
    (root / "README.md").write_text("docs")
    first = implementation_identity(root, ["score.py"])
    (root / "README.md").write_text("new docs")
    assert implementation_identity(root, ["score.py"]) == first
    copied = tmp_path / "relocated"
    shutil.copytree(root, copied)
    assert implementation_identity(copied, ["score.py"]) == first
    (root / "normalizer.py").write_text("def normalize(x): return x.lower()\n")
    assert implementation_identity(root, ["score.py"])["sha256"] != first["sha256"]


@pytest.mark.parametrize(
    "change",
    [
        {"role": "reporting"},
        {"parameters": {"model": "new"}},
        {"parameters": {"tokenizer": "new"}},
        {"parameters": {"pool": "new"}},
        {"inputs": {"training_labels": "b" * 64}},
        {"dependencies": {"torch": "new"}},
    ],
)
def test_material_changes_never_share_identity(change):
    assert identity()["artifact_id"] != identity(**change)["artifact_id"]


def test_repair_reuses_evidence_and_recomputes_only_descendants(tmp_path):
    store = ArtifactStore(tmp_path)
    evidence = identity("evidence")
    scores = identity("pair_scores", parents=[evidence["artifact_id"]])
    evaluation = identity("evaluation", parents=[scores["artifact_id"]])
    expected = {key["stage"]: key for key in (evidence, scores, evaluation)}
    previous = {stage: key["artifact_id"] for stage, key in expected.items()}
    for key in expected.values():
        store.publish(key, {"value": key["stage"].encode()})
    plan = build_reuse_plan(
        store,
        expected,
        previous,
        changed_stages=["pair_scores"],
        estimates={"evidence": 100, "pair_scores": 2, "evaluation": 1},
    )
    assert [row["action"] for row in plan["stages"]] == ["reuse", "recompute", "recompute"]
    assert plan["estimated_incremental_cost"] == 3
    evaluator = build_reuse_plan(store, expected, previous, changed_stages=["evaluation"])
    assert [row["action"] for row in evaluator["stages"]] == ["reuse", "reuse", "recompute"]


def test_legacy_import_keeps_original_unknown_provenance(tmp_path):
    (tmp_path / "experiment_manifest.json").write_text('{"fingerprint":"old-v1"}')
    record = legacy_import_record(tmp_path)
    assert record["original_manifests"]["experiment_manifest.json"]["fingerprint"] == "old-v1"
    assert record["reused"] == []
    assert record["source_hashes"]


def test_result_set_rejects_duplicates_missing_controls_and_stale_selection():
    rows = [
        dict(
            cell_id=cell,
            artifact_id="a" * 64,
            design_id="design",
            semantics_id="repair-1",
            selection_id="frozen",
        )
        for cell in ("control", "arm")
    ]
    options = dict(expected_cells=["control", "arm"], design_id="design", selection_id="frozen")
    validate_current_result_set(rows, **options)
    for invalid in (
        rows + rows[:1],
        rows[1:],
        [rows[0], {**rows[1], "semantics_id": "repair-0"}],
        [rows[0], {**rows[1], "selection_id": "refrozen"}],
    ):
        with pytest.raises(ValueError):
            validate_current_result_set(invalid, **options)
    with pytest.raises(ValueError, match="invalidated"):
        validate_current_result_set(rows, invalidated_artifacts=["a" * 64], **options)


def test_untrusted_output_paths_and_symlinks_rejected(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    key = identity()
    with pytest.raises(ValueError, match="Unsafe"):
        store.publish(key, {"../escape": b"bad"})
    store.publish(key, {"nested/data": b"safe"})
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "nested").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.restore(key["artifact_id"], destination)
