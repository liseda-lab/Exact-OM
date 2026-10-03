"""Completed fitting retention preserves science and copy isolation on future recovery."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from exact.experiments.recovery import ArtifactStore, stage_identity
from exact.utils.fitted_artifacts import freeze_json
from tools.deduplicate_fitting import deduplicate_completed


def binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / "experiments-v2/completed"
    runtime = root / "runtime/campaign"
    cell = runtime / "screen/runs/E23/arm/task/seed-17"
    cell.mkdir(parents=True)
    identity = stage_identity(
        "extraction",
        parameters={},
        inputs={},
        role="development",
        entity_kind="class",
        implementation={"fixture": "unchanged"},
        dependencies={},
        seed=17,
    )
    store = ArtifactStore(runtime)
    outputs = {
        "fitting/id/training_scores.json": b'{"rows":[1,2,3]}\n',
        "fitting/id/abc.json": b'{"rows":[1,2]}\n',
        "fitting/id/model.pt": b"weights",
        "stats/file.json": b'{"cost":2}\n',
    }
    manifest = store.publish(identity, outputs)
    store.restore(identity["artifact_id"], cell)
    report = cell / "experiment_manifest.json"
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "extraction_complete": True,
                "recovery": {"artifacts": {"extraction": identity["artifact_id"]}},
            }
        )
    )
    selected = runtime / "screen/selection.json"
    selected.write_text("{}")
    complete = root / "completion.json"
    complete.write_text(
        json.dumps(
            {
                "status": "complete",
                "exit_code": 0,
                "selection": binding(selected),
                "manifests": [binding(report)],
            }
        )
    )
    return root, store, cell, manifest, complete


def test_completed_dedup_preserves_bytes_and_all_bindings(saved, tmp_path):
    root, store, cell, manifest, complete = saved
    completion = complete.read_bytes()
    planned = deduplicate_completed(root)
    assert planned["planned_reclaim_bytes"] > 0
    assert len(planned["files"]) == 2
    assert all(row["action"] == "link" for row in planned["files"])
    result = deduplicate_completed(root, apply=True)
    assert (
        result["status"] == "complete"
        and result["reclaimed_bytes"] == planned["planned_reclaim_bytes"]
    )
    assert complete.read_bytes() == completion
    for row in result["files"]:
        path, blob = Path(row["path"]), Path(row["blob"])
        assert path.stat().st_ino == blob.stat().st_ino
        assert not path.stat().st_mode & 0o222
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    store.verify(manifest["identity"]["artifact_id"])
    assert deduplicate_completed(root, apply=True)["reclaimed_bytes"] == 0
    fresh = tmp_path / "fresh"
    store.restore(manifest["identity"]["artifact_id"], fresh)
    (fresh / "fitting/id/abc.json").write_text("independent new attempt")
    assert (cell / "fitting/id/abc.json").read_bytes() == b'{"rows":[1,2]}\n'
    store.verify(manifest["identity"]["artifact_id"])


def test_atomic_replacement_does_not_mutate_canonical_blob(saved):
    root, store, cell, manifest, _ = saved
    deduplicate_completed(root, apply=True)
    path = cell / "fitting/id/abc.json"
    freeze_json(path.parent / "new.json", {"rows": [4]})
    os.replace(path.parent / "new.json", path)
    original = manifest["outputs"]["fitting/id/abc.json"]
    assert store._blob(original["sha256"]).read_bytes() == b'{"rows":[1,2]}\n'


def test_incomplete_or_external_finalizer_is_untouched(saved, tmp_path):
    root, store, cell, manifest, complete = saved
    value = json.loads(complete.read_text())
    value["status"] = "failed"
    complete.write_text(json.dumps(value))
    assert deduplicate_completed(root, apply=True)["status"] == "skipped"
    assert (cell / "fitting/id/abc.json").stat().st_nlink == 1
    value["status"] = "complete"
    value["selection"] = {
        "path": str(tmp_path / "external/screen/selection.json"),
        "sha256": "unused",
    }
    complete.write_text(json.dumps(value))
    assert deduplicate_completed(root, apply=True)["status"] == "skipped"


def test_any_corrupt_candidate_rejects_before_linking_other_files(saved):
    root, store, cell, manifest, _ = saved
    (cell / "fitting/id/training_scores.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size changed"):
        deduplicate_completed(root, apply=True)
    assert (cell / "fitting/id/abc.json").stat().st_nlink == 1


def test_live_worker_lock_prevents_cleanup(saved):
    import fcntl

    root, _, _, _, _ = saved
    with (root / "worker.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            deduplicate_completed(root, apply=True)
