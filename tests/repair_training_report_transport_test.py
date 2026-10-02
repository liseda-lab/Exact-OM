"""A full v3 acquisition report must survive the real bounded worker transport."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.repair.workers import bounded_call
from tools.repair import train
from tools.repair.training_report import publish_report, read_report


def report_fixture():
    return {
        "schema": "exact-repair/training/v3",
        "status": "complete",
        "selected_epoch": 5,
        "history": [{"epoch": 5, "selection_criterion": [-1.0, -0.8, 6]}],
        "acquisition_rounds": [{"evidence": "x" * (17 * 1024 * 1024)}],
        "coverage": {"unknown": None, "unvisited": 33, "scheduled": 128},
    }


def publish_large_training_result(directory):
    # bounded_call uses a fresh spawned interpreter; install the fixture there.
    import torch

    model = SimpleNamespace(
        metadata=(), config={"revision": "v3"}, state_dict=lambda: {"weight": torch.ones(1)}
    )
    original = train.train_cases
    try:
        train.train_cases = lambda *args, **kwargs: (model, report_fixture())
        return train._train_payload(
            [], [], {"revision": "v3", "checkpoint_path": directory / "training-state.pt"}
        )
    finally:
        train.train_cases = original


def test_large_report_survives_real_worker_with_small_result(tmp_path):
    torch = pytest.importorskip("torch")
    expected = report_fixture()
    failed = bounded_call(report_fixture, timeout=10)
    assert failed.status == "error" and "transport frame limit" in failed.detail
    result = bounded_call(
        publish_large_training_result,
        tmp_path,
        timeout=15,
    )
    assert result.status == "complete", result.detail
    assert len(pickle.dumps(result.value)) < 2048
    checkpoint, report = result.value
    assert report["size_bytes"] > 16 * 1024 * 1024
    assert read_report(report, tmp_path / "reports") == expected
    assert torch.equal(
        train._read_training_checkpoint(checkpoint, tmp_path / "checkpoints")["state_dict"][
            "weight"
        ],
        torch.ones(1),
    )


def test_report_artifact_rejects_corruption_missing_and_wrong_directory(tmp_path):
    expected = {"schema": "exact-repair/training/v3", "status": "complete"}
    artifact = publish_report(expected, tmp_path)
    assert read_report(artifact, tmp_path) == expected
    with pytest.raises(ValueError, match="outside"):
        read_report(artifact, tmp_path / "other")
    with pytest.raises(ValueError, match="size mismatch"):
        read_report({**artifact, "size_bytes": artifact["size_bytes"] + 1}, tmp_path)
    path = Path(artifact["path"])
    original = path.read_bytes()
    path.write_bytes(original.replace(b"complete", b"tampered"))
    with pytest.raises(ValueError, match="integrity mismatch"):
        read_report(artifact, tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        read_report(artifact, tmp_path)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes(original)
    path.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="outside"):
        read_report(artifact, tmp_path)


def test_report_transfer_preserves_json_output_semantics(tmp_path):
    report = {"schema": "exact-repair/training/v3", "history": [(1, None)], "unknown": None}
    assert read_report(publish_report(report, tmp_path), tmp_path) == json.loads(json.dumps(report))
    with pytest.raises(ValueError, match="schema v3"):
        read_report(publish_report({"schema": "legacy"}, tmp_path), tmp_path)
