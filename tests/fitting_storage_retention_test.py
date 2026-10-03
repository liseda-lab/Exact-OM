"""Training feature storage retires only redundant, durably aggregated shards."""

import json

import pandas as pd
import pytest

from exact.impl.trainer import fitting
from tests.grouped_fitting_test import (
    TinyDataset,
    TinyScorer,
    selector,
    tiny_runner,
    training_rows,
)


def runner_at(root, scorer=None):
    root.mkdir(parents=True, exist_ok=True)
    frame = training_rows()
    pool, reference = root / "training.tsv", root / "reference.tsv"
    frame[["Src", "Tgt"]].to_csv(pool, sep="\t", index=False)
    frame[frame.Tgt.str.endswith("0")][["Src", "Tgt"]].to_csv(reference, sep="\t", index=False)
    head = selector()
    head.training_reference_file_path = str(reference)
    runner = tiny_runner(
        root,
        TinyDataset(pd.DataFrame({"Src": ["report"], "Tgt": ["report-0"]})),
        scorer or TinyScorer(),
        head,
    )
    runner.supervision_config = {"negative_label_policy": "complete_reference"}
    runner.training_candidates_file_path = pool
    return runner


def test_aggregate_is_unchanged_and_published_before_only_consumed_shards_retire(
    tmp_path, monkeypatch
):
    freeze = fitting.freeze_json
    snapshots, reads = [], []

    def publish(path, payload, **kwargs):
        if path.name == "training_scores.json":
            shards = list(path.parent.glob("*.json"))
            assert len(shards) == 5
            rows = [row for shard in shards for row in json.loads(shard.read_text())["rows"]]
            assert sorted(rows, key=lambda row: (row["Src"], row["Tgt"])) == payload["rows"]
            snapshots.append(json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n")
            freeze(path.parent / "unrelated.json", {"keep": True})
            result = freeze(path, payload, **kwargs)
            assert path.read_bytes() == snapshots[-1]
            assert all(shard.is_file() for shard in shards)
            return result
        return freeze(path, payload, **kwargs)

    def shared(*args, **kwargs):
        assert "rows" not in kwargs, "A durable aggregate must not be serialized into SQL again"
        reads.append(True)
        return None

    monkeypatch.setattr(fitting, "freeze_json", publish)
    monkeypatch.setattr("exact.experiments.numerical_cache.shared_training_scores", shared)
    runner_at(tmp_path).fit_training_pool(batch_size=5)
    path = next(tmp_path.glob("fitting/*/training_scores.json"))
    assert path.read_bytes() == snapshots[0]
    assert (path.parent / "unrelated.json").is_file()
    assert not [p for p in path.parent.glob("*.json") if len(p.stem) == 64]
    restarted = runner_at(tmp_path)
    restarted.fit_training_pool(batch_size=5)
    assert restarted.model.calls == []
    assert path.read_bytes() == snapshots[0]
    assert reads == [True]


def test_failed_aggregate_preserves_all_shards_and_resumes_without_scoring(tmp_path, monkeypatch):
    freeze = fitting.freeze_json

    def fail_aggregate(path, payload, **kwargs):
        if path.name == "training_scores.json":
            return freeze(path, payload, max_bytes=1)
        return freeze(path, payload, **kwargs)

    monkeypatch.setattr(fitting, "freeze_json", fail_aggregate)
    with pytest.raises(ValueError, match="safety limit"):
        runner_at(tmp_path).fit_training_pool(batch_size=5)
    shards = list(tmp_path.glob("fitting/*/*.json"))
    assert len(shards) == 5
    rows = sorted(
        (row for path in shards for row in json.loads(path.read_text())["rows"]),
        key=lambda row: (row["Src"], row["Tgt"]),
    )
    assert not list(tmp_path.glob("fitting/*/training_scores.json*"))
    monkeypatch.setattr(fitting, "freeze_json", freeze)
    restarted = runner_at(tmp_path)
    restarted.fit_training_pool(batch_size=5)
    assert restarted.model.calls == []
    assert (
        json.loads(next(tmp_path.glob("fitting/*/training_scores.json")).read_text())["rows"]
        == rows
    )
    assert not any(path.exists() for path in shards)


def test_interrupted_scoring_retains_completed_sources_and_resumes_remaining(tmp_path):
    class Interrupted(TinyScorer):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            if len(self.calls) == 2:
                raise RuntimeError("Interrupted fixture scoring")
            return result

    with pytest.raises(RuntimeError, match="Interrupted"):
        runner_at(tmp_path, Interrupted()).fit_training_pool(batch_size=5)
    shards = list(tmp_path.glob("fitting/*/*.json"))
    assert len(shards) == 1
    saved = json.loads(shards[0].read_text())
    restarted = runner_at(tmp_path)
    restarted.fit_training_pool(batch_size=5)
    assert len(restarted.model.calls) == 4
    rescored = {source for call in restarted.model.calls for source in call["src_iris"]}
    assert not rescored.intersection(saved["source_ids"])
    rows = json.loads(next(tmp_path.glob("fitting/*/training_scores.json")).read_text())["rows"]
    assert rows[: len(saved["rows"])] == saved["rows"]
    assert len(rows) == 18
    assert not shards[0].exists()


def test_legacy_shared_aggregate_remains_readable_without_republishing(tmp_path, monkeypatch):
    runner_at(tmp_path / "original").fit_training_pool(batch_size=5)
    original = next((tmp_path / "original").glob("fitting/*/training_scores.json"))
    saved = original.read_bytes()
    rows = json.loads(saved)["rows"]

    def legacy_rows(*args, **kwargs):
        assert "rows" not in kwargs
        return rows

    monkeypatch.setattr("exact.experiments.numerical_cache.shared_training_scores", legacy_rows)
    restarted = runner_at(tmp_path / "legacy-replay")
    restarted.fit_training_pool(batch_size=5)
    assert restarted.model.calls == []
    restored = next((tmp_path / "legacy-replay").glob("fitting/*/training_scores.json"))
    assert restored.read_bytes() == saved
