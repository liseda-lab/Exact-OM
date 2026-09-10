"""Offline lock creation validates real assets and preserves historical identity."""

import pytest
import yaml

from tools import prepare_experiment_locks as locks


def _snapshot(root, model, *, revision="1" * 40, weights=True):
    folder = root / ("models--" + model.replace("/", "--"))
    snapshot = folder / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (folder / "refs").mkdir(exist_ok=True)
    (folder / "refs/main").write_text(revision)
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "config.json").write_text("{}")
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"fixture weights")
    return snapshot


def test_cached_revision_checks_weights_shards_and_content(tmp_path):
    snapshot = _snapshot(tmp_path, "example/encoder")
    first = locks.cached_model("example/encoder", tmp_path)
    assert first["resolved_revision"] == "1" * 40
    (snapshot / "model.safetensors").write_bytes(b"different bytes")
    assert (
        locks.cached_model("example/encoder", tmp_path)["artifact_sha256"]
        != first["artifact_sha256"]
    )
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"missing.safetensors"}}'
    )
    with pytest.raises(ValueError, match="shards are missing"):
        locks.cached_model("example/encoder", tmp_path)
    (snapshot / "model.safetensors.index.json").unlink()
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="weights are missing"):
        locks.cached_model("example/encoder", tmp_path)
    assert locks.cached_model("example/encoder", tmp_path, tokenizer_only=True)["tokenizer_only"]
    with pytest.raises(FileNotFoundError, match="not cached"):
        locks.cached_model("example/encoder", tmp_path, revision="2" * 40)


def test_offline_baseline_freeze_is_immutable_and_keeps_r0(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(
        socket.socket, "connect", lambda *args, **kwargs: pytest.fail("network used")
    )
    monkeypatch.setattr(
        locks, "_source_tree_fingerprint", lambda root: {"sha256": "a" * 64, "files": 1}
    )
    cache = tmp_path / "cache"
    for model in (
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "BAAI/bge-large-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "Xenova/gpt-4o",
        "BAAI/bge-reranker-large",
    ):
        _snapshot(cache, model, weights=model != "Xenova/gpt-4o")
    parent = tmp_path / "R_0.yaml"
    parent.write_text(
        yaml.safe_dump(
            {
                "baseline_id": "R_0",
                "parent": "B0",
                "exact_om_version": "2.1.0",
                "pyowlcore_version": "0.2.0",
                "config": "historical.yaml",
                "config_sha256": "b" * 64,
                "source_commit": "c" * 40,
                "source_tree_sha256": "d" * 64,
                "source_tree_files": 1,
                "status": "frozen_configuration",
                "experiment_flags": "disabled",
                "note": "Historical evidence, not current code",
            }
        )
    )
    before = parent.read_bytes()
    kwargs = dict(
        cache=cache,
        parent_path=parent,
        models=("BAAI/bge-reranker-large",),
        note="Corrected local evaluation and hosted-only routing",
    )
    output = tmp_path / "locks"
    first = locks.prepare(locks.ROOT / "exact/default_config.yaml", output, **kwargs)
    second = locks.prepare(locks.ROOT / "exact/default_config.yaml", output, **kwargs)
    assert first == second
    assert parent.read_bytes() == before
    baseline = locks.load_baseline(output / "R_v2.yaml")
    assert baseline.parent == "R_0"
    assert baseline.source_tree_sha256 == "a" * 64
    model_lock = locks._validate_model_lock(output / "models.lock.yaml")
    assert len(model_lock["models"]) == 5
    config = yaml.safe_load((output / "R_v2.config.yaml").read_text())
    assert config["candidates"]["encoder_revision"] == "1" * 40
    hosted = config["llm"]["profiles"]["openrouter_gpt4o_mini"]
    assert hosted["model"] == "openai/gpt-4o-mini-2024-07-18"
    assert hosted["revision"] == "2024-07-18"
    assert hosted["provider"] == {
        "only": ["OpenAI"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert config["llm"]["profiles"]["openrouter_gpt4o_mini"]["tokenizer_revision"] == "1" * 40
    assert all(
        value in {None, "openrouter_gpt4o_mini"} for value in config["llm"]["routing"].values()
    )
    with pytest.raises(ValueError, match="immutable lock output differs"):
        locks.prepare(
            locks.ROOT / "exact/default_config.yaml",
            output,
            **{**kwargs, "note": "Changed history"}
        )
