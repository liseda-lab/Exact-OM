"""Batch cache identity, persistent I/O and reference-only training discovery."""
import json
from types import SimpleNamespace
import pytest
import torch
from exact.experiments import runtime
from exact.experiments.encoder_store import encoder_store, close_encoder_stores
from exact.experiments.numerical_cache import cached_numerical, training_score_reference
from tests.numerical_cache_test import Probe, runtime as bind_runtime
from tests.encoder_cache_migration_test import Config, Tokenizer, scorer


def test_hosted_batches_reuse_channels_but_never_full_forward(tmp_path, monkeypatch):
    bind_runtime(tmp_path, monkeypatch)
    model = Probe()
    model.use_llm = True
    model.llm_experiment_config = {"gate": {"mode": "fixed"}}
    first = model.forward(torch.tensor([0.2]))
    second = model.forward(torch.tensor([0.2]))
    assert model.forward_calls == 2 and model.channel_calls == 1
    assert torch.equal(first["score"], second["score"])
    assert not second["numerical_cache"]["full_forward_replayed"]
    assert model._numerical_scope_rebuilds == 2
    assert not model._numerical_cache.connection.in_transaction
    assert not hasattr(model, "_numerical_channel_scopes")


def test_hosted_batch_is_bounded_and_releases_writer_before_wait(tmp_path, monkeypatch):
    bind_runtime(tmp_path, monkeypatch)
    class Hosted(Probe):
        use_llm = True
        llm_experiment_config = {"gate": {"mode": "fixed"}}
        @torch.inference_mode()
        @cached_numerical()
        def forward(self, values):
            for _ in range(20):
                self.channel(values)
            assert not self._numerical_cache.connection.in_transaction
            self._numerical_cache.flush()
            assert not self._numerical_cache.connection.in_transaction
            return {"score": values}
    model = Hosted()
    model.forward(torch.tensor([0.2]))
    assert model.channel_calls == 1
    assert model._numerical_scope_rebuilds == 1
    assert model._numerical_cache.transactions == 1


def test_nested_context_and_scorer_error_cleanup(tmp_path, monkeypatch):
    bind_runtime(tmp_path, monkeypatch)
    model = Probe()
    previous = {"outer": "scope"}
    model._numerical_channel_scopes = previous
    with pytest.raises(TypeError):
        model.forward("invalid")
    assert model.forward_calls == 1
    assert model._numerical_channel_scopes is previous
    assert model._numerical_cache.depth == 0
    assert not model._numerical_cache.connection.in_transaction


def test_encoder_deduplicates_orders_and_uses_one_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_DIR", str(tmp_path))
    close_encoder_stores()
    calls = []
    def compute(texts):
        calls.append(texts)
        return torch.tensor([[float(ord(text)), 2.] for text in texts])
    model, tok, current = SimpleNamespace(config=Config(), training=False), Tokenizer(), scorer()
    cold = runtime.cached_encoder_rows(current, tok, model, ["a", "b", "a"], 16, compute)
    store = encoder_store(tmp_path / "vectors.sqlite3")
    assert calls == [["a", "b"]]
    assert store.counts["connections"] == store.counts["transactions"] == store.counts["lookups"] == 1
    warm = runtime.cached_encoder_rows(current, tok, model, ["b", "a", "b"], 16, compute)
    assert torch.equal(warm, cold[[1, 0, 1]])
    assert store is encoder_store(tmp_path / "vectors.sqlite3")
    assert store.counts["lookups"] == 2 and len(calls) == 1
    close_encoder_stores()
    reopened = runtime.cached_encoder_rows(current, tok, model, ["a"], 16, compute)
    assert torch.equal(reopened, cold[:1]) and len(calls) == 1


@pytest.mark.parametrize("damage", ["checksum", "shape", "dtype", "truncated"])
def test_encoder_corruption_is_a_miss(tmp_path, monkeypatch, damage):
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_DIR", str(tmp_path))
    model, tok, current = SimpleNamespace(config=Config(), training=False), Tokenizer(), scorer()
    calls = []
    def compute(texts):
        calls.append(texts)
        return torch.tensor([[1., 2.] for _ in texts])
    runtime.cached_encoder_rows(current, tok, model, ["a"], 16, compute)
    store = encoder_store(tmp_path / "vectors.sqlite3")
    if damage == "checksum":
        store.db.execute("UPDATE vectors SET sha256='incorrect'")
    elif damage == "shape":
        store.db.execute("UPDATE vectors SET shape=?", (json.dumps({"shape": [3], "dtype": "torch.float32"}),))
    elif damage == "dtype":
        store.db.execute("UPDATE vectors SET shape=?", (json.dumps({"shape": [2], "dtype": "torch.float16"}),))
    else:
        store.db.execute("UPDATE vectors SET raw=x'00'")
    store.db.commit()
    actual = runtime.cached_encoder_rows(current, tok, model, ["a", "a"], 16, compute)
    assert calls == [["a"], ["a"]]
    assert torch.equal(actual, torch.tensor([[1., 2.], [1., 2.]]))
    assert store.counts["corrupt"] == 1


def test_resident_cpu_cache_bounds_physical_storage(tmp_path, monkeypatch):
    from collections import OrderedDict
    monkeypatch.setenv("EXACT_CPU_EMBEDDING_MAX_BYTES", "16")
    current, cache = scorer(), OrderedDict()
    large = torch.ones(100, 2)
    for index in range(3):
        current._cache_store(cache, str(index), large[index], None)
    assert len(cache) == 2
    assert sum(value.untyped_storage().nbytes() for value in cache.values()) == 16


def test_training_reference_discovery_integrity_and_authorization(tmp_path, monkeypatch):
    bind_runtime(tmp_path, monkeypatch)
    path = tmp_path / "producer" / "training_scores.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"identity": "training", "rows": [{"Src": "train", "Tgt": "t", "score": .5}]}))
    application = {"source_ids": ["dev"], "negative_label_policy": "confirmed_negatives"}
    first = training_score_reference(Probe(), "training", application, 8, aggregate_path=path)
    second = training_score_reference(Probe(), "training", {**application, "source_ids": ["other-dev"]}, 8)
    assert first == second
    assert not list((tmp_path / "numerical-cache").glob("*.sqlite"))
    assert len(list((tmp_path / "numerical-cache/training-aggregates").glob("*.json"))) == 1
    assert training_score_reference(Probe(), "changed-labels", application, 8) is None
    assert training_score_reference(Probe(), "training", application, 16) is None
    with pytest.raises(ValueError, match="overlaps"):
        training_score_reference(Probe(), "training", {**application, "source_ids": ["train"]}, 8)
    path.write_text("corrupt")
    assert training_score_reference(Probe(), "training", application, 8) is None
