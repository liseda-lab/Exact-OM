"""Batch cache identity, persistent I/O and reference-only training discovery."""
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from exact.experiments import runtime
from exact.experiments.encoder_store import encoder_store, close_encoder_stores
from exact.experiments.numerical_cache import (
    NumericalCache, cached_numerical, publish_training_reference,
    training_content_contract, training_score_reference,
)
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


def test_training_reference_rejects_overlap_before_publication_and_allows_only_uri_relocation(tmp_path, monkeypatch):
    bind_runtime(tmp_path, monkeypatch)
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"identity": "training", "rows": [{"Src": "train", "Tgt": "t"}]}))
    application = {"source_ids": ["train"]}
    with pytest.raises(ValueError, match="overlaps"):
        training_score_reference(Probe(), "training", application, 8, aggregate_path=path)
    assert not list((tmp_path / "numerical-cache/training-aggregates").glob("*.json"))
    application["source_ids"] = ["dev"]
    first = training_score_reference(Probe(), "training", application, 8, aggregate_path=path)
    local = tmp_path / "local-reference.json"
    publish_training_reference(local, first)
    relocated = tmp_path / "relocated.json"
    relocated.write_bytes(path.read_bytes())
    path.unlink()
    moved = training_score_reference(Probe(), "training", application, 8, aggregate_path=relocated)
    assert moved["aggregate_uri"] != first["aggregate_uri"]
    assert moved["semantic_key"] == first["semantic_key"] and moved["sha256"] == first["sha256"]
    publish_training_reference(local, moved)
    assert json.loads(local.read_text()) == moved
    index = next((tmp_path / "numerical-cache/training-aggregates").glob("*.json"))
    damaged = json.loads(index.read_text())
    damaged["upstream"]["training_identity"] = "other-labels"
    index.write_text(json.dumps(damaged))
    assert training_score_reference(Probe(), "training", application, 8, aggregate_path=relocated) is None
    assert json.loads(index.read_text()) == damaged


def test_optional_numerical_write_failure_never_retries_scoring(tmp_path, monkeypatch):
    bind_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(NumericalCache, "put", lambda *args: (_ for _ in ()).throw(RuntimeError("disk serializer")))
    model = Probe()
    result = model.forward(torch.tensor([0.4]))
    assert model.forward_calls == model.channel_calls == 1
    assert result["numerical_cache"]["skipped_writes"] == 2
    monkeypatch.setenv("EXACT_NUMERICAL_CACHE", "0")
    with pytest.raises(TypeError):
        model.forward("bad input")
    assert model.forward_calls == model.channel_calls == 2


def test_numerical_physical_cap_includes_sqlite_overhead_and_compact_tensor_storage(tmp_path):
    cache = NumericalCache(tmp_path)
    before = cache._disk_bytes()
    cache.disk_limit = before + 1
    cache.put("s", "op", "k", torch.ones(2))
    assert cache.stats()["entries"] == 0 and cache.skipped == 1
    cache.disk_limit = 10**7
    large = torch.randn(10000, 32)
    cache.put("s", "op", "view", large[17])
    assert cache.stats()["payload_bytes"] < 4096
    assert torch.equal(cache.get("s", "op", "view", "cpu"), large[17])
    assert cache.stats()["disk_bytes"] <= cache.disk_limit
    cache.close()


def test_numerical_reader_observes_only_committed_batches(tmp_path):
    writer, reader = NumericalCache(tmp_path), NumericalCache(tmp_path)
    with writer.batch():
        writer.put("s", "op", "k", [1, 2])
        assert reader.get("s", "op", "k", "cpu") is None
        assert not writer.connection.in_transaction
    assert reader.get("s", "op", "k", "cpu") == [1, 2]
    writer.close()
    reader.close()


def test_encoder_cap_accounts_for_many_small_row_keys_and_page_overhead(tmp_path, monkeypatch):
    store = encoder_store(tmp_path / "vectors.sqlite3")
    before = sum(path.stat().st_size for path in store.path.parent.glob(store.path.name + "*"))
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_MAX_BYTES", str(before + 65536 + 2 * 1000 * 8))
    store.publish({str(index).zfill(64): torch.ones(2) for index in range(1000)})
    assert store.db.execute("SELECT count(*) FROM vectors").fetchone()[0] == 0
    assert store.counts["skipped_writes"] == 1000 and not store.db.in_transaction


def test_encoder_bulk_2048_byte_vectors_respect_db_and_wal_cap(tmp_path, monkeypatch):
    limit = 6 * 1024**2
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_MAX_BYTES", str(limit))
    store = encoder_store(tmp_path / "vectors.sqlite3")
    # Exercise a commit that checkpoints, retaining both DB and WAL allocation.
    store.db.execute("PRAGMA wal_autocheckpoint=1")
    vector = torch.arange(1024, dtype=torch.float16)
    existing = {str(index).zfill(64): vector for index in range(32)}
    store.publish(existing)
    assert store.db.execute("SELECT count(*) FROM vectors").fetchone()[0] == 32
    assert Path(str(store.path) + "-wal").stat().st_size > 0

    bulk = {str(index).zfill(64): vector for index in range(32, 1056)}
    store.publish(bulk)
    physical = sum(path.stat().st_size for path in store.path.parent.glob(store.path.name + "*"))
    assert physical <= limit
    assert store.db.execute("SELECT count(*) FROM vectors").fetchone()[0] == 32
    assert store.counts["skipped_writes"] == 1024 and not store.db.in_transaction
    assert torch.equal(store.lookup(["0".zfill(64)], torch.float16, 1024)["0".zfill(64)], vector)
    assert not store.lookup(["32".zfill(64)], torch.float16, 1024)


def test_encoder_returns_complete_vectors_when_bulk_publication_exceeds_cap(tmp_path, monkeypatch):
    class WideConfig(Config):
        def to_dict(self):
            return {"hidden_size": 1024, "_commit_hash": self._commit_hash}

    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_MAX_BYTES", str(6 * 1024**2))
    current = scorer()
    current._cache_tensor_dtype = torch.float16
    model = SimpleNamespace(config=WideConfig(), training=False)
    texts = [str(index) for index in range(1024)]
    calls = []

    def compute(rows):
        calls.append(list(rows))
        return torch.tensor([float(text) for text in rows], dtype=torch.float16)[:, None].expand(-1, 1024)

    actual = runtime.cached_encoder_rows(current, Tokenizer(), model, texts + [texts[0]], 16, compute)
    assert calls == [texts]
    assert actual.shape == (1025, 1024)
    assert torch.equal(actual[:-1, 0], torch.arange(1024, dtype=torch.float16))
    assert torch.equal(actual[-1], actual[0])
    store = encoder_store(tmp_path / "vectors.sqlite3")
    assert store.db.execute("SELECT count(*) FROM vectors").fetchone()[0] == 0
    assert store.counts["skipped_writes"] == 1024 and not store.db.in_transaction


def _fork_encoder_probe(path, pipe):
    try:
        store = encoder_store(path)
        rows = store.lookup(["key"], torch.float32, 2)
        pipe.send((store.pid, list(rows), store.counts["connections"]))
    finally:
        close_encoder_stores()
        pipe.close()


def test_fork_reopens_process_owned_encoder_connections(tmp_path):
    store = encoder_store(tmp_path / "vectors.sqlite3")
    store.publish({"key": torch.tensor([1., 2.])})
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_fork_encoder_probe, args=(store.path, child))
    process.start()
    child.close()
    assert parent.poll(10)
    pid, keys, connections = parent.recv()
    process.join(10)
    assert process.exitcode == 0 and pid != os.getpid() and keys == ["key"] and connections == 1
    assert store.pid == os.getpid() and torch.equal(store.lookup(["key"], torch.float32)["key"], torch.tensor([1., 2.]))
    parent.close()


def test_actual_encoder_weight_and_tokenizer_mutations_invalidate_identities(monkeypatch):
    current, tokenizer = scorer(), Tokenizer()
    model = torch.nn.Linear(2, 2, bias=False).eval()
    model.config = Config()
    initial = runtime.encoder_identity(current, tokenizer, model, 16)
    with torch.no_grad():
        model.weight.add_(1)
    changed = runtime.encoder_identity(current, tokenizer, model, 16)
    assert initial != changed
    tokenizer.padding_side = "left"
    padded = runtime.encoder_identity(current, tokenizer, model, 16)
    assert padded != changed
    tokenizer.init_kwargs = {"do_lower_case": False}
    configured = runtime.encoder_identity(current, tokenizer, model, 16)
    assert configured != padded
    vocab = {"a": 0, "b": 1}
    tokenizer.get_vocab = lambda: vocab
    current._numerical_channel_scopes = {}
    before = runtime.encoder_identity(current, tokenizer, model, 16)
    vocab["c"] = 2
    current._numerical_channel_scopes = {}  # Beginning of the next outer batch.
    assert runtime.encoder_identity(current, tokenizer, model, 16) != before
    del current._numerical_channel_scopes
    with torch.inference_mode():
        unversioned = torch.nn.Linear(2, 2, bias=False).eval()
    unversioned.config = Config()
    assert runtime.encoder_identity(current, tokenizer, unversioned, 16) is None


def test_resident_encoder_cache_cannot_survive_loss_of_weight_identity(monkeypatch):
    from collections import OrderedDict
    current, tokenizer = scorer(), Tokenizer()
    model = torch.nn.Linear(2, 2, bias=False).eval()
    model.config = Config()
    calls, cache = [], OrderedDict()
    def compute(tokenizer, encoder, texts, max_len):
        calls.append(list(texts))
        return torch.full((len(texts), 2), float(len(calls)))
    current._encode_texts = compute
    assert current._encode_with_cache(["same"], tokenizer, model, 16, cache, None).tolist() == [[1., 1.]]
    assert current._encode_with_cache(["same"], tokenizer, model, 16, cache, None).tolist() == [[1., 1.]]
    model.train()
    assert current._encode_with_cache(["same", "same"], tokenizer, model, 16, cache, None).tolist() == [[2., 2.], [2., 2.]]
    assert calls == [["same"], ["same", "same"]] and not cache
    model.eval()
    model.config._commit_hash = None
    current._encode_with_cache(["same"], tokenizer, model, 16, cache, None)
    current._encode_with_cache(["same"], tokenizer, model, 16, cache, None)
    assert len(calls) == 4


def native_training_model(tmp_path, *, seed=17, local=False):
    from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
    from tests.pair_adaptive_experiments_test import _scorer
    source, target = tmp_path / "source.ofn", tmp_path / "target.ofn"
    source.write_text("Ontology(Declaration(Class(<urn:s>)))")
    target.write_text("Ontology(Declaration(Class(<urn:t>)))")
    dataset = PairAdaptiveContextDataset(output_path=tmp_path / "dataset", cache_ok=False,
        verbaliser_name=None, verbalization_mode="deterministic", request_seed=seed,
        filter_exact_matches=local, drop_exact_match_sources=local)
    dataset.load_ontologies(source, target)
    model = _scorer(request_seed=seed)
    model._attached_dataset = dataset
    model.dataset_signature = dataset.dataset_signature
    return model


def test_known_training_contract_reuses_reporting_modes_seeds_but_binds_actual_dependencies(tmp_path, monkeypatch):
    bind_runtime(tmp_path, monkeypatch)
    model = native_training_model(tmp_path)
    first = training_content_contract(model)
    assert first is not None
    model.request_seed = model._attached_dataset.request_seed = 29
    model._attached_dataset._filter_exact_matches = True
    model._attached_dataset._drop_exact_match_sources = True
    bind_runtime(tmp_path, monkeypatch, seed=29, role="local_ranking", inputs={
        "source": "source-bytes", "target": "target-bytes", "candidates": "other-reporting-pool",
    })
    assert training_content_contract(model) == first
    previous_share = model._attached_dataset._candidate_share_k
    model._attached_dataset._candidate_share_k = previous_share + 1
    assert training_content_contract(model) != first
    model._attached_dataset._candidate_share_k = previous_share
    model._exact_anchor_src_to_tgt = {"urn:s": {"urn:t"}}
    assert training_content_contract(model) != first
    model._exact_anchor_src_to_tgt = {}
    model._attached_dataset.verbalization_mode = "llm"
    assert training_content_contract(model) is None
    model._attached_dataset._verbalization_templates = {"r": "first template"}
    templates = training_content_contract(model)
    model._attached_dataset._verbalization_templates["r"] = "second template"
    assert training_content_contract(model) != templates
    model.graph_config["hierarchy_removal"] = 0.5
    shuffled = training_content_contract(model)
    model.request_seed = 43
    assert training_content_contract(model) != shuffled
    assert training_content_contract(Probe()) is None


def test_known_training_aggregate_is_discovered_across_reporting_modes(tmp_path, monkeypatch):
    bind_runtime(tmp_path, monkeypatch)
    model = native_training_model(tmp_path)
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"identity": "same-ordered-train-rows-labels-batches", "rows": [{"Src": "train", "Tgt": "t"}]}))
    application = {"dataset_signature": model.dataset_signature, "source_ids": ["dev"]}
    first = training_score_reference(model, "same-ordered-train-rows-labels-batches", application, 8, aggregate_path=path)
    assert first is not None
    bind_runtime(tmp_path, monkeypatch, seed=29, role="local_ranking", inputs={
        "source": "source-bytes", "target": "target-bytes", "candidates": "other-reporting-pool",
    })
    model.request_seed = model._attached_dataset.request_seed = 29
    model._attached_dataset._filter_exact_matches = True
    assert training_score_reference(model, "same-ordered-train-rows-labels-batches", application, 8) == first
    assert training_score_reference(model, "different-training-labels", application, 8) is None
