"""Resident rows avoid repeated copies while keeping dtype, order and bounded storage."""

import torch

from exact.impl.models.scorer_common import _DeviceEmbeddingRows


def test_resident_rows_preserve_duplicates_order_and_replacement():
    first = torch.tensor([0.125, 0.875], dtype=torch.float16)
    second = torch.tensor([-0.25, 0.25], dtype=torch.float16)
    cache = _DeviceEmbeddingRows("cpu", 16)
    assert torch.equal(
        cache.stack(["a", "b", "a"], [first, second, first]), torch.stack([first, second, first])
    )
    resident = cache.rows["a"][1]
    assert torch.equal(cache.stack(["b", "a"], [second, first]), torch.stack([second, first]))
    assert cache.rows["a"][1] is resident
    replacement = torch.tensor([1.0, 0.0], dtype=torch.float16)
    assert torch.equal(cache.stack(["a"], [replacement]), replacement[None])
    assert cache.rows["a"][1] is not resident
    assert cache.bytes == 8


def test_resident_rows_limit_owns_only_individual_row_storage():
    values = [torch.tensor([float(i), 1.0]) for i in range(16)]
    cache = _DeviceEmbeddingRows("cpu", 16)
    assert torch.equal(cache.stack(list(range(16)), values), torch.stack(values))
    assert cache.bytes == 16 and len(cache.rows) == 2
    assert all(row.untyped_storage().nbytes() == 8 for _, row in cache.rows.values())
    uncached = _DeviceEmbeddingRows("cpu", 0)
    assert torch.equal(uncached.stack([0, 1], values[:2]), torch.stack(values[:2]))
    assert not uncached.rows and uncached.bytes == 0


def test_pair_prefetch_preserves_uneven_missing_and_tied_evidence(monkeypatch):
    from types import SimpleNamespace

    from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset

    class EvidenceDataset(_TinyDataset):
        def get_entity_features(self, iri, side):
            if iri == "empty":
                return {"labels": [], "hierarchy": {}, "object_triples": [], "attributes": []}
            return {
                "labels": ["same", "synonym"],
                "hierarchy": {
                    "is_a": [["same", "is a", "parent", 0.4], ["same", "is a", "other", 0.2]]
                },
                "object_triples": [{"triple": ["same", "has part", "part"], "score": 0.4}],
                "attributes": [{"prop": "definition", "text": "same definition"}],
            }

    def model():
        from tests.encoder_cache_migration_test import Tokenizer
        scorer = _scorer(return_explanations=True)
        scorer.use_lexical = scorer.use_context = True
        pinned = SimpleNamespace(hidden_size=3, _commit_hash="a" * 40,
                                 to_dict=lambda: {"hidden_size": 3, "_commit_hash": "a" * 40})
        scorer.lex_model = SimpleNamespace(config=pinned, training=False)
        scorer.ctx_model = SimpleNamespace(config=pinned, training=False)
        scorer.lex_tok = scorer.ctx_tok = Tokenizer()
        calls = []

        def encode(tokenizer, encoder, texts, max_len):
            calls.append(list(texts))
            return torch.tensor([[len(text) + 1, sum(map(ord, text)) % 17, 2.0] for text in texts])

        scorer._encode_texts = encode
        scorer.attach_dataset(EvidenceDataset())
        return scorer, calls

    args = {
        "src_iris": ["s", "s", "empty"],
        "tgt_iris": ["t", "other", "empty"],
        "src_label_lists": [["same", "same", "synonym"], ["same"], []],
        "tgt_label_lists": [["same"], ["same", "synonym"], []],
    }
    monkeypatch.delenv("EXACT_EXPERIMENT_RUNTIME", raising=False)
    monkeypatch.delenv("EXACT_EMBEDDING_CACHE_DIR", raising=False)
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH", "0")
    plain, plain_calls = model()
    expected = plain(**args)
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH", "1")
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH_BATCH_SIZE", "64")
    prefetched, calls = model()
    actual = prefetched(**args)
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, actual[name]), name
    assert expected["explanations"] == actual["explanations"]
    assert len(calls) < len(plain_calls)
    assert any("parent" in batch and "has part" in batch for batch in calls)
    assert all(len(batch) <= 64 for batch in calls)


def test_prefetch_bounds_distinct_texts_and_disables_without_encoding(monkeypatch):
    from types import SimpleNamespace

    from tests.pair_adaptive_experiments_test import _scorer

    scorer = _scorer()
    scorer.use_lexical = True
    scorer.lex_model = SimpleNamespace()
    calls = []
    scorer.encode_labels_batch = lambda texts: calls.append(texts)
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH_MAX_TEXTS", "3")
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH_BATCH_SIZE", "2")
    scorer._prefetch_evidence_embeddings([], [["a", "a", "b", "c", "d"]])
    assert calls == [["a", "b"], ["c"]]
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH", "0")
    scorer._prefetch_evidence_embeddings([], [["new"]])
    assert calls == [["a", "b"], ["c"]]


def test_prefetch_oom_retries_smaller_batches_without_dropping_texts(monkeypatch):
    from types import SimpleNamespace

    from tests.pair_adaptive_experiments_test import _scorer

    scorer = _scorer()
    scorer.use_lexical = True
    scorer.lex_model = SimpleNamespace()
    encoded = []

    def encode(texts):
        if len(texts) > 2:
            raise torch.cuda.OutOfMemoryError("fixture prefetch pressure")
        encoded.extend(texts)

    scorer.encode_labels_batch = encode
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH_BATCH_SIZE", "8")
    scorer._prefetch_evidence_embeddings([], [["a", "b", "c", "d", "e"]])
    assert encoded == ["a", "b", "c", "d", "e"]
