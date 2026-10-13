"""Batch execution must preserve first-touch shapes and all discrete evidence."""

from types import SimpleNamespace

import pytest
import torch

from exact.impl.models.pair_adaptive_batch import (
    _batch_matrices,
    call_prepared,
    execution_options,
    prepare_structural_block,
    score_evidence_job,
)
from exact.impl.models.pair_adaptive_channels import _host_lists
from tests.pair_adaptive_experiments_test import _evidence_forward, _evidence_scorer


def _assert_equal_output(expected, actual):
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, actual[key], rtol=0, atol=0, equal_nan=True)
    assert expected["explanations"] == actual["explanations"]
    assert expected["llm_evidence_packets"] == actual["llm_evidence_packets"]


@pytest.mark.parametrize("block", [1, 2, 32])
def test_strict_batches_preserve_shape_sensitive_first_touch_and_replay(monkeypatch, block):
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "throughput_fixture")
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", "qualification-sensitive-encoder")
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH", "0")
    monkeypatch.delenv("EXACT_EXPERIMENT_RUNTIME", raising=False)

    def make_scorer():
        scorer, _ = _evidence_scorer(monkeypatch, attr={"enabled": True, "bank": "full"})
        rows = {"labels": {}, "contexts": {}}
        calls = []

        def encode(texts, kind):
            # Model finite-precision behavior depends on batch/padding and cache
            # first-touch order. A text-only fixed-vector mock misses this defect.
            missing = list(dict.fromkeys(text for text in texts if text not in rows[kind]))
            if missing:
                calls.append((kind, tuple(missing)))
                width = max(map(len, missing))
                for text in missing:
                    rows[kind][text] = torch.tensor(
                        [
                            1.0 + len(missing) * 0.003,
                            0.5 + width * 0.001,
                            0.25 + (sum(map(ord, text)) % 37) * 0.01,
                        ]
                    )
            return torch.stack([rows[kind][text] for text in texts])

        scorer.encode_labels_batch = lambda texts: encode(texts, "labels")
        scorer.encode_contexts_batch = lambda texts: encode(texts, "contexts")
        return scorer, calls

    monkeypatch.setenv("EXACT_PAIR_CONTEXT_BATCHING", "0")
    plain, plain_calls = make_scorer()
    cold = _evidence_forward(plain)
    warm = _evidence_forward(plain)
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_BATCHING", "1")
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_BLOCK_PAIRS", str(block))
    strict, strict_calls = make_scorer()
    _assert_equal_output(cold, _evidence_forward(strict))
    _assert_equal_output(warm, _evidence_forward(strict))
    assert strict_calls == plain_calls


@pytest.mark.parametrize("missing", [None, "attributes", "similarity", "hierarchy"])
def test_staged_blocks_only_compute_channel_cache_misses(monkeypatch, missing):
    import exact.experiments.numerical_cache as numerical

    monkeypatch.setenv("EXACT_PAIR_CONTEXT_BATCHING", "0")
    scorer, _ = _evidence_scorer(monkeypatch, attr={"enabled": True, "bank": "full"})
    dataset = scorer._attached_dataset
    src, tgt = dataset.get_entity_features("source", "src"), dataset.get_entity_features(
        "target", "tgt"
    )
    hierarchy = scorer._score_hierarchy_family(
        "is_a",
        src["hierarchy"]["is_a"],
        tgt["hierarchy"]["is_a"],
        src_iri="source",
        tgt_iri="target",
    )
    similarity = scorer._score_similarity_channel(src["object_triples"], tgt["object_triples"])
    attributes = scorer._score_attribute_channel(
        src["attributes"],
        tgt["attributes"],
        src["labels"],
        tgt["labels"],
        {"is_a": hierarchy},
        similarity,
    )
    payloads = {"hierarchy": hierarchy, "similarity": similarity, "attributes": attributes}
    methods = {
        "_score_hierarchy_family": "hierarchy",
        "_score_similarity_channel": "similarity",
        "_score_attribute_channel": "attributes",
    }
    seen = []

    def peek(method, *args, **kwargs):
        name = methods[method.__name__]
        seen.append(name)
        return None if name == missing else payloads[name]

    monkeypatch.setattr(numerical, "peek_channel", peek)
    for name, job_name in (
        ("hierarchy", "_hierarchy_family_job"),
        ("similarity", "_similarity_channel_job"),
        ("attributes", "_attribute_channel_job"),
    ):
        if name != missing:
            monkeypatch.setattr(
                scorer, job_name, lambda *_a, **_k: pytest.fail("cached job executed")
            )
    row = prepare_structural_block(scorer, [("source", "target", src, tgt)], ["is_a"])[0]
    assert seen == ["hierarchy", "similarity", "attributes"]
    for name, payload in payloads.items():
        actual = row["hierarchy"]["is_a"] if name == "hierarchy" else row[name]
        if name != missing:
            assert actual is payload
            assert (name, "is_a" if name == "hierarchy" else None) in row["cache_hits"]
        else:
            assert actual["src_selected"] == payload["src_selected"]
            assert actual["tgt_selected"] == payload["tgt_selected"]
            assert actual["score"] == payload["score"]


def test_prepared_cache_hit_does_not_repeat_cache_lookup():
    payload = {"score": 0.8675}

    def forbidden():
        pytest.fail("validated channel cache hit was looked up twice")

    assert (
        call_prepared(SimpleNamespace(), "attributes", forbidden, payload, cache_hit=True)
        is payload
    )


@pytest.mark.parametrize("kernel", ["mm", "bmm"])
def test_matrix_batches_keep_device_and_empty_shapes(monkeypatch, kernel):
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_MATRIX_KERNEL", kernel)
    scorer = SimpleNamespace(
        device="meta",
        use_context=True,
        encode_labels_batch=lambda texts: torch.ones((len(texts), 4), device="meta"),
        _sim01=lambda value: (value + 1.0) / 2.0,
    )
    matrices = _batch_matrices(
        scorer,
        [
            ("label_matrix", ["a", "b"], ["c"]),
            ("label_matrix", ["b"], ["a", "c"]),
            ("label_matrix", [], ["c"]),
        ],
        "label_matrix",
    )
    assert [tuple(matrix.shape) for matrix in matrices] == [(2, 1), (1, 2), (0, 1)]
    assert all(matrix.device.type == "meta" for matrix in matrices)


@pytest.mark.parametrize("encoder", ["legacy", "batch"])
@pytest.mark.parametrize("kernel", ["mm", "bmm"])
def test_matrix_ablation_keeps_original_normalization_shapes(monkeypatch, encoder, kernel):
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_ENCODER", encoder)
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_MATRIX_KERNEL", kernel)
    calls = []
    values = {"a": [1.0, 0.25], "b": [0.2, 1.0], "c": [0.5, 0.5]}

    def encode(texts):
        calls.append(tuple(texts))
        return torch.tensor([values[text] for text in texts])

    scorer = SimpleNamespace(
        device="cpu", use_context=True, encode_labels_batch=encode, _sim01=lambda x: x
    )
    requests = [("label_matrix", ["a", "b"], ["c"]), ("label_matrix", ["b"], ["a", "c"])]
    matrices = _batch_matrices(scorer, requests, "label_matrix")
    for matrix, (_, left, right) in zip(matrices, requests):
        lvec = torch.nn.functional.normalize(torch.tensor([values[t] for t in left]), dim=-1)
        rvec = torch.nn.functional.normalize(torch.tensor([values[t] for t in right]), dim=-1)
        torch.testing.assert_close(matrix, lvec @ rvec.T, rtol=0, atol=0)
    assert calls == (
        [("a", "b"), ("c",), ("b",), ("a", "c")] if encoder == "legacy" else [("a", "b", "c")]
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_packed_host_reductions_preserve_values_and_transfer_once(monkeypatch, dtype):
    values = [
        torch.tensor([0.71142578125, 0.7109375], dtype=dtype),
        torch.empty(0, dtype=dtype),
        torch.tensor([0.8675, 0.0, 1.0], dtype=dtype),
    ]
    expected = [value.tolist() for value in values]
    transfers = []
    cpu = torch.Tensor.cpu

    def counted(self, *args, **kwargs):
        transfers.append(self.numel())
        return cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted)
    assert _host_lists(*values) == expected
    assert transfers == [5]


def test_ablation_switches_reject_misspellings(monkeypatch):
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_ENCODER", "batched")
    with pytest.raises(ValueError, match="EXACT_PAIR_CONTEXT_ENCODER"):
        execution_options()


@pytest.mark.parametrize(
    "flag", ["EXACT_PAIR_CONTEXT_BATCHING", "EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY"]
)
def test_direct_evidence_helpers_enforce_qualification_admission(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_ENCODER", "batch")
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "reporting")
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", "scientific")
    scorer = SimpleNamespace(_prepared_evidence_result=("hierarchy", {"score": 1.0}))

    def job():
        pytest.fail("unadmitted generator executed")
        yield  # pragma: no cover

    pending = job()
    try:
        with pytest.raises(RuntimeError, match="only admitted"):
            score_evidence_job(scorer, "hierarchy", pending)
        with pytest.raises(RuntimeError, match="only admitted"):
            prepare_structural_block(scorer, [], [])
    finally:
        pending.close()
