"""Admission, isolation and bounded replay contracts without allocating a GPU."""

from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding, BertConfig, BertModel
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.bert import modeling_bert

from exact.experiments import runtime
from exact.impl.models import encoder_graph as graphs
from tests.encoder_cache_migration_test import scorer
from tests.encoder_tokenizer_identity_test import TinyEncoder, tokenizer


@pytest.fixture
def qualification(monkeypatch):
    monkeypatch.setenv("EXACT_ENCODER_CUDA_GRAPHS", "1")
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "throughput_fixture")
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", "qualification-graph-unit")
    monkeypatch.setenv("EXACT_ENCODER_GRAPH_MIN_CALLS", "2")


def tokens(mask=None):
    return BatchEncoding(
        {
            "input_ids": torch.tensor([[2, 3, 1], [4, 1, 0]]),
            "attention_mask": torch.tensor(mask or [[1, 1, 1], [1, 1, 0]]),
            "token_type_ids": torch.zeros(2, 3, dtype=torch.long),
        }
    )


@pytest.mark.parametrize("entry", ["identity", "direct_encode", "direct_graph", "admission"])
def test_graph_flag_rejected_outside_qualification(monkeypatch, qualification, entry):
    from exact.impl.models.pair_adaptive_batch import validate_batching_admission

    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "scientific")
    current, model, tok = scorer(), TinyEncoder(), tokenizer()
    calls = {
        "identity": lambda: runtime.encoder_identity(current, tok, model, 8),
        "direct_encode": lambda: current._encode_texts(tok, model, ["hello"], 8),
        "direct_graph": lambda: graphs.encode_with_graph(
            current, model, tokens(), lambda _: pytest.fail("encoded")
        ),
        "admission": validate_batching_admission,
    }
    with pytest.raises(RuntimeError, match="only admitted"):
        calls[entry]()
    assert not model.batches


@pytest.mark.parametrize("entry", ["cached", "probe"])
def test_cached_channel_cannot_hide_unadmitted_graph_mode(
    tmp_path, monkeypatch, qualification, entry
):
    from tests.numerical_cache_test import Probe, runtime as setup
    from exact.experiments.numerical_cache import peek_channel

    setup(tmp_path, monkeypatch)
    monkeypatch.setattr(Probe.channel, "__module__", "exact.impl.models.pair_adaptive_channels")
    monkeypatch.setattr(
        Probe.channel.__wrapped__, "__module__", "exact.impl.models.pair_adaptive_channels"
    )
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "throughput_fixture")
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", "qualification-graph-unit")
    current = Probe()
    current.forward(torch.tensor([0.2]))
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "scientific")
    with torch.inference_mode(), pytest.raises(RuntimeError, match="only admitted"):
        if entry == "cached":
            current.channel(torch.tensor([0.2]))
        else:
            peek_channel(current.channel, torch.tensor([0.2]))


def test_execution_keys_separate_graph_and_limits(tmp_path, monkeypatch, qualification):
    from tests.numerical_cache_test import Probe, runtime as setup

    setup(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "throughput_fixture")
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", "qualification-graph-unit")
    current = Probe()
    for mode, limit in [("0", "16"), ("1", "16"), ("1", "8")]:
        monkeypatch.setenv("EXACT_ENCODER_CUDA_GRAPHS", mode)
        monkeypatch.setenv("EXACT_ENCODER_GRAPH_MAX_SHAPES", limit)
        current.forward(torch.tensor([0.2]))
    assert current.forward_calls == current.channel_calls == 3


def test_observer_counts_logical_encoder_once_on_cpu_fallback(qualification):
    current, model, tok = scorer(), TinyEncoder(), tokenizer()
    seen = []
    current._encoder_observer = lambda model, rows: seen.append(
        (model, rows["input_ids"].device.type)
    )
    actual = current._encode_texts(tok, model, ["hello", "hello world"], 8)
    assert actual.shape == (2, 2) and seen == [(model, "cpu")]
    assert len(model.batches) == 1
    assert graphs.encoder_graph_stats(current)["fallback_reasons"] == {"non_cuda": 1}


@pytest.mark.parametrize("all_ones", [True, False])
def test_model_local_mask_specialization_matches_original_cpu_without_mutation(all_ones):
    config = BertConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=10,
        max_position_embeddings=16,
    )
    config._attn_implementation = "sdpa"
    model = BertModel(config).eval()
    original = modeling_bert.BertModel.forward
    helper = modeling_bert._prepare_4d_attention_mask_for_sdpa
    inputs = tokens([[1, 1, 1], [1, 1, 1]] if all_ones else None)
    specialized = graphs._specialized_forward(
        original,
        AttentionMaskConverter._expand_mask,
        shape=(2, 3),
        dtype=torch.float32,
        all_ones=all_ones,
    )
    with torch.inference_mode():
        expected = model(**inputs).last_hidden_state
        actual = specialized(model, **inputs).last_hidden_state
    assert graphs._same_tensor(expected, actual)
    assert modeling_bert._prepare_4d_attention_mask_for_sdpa is helper
    assert model.forward.__func__ is original
    changed = tokens()
    changed["input_ids"] = changed["input_ids"][:, :2]
    changed["attention_mask"] = changed["attention_mask"][:, :2]
    changed["token_type_ids"] = changed["token_type_ids"][:, :2]
    with torch.inference_mode(), pytest.raises(RuntimeError, match="shape/dtype"):
        specialized(model, **changed)


def test_input_keys_preserve_exact_shapes_mask_branch_and_changed_content():
    before = tokens()
    changed = tokens()
    changed["input_ids"][0, 0] = 5
    changed["attention_mask"][0, -1] = 0
    assert graphs._input_key(before) == graphs._input_key(changed)
    assert graphs._input_key(tokens([[1, 1, 1], [1, 1, 1]])) != graphs._input_key(before)
    invalid = tokens()
    invalid["attention_mask"][0, 0] = 2
    assert graphs._input_key(invalid) is None
    invalid = tokens()
    invalid["input_ids"] = invalid["input_ids"].float()
    assert graphs._input_key(invalid) is None
    invalid = tokens()
    invalid["input_ids"] = invalid["input_ids"].T.contiguous().T
    assert graphs._input_key(invalid) is None


def test_model_signature_rejects_hooks_training_mutations_and_unversioned_tensors(monkeypatch):
    current = scorer()
    config = BertConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=10,
    )
    config._attn_implementation = "sdpa"
    config._commit_hash = "a" * 40
    model = BertModel(config).eval()
    with torch.inference_mode():
        original = graphs._model_signature(current, model)
        assert original is not None
        handle = model.register_forward_pre_hook(lambda *_: None)
        assert graphs._model_signature(current, model) is None
        handle.remove()
        model.embeddings.LayerNorm.eps *= 2
        scalar_changed = graphs._model_signature(current, model)
        assert scalar_changed != original
        model.train()
        assert graphs._model_signature(current, model) is None
        model.eval()
        model.embeddings.word_embeddings.weight.add_(1)
        weight_changed = graphs._model_signature(current, model)
        assert weight_changed != scalar_changed
        model.config.layer_norm_eps *= 2
        assert graphs._model_signature(current, model) != weight_changed
        model.embeddings.word_embeddings.max_norm = 1.0
        assert graphs._model_signature(current, model) is None
        model.embeddings.word_embeddings.max_norm = None
        model.embeddings.word_embeddings.weight = torch.nn.Parameter(
            torch.ones_like(model.embeddings.word_embeddings.weight)
        )
        assert graphs._model_signature(current, model) is None


class FakeEntry:
    def __init__(self, size):
        self.size, self.closed = size, False

    def replay(self, inputs):
        assert not self.closed
        return inputs["input_ids"].clone()

    def close(self):
        self.closed = True


@pytest.fixture
def simulated(qualification, monkeypatch):
    current, model = scorer(), object()
    current.device = "cuda"
    state = {"signature": "original", "captures": [], "eager": 0, "oom": False, "size": 80}
    monkeypatch.setattr(graphs, "_model_signature", lambda *_: state["signature"])
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=5))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (16 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(BatchEncoding, "to", lambda self, *_: self)

    def capture(_scorer, _model, inputs, _all_ones, _eager, stats):
        if state["oom"]:
            raise torch.cuda.OutOfMemoryError("simulated capture pressure")
        entry = FakeEntry(state["size"])
        state["captures"].append(entry)
        return entry, inputs["input_ids"].clone()

    def eager(inputs):
        state["eager"] += 1
        return inputs["input_ids"].clone()

    monkeypatch.setattr(graphs, "_capture", capture)
    return (
        current,
        model,
        state,
        lambda values=None: graphs.encode_with_graph(current, model, values or tokens(), eager),
    )


def test_repeated_exact_shape_captures_then_replays_changed_inputs_without_alias(simulated):
    current, _, state, run = simulated
    first = run()
    second = run()
    changed = tokens()
    changed["input_ids"][0, 0] = 9
    third = run(changed)
    assert state["eager"] == 1 and len(state["captures"]) == 1
    assert torch.equal(first, second) and third[0, 0] == 9 and first[0, 0] == 2
    stats = graphs.encoder_graph_stats(current)
    assert stats["captures"] == stats["replays"] == stats["cached_shapes"] == 1
    assert stats["retained_bytes"] == 80


def test_shape_and_byte_limits_do_not_repeat_rejected_capture(simulated, monkeypatch):
    current, _, state, run = simulated
    monkeypatch.setenv("EXACT_ENCODER_GRAPH_MAX_BYTES", "50")
    run()
    run()
    run()
    assert len(state["captures"]) == 1 and state["captures"][0].closed
    assert graphs.encoder_graph_stats(current)["retained_bytes"] == 0
    assert state["eager"] == 2
    graphs.clear_encoder_graphs(current, reset_stats=True)
    monkeypatch.setenv("EXACT_ENCODER_GRAPH_MAX_BYTES", "1000")
    monkeypatch.setenv("EXACT_ENCODER_GRAPH_MAX_SHAPES", "1")
    run()
    run()
    run(tokens([[1, 1, 1], [1, 1, 1]]))
    run(tokens([[1, 1, 1], [1, 1, 1]]))
    assert graphs.encoder_graph_stats(current)["cached_shapes"] == 1
    assert graphs.encoder_graph_stats(current)["fallback_reasons"]["shape_limit"] == 1


def test_model_changes_release_old_graph_and_oom_falls_back_once(simulated):
    current, _, state, run = simulated
    run()
    run()
    old = state["captures"][0]
    state["signature"] = "changed_weights"
    run()
    assert old.closed and graphs.encoder_graph_stats(current)["cached_shapes"] == 0
    state["oom"] = True
    run()
    run()
    stats = graphs.encoder_graph_stats(current)
    assert stats["capture_ooms"] == 1 and stats["retained_bytes"] == 0
    assert stats["fallback_reasons"]["rejected_capture"] == 1


def test_inherited_graph_state_fails_before_any_cuda_use(simulated, monkeypatch):
    current, _, _, run = simulated
    run()
    monkeypatch.setattr(graphs.os, "getpid", lambda: current._encoder_graph_cache.pid + 1)
    with pytest.raises(RuntimeError, match="after fork"):
        run()


def test_invalid_configuration_and_unexpected_errors_are_not_silently_hidden(
    simulated, monkeypatch
):
    _, _, _, run = simulated
    monkeypatch.setenv("EXACT_ENCODER_GRAPH_MAX_SHAPES", "0")
    with pytest.raises(ValueError, match="positive"):
        run()
    monkeypatch.setenv("EXACT_ENCODER_GRAPH_MAX_SHAPES", "2")
    run()

    def application_failure(*_args):
        raise RuntimeError("unexpected application failure")

    monkeypatch.setattr(graphs, "_capture", application_failure)
    with pytest.raises(RuntimeError, match="unexpected application"):
        run()


def test_explicit_capture_rejection_returns_verified_eager_and_disables_shape(
    simulated, monkeypatch
):
    current, _, _, run = simulated
    expected = torch.full((2, 3), 17)

    def reject(*_args):
        raise graphs._CaptureRejected("output_mismatch", expected)

    monkeypatch.setattr(graphs, "_capture", reject)
    run()
    assert torch.equal(run(), expected)
    run()
    stats = graphs.encoder_graph_stats(current)
    assert stats["capture_rejections"] == 1 and stats["cached_shapes"] == 0
    assert stats["fallback_reasons"]["output_mismatch"] == 1
    assert stats["fallback_reasons"]["rejected_capture"] == 1
