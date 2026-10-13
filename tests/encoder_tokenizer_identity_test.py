"""Real tokenizer invocation state must not masquerade as changed encoder semantics."""

import json
from collections import OrderedDict
from functools import wraps
from types import SimpleNamespace

import pytest
import torch
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)
from transformers import PreTrainedTokenizerFast

from exact.experiments import runtime
from tests.encoder_cache_migration_test import Config, scorer

VOCAB = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "HELLO": 4, "new": 5, "[SEP]": 6}


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.weight = torch.nn.Parameter(
            torch.arange(len(VOCAB) * 2, dtype=torch.float32).reshape(-1, 2), requires_grad=False
        )
        self.batches = []
        self.eval()

    def forward(self, input_ids, attention_mask, **kwargs):
        self.batches.append(input_ids.detach().cpu().tolist())
        return SimpleNamespace(last_hidden_state=self.weight[input_ids])


def tokenizer():
    backend = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")


@pytest.fixture
def qualification(monkeypatch):
    monkeypatch.setenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "1")
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "throughput_fixture")
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", "qualification-tokenizer-identity")


@pytest.mark.parametrize("observed", [False, True])
def test_real_tokenizer_reuses_across_different_batches_and_semantic_change_misses(
    tmp_path, monkeypatch, qualification, observed
):
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_DIR", str(tmp_path / "vectors"))
    current, tok, model, resident = scorer(), tokenizer(), TinyEncoder(), OrderedDict()
    if observed:
        original = current._encode_texts

        @wraps(original)
        def observation(*args, **kwargs):
            return original(*args, **kwargs)

        current._encode_texts = observation

    def forward(texts):
        current._numerical_channel_scopes = {}  # The actual outer-forward boundary.
        return current._encode_with_cache(texts, tok, model, 8, resident, None)

    assert json.loads(tok.backend_tokenizer.to_str())["padding"] is None
    cold = forward(["HELLO", "hello world"])
    assert json.loads(tok.backend_tokenizer.to_str())["padding"] is not None
    partial = forward(["HELLO", "new", "hello world"])
    torch.testing.assert_close(partial[[0, 2]], cold, rtol=0, atol=0)
    assert [len(batch) for batch in model.batches] == [2, 1]
    resident.clear()  # Prove the durable rows, not just resident vectors, are reused.
    warm = forward(["hello world", "new", "HELLO"])
    torch.testing.assert_close(warm, partial[[2, 1, 0]], rtol=0, atol=0)
    assert [len(batch) for batch in model.batches] == [2, 1]

    tok.backend_tokenizer.normalizer = normalizers.Lowercase()
    changed = forward(["HELLO"])
    assert [len(batch) for batch in model.batches] == [2, 1, 1]
    torch.testing.assert_close(changed[0], model.weight[VOCAB["hello"]], rtol=0, atol=0)
    assert not torch.equal(changed[0], cold[0])


def test_fresh_cold_outputs_match_legacy_control(tmp_path, monkeypatch, qualification):
    outputs = []
    for mode in ("0", "1"):
        monkeypatch.setenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", mode)
        monkeypatch.setenv("EXACT_EMBEDDING_CACHE_DIR", str(tmp_path / mode))
        current, tok, model = scorer(), tokenizer(), TinyEncoder()
        current._numerical_channel_scopes = {}
        outputs.append(
            current._encode_with_cache(
                ["HELLO", "hello world", "new"], tok, model, 8, OrderedDict(), None
            )
        )
        assert len(model.batches) == 1
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("custom", [False, True])
def test_legacy_and_custom_encoders_retain_backend_invocation_state(
    monkeypatch, qualification, custom
):
    current, tok, model = scorer(), tokenizer(), TinyEncoder()
    if custom:

        def encode(tokenizer, model, texts, max_len):
            return tokenizer(texts)  # Relies on backend defaults, not the known contract.

        current._encode_texts = encode
    else:
        monkeypatch.setenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "0")
    initial = runtime.encoder_identity(current, tok, model, 8)
    tok(["hello", "hello world"], padding=True, truncation=True, max_length=8)
    assert runtime.encoder_identity(current, tok, model, 8) != initial


@pytest.mark.parametrize(
    "change",
    [
        "vocabulary",
        "added_vocabulary",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "special_token",
        "padding_side",
        "truncation_side",
        "max_len",
    ],
)
def test_semantic_tokenizer_components_remain_bound(qualification, change):
    current, tok, model = scorer(), tokenizer(), TinyEncoder()
    before = runtime.encoder_identity(current, tok, model, 8)
    length = 8
    if change == "vocabulary":
        tok.backend_tokenizer.model = models.WordLevel(
            {**VOCAB, "hello": VOCAB["world"], "world": VOCAB["hello"]}, unk_token="[UNK]"
        )
    elif change == "added_vocabulary":
        tok.add_tokens(["additional"])
    elif change == "normalizer":
        tok.backend_tokenizer.normalizer = normalizers.Lowercase()
    elif change == "pre_tokenizer":
        tok.backend_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    elif change == "post_processor":
        tok.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="$A [SEP]", special_tokens=[("[SEP]", VOCAB["[SEP]"])]
        )
    elif change == "decoder":
        tok.backend_tokenizer.decoder = decoders.WordPiece(prefix="##")
    elif change == "special_token":
        tok.pad_token = "[UNK]"
    elif change == "padding_side":
        tok.padding_side = "left"
    elif change == "truncation_side":
        tok.truncation_side = "left"
    else:
        length = 16
    assert runtime.encoder_identity(current, tok, model, length) != before


def test_stable_identity_requires_qualification_namespace(monkeypatch, qualification):
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "scientific")
    with pytest.raises(RuntimeError, match="only admitted"):
        runtime.encoder_identity(scorer(), tokenizer(), TinyEncoder(), 8)


def test_mode_switch_cannot_replay_cached_scores(tmp_path, monkeypatch, qualification):
    from tests.numerical_cache_test import Probe
    from tests.numerical_cache_test import runtime as configure_runtime

    configure_runtime(tmp_path, monkeypatch)
    current = Probe()
    monkeypatch.setenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "0")
    current.forward(torch.tensor([0.2]))
    monkeypatch.setenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "1")
    current.forward(torch.tensor([0.2]))
    assert current.forward_calls == current.channel_calls == 2
