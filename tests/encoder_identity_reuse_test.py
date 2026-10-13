"""Resident and persistent lookups share one invocation's validated identity."""

from collections import OrderedDict
from functools import wraps

import pytest
import torch
from tokenizers import normalizers

from exact.experiments import runtime
from tests.encoder_cache_migration_test import scorer, setup_cache
from tests.encoder_tokenizer_identity_test import TinyEncoder, tokenizer


@pytest.fixture
def encoder(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_DIR", str(tmp_path / "vectors"))
    monkeypatch.setenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "0")
    current, tok, model = scorer(), tokenizer(), TinyEncoder()
    # Match the known encoder call's invocation settings; changing semantic
    # tokenizer settings below still changes the default (unnormalized) identity.
    tok(["hello"], padding=True, truncation=True, max_length=8)
    resident, identities = OrderedDict(), []
    original = runtime.encoder_identity

    def counted(*args, **kwargs):
        identity = original(*args, **kwargs)
        identities.append(identity)
        return identity

    monkeypatch.setattr(runtime, "encoder_identity", counted)

    def forward(texts):
        current._numerical_channel_scopes = {}
        return current._encode_with_cache(texts, tok, model, 8, resident, None)

    return current, tok, model, resident, identities, forward


def test_cold_durable_partial_and_resident_hits_validate_once_each(encoder):
    _, _, model, resident, identities, forward = encoder
    cold = forward(["hello", "hello world", "hello"])
    assert len(identities) == 1
    assert [len(batch) for batch in model.batches] == [2]

    resident.clear()
    durable = forward(["hello world", "hello"])
    assert len(identities) == 2
    torch.testing.assert_close(durable, cold[[1, 0]], rtol=0, atol=0)
    assert [len(batch) for batch in model.batches] == [2]

    partial = forward(["HELLO", "hello"])
    assert len(identities) == 3
    assert [len(batch) for batch in model.batches] == [2, 1]
    warm = forward(["hello", "HELLO"])
    assert len(identities) == 4
    assert [len(batch) for batch in model.batches] == [2, 1]
    torch.testing.assert_close(warm, partial[[1, 0]], rtol=0, atol=0)
    assert len(set(identities)) == 1


@pytest.mark.parametrize("mutation", ["weights", "tokenizer"])
def test_new_invocation_rejects_stale_resident_and_durable_rows(encoder, mutation):
    _, tok, model, _, identities, forward = encoder
    cold = forward(["HELLO"])
    if mutation == "weights":
        with torch.no_grad():
            model.weight.add_(1)
    else:
        tok.backend_tokenizer.normalizer = normalizers.Lowercase()
    changed = forward(["HELLO"])
    assert len(identities) == 2 and identities[0] != identities[1]
    assert [len(batch) for batch in model.batches] == [1, 1]
    assert not torch.equal(cold, changed)


def test_public_entrypoint_still_validates_its_own_identity(encoder):
    current, tok, model, _, identities, _ = encoder

    def public():
        current._numerical_channel_scopes = {}
        return runtime.cached_encoder_rows(
            current, tok, model, ["hello"], 8,
            lambda rows: current._encode_texts(tok, model, rows, 8),
        )

    cold = public()
    warm = public()
    assert len(identities) == 2
    torch.testing.assert_close(cold, warm, rtol=0, atol=0)
    assert len(model.batches) == 1
    with torch.no_grad():
        model.weight.add_(1)
    changed = public()
    assert len(identities) == 3 and identities[-1] != identities[0]
    assert len(model.batches) == 2
    assert not torch.equal(cold, changed)


def test_unpinned_model_computes_without_persistent_cache(encoder, tmp_path):
    _, _, model, _, identities, forward = encoder
    model.config._commit_hash = None
    first = forward(["hello"])
    second = forward(["hello"])
    assert identities == [None, None]
    assert len(model.batches) == 2
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not (tmp_path / "vectors/vectors.sqlite3").exists()


def test_training_bypasses_resident_and_persistent_rows(encoder, monkeypatch):
    _, _, model, resident, identities, forward = encoder
    forward(["hello", "hello world"])
    assert resident and len(model.batches) == 1

    def forbidden(*_args, **_kwargs):
        pytest.fail("Training must not use persistent encoder rows")

    monkeypatch.setattr(runtime, "_cached_encoder_rows_from_identity", forbidden)
    model.train()
    first = forward(["hello", "hello world", "hello"])
    second = forward(["hello", "hello world", "hello"])
    assert identities[1:] == [None, None]
    assert [len(batch) for batch in model.batches] == [2, 3, 3]
    assert model.batches[1] == model.batches[2]
    assert not resident
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_resident_entrypoint_retains_verified_legacy_row_migration(tmp_path, monkeypatch):
    model, tok, expected, _ = setup_cache(tmp_path, monkeypatch)
    current = scorer()

    @wraps(current._encode_texts)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Verified historical vectors must not be re-encoded")

    current._encode_texts = forbidden
    actual = current._encode_with_cache(["b", "a", "b"], tok, model, 16, OrderedDict(), None)
    torch.testing.assert_close(actual, expected[[1, 0, 1]], rtol=0, atol=0)
