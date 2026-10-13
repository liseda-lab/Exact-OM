"""Tokenizer identity changes must not recopy unchanged pinned model weights."""

from copy import deepcopy
import hashlib
import json

import pytest
import torch

from exact.experiments import runtime
from tests.encoder_cache_migration_test import scorer
from tests.encoder_tokenizer_identity_test import TinyEncoder, tokenizer


@pytest.fixture
def counted_model(monkeypatch):
    monkeypatch.setenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "0")
    current, tok, model = scorer(), tokenizer(), TinyEncoder()
    model.register_buffer("scale", torch.tensor([1.0, 2.0]))
    counts = {"cpu_calls": 0, "cpu_bytes": 0, "numpy_bytes": 0}
    cpu, numpy = torch.Tensor.cpu, torch.Tensor.numpy

    def counted_cpu(value, *args, **kwargs):
        counts["cpu_calls"] += 1
        counts["cpu_bytes"] += value.numel() * value.element_size()
        return cpu(value, *args, **kwargs)

    def counted_numpy(value, *args, **kwargs):
        array = numpy(value, *args, **kwargs)
        counts["numpy_bytes"] += array.nbytes
        return array

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    monkeypatch.setattr(torch.Tensor, "numpy", counted_numpy)
    return current, tok, model, counts


def test_tokenizer_and_namespace_changes_preserve_legacy_identity_without_weight_copies(
    counted_model, monkeypatch
):
    current, tok, model, counts = counted_model
    payloads = []
    original_hash = runtime._hash

    def capture_payload(value):
        if isinstance(value, dict) and "weights_sha256" in value:
            payloads.append(deepcopy(value))
        return original_hash(value)

    monkeypatch.setattr(runtime, "_hash", capture_payload)
    initial = runtime.encoder_identity(current, tok, model, 8)
    size = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    assert counts == {"cpu_calls": 2, "cpu_bytes": size, "numpy_bytes": size}
    original = payloads[-1]
    # Independently reproduce the previous header/byte stream without exercising
    # the observed tensor materialization path again.
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(bytes(tensor.detach().contiguous().view(torch.uint8).flatten().tolist()))
    assert original["weights_sha256"] == digest.hexdigest()
    assert initial == (original_hash(original), None)

    tok(["hello", "hello world"], padding=True, truncation=True, max_length=8)
    changed = runtime.encoder_identity(current, tok, model, 8)
    expected = {**original, "tokenizer_backend": tok.backend_tokenizer.to_str()}
    assert changed != initial
    assert payloads[-1] == expected
    assert changed == (original_hash(expected), None)
    assert counts == {"cpu_calls": 2, "cpu_bytes": size, "numpy_bytes": size}

    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", "different-weight-reuse-scope")
    scoped = runtime.encoder_identity(current, tok, model, 8)
    expected["namespace"] = "different-weight-reuse-scope"
    assert scoped == (original_hash(expected), None)
    assert counts == {"cpu_calls": 2, "cpu_bytes": size, "numpy_bytes": size}
    assert set(model.state_dict()) == {"weight", "scale"}


@pytest.mark.parametrize("change", ["parameter", "buffer", "replacement", "model"])
def test_changed_or_replaced_weights_are_materialized_again(counted_model, change):
    current, tok, model, counts = counted_model
    initial = runtime.encoder_identity(current, tok, model, 8)
    first_bytes = counts["cpu_bytes"]
    with torch.no_grad():
        if change == "parameter":
            model.weight.add_(1)
        elif change == "buffer":
            model.scale.add_(1)
        elif change == "replacement":
            model.weight = torch.nn.Parameter(model.weight.clone(), requires_grad=False)
        else:
            # A copied model also copies ordinary Python memo attributes; tensor
            # object/storage identity must reject that inherited digest entry.
            model = deepcopy(model)
    changed = runtime.encoder_identity(current, tok, model, 8)
    assert counts == {
        "cpu_calls": 4,
        "cpu_bytes": first_bytes * 2,
        "numpy_bytes": first_bytes * 2,
    }
    assert (changed == initial) is (change in {"replacement", "model"})


def test_inherited_process_memos_rehash_without_changing_identity(counted_model, monkeypatch):
    current, tok, model, counts = counted_model
    initial = runtime.encoder_identity(current, tok, model, 8)
    size = counts["cpu_bytes"]
    next_pid = runtime.os.getpid() + 1
    monkeypatch.setattr(runtime.os, "getpid", lambda: next_pid)
    assert runtime.encoder_identity(current, tok, model, 8) == initial
    assert counts == {"cpu_calls": 4, "cpu_bytes": size * 2, "numpy_bytes": size * 2}


def test_unversioned_inference_tensor_cannot_reuse_weight_memo(counted_model):
    current, tok, model, counts = counted_model
    assert runtime.encoder_identity(current, tok, model, 8) is not None
    previous = dict(counts)
    with torch.inference_mode():
        model.weight = torch.nn.Parameter(torch.ones_like(model.weight), requires_grad=False)
    assert runtime.encoder_identity(current, tok, model, 8) is None
    assert counts == previous
