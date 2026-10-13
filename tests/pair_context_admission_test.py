"""Scientific batching is explicit, fixed, isolated and checked before replay."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from exact.experiments import numerical_cache
from exact.experiments import runtime as encoder_runtime
from exact.impl.models.pair_adaptive_batch import (
    BATCHING_RECIPE,
    SCIENTIFIC_BATCHING_ENV,
    scientific_execution_payload,
    validate_batching_admission,
)
from exact.impl.models.pair_adaptive_channels import PairAdaptiveChannelsMixin
from tests.encoder_cache_migration_test import Config, Tokenizer, scorer
from tests.numerical_cache_test import Probe, runtime
from tests.pair_adaptive_experiments_test import _evidence_forward, _evidence_scorer


@pytest.fixture
def admitted(monkeypatch):
    for name, value in SCIENTIFIC_BATCHING_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_RECIPE", BATCHING_RECIPE)
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "development")
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", f"scientific-{BATCHING_RECIPE}-unit")


def test_fixed_scientific_recipe_runs_evidence_and_binds_checkpoint(admitted, monkeypatch):
    monkeypatch.delenv("EXACT_EXPERIMENT_RUNTIME", raising=False)
    current, _ = _evidence_scorer(monkeypatch, attr={"enabled": True, "bank": "full"})
    validate_batching_admission()
    payload = current.runtime_fingerprint_payload()
    assert payload["pair_context_execution"] == scientific_execution_payload()
    assert payload["pair_context_execution"]["settings"] == SCIENTIFIC_BATCHING_ENV
    output = _evidence_forward(current)
    assert output["explanations"]
    assert torch.isfinite(output["S_final"]).all()

    monkeypatch.delenv("EXACT_PAIR_CONTEXT_RECIPE")
    unmarked = current.runtime_fingerprint_payload()
    assert "pair_context_execution" not in unmarked
    assert {k: v for k, v in payload.items() if k != "pair_context_execution"} == unmarked


@pytest.mark.parametrize("name", list(SCIENTIFIC_BATCHING_ENV))
def test_every_recipe_setting_is_required_and_cannot_silently_fall_back(
    admitted, monkeypatch, name
):
    monkeypatch.delenv(name)
    with pytest.raises(ValueError, match="requires its fixed settings"):
        validate_batching_admission()


@pytest.mark.parametrize(
    "name,value",
    [
        ("EXACT_PAIR_CONTEXT_BLOCK_PAIRS", "64"),
        ("EXACT_PAIR_CONTEXT_TEXT_BATCH", "128"),
        ("EXACT_PAIR_CONTEXT_MATRIX_KERNEL", "bmm"),
        ("EXACT_PAIR_CONTEXT_ARITHMETIC", "cpu"),
        ("EXACT_PAIR_CONTEXT_ENCODER", "legacy"),
        ("EXACT_PAIR_CONTEXT_SCHEDULE", "legacy"),
        ("EXACT_PAIR_CONTEXT_BATCHING", "0"),
        ("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "1"),
        ("EXACT_ENCODER_CUDA_GRAPHS", "1"),
    ],
)
def test_scientific_admission_does_not_enable_unreviewed_variants(
    admitted, monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    with pytest.raises((ValueError, RuntimeError), match="fixed settings|only admitted"):
        validate_batching_admission()


@pytest.mark.parametrize("role", ["reporting", "test", "throughput_fixture", ""])
def test_admission_remains_development_only(admitted, monkeypatch, role):
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", role)
    with pytest.raises(RuntimeError, match="only admitted for development"):
        validate_batching_admission()


@pytest.mark.parametrize(
    "namespace", ["scientific", "qualification-unit", f"scientific-{BATCHING_RECIPE}-", ""]
)
def test_scientific_recipe_requires_its_own_named_namespace(admitted, monkeypatch, namespace):
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", namespace)
    with pytest.raises(RuntimeError, match="isolated"):
        validate_batching_admission()


def test_unknown_recipe_fails_before_scoring(admitted, monkeypatch):
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_RECIPE", "unreviewed")
    with pytest.raises(ValueError, match="Unknown EXACT_PAIR_CONTEXT_RECIPE"):
        validate_batching_admission()


@pytest.mark.parametrize("operation", ["channel", "probe"])
def test_mutated_recipe_cannot_replay_a_channel_hit(admitted, tmp_path, monkeypatch, operation):
    payload, reads = {"score": 0.75}, []
    cache = SimpleNamespace(
        pid=os.getpid(),
        path=tmp_path / "numerical/entries.sqlite",
        get=lambda *_args: (reads.append(True), payload)[1],
        batch=nullcontext,
    )
    model = SimpleNamespace(_numerical_cache=cache, device="cpu")
    method = PairAdaptiveChannelsMixin._score_label_channel.__get__(model)
    monkeypatch.setattr(numerical_cache, "_scope", lambda *_a, **_k: (tmp_path, "fixed"))
    invoke = (
        method
        if operation == "channel"
        else lambda *args: numerical_cache.peek_channel(method, *args)
    )
    with torch.inference_mode():
        assert invoke(["source"], ["target"]) is payload
        monkeypatch.setenv("EXACT_PAIR_CONTEXT_TEXT_BATCH", "128")
        with pytest.raises(ValueError, match="fixed settings"):
            invoke(["source"], ["target"])
    assert reads == [True]


def test_recipe_marker_invalidates_existing_batch_scopes(admitted, tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    model = Probe()
    model._numerical_channel_scopes = {}
    monkeypatch.delenv("EXACT_PAIR_CONTEXT_RECIPE")
    with torch.inference_mode():
        before = model.channel(torch.tensor([0.2]))
        monkeypatch.setenv("EXACT_PAIR_CONTEXT_RECIPE", BATCHING_RECIPE)
        after = model.channel(torch.tensor([0.2]))
        replay = model.channel(torch.tensor([0.2]))
    assert model.channel_calls == 2
    assert torch.equal(before["raw"], after["raw"])
    assert torch.equal(after["raw"], replay["raw"])


def test_encoder_identity_binds_admission_and_rejects_changed_recipe(admitted, monkeypatch):
    current, tokenizer = scorer(), Tokenizer()
    model = torch.nn.Linear(2, 2, bias=False).eval()
    model.config = Config()
    monkeypatch.delenv("EXACT_PAIR_CONTEXT_RECIPE")
    before = encoder_runtime.encoder_identity(current, tokenizer, model, 16)
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_RECIPE", BATCHING_RECIPE)
    after = encoder_runtime.encoder_identity(current, tokenizer, model, 16)
    assert before is not None and after is not None and before != after
    assert encoder_runtime.encoder_identity(current, tokenizer, model, 16) == after
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_TEXT_BATCH", "128")
    with pytest.raises(ValueError, match="fixed settings"):
        encoder_runtime.encoder_identity(current, tokenizer, model, 16)
