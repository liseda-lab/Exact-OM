"""Numerical caches preserve outputs and invalidate changed evidence/roles/precision."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from exact.experiments.numerical_cache import (
    NumericalCache,
    cached_numerical,
    shared_training_scores,
)
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset


def runtime(tmp_path, monkeypatch, **changes):
    identity = {
        "parameters": {
            "matching": {"extraction": {"mode": "greedy"}, "fusion": {"gamma": 2, "tau": 0.5}},
            "selector": {"accept_training": "winner_only"},
        },
        "inputs": {"source": "source-bytes", "target": "target-bytes", "candidates": "pool"},
        "implementation": {"code": "fixed-code"},
        "dependencies": {"torch": "pinned", "native": "pinned"},
        "role": "development",
        "entity_kind": "class",
        "seed": 17,
    }
    identity.update(changes)
    path = tmp_path / f"runtime-{len(list(tmp_path.glob('runtime*')))}.json"
    path.write_text(json.dumps({"root": str(tmp_path), "identity": identity}))
    monkeypatch.setenv("EXACT_EXPERIMENT_RUNTIME", str(path))
    return identity


class Probe:
    device = "cpu"
    fp16 = use_llm = False
    beta, threshold = 0.8, 0.5
    llm_experiment_config = {"gate": {"mode": "off"}}
    _attached_dataset = SimpleNamespace(cache_fingerprint="frozen-dataset")

    def __init__(self, gamma=2, tau=0.5):
        self.gamma, self.tau = gamma, tau
        self.forward_calls = self.channel_calls = 0

    def runtime_fingerprint_payload(self):
        return {
            "pair_adaptive_channels": {
                "experiments": {"fusion": {"gamma": self.gamma, "tau": self.tau}}
            }
        }

    @cached_numerical(channels=True)
    def channel(self, values):
        self.channel_calls += 1
        return {"raw": values + self.tau, "evidence": ["frozen fact"], "nested": (values.bool(),)}

    @torch.inference_mode()
    @cached_numerical()
    def forward(self, values):
        self.forward_calls += 1
        raw = self.channel(values)
        return {"score": raw["raw"] ** self.gamma, "trace": {"gamma": self.gamma, **raw}}


def test_extraction_and_acceptance_reuse_scores_while_gamma_replays_raw_channels(
    tmp_path, monkeypatch
):
    identity = runtime(tmp_path, monkeypatch)
    first = Probe()
    cold = first.forward(torch.tensor([0.2, 0.3]))
    changed = deepcopy(identity)
    changed["parameters"]["matching"]["extraction"]["mode"] = "assignment_accepted_utility"
    changed["parameters"]["selector"]["accept_training"] = "winner_plus_runnerup"
    runtime(tmp_path, monkeypatch, **changed)
    replay = Probe()
    warm = replay.forward(torch.tensor([0.2, 0.3]))
    assert replay.forward_calls == replay.channel_calls == 0
    assert torch.equal(cold["score"], warm["score"])
    assert warm["numerical_cache"]["full_forward_replayed"] is True
    assert warm["numerical_cache"]["new_forward_calls"] == 0
    assert cold["numerical_cache"]["new_forward_calls"] == 1
    assert warm["trace"]["evidence"] == cold["trace"]["evidence"]
    changed["parameters"]["matching"]["fusion"]["gamma"] = 3
    runtime(tmp_path, monkeypatch, **changed)
    variant = Probe(gamma=3)
    result = variant.forward(torch.tensor([0.2, 0.3]))
    assert variant.forward_calls == 1 and variant.channel_calls == 0
    assert torch.equal(result["score"], torch.tensor([0.7, 0.8]) ** 3)
    assert result["trace"]["gamma"] == 3
    assert result["numerical_cache"]["hits"] >= 1


@pytest.mark.parametrize(
    "mutation", ["tau", "input", "code", "native", "role", "seed", "precision", "arguments"]
)
def test_upstream_or_numerical_changes_invalidate_both_levels(tmp_path, monkeypatch, mutation):
    identity = runtime(tmp_path, monkeypatch)
    Probe().forward(torch.tensor([0.2]))
    model, values = Probe(), torch.tensor([0.2])
    if mutation == "tau":
        model.tau = 0.4
    elif mutation == "input":
        identity["inputs"]["source"] = "changed"
    elif mutation == "code":
        identity["implementation"]["code"] = "changed"
    elif mutation == "native":
        identity["dependencies"]["native"] = "changed"
    elif mutation in {"role", "seed"}:
        identity[mutation] = "reporting" if mutation == "role" else 29
    elif mutation == "precision":
        model.fp16 = True
    else:
        values = torch.tensor([0.3])
    runtime(tmp_path, monkeypatch, **identity)
    model.forward(values)
    assert model.forward_calls == model.channel_calls == 1


def test_corrupt_entries_and_storage_limit_recompute_without_suppressing_real_errors(
    tmp_path, monkeypatch
):
    runtime(tmp_path, monkeypatch)
    model = Probe()
    expected = model.forward(torch.tensor([0.2]))["score"]
    model._numerical_cache.connection.execute("UPDATE entries SET payload=x'00'")
    model._numerical_cache.connection.commit()
    actual = Probe().forward(torch.tensor([0.2]))
    assert torch.equal(actual["score"], expected)
    monkeypatch.setenv("EXACT_NUMERICAL_CACHE_MAX_BYTES", "0")
    bounded = Probe(tau=0.123)
    first = bounded.forward(torch.tensor([0.4]))
    bounded.forward(torch.tensor([0.4]))
    assert bounded.forward_calls == bounded.channel_calls == 2
    assert first["numerical_cache"]["skipped_writes"] >= 2
    assert first["numerical_cache"]["limit_bytes"] == 0
    with pytest.raises(TypeError):
        bounded.forward("invalid-scorer-input")
    assert bounded.forward_calls == 3  # The failing real computation was not retried.


def test_training_frames_share_features_but_preserve_role_labels_and_recipe(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    binding = {"source_ids": ["dev"], "negative_label_policy": "confirmed_negatives"}
    rows = [{"Src": "train", "Tgt": "target", "score": 0.6, "confirmed_label": 1}]
    first = Probe()
    shared_training_scores(first, "training-content-hash", binding, 8, rows=rows)
    assert shared_training_scores(Probe(), "training-content-hash", binding, 8) == rows
    assert shared_training_scores(Probe(), "changed-label-content-hash", binding, 8) is None
    assert shared_training_scores(Probe(), "training-content-hash", binding, 16) is None
    assert shared_training_scores(Probe(gamma=3), "training-content-hash", binding, 8) is None
    assert not hasattr(first, "_numerical_scoring_role")
    first.forward(torch.tensor([0.2]))
    first._numerical_scoring_role = "train"
    first.forward(torch.tensor([0.2]))
    assert first.forward_calls == 2


def test_real_pair_scorer_replays_tensor_scores_and_explanations(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    args = {
        "src_iris": ["s"],
        "tgt_iris": ["t"],
        "src_label_lists": [["heart valve"]],
        "tgt_label_lists": [["heart valve"]],
    }
    first = _scorer(return_explanations=True, strsim={"enabled": True})
    first.attach_dataset(_TinyDataset())
    expected = first(**args)
    replay = _scorer(return_explanations=True, strsim={"enabled": True})
    replay.attach_dataset(_TinyDataset())
    monkeypatch.setattr(
        replay, "_score_label_channel", lambda *args: pytest.fail("recomputed frozen evidence")
    )
    actual = replay(**args)
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, actual[name]), name
    assert expected["explanations"] == actual["explanations"]
    assert actual["backend_usage"] == {}
    assert actual["numerical_cache"]["new_forward_calls"] == 0
    assert actual["numerical_cache"]["hits"] == 1
    assert (tmp_path / "diagnostics/numerical-cache.json").is_file()


def test_raw_channel_scope_is_reused_within_forward_only(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    model = Probe()
    calls = []
    original = model.runtime_fingerprint_payload
    model.runtime_fingerprint_payload = lambda: (calls.append(True), original())[1]
    model._numerical_channel_scopes = {}
    with torch.inference_mode():
        model.channel(torch.tensor([0.2]))
        model.channel(torch.tensor([0.3]))
        assert len(calls) == 1
        model.tau = 0.4
        model.channel(torch.tensor([0.2]))
        assert len(calls) == 2
    del model._numerical_channel_scopes
    model.forward(torch.tensor([0.2]))
    assert len(calls) == 4
    assert not hasattr(model, "_numerical_channel_scopes")


def test_payload_limit_is_shared_by_independent_writers(tmp_path):
    first, second = NumericalCache(tmp_path), NumericalCache(tmp_path)
    with first.batch():
        first.put("scope", "operation", "first", {"value": "evidence"})
    second.limit = first.bytes
    with second.batch():
        second.put("scope", "operation", "second", {"value": "evidence"})
    assert second.stats()["entries"] == 1
    assert second.stats()["payload_bytes"] <= second.limit
    assert second.skipped == 1
