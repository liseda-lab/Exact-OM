from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from exact.impl.datasets.contextgraph import ContextDataset
from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer


class _RelationGraph:
    def get_example_triples(self, *args, **kwargs):
        return {"part_of": [("heart", "part_of", "body")]}


def _components(tmp_path: Path):
    dataset = ContextDataset(
        output_path=tmp_path,
        device="cpu",
        verbalization_mode="llm",
        llm_profiles={"hosted": {"backend": "openrouter", "model": "test-model"}},
        llm_routing={"verbaliser_profile": "hosted"},
        request_seed=17,
    )
    dataset._source_graph = _RelationGraph()
    dataset._target_graph = _RelationGraph()
    scorer = PairAdaptiveSemanticScorer(
        use_lexical=False,
        use_context=False,
        use_llm=False,
        llm_model_name=None,
        generate_llm_rationales=False,
        llm_experiment_config={"enabled": True, "gate": {"mode": "off"}},
        persist_cache_to_disk=False,
        device="cpu",
    )
    scorer.use_context = True
    scorer.attach_dataset(dataset)
    return dataset, scorer


def _difference(scorer):
    item = {"triple": ("heart", "part_of", "body"), "score": 1.0}
    return scorer._score_difference_channel([item], [item], torch.zeros((1, 1)))


def test_decision_gate_off_still_uses_and_caches_relation_templates(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    dataset, scorer = _components(tmp_path)
    requests = []

    def chat_completion(**kwargs):
        requests.append(kwargs)
        return {
            "choices": [{"message": {"content": json.dumps({"template": "$SRC is part of $TGT."})}}]
        }

    monkeypatch.setattr(dataset._llm_router.hosted, "chat_completion", chat_completion)
    result = _difference(scorer)
    assert result["src_sentences"] == ["heart is part of body."]
    assert result["tgt_sentences"] == ["heart is part of body."]
    assert len(requests) == 1
    assert requests[0]["role"] == "verbaliser"
    assert requests[0]["seed"] == 17
    assert dataset._template_cache_matches()
    assert scorer.generate_final_rationales_for_records([{}]) == [""]

    def forbidden(**kwargs):
        raise AssertionError("persisted relation template should avoid another hosted call")

    dataset._verbalization_templates = None
    monkeypatch.setattr(dataset._llm_router.hosted, "chat_completion", forbidden)
    assert _difference(scorer)["src_sentences"] == result["src_sentences"]


def test_relation_template_failure_propagates_with_decision_gate_off(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    dataset, scorer = _components(tmp_path)

    def unauthorized(**kwargs):
        raise RuntimeError("OpenRouter HTTP 401")

    monkeypatch.setattr(dataset._llm_router.hosted, "chat_completion", unauthorized)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        _difference(scorer)
    assert not dataset._verb_temp_path.exists()
    assert not dataset._verb_temp_meta_path.exists()
