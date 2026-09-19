"""Ambiguity compares distinct target concepts, independently of duplicate labels."""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer


class _Scorer(PairAdaptiveSemanticScorer):
    def __init__(self, mode, targets):
        super().__init__(
            use_lexical=False,
            use_context=False,
            use_llm=False,
            llm_model_name=None,
            persist_cache_to_disk=False,
            device="cpu",
            lex={"enabled": True, "quality": mode, "deduplicate_labels": True},
        )
        self.use_lexical = self.use_context = True
        self._attached_dataset = SimpleNamespace(
            dataframe=pd.DataFrame({"Src": ["s"] * len(targets), "Tgt": targets})
        )

    def encode_labels_batch(self, labels):
        vectors = {"source": [1.0, 0.0], "A": [1.0, 0.0], "B": [0.8, 0.6]}
        return torch.tensor([vectors[label] for label in labels])

    def encode_contexts_batch(self, labels):
        vectors = {"source": [1.0, 0.0], "A": [0.8, 0.6], "B": [1.0, 0.0]}
        return torch.tensor([vectors[label] for label in labels])

    def score(self, source_labels, target_labels, targets):
        scores, _, _, payloads = self._score_label_channel(source_labels, target_labels)
        quality = self._candidate_quality(
            ["s"] * len(targets), targets, source_labels, target_labels, scores, payloads
        )
        return quality, payloads


def test_candidate_margin_ignores_duplicated_synonyms_and_target_rows():
    first = _Scorer("candidate_margin", ["a", "b"])
    q1, p1 = first.score([["source"], ["source"]], [["A"], ["B"]], ["a", "b"])
    second = _Scorer("candidate_margin", ["a", "b", "a"])
    q2, p2 = second.score(
        [["source", "source"]] * 3, [["A", "A"], ["B", "B"], ["A"]], ["a", "b", "a"]
    )
    assert torch.allclose(q1, torch.tensor([0.1, 0.1]))
    assert torch.allclose(q2, q1[0].expand(3))
    assert p1[0]["candidate_count"] == p2[0]["candidate_count"] == 2


def test_encoder_agreement_requires_same_best_target_not_similar_maxima():
    model = _Scorer("encoder_agreement", ["a", "b"])
    quality, payloads = model.score([["source"], ["source"]], [["A"], ["B"]], ["a", "b"])
    assert quality.tolist() == [0, 0]
    assert payloads[0]["lexical_best_target"] == "a"
    assert payloads[0]["context_best_target"] == "b"
    assert payloads[0]["encoder_score_similarity"] > 0.8


def test_singleton_quality_is_undefined_and_partial_source_batch_rejected():
    singleton = _Scorer("candidate_margin", ["a"])
    quality, payload = singleton.score([["source"]], [["A"]], ["a"])
    assert quality.tolist() == [0]
    assert not payload[0]["candidate_quality_defined"]
    partial = _Scorer("candidate_margin", ["a", "b"])
    with pytest.raises(ValueError, match="complete frozen source pool"):
        partial.score([["source"]], [["A"]], ["a"])


@pytest.mark.parametrize("mode", ["candidate_margin", "encoder_agreement"])
def test_candidate_quality_uses_complete_scored_pool_after_exact_prefilter(mode):
    model = _Scorer(mode, ["exact", "a", "b"])
    frame = model._attached_dataset.dataframe
    frame["SrcKind"] = frame["TgtKind"] = "class"
    frame["inference"] = [False, True, True]
    frame["prefiltered"] = [True, False, False]
    model._attached_dataset._active_dataframe = lambda: frame.loc[frame.inference]

    quality, payloads = model.score([["source"], ["source"]], [["A"], ["B"]], ["a", "b"])
    expected = 0.1 if mode == "candidate_margin" else 0.0
    torch.testing.assert_close(quality, torch.full((2,), expected))
    assert all(payload["candidate_count"] == 2 for payload in payloads)
    assert all(set(payload["candidate_scores"]) == {"a", "b"} for payload in payloads)
    with pytest.raises(ValueError, match="complete frozen source pool"):
        model.score([["source"]], [["A"]], ["a"])
