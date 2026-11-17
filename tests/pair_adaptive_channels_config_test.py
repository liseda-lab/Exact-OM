from __future__ import annotations

import pytest
import torch

from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer


def _scorer(**kwargs) -> PairAdaptiveSemanticScorer:
    return PairAdaptiveSemanticScorer(
        use_lexical=False,
        use_context=False,
        use_llm=False,
        llm_model_name=None,
        persist_cache_to_disk=False,
        device="cpu",
        **kwargs,
    )


def test_matching_channel_defaults_and_overrides_drive_channel_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = _scorer()
    assert (
        default.hierarchy_embedding_weight,
        default.hierarchy_support_weight,
        default.similarity_embedding_weight,
        default.similarity_support_weight,
        default.similarity_per_relation_cap,
        default.difference_per_relation_cap,
        default.stability_factor,
        default.attribute_information_word_cap,
        default.attribute_score_floor,
        default.uncertainty_indecision_scale,
        default.uncertainty_disagreement_quality_power,
    ) == (0.5, 0.5, 0.5, 0.5, 2, 3, 2.0, 20, 0.5, 2.0, 0.5)
    assert default.attribute_property_weights == {
        "definition": 1.0,
        "identifier": 0.8,
        "comment": 0.6,
        "other": 0.5,
    }

    configured = _scorer(
        hierarchy_embedding_weight=1.0,
        hierarchy_support_weight=0.0,
        similarity_embedding_weight=0.0,
        similarity_support_weight=1.0,
        similarity_per_relation_cap=1,
        difference_per_relation_cap=1,
        stability_factor=0.0,
        attribute_property_weights={
            "definition": 0.25,
            "identifier": 0.3,
            "comment": 0.4,
            "other": 0.2,
        },
        attribute_information_word_cap=1,
        attribute_score_floor=0.8,
        uncertainty_indecision_scale=1.0,
        uncertainty_disagreement_quality_power=1.0,
    )
    configured.use_context = True
    monkeypatch.setattr(
        configured,
        "_encode_label_matrix",
        lambda left, right: torch.full((len(left), len(right)), 0.8),
    )
    monkeypatch.setattr(
        configured,
        "_context_similarity_from_sentences",
        lambda *args, **kwargs: 0.2,
    )

    hierarchy_item = {
        "triple": ("child", "subClassOf", "parent"),
        "specificity": 0.5,
    }
    hierarchy = configured._score_hierarchy_family("is_a", [hierarchy_item], [hierarchy_item])
    assert hierarchy["score"] == pytest.approx(0.2)

    object_items = [{"triple": (f"s{i}", "relatedTo", f"o{i}"), "score": 1.0} for i in range(3)]
    similarity = configured._score_similarity_channel(object_items, object_items)
    assert len(similarity["src_selected"]) == 1
    assert len(similarity["tgt_selected"]) == 1
    assert similarity["score"] == pytest.approx(0.8)
    assert similarity["stability"] == pytest.approx(1.0)

    difference = configured._score_difference_channel(
        object_items,
        object_items,
        support_mat=torch.zeros((3, 3)),
    )
    assert len(difference["src_selected"]) == 1
    assert len(difference["tgt_selected"]) == 1

    monkeypatch.setattr(
        configured,
        "_encode_context_matrix",
        lambda left, right: torch.full((len(left), len(right)), 0.1),
    )
    attribute = {"prop": "definition", "value": "short text", "text": "short text"}
    attribute_payload = configured._score_attribute_channel(
        [attribute],
        [attribute],
        ["source"],
        ["target"],
        {},
        {},
    )
    assert configured._attribute_weight(attribute) == pytest.approx(0.25)
    assert attribute_payload["score"] == pytest.approx(0.8)

    U_ind, U_dis = configured._uncertainty_components(
        torch.tensor([0.5]),
        torch.tensor([0.8]),
        torch.tensor([0.2]),
        torch.tensor([0.5]),
        torch.tensor([0.5]),
    )
    assert U_ind.item() == pytest.approx(0.5)
    assert U_dis.item() == pytest.approx(0.15)
