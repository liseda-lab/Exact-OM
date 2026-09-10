"""Repeated fact provenance must not crowd unique evidence out of the attribute cap."""

import pytest
import torch

from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer


def scorer(dedup):
    model = PairAdaptiveSemanticScorer(
        use_lexical=False,
        use_context=False,
        use_llm=False,
        llm_model_name=None,
        persist_cache_to_disk=False,
        device="cpu",
        max_attr_items=2,
        attr={"enabled": True, "bank": "attrs_labels", "provenance_dedup": dedup},
    )
    model.use_context = True
    model._encode_context_matrix = lambda left, right: torch.tensor(
        [[float(a == b) for b in right] for a in left]
    )
    return model


def attr(value, **kwargs):
    return {
        "prop_iri": "urn:definition",
        "prop": "definition",
        "value": value,
        "text": value,
        **kwargs,
    }


def score(model, source):
    return model._score_attribute_channel(source, [attr("a"), attr("b")], [], [], {}, {})


def test_repeated_facts_preserve_mass_and_unique_evidence_before_cap():
    model = scorer(True)
    original = score(model, [attr("a"), attr("b")])
    repeated = score(model, [attr("a"), attr("a"), attr("a"), attr("b")])
    for key in ("score", "quality", "coverage", "informativeness", "stability"):
        assert repeated[key] == pytest.approx(original[key])
    assert [row["value"] for row in repeated["src_selected"]] == ["a", "b"]
    assert [
        row["value"]
        for row in score(scorer(False), [attr("a"), attr("a"), attr("b")])["src_selected"]
    ] == ["a", "a"]


def test_dedup_retains_literal_languages_as_distinct_facts():
    result = score(scorer(True), [attr("a", language="en"), attr("a", language="pt")])
    assert {row["language"] for row in result["src_selected"]} == {"en", "pt"}
    assert all(row["provenance_items"] for row in result["src_selected"])
