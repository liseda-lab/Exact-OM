"""Bound whole-fact few-shot inputs without losing candidate identities or labels."""

import json
from copy import deepcopy

import pytest

from exact.impl.models.selector.llm_learning import (
    EXEMPLAR_RENDERING,
    _compact_exemplars,
    _exemplar_evidence,
)
from exact.llm.prompt_budget import PromptBudgetError


class FixtureTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return list(range((len(text.encode("utf-8")) + 3) // 4))


def examples():
    rows = []
    for source in range(3):
        candidates = []
        for target in range(5):
            facts = [
                {
                    "side": "source",
                    "group": "attribute",
                    "property_iri": "urn:definition",
                    "text": "A very long but indivisible definition. " * 2000,
                    "datatype": "xsd:string",
                    "item_id": "transport-provenance-" * 100,
                },
                {
                    "side": "target",
                    "group": "hierarchy:child",
                    "triple": [f"Target {target}", "subClassOf", "Disease"],
                    "item_id": f"fact-{source}-{target}",
                    "subject_iri": f"urn:target:{target}",
                },
            ]
            facts.extend(
                {
                    "side": "target",
                    "group": "attribute",
                    "property_iri": "urn:definition",
                    "text": f"Complete sentence {index}: " + "medical fact " * 20,
                    "language": "en",
                }
                for index in range(30)
            )
            candidates.append(
                {
                    "target": f"urn:target:{source}:{target}",
                    "equivalent": target == 0,
                    "evidence": json.dumps(
                        {
                            "source_label": f"Source {source}",
                            "target_label": f"Target {target}",
                            "facts": facts,
                        }
                    ),
                }
            )
        rows.append({"source": f"urn:source:{source}", "candidates": candidates})
    return rows


def test_large_three_by_five_examples_are_bounded_complete_and_deterministic():
    rows, tokenizer = examples(), FixtureTokenizer()
    original = deepcopy(rows)
    text = _compact_exemplars(rows, tokenizer)
    assert len(tokenizer.encode(text)) <= EXEMPLAR_RENDERING["max_input_tokens"]
    assert len(text.encode()) <= EXEMPLAR_RENDERING["max_utf8_bytes"]
    assert text == _compact_exemplars(rows, tokenizer) and rows == original
    rendered = json.loads(text.split("\n", 2)[2])
    assert [row["source"] for row in rendered] == [row["source"] for row in rows]
    for expected, actual in zip(rows, rendered):
        assert len(actual["candidates"]) == 5
        for candidate, compact in zip(expected["candidates"], actual["candidates"]):
            packet, allowed = _exemplar_evidence(candidate)
            assert actual["source_label"] == packet["source_label"]
            for field in ("target", "equivalent"):
                assert compact[field] == candidate[field]
            assert compact["target_label"] == packet["target_label"]
            assert compact["facts"] and all(fact in allowed for fact in compact["facts"])
            assert compact["omitted_facts"] + len(compact["facts"]) == len(allowed)
            assert len({json.dumps(fact) for fact in compact["facts"]}) == len(compact["facts"])
            assert any("triple" in fact for fact in compact["facts"])
    assert "transport-provenance" not in text
    assert "A very long but indivisible" not in text  # No partial definition is fabricated.


def test_projection_preserves_literal_and_contradiction_semantics_and_deduplicates():
    candidate = examples()[0]["candidates"][0]
    raw = {
        "side": "source",
        "group": "attribute",
        "property_iri": "urn:p",
        "text": "false",
        "datatype": "xsd:boolean",
        "language": "en",
        "state": "observed",
        "contradiction_semantics": "unobserved_counterpart",
        "item_id": "first",
    }
    candidate["evidence"] = {
        "source_label": "source",
        "target_label": "target",
        "facts": [raw, {**raw, "item_id": "duplicate-provenance"}],
    }
    _, facts = _exemplar_evidence(candidate)
    assert len(facts) == 1 and "item_id" not in facts[0]
    assert facts[0]["property"] == raw["property_iri"]
    for field in ("text", "datatype", "language", "state", "contradiction_semantics"):
        assert facts[0][field] == raw[field]


@pytest.mark.parametrize("oversize", ["identity", "all_facts"])
def test_unrepresentable_examples_fail_instead_of_dropping_candidates(oversize):
    rows = examples()
    if oversize == "identity":
        rows[0]["source"] = "x" * 20000
    else:
        candidate = rows[0]["candidates"][0]
        packet = json.loads(candidate["evidence"])
        packet["facts"] = packet["facts"][:1]
        candidate["evidence"] = packet
    with pytest.raises(PromptBudgetError, match="exemplar"):
        _compact_exemplars(rows, FixtureTokenizer())


def test_genuinely_empty_evidence_remains_explicit_without_invented_facts():
    rows = examples()[:1]
    candidate = rows[0]["candidates"][0]
    packet = json.loads(candidate["evidence"])
    packet["facts"] = []
    candidate["evidence"] = packet
    rendered = json.loads(_compact_exemplars(rows, FixtureTokenizer()).split("\n", 2)[2])
    assert rendered[0]["candidates"][0]["facts"] == []
    assert rendered[0]["candidates"][0]["omitted_facts"] == 0


def test_candidate_specific_source_alias_is_preserved_without_relabeling_other_pairs():
    rows = examples()[:1]
    candidate = rows[0]["candidates"][1]
    packet = json.loads(candidate["evidence"])
    packet["source_label"] = "Another valid source alias"
    candidate["evidence"] = packet
    rendered = json.loads(_compact_exemplars(rows, FixtureTokenizer()).split("\n", 2)[2])[0]
    assert rendered["source_label"] == "Source 0"
    assert "source_label" not in rendered["candidates"][0]
    assert rendered["candidates"][1]["source_label"] == "Another valid source alias"
