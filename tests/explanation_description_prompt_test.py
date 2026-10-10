"""Prose instructions must reach the real transport boundary before a paid dispatch."""

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from exact_inspect.contracts import EntityRef, canonical_hash, file_hash
from exact_inspect.descriptions import (
    COMPARISON_PROSE_PROMPT_V1,
    COMPARISON_PROSE_PROMPT_V2,
    PROFILE_PROSE_PROMPT_V1,
    PROFILE_PROSE_PROMPT_V2,
)
from exact_inspect.generation import (
    Claim,
    ExplanationJobs,
    ExplanationOutput,
    FactPacket,
    GenerationProfile,
    parse_provider_output,
)


@pytest.mark.parametrize("task", ["entity_profile", "pair_comparison"])
def test_versioned_output_shape_reaches_outbound_transport_without_reusing_old_cache(
    task, tmp_path, monkeypatch
):
    import exact.llm.routing as routing

    calls, closes = [], []

    class Transport:
        def __init__(self):
            self._client = self

        def close(self):
            closes.append(True)

        def chat_completion(self, binding, messages, max_tokens, **kwargs):
            calls.append(messages)
            return {"choices": [{"message": {"content": json.dumps(output)}}]}

    monkeypatch.setattr(routing, "OpenRouterClient", Transport)
    entity = EntityRef(ontology_version_id="urn:ontology", iri="urn:A", kind="class")
    fact = {
        "fact_id": "fact:recorded-definition",
        "subject": entity.model_dump(),
        "category": "definitions",
        "value": {"term_type": "literal", "lexical_form": "Recorded original meaning."},
    }
    packet = FactPacket(
        task=task,
        entities=[entity]
        + ([entity.model_copy(update={"iri": "urn:B"})] if task == "pair_comparison" else []),
        context_hashes=["sha256:" + "a" * 64],
        policy_hash="sha256:" + "b" * 64,
        facts=[fact],
        missingness=[],
        selection={"description_contract": "exact-description/2", "description_prose": True},
    )
    output = {
        "claims": [
            {
                "text": fact["value"]["lexical_form"],
                "fact_ids": [fact["fact_id"]],
                "category": "key_fact",
            }
        ],
        "limitations": [],
        "relation": None,
    }
    old_prompt, new_prompt = (
        (PROFILE_PROSE_PROMPT_V1, PROFILE_PROSE_PROMPT_V2)
        if task == "entity_profile"
        else (COMPARISON_PROSE_PROMPT_V1, COMPARISON_PROSE_PROMPT_V2)
    )
    jobs = ExplanationJobs(tmp_path / "jobs")
    profile = GenerationProfile(name="synthetic-transport", model="synthetic-model")
    previous = jobs.generate(packet, profile, prompt=old_prompt, max_repairs=0)
    old_directory = jobs.root / previous["explanation_id"][7:]
    frozen = {p.name: file_hash(p) for p in old_directory.iterdir() if p.is_file()}
    result = jobs.generate(packet, profile, max_repairs=0)
    assert len(calls) == len(closes) == 2
    assert result["explanation_id"] != previous["explanation_id"]
    assert result["manifest"]["prompt_hash"] == canonical_hash(new_prompt)
    assert previous["manifest"]["prompt_hash"] == canonical_hash(old_prompt)
    assert calls[0][0]["content"] == old_prompt
    sent = calls[1][0]["content"]
    assert sent == new_prompt and "exact-description-prose/2" in sent
    example = json.loads(next(line for line in sent.splitlines() if line.startswith('{"claims"')))
    ExplanationOutput.model_validate(example)
    assert set(example) == {"claims", "limitations", "relation"}
    assert set(example["claims"][0]) == {"text", "fact_ids", "category"}
    for category in get_args(Claim.model_fields["category"].annotation):
        assert category in sent
    for instruction in (
        "Each claim must be an object",
        "never a scalar",
        "Do not use claim instead of text",
        "Inline\ncitations cannot replace fact_ids",
        "Do not return claim_id or any additional fields",
    ):
        assert instruction in sent
    assert json.loads(calls[1][1]["content"])["facts"] == packet.facts
    assert jobs.generate(packet, profile, prompt=old_prompt, max_repairs=0) == previous
    assert jobs.generate(packet, profile, max_repairs=0) == result
    assert len(calls) == 2
    assert {p.name: file_hash(p) for p in old_directory.iterdir() if p.is_file()} == frozen


@pytest.mark.parametrize(
    "claim",
    [
        "A scalar claim [fact:one].",
        {"text": "Missing category.", "fact_ids": ["fact:one"]},
        {"claim": "Wrong field.", "category": "scope", "fact_id": "fact:one"},
    ],
)
def test_malformed_saved_prose_is_not_normalized_into_a_reviewable_claim(claim):
    with pytest.raises(ValidationError):
        parse_provider_output(json.dumps({"claims": [claim], "limitations": [], "relation": None}))
