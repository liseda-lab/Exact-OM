"""Prose instructions must reach the real transport boundary before a paid dispatch."""

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from exact_inspect.contracts import EntityRef, canonical_hash, file_hash
from exact_inspect.descriptions import (
    COMPARISON_PROSE_PROMPT_V2,
    COMPARISON_PROSE_PROMPT_V5,
    PROFILE_PROSE_PROMPT_V2,
    PROFILE_PROSE_PROMPT_V5,
)
from exact_inspect.generation import (
    Claim,
    ExplanationJobs,
    ExplanationOutput,
    FactPacket,
    GenerationProfile,
    parse_provider_output,
    prose_response_format,
)


@pytest.mark.parametrize("task", ["entity_profile", "pair_comparison"])
def test_versioned_output_shape_reaches_outbound_transport_without_reusing_old_cache(
    task, tmp_path, monkeypatch
):
    import exact.llm.routing as routing

    calls, closes, wire_options = [], [], []

    class Transport:
        def __init__(self):
            self._client = self

        def close(self):
            closes.append(True)

        def chat_completion(self, binding, messages, max_tokens, **kwargs):
            calls.append(messages)
            wire_options.append(kwargs)
            return {"choices": [{"message": {"content": json.dumps(output)}}]}

    monkeypatch.setattr(routing, "OpenRouterClient", Transport)
    entity = EntityRef(ontology_version_id="urn:ontology", iri="urn:A", kind="class")
    fact = {
        "fact_id": "fact:recorded-definition",
        "subject": entity.model_dump(),
        "category": "definitions",
        "origins": [{"document_sha256": "frozen-original-document"}],
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
        (PROFILE_PROSE_PROMPT_V2, PROFILE_PROSE_PROMPT_V5)
        if task == "entity_profile"
        else (COMPARISON_PROSE_PROMPT_V2, COMPARISON_PROSE_PROMPT_V5)
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
    assert sent == new_prompt
    assert "response_format" not in wire_options[0]
    assert wire_options[1]["response_format"] == prose_response_format()
    wire_claim = prose_response_format()["json_schema"]["schema"]["properties"]["claims"]["items"]
    assert set(wire_claim["required"]) == {"text", "fact_ids", "category"}
    assert wire_claim["additionalProperties"] is False
    assert set(wire_claim["properties"]["category"]["enum"]) == set(
        get_args(Claim.model_fields["category"].annotation)
    )
    assert "definition" not in wire_claim["properties"]["category"]["enum"]
    assert "ONLY the provided" in sent
    assert "label citation cannot support details" in sent
    assert "ONLY in fact_ids" in sent
    sent_packet = json.loads(calls[1][1]["content"])
    assert sent_packet["facts"] == [
        {k: v for k, v in f.items() if k in {"fact_id", "subject", "category", "value"}}
        for f in packet.facts
    ]
    assert "origins" in json.loads(calls[0][1]["content"])["facts"][0]
    saved_request = json.loads(
        (jobs.root / result["explanation_id"][7:] / "request.json").read_bytes()
    )
    assert saved_request["packet"]["facts"] == packet.facts
    assert "supported_description_templates" not in sent_packet
    assert "supported_comparison_templates" not in sent_packet
    assert "supported_description_templates" in json.loads(calls[0][1]["content"])
    assert "Labels/synonyms alone support only recorded naming" in sent
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
