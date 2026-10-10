"""Synthetic review receipts exercise an offline boundary, never semantic self-approval."""

import copy
import json

import pyowl_core as core
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from exact_inspect.artifacts import atomic_json, publish_bundle
from exact_inspect.context import build_context_package
from exact_inspect.contracts import (
    EntityRef,
    VisibilityPolicy,
    canonical_hash,
    canonical_json,
    file_hash,
)
from exact_inspect.description_review import (
    admit_independent_review,
    build_review_request,
)
from exact_inspect.descriptions import (
    prepare_description_packet,
    validate_prepared_support,
)
from exact_inspect.generation import ExplanationJobs, GenerationProfile, entity_packet
from exact_inspect.models import GeneratedExplanationResponse
from exact_inspect.service import create_prepared_app
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.resources import ExplanationResource
from exact_inspect.study.workspace import _admit_explanation


@pytest.fixture
def reviewed_fixture(tmp_path):
    context = build_context_package(
        core.load_snapshot(b"""Ontology(
        Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
        Declaration(ObjectProperty(<urn:p>))
        SubClassOf(<urn:A> ObjectSomeValuesFrom(<urn:p> <urn:B>)))"""),
        tmp_path / "context",
    )
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri="urn:A", kind="class")
    policy = VisibilityPolicy(ontology_ids=(context.ontology_version_id,))
    facts = context.facts(entity, category="restrictions", policy=policy)["items"]
    packet = prepare_description_packet(
        entity_packet(
            entity,
            facts,
            context_hash=context.manifest["artifacts"]["context.sqlite"],
            policy=policy,
        ),
        context,
        policy,
    )
    packet = packet.model_copy(
        update={"selection": {**packet.selection, "description_prose": True}}
    )
    claims = [
        {
            "text": "An instance of A must have a p relationship to at least one instance of B.",
            "fact_ids": [facts[0]["fact_id"]],
            "category": "scope",
        },
        {
            "text": "Instances of A cannot have any p relationship.",
            "fact_ids": [facts[0]["fact_id"]],
            "category": "scope",
        },
    ]
    raw = {
        "model": "draft-provider",
        "choices": [
            {
                "message": {
                    "content": json.dumps({"claims": claims, "limitations": [], "relation": None})
                }
            }
        ],
    }
    calls = []

    def provider(messages, profile):
        calls.append(messages)
        return raw

    jobs = ExplanationJobs(tmp_path / "jobs")
    result = jobs.generate(packet, GenerationProfile(name="test", model="test"), provider=provider)
    request = build_review_request(result, raw)
    receipt = {
        "contract_version": "exact-description-review/1",
        **request["bindings"],
        "reviewer_id": "synthetic-independent-test-reviewer",
        "reviewer_kind": "independent_agent",
        "review_provenance": "Synthetic test decision supplied outside the provider response; this is not a real semantic audit.",
        "decisions": [
            {
                "ordinal": i,
                **entry,
                "decision": "approved" if i == 0 else "rejected",
                "support_explanation": (
                    "The original existential restriction requires one p successor in B."
                    if i == 0
                    else "This negation contradicts the cited existential restriction and is rejected."
                ),
            }
            for i, entry in enumerate(request["claim_inventory"])
        ],
    }
    return context, policy, packet, result, raw, request, receipt, calls


def test_prose_remains_fallback_until_separate_review_and_cache_is_immutable(
    reviewed_fixture, tmp_path
):
    context, policy, packet, source, raw, request, receipt, calls = reviewed_fixture
    assert len(calls) == 1
    assert "separate reviewer" in calls[0][0]["content"]
    assert source["preparation_kind"] == "factual_fallback"
    before = canonical_json(source)
    saved = tmp_path / "jobs" / source["explanation_id"][7:] / "explanation.json"
    frozen = file_hash(saved)
    result = admit_independent_review(source, raw, receipt)
    assert result["explanation_id"] != source["explanation_id"]
    assert result["support_status"] == "independently_reviewed"
    assert len(result["claims"]) == 1 and "cannot" not in result["claims"][0]["text"]
    assert canonical_json(source) == before and file_hash(saved) == frozen
    GeneratedExplanationResponse.model_validate(result)
    validate_prepared_support(result, {context.ontology_version_id: context}, policy)
    assert admit_independent_review(source, raw, receipt) == result


def test_review_binds_every_input_and_selected_claim(reviewed_fixture):
    context, policy, packet, source, raw, request, receipt, _ = reviewed_fixture
    for key in ("packet_hash", "policy_hash", "response_hash", "input_hash", "claims_hash"):
        forged = copy.deepcopy(receipt)
        forged[key] = "sha256:" + "0" * 64
        with pytest.raises(ValueError):
            admit_independent_review(source, raw, forged)
    changed = copy.deepcopy(raw)
    changed["model"] = "another-provider"
    with pytest.raises(ValueError, match="source input bindings"):
        admit_independent_review(source, changed, receipt)
    rejected = copy.deepcopy(receipt)
    rejected["decisions"][0]["decision"] = "rejected"
    with pytest.raises(ValueError, match="approved inventory"):
        admit_independent_review(source, raw, rejected)
    result = admit_independent_review(source, raw, receipt)
    result["claims"][0]["text"] = "A is equivalent to B."
    with pytest.raises(ValidationError, match="content or original support"):
        GeneratedExplanationResponse.model_validate(result)


def test_provider_cannot_supply_its_own_review_receipt(reviewed_fixture):
    _, _, _, source, raw, _, receipt, _ = reviewed_fixture
    embedded = copy.deepcopy(raw)
    content = json.loads(embedded["choices"][0]["message"]["content"])
    content["independent_review"] = receipt
    embedded["choices"][0]["message"]["content"] = json.dumps(content)
    modified = copy.deepcopy(source)
    modified["manifest"]["response_hash"] = canonical_hash(embedded)
    with pytest.raises(ValidationError, match="Extra inputs"):
        build_review_request(modified, embedded)


def test_originals_policy_resource_and_workspace_are_rechecked(reviewed_fixture):
    context, policy, packet, source, raw, _, receipt, _ = reviewed_fixture
    result = admit_independent_review(source, raw, receipt)
    contexts = {context.ontology_version_id: context}
    resource = build_explanation_resource(
        contexts, packet.entities, policy=policy, packets=[packet], explanations=[result]
    )
    claim = resource.entity_profiles[0]
    assert (
        claim.grounding == "independent_review"
        and claim.reviewer_receipt in resource.independent_reviews
    )
    case = {"source": packet.entities[0].model_dump(), "candidates": []}
    assert (
        _admit_explanation(result, case, policy, contexts)["support_status"]
        == "independently_reviewed"
    )
    forged = resource.model_dump(mode="json")
    forged["entity_profiles"][0]["text"] = "Invented prose."
    with pytest.raises(ValueError, match="content or original support"):
        ExplanationResource.model_validate(forged)
    unsupported_policy = VisibilityPolicy(
        ontology_ids=(context.ontology_version_id,), categories=("labels",)
    )
    with pytest.raises(ValueError, match="frozen scope"):
        validate_prepared_support(result, contexts, unsupported_policy)


@pytest.mark.parametrize(
    "edited_field",
    [
        "claims",
        "limitations",
        "support_reasons",
        "fixture_provenance",
        "manifest_failed",
        "manifest_ambiguous",
    ],
)
def test_actual_http_serves_reviewed_prose_and_rejects_edited_claims(
    reviewed_fixture, tmp_path, edited_field
):
    context, policy, _, source, raw, _, receipt, _ = reviewed_fixture
    result = admit_independent_review(source, raw, receipt)
    path = tmp_path / "reviewed.json"
    endpoint = "/api/v1/explanations/" + result["explanation_id"]

    def app():
        package = publish_bundle(
            tmp_path,
            audience="local",
            policy=policy,
            ontologies={context.ontology_version_id: "context"},
            explanations={result["explanation_id"]: "reviewed.json"},
        )
        return create_prepared_app(package)

    atomic_json(path, result)
    with TestClient(app()) as client:
        response = client.get(endpoint)
        assert response.status_code == 200, response.text
        assert response.json()["support_status"] == "independently_reviewed"
    if edited_field == "claims":
        result["claims"][0]["text"] = "Invented unsupported definition."
    elif edited_field == "fixture_provenance":
        result[edited_field] = "Invented unsupported definition."
    elif edited_field.startswith("manifest_"):
        result["manifest"]["status"] = edited_field.removeprefix("manifest_")
    else:
        result[edited_field] = ["Invented unsupported definition."]
    with pytest.raises(ValueError):
        validate_prepared_support(result, {context.ontology_version_id: context}, policy)
    atomic_json(path, result)
    with TestClient(app()) as client:
        response = client.get(endpoint)
        assert response.status_code == 409, response.text
        assert "Invented" not in response.text


def test_valid_receipt_cannot_override_the_admitted_original_ast(reviewed_fixture):
    context, policy, _, source, raw, _, receipt, _ = reviewed_fixture
    forged_source = copy.deepcopy(source)
    forged_packet = forged_source["support_packet"]
    forged_packet["facts"][0]["value"]["ast"]["super_class"]["filler"]["iri"][
        "value"
    ] = "urn:Invented"
    forged_source["manifest"]["packet_hash"] = canonical_hash(forged_packet)
    request = build_review_request(forged_source, raw)
    forged_receipt = {**receipt, **request["bindings"]}
    reviewed = admit_independent_review(forged_source, raw, forged_receipt)
    GeneratedExplanationResponse.model_validate(reviewed)
    with pytest.raises(ValueError, match="AST differs from original"):
        validate_prepared_support(reviewed, {context.ontology_version_id: context}, policy)


@pytest.mark.parametrize("mutation", ["digest", "manifest", "receipt", "packet", "status", "extra"])
def test_study_review_support_binds_the_actual_generation_manifest(reviewed_fixture, mutation):
    context, policy, packet, source, raw, _, receipt, _ = reviewed_fixture
    result = admit_independent_review(source, raw, receipt)
    resource = build_explanation_resource(
        {context.ontology_version_id: context},
        packet.entities,
        policy=policy,
        packets=[packet],
        explanations=[result],
    ).model_dump(mode="json")
    support = next(iter(resource["independent_reviews"].values()))
    assert support["generation_manifest"] == result["manifest"]
    claim = resource["entity_profiles"][0]
    if mutation == "digest":
        support["generation_manifest_sha256"] = "0" * 64
        claim["generation_manifest_sha256"] = "0" * 64
    else:
        if mutation == "manifest":
            support["generation_manifest"]["requested_model"] = "unreviewed-provider-model"
        elif mutation == "receipt":
            support["generation_manifest"]["description_review_hash"] = "sha256:" + "0" * 64
        elif mutation == "packet":
            support["packet"]["language"] = "another-language"
            support["generation_manifest"]["packet_hash"] = canonical_hash(support["packet"])
        elif mutation == "status":
            support["generation_manifest"]["status"] = "failed"
        elif mutation == "extra":
            support["generation_manifest"]["unadmitted"] = "raw unreviewed content"
        # Recomputing both declared hashes cannot substitute different source bytes.
        support["generation_manifest_sha256"] = canonical_hash(
            support["generation_manifest"]
        ).removeprefix("sha256:")
        claim["generation_manifest_sha256"] = support["generation_manifest_sha256"]
    with pytest.raises(ValueError):
        ExplanationResource.model_validate(resource)


def test_nested_review_model_cannot_bypass_generation_binding(reviewed_fixture):
    context, policy, packet, source, raw, _, receipt, _ = reviewed_fixture
    result = admit_independent_review(source, raw, receipt)
    resource = build_explanation_resource(
        {context.ontology_version_id: context},
        packet.entities,
        policy=policy,
        packets=[packet],
        explanations=[result],
    )
    payload = resource.model_dump(mode="json")
    payload["independent_reviews"] = resource.independent_reviews
    support = next(iter(resource.independent_reviews.values()))
    support.generation_manifest["requested_model"] = "changed-after-model-validation"
    with pytest.raises(ValueError, match="immutable review bindings"):
        ExplanationResource.model_validate(payload)
