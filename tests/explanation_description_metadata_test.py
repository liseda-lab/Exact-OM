"""Description metadata follows selected support and preserves frozen legacy records."""

import copy

import pyowl_core as core
import pytest
from pydantic import ValidationError

from exact_inspect.context import build_context_package
from exact_inspect.contracts import EntityRef, VisibilityPolicy, canonical_json
from exact_inspect.descriptions import (
    description_templates,
    preparation_metadata,
    prepare_description_packet,
)
from exact_inspect.generation import (
    Claim,
    ExplanationJobs,
    ExplanationOutput,
    GenerationProfile,
    comparison_packet,
    comparison_templates,
    entity_packet,
)
from exact_inspect.models import GeneratedExplanationResponse
from exact_inspect.study.builder import build_explanation_resource


def prepare(tmp_path, ontology):
    context = build_context_package(core.load_snapshot(ontology), tmp_path / "context")
    policy = VisibilityPolicy(ontology_ids=(context.ontology_version_id,))
    return context, policy


def packet_for(context, policy, iri, category):
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri=iri, kind="class")
    base = entity_packet(
        entity,
        context.facts(entity, category=category, policy=policy)["items"],
        context_hash=context.manifest["artifacts"]["context.sqlite"],
        policy=policy,
    )
    return prepare_description_packet(base, context, policy)


def generate(tmp_path, packet, claims):
    output = ExplanationOutput(claims=claims)
    return ExplanationJobs(tmp_path / "jobs").generate(
        packet,
        GenerationProfile(name="test", model="test"),
        provider=lambda *_: {"choices": [{"message": {"content": output.model_dump_json()}}]},
        max_repairs=0,
    )


def test_label_extract_has_exact_support_and_legacy_null_projection_is_read_only(tmp_path):
    context, policy = prepare(
        tmp_path,
        b"Ontology(Declaration(Class(<urn:A>)) "
        b'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Only a label"))',
    )
    packet = packet_for(context, policy, "urn:A", "labels")
    claims = [
        Claim(text="Only a label", fact_ids=[packet.facts[0]["fact_id"]], category="key_fact")
    ]
    result = generate(tmp_path, packet, claims)
    assert result["preparation_kind"] == "fact_extract"
    assert result["support_status"] == "exact_support"
    GeneratedExplanationResponse.model_validate(result)
    legacy = generate(tmp_path, packet.model_copy(update={"selection": {}}), claims)
    before = canonical_json(legacy)
    projected = GeneratedExplanationResponse.model_validate(legacy).model_dump(mode="json")
    assert projected["preparation_kind"] is None
    assert preparation_metadata(projected)["preparation_kind"] == "fact_extract"
    assert preparation_metadata(projected)["support_status"] == "exact_support"
    assert (
        preparation_metadata({**projected, "preparation_kind": "fact_extract"})["support_status"]
        == "exact_support"
    )
    assert canonical_json(legacy) == before
    assert canonical_json(projected["manifest"]) == canonical_json(legacy["manifest"])

    for forged in (copy.deepcopy(result), copy.deepcopy(legacy)):
        forged["preparation_kind"] = "grounded_description"
        forged["support_status"] = "typed_template_support"
        with pytest.raises(ValidationError, match="Preparation metadata|versioned support"):
            GeneratedExplanationResponse.model_validate(forged)
        with pytest.raises(ValueError, match="Preparation metadata|versioned support"):
            preparation_metadata(forged)


def test_short_comparison_is_typed_description_and_admits_at_resource_boundary(tmp_path):
    source = "A source definition with detail. " * 12
    target = "A target definition with different detail. " * 12
    owl = f"""Ontology(Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
    AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "{source}")
    AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:B> "{target}"))"""
    context, policy = prepare(tmp_path, owl.encode())
    a = packet_for(context, policy, "urn:A", "definitions")
    b = packet_for(context, policy, "urn:B", "definitions")
    profiles = [generate(tmp_path, p, description_templates(p)) for p in (a, b)]
    packet = comparison_packet(a, b, profiles)
    templates = comparison_templates(packet)
    result = generate(tmp_path, packet, templates)
    assert len(templates) == 1 and len(templates[0].text) < 500
    assert "excerpt" in templates[0].text and "…" in templates[0].text
    assert result["preparation_kind"] == "grounded_description"
    assert result["support_status"] == "typed_template_support"
    GeneratedExplanationResponse.model_validate(result)
    resource = build_explanation_resource(
        {context.ontology_version_id: context},
        packet.entities,
        policy=policy,
        packets=[packet],
        explanations=[result],
    )
    assert resource.pair_comparison[0].grounding == "typed_description"
    assert resource.pair_comparison[0].preparation_kind == "grounded_description"
    forged = copy.deepcopy(result)
    forged["claims"][0]["fact_ids"] = [packet.facts[0]["fact_id"]]
    with pytest.raises(ValidationError, match="supported selected claims"):
        GeneratedExplanationResponse.model_validate(forged)


def test_fallback_status_cannot_be_promoted_by_available_templates(tmp_path):
    context, policy = prepare(
        tmp_path,
        b"Ontology(Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>)) "
        b"SubClassOf(<urn:A> <urn:B>))",
    )
    packet = packet_for(context, policy, "urn:A", "hierarchy")
    assert description_templates(packet)
    forged = Claim(
        text="An invented definition", fact_ids=[packet.facts[0]["fact_id"]], category="meaning"
    )
    result = generate(tmp_path, packet, [forged])
    assert result["manifest"]["status"] == "failed"
    assert (result["preparation_kind"], result["support_status"]) == (
        "factual_fallback",
        "fallback",
    )
    GeneratedExplanationResponse.model_validate(result)
    result["preparation_kind"] = "grounded_description"
    result["support_status"] = "typed_template_support"
    with pytest.raises(ValidationError, match="supported selected claims"):
        GeneratedExplanationResponse.model_validate(result)


def test_one_sided_label_is_not_promoted_to_comparative_description(tmp_path):
    context, policy = prepare(
        tmp_path,
        b"Ontology(Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>)) "
        b'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Only a label"))',
    )
    a = packet_for(context, policy, "urn:A", "labels")
    b = packet_for(context, policy, "urn:B", "labels")
    profiles = [generate(tmp_path, p, []) for p in (a, b)]
    packet = comparison_packet(a, b, profiles)
    assert comparison_templates(packet) == []
    legacy = packet.model_copy(update={"selection": {}})
    assert len(comparison_templates(legacy)) == 1
    result = generate(tmp_path, packet, [])
    assert (result["preparation_kind"], result["support_status"]) == (
        "fact_extract",
        "exact_support",
    )


def test_actual_http_and_workspace_reject_modern_labels_on_legacy_artifact(tmp_path):
    from fastapi.testclient import TestClient

    from exact_inspect.artifacts import atomic_json, publish_bundle
    from exact_inspect.contracts import file_hash
    from exact_inspect.service import create_prepared_app
    from exact_inspect.study.workspace import _admit_explanation

    context, policy = prepare(
        tmp_path,
        b"Ontology(Declaration(Class(<urn:A>)) "
        b'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Only a label"))',
    )
    packet = packet_for(context, policy, "urn:A", "labels")
    claims = [
        Claim(text="Only a label", fact_ids=[packet.facts[0]["fact_id"]], category="key_fact")
    ]
    legacy = generate(tmp_path, packet.model_copy(update={"selection": {}}), claims)
    path = tmp_path / "prepared.json"
    endpoint = "/api/v1/explanations/" + legacy["explanation_id"]

    def app():
        package = publish_bundle(
            tmp_path,
            audience="local",
            policy=policy,
            ontologies={context.ontology_version_id: "context"},
            explanations={legacy["explanation_id"]: "prepared.json"},
        )
        return create_prepared_app(package)

    atomic_json(path, legacy)
    frozen = file_hash(path)
    with TestClient(app()) as client:
        result = client.get(endpoint)
        assert result.status_code == 200, result.text
        assert result.json()["preparation_kind"] == "fact_extract"
        assert result.json()["support_status"] == "exact_support"
    assert file_hash(path) == frozen
    forged = {
        **legacy,
        "preparation_kind": "grounded_description",
        "support_status": "typed_template_support",
    }
    atomic_json(path, forged)
    with TestClient(app()) as client:
        result = client.get(endpoint)
        assert result.status_code == 409, result.text
        assert result.json()["code"] == "explanation_invalid_support"
        assert "Only a label" not in result.text
    case = {"source": packet.entities[0].model_dump(), "candidates": []}
    with pytest.raises(ValueError, match="versioned support packet"):
        _admit_explanation(forged, case, policy, {context.ontology_version_id: context})
