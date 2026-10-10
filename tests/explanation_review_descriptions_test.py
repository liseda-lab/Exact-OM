"""The new description boundary verifies meaning, not merely citation membership."""

import copy

import pyowl_core as core
import pytest

from exact_inspect.context import build_context_package
from exact_inspect.contracts import EntityRef, VisibilityPolicy
from exact_inspect.descriptions import (
    description_templates,
    prepare_description_packet,
    preparation_metadata,
)
from exact_inspect.generation import ExplanationOutput, entity_packet, grounding


@pytest.fixture
def packet(tmp_path):
    context = build_context_package(
        core.load_snapshot(b"""Ontology(
      Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
      Declaration(ObjectProperty(<urn:p>))
      AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "A label")
      SubClassOf(<urn:A> ObjectAllValuesFrom(<urn:p> <urn:B>))
      SubClassOf(<urn:A> ObjectSomeValuesFrom(<urn:p> <urn:B>))
      SubClassOf(<urn:A> ObjectMinCardinality(2 <urn:p> <urn:B>))
    )"""),
        tmp_path / "context",
    )
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri="urn:A", kind="class")
    policy = VisibilityPolicy(ontology_ids=(context.ontology_version_id,))
    facts = context.facts(entity, category="restrictions", policy=policy)["items"]
    base = entity_packet(
        entity, facts, context_hash=context.manifest["artifacts"]["context.sqlite"], policy=policy
    )
    return prepare_description_packet(base, context, policy)


def test_missing_definition_gets_necessary_scope_without_equivalence(packet):
    templates = description_templates(packet)
    assert len(templates) == 3
    text = " ".join(c.text for c in templates)
    assert "without requiring any such value" in text
    assert "at least one" in text and "at least 2 distinct" in text
    assert "necessary condition" in text and "equivalence" not in text
    assert grounding(ExplanationOutput(claims=templates), packet)[0] == "validated"
    forged = templates[0].model_copy(update={"text": "This entity is equivalent to urn:B."})
    assert grounding(ExplanationOutput(claims=[forged]), packet)[0] == "unverified"


def test_description_contract_does_not_change_legacy_support(packet):
    legacy = packet.model_copy(update={"selection": {}})
    assert description_templates(legacy) == []
    assert (
        grounding(ExplanationOutput(claims=description_templates(packet)), legacy)[0]
        == "unverified"
    )
    value = {"manifest": {"status": "validated"}, "claims": []}
    before = copy.deepcopy(value)
    assert preparation_metadata(value)["preparation_kind"] == "fact_extract"
    assert value == before


def test_label_alone_is_not_a_description(packet):
    empty = packet.model_copy(update={"facts": []})
    assert description_templates(empty) == []


def test_punned_property_cannot_receive_a_class_necessary_condition(packet):
    facts = copy.deepcopy(packet.facts)
    for fact in facts:
        fact["subject"]["kind"] = "object_property"
    assert description_templates(packet.model_copy(update={"facts": facts})) == []


def test_prepared_description_is_revalidated_at_actual_api(packet, tmp_path):
    from fastapi.testclient import TestClient
    from exact_inspect.artifacts import atomic_json, publish_bundle
    from exact_inspect.context import OntologyContext
    from exact_inspect.generation import ExplanationJobs, GenerationProfile
    from exact_inspect.service import create_prepared_app as create_app
    from exact_inspect.study.builder import build_explanation_resource
    from exact_inspect.descriptions import validate_prepared_support

    context = OntologyContext(tmp_path / "context")
    policy = VisibilityPolicy(ontology_ids=(context.ontology_version_id,))
    output = ExplanationOutput(claims=description_templates(packet))
    calls = []

    def provider(messages, profile):
        calls.append(messages)
        return {
            "choices": [{"message": {"content": output.model_dump_json()}}],
            "model": "test-provider",
        }

    jobs = ExplanationJobs(tmp_path / "jobs")
    result = jobs.generate(packet, GenerationProfile(name="test", model="test"), provider=provider)
    assert result["preparation_kind"] == "grounded_description"
    assert len(calls) == 1
    assert (
        jobs.generate(packet, GenerationProfile(name="test", model="test"), provider=provider)
        == result
    )
    assert len(calls) == 1
    validate_prepared_support(result, {context.ontology_version_id: context}, policy)
    resource = build_explanation_resource(
        {context.ontology_version_id: context},
        packet.entities,
        policy=policy,
        packets=[packet],
        explanations=[result],
    )
    assert all(c.grounding == "typed_description" for c in resource.entity_profiles)
    atomic_json(tmp_path / "prepared.json", result)

    def bundle():
        return publish_bundle(
            tmp_path,
            audience="local",
            policy=policy,
            ontologies={context.ontology_version_id: "context"},
            explanations={result["explanation_id"]: "prepared.json"},
        )

    with TestClient(create_app(package=bundle())) as client:
        response = client.get("/api/v1/explanations/" + result["explanation_id"])
        assert response.status_code == 200, response.text
        assert response.json()["preparation_kind"] == "grounded_description"
    result["claims"][0]["text"] = "This entity is equivalent to urn:B."
    atomic_json(tmp_path / "prepared.json", result)
    with TestClient(create_app(package=bundle())) as client:
        response = client.get("/api/v1/explanations/" + result["explanation_id"])
        assert response.status_code == 409, response.text
        assert "equivalent to urn:B" not in response.text
