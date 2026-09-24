"""Participant resource admission checks original syntax, provenance and policy recursively."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pyowl_core as core
import pytest

from exact_inspect.context_semantics import _category, _iri, typed_node
from exact_inspect.study.owl_ast import validate_axiom
from exact_inspect.study.resources import (
    ExplanationResource,
    validate_explanation_resource,
)
from tests.explanation_study_test import publication


@pytest.fixture
def resource(tmp_path):
    definition = publication(tmp_path).definition.model_dump(mode="json")
    return json.loads((tmp_path / "context").read_bytes()), definition


def ast(annotations=""):
    snapshot = core.load_snapshot(
        (f"Ontology(SubClassOf({annotations} <urn:target:1> <urn:target:2>))").encode(),
        options=core.LoadOptions(imports=core.ImportPolicy.IGNORE),
    )
    return typed_node(next(snapshot.iter_axioms()))


def with_expression(payload, expression):
    payload = copy.deepcopy(payload)
    fact = payload["facts"][0]
    fact["category"] = _category(expression)
    fact["predicate_iri"] = (
        _iri(expression.get("property")) if expression["type"] == "AnnotationAssertion" else None
    )
    fact["value"] = {
        "term_type": "expression_ref",
        "expression_id": fact["axiom_ref"],
        "ast": expression,
    }
    return payload


@pytest.mark.parametrize(
    "violation",
    [
        "nested_xref",
        "category_spoof",
        "predicate_spoof",
        "missing_predicate",
        "extra_ast",
        "extra_nested_ast",
        "wrong_constructor_type",
        "extra_origin",
        "extra_span",
        "reversed_span",
        "original_syntax",
    ],
)
def test_nested_admission_rejects_prohibited_or_untyped_content(resource, violation):
    payload, definition = resource
    if violation in {
        "nested_xref",
        "category_spoof",
        "extra_ast",
        "extra_nested_ast",
        "wrong_constructor_type",
    }:
        expression = ast(
            'Annotation(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> "forbidden-target")'
            if violation == "nested_xref"
            else ""
        )
        payload = with_expression(payload, expression)
    fact = payload["facts"][0]
    if violation == "category_spoof":
        fact["category"] = "definitions"
    elif violation == "predicate_spoof":
        fact["predicate_iri"] = "http://www.geneontology.org/formats/oboInOwl#hasDbXref"
    elif violation == "missing_predicate":
        fact["predicate_iri"] = None
    elif violation == "extra_ast":
        fact["value"]["ast"]["answer_key"] = "private"
    elif violation == "extra_nested_ast":
        fact["value"]["ast"]["sub_class"]["iri"]["answer_key"] = "private"
    elif violation == "wrong_constructor_type":
        fact["value"]["ast"]["sub_class"] = {"type": "IRI", "value": "urn:target:1"}
    elif violation == "extra_origin":
        fact["origins"][0]["answer_key"] = "private"
    elif violation == "extra_span":
        fact["origins"][0]["source_span"] = {
            "start": 0,
            "end": 1,
            "unit": "byte",
            "answer_key": "private",
        }
    elif violation == "reversed_span":
        fact["origins"][0]["source_span"] = {"start": 2, "end": 1, "unit": "byte"}
    elif violation == "original_syntax":
        fact["value"] = {
            "term_type": "original_syntax",
            "syntax": 'AnnotationAssertion(<urn:xref> <urn:A> "private")',
            "syntax_format": "functional",
        }
    with pytest.raises(ValueError):
        validate_explanation_resource(json.dumps(payload), definition)


def test_valid_structural_original_and_typed_provenance_remain_available(resource):
    payload, definition = resource
    payload = with_expression(payload, ast())
    result = validate_explanation_resource(json.dumps(payload), definition)
    assert result.facts[0].value.ast == payload["facts"][0]["value"]["ast"]
    assert result.facts[0].origins[0].document_sha256.startswith("sha256:")


def test_frozen_ast_registry_matches_public_pyowl_contract_and_native_constructors():
    import exact_inspect.study.owl_ast as admission
    from tests.explanation_context_test import OWL

    root = Path(__file__).resolve().parents[1]
    expected = runpy.run_path(str(root / "tools/generate_study_ast_schema.py"))["schema"]()
    assert admission._SCHEMA == expected
    snapshot = core.load_snapshot(OWL, options=core.LoadOptions(imports=core.ImportPolicy.IGNORE))
    for axiom in snapshot.iter_axioms():
        validate_axiom(typed_node(axiom))


def test_semantic_root_category_cannot_hide_direct_mapping_annotation(resource):
    payload, definition = resource
    snapshot = core.load_snapshot(
        b'Ontology(AnnotationAssertion(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> <urn:target:1> "forbidden-target"))'
    )
    payload = with_expression(payload, typed_node(next(snapshot.iter_axioms())))
    payload["facts"][0]["category"] = "definitions"
    payload["facts"][0]["predicate_iri"] = "http://purl.obolibrary.org/obo/IAO_0000115"
    with pytest.raises(ValueError, match="category"):
        ExplanationResource.model_validate(payload)


def test_study_projection_preserves_allowed_qualifiers_and_withholds_nested_mappings(tmp_path):
    from exact_inspect.context import build_context_package
    from exact_inspect.contracts import EntityRef, VisibilityPolicy
    from exact_inspect.study.builder import build_explanation_resource

    source = b"""Ontology(<urn:qualified-study>
Declaration(Class(<urn:A>))
AnnotationAssertion(Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P378> "Recorded definition source") Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P207> "forbidden-direct") <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P97> <urn:A> "Exact qualified definition")
AnnotationAssertion(Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P383> "PT") Annotation(Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P207> "forbidden-nested") <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P384> "NCI") <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P90> <urn:A> "Exact qualified synonym")
)"""
    snapshot = core.load_snapshot(source)
    context = build_context_package(snapshot, tmp_path / "qualified")
    policy = VisibilityPolicy()
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri="urn:A", kind="class")
    resource = build_explanation_resource(
        {context.ontology_version_id: context}, [entity], policy=policy
    )
    study = {
        "policy_hash": policy.policy_hash.removeprefix("sha256:"),
        "visibility_policy": policy.model_dump(mode="json"),
    }
    admitted = validate_explanation_resource(resource.model_dump_json(), study)
    assert "forbidden-" not in admitted.model_dump_json()
    for fact in admitted.facts:
        original = context.axiom(fact.axiom_ref, policy=policy)
        assert fact.qualifiers == original["ast"]["annotations"]
    definition = next(fact for fact in admitted.facts if fact.category == "definitions")
    synonym = next(fact for fact in admitted.facts if fact.category == "synonyms")
    assert definition.qualifiers[0]["value"]["lexical_form"] == "Recorded definition source"
    assert {q["value"]["lexical_form"] for q in synonym.qualifiers} == {"PT", "NCI"}
    assert ExplanationResource.model_validate_json(admitted.model_dump_json()) == admitted

    unrestricted = VisibilityPolicy(
        categories=(*policy.categories, "xrefs"), allow_mapping_xrefs=True
    )
    unfiltered = context.axiom(synonym.axiom_ref, policy=unrestricted)["ast"]["annotations"]
    forged = json.loads(admitted.model_dump_json())
    next(fact for fact in forged["facts"] if fact["category"] == "synonyms")[
        "qualifiers"
    ] = unfiltered
    with pytest.raises(ValueError, match="qualifier"):
        validate_explanation_resource(json.dumps(forged), study)
    forged = json.loads(admitted.model_dump_json())
    forged["facts"][0]["qualifiers"][0]["answer_key"] = "private"
    with pytest.raises(ValueError, match="extra fields"):
        validate_explanation_resource(json.dumps(forged), study)
