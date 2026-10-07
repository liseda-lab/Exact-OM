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


def comparison_resource(tmp_path, *, shared_pair=False):
    from exact_inspect.context import build_context_package
    from exact_inspect.contracts import EntityRef, VisibilityPolicy, canonical_hash
    from exact_inspect.generation import FactPacket, comparison_templates
    from exact_inspect.study.builder import build_explanation_resource

    context = build_context_package(
        core.load_snapshot(
            b"""Ontology(
Declaration(Class(<urn:source>)) Declaration(Class(<urn:child>)) Declaration(Class(<urn:parent>))
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:source> "source")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:child> "child")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:parent> "parent")
SubClassOf(<urn:child> <urn:parent>)
)"""
        ),
        tmp_path / "shared-context",
    )
    policy = VisibilityPolicy()
    entities = [
        EntityRef(ontology_version_id=context.ontology_version_id, iri=iri, kind="class")
        for iri in ("urn:source", "urn:child", "urn:parent")
    ]
    packets, outputs = [], []
    pairs = (
        [(entities[1], entities[2])]
        if shared_pair
        else [(entities[0], entities[1]), (entities[0], entities[2])]
    )
    for pair in pairs:
        packet = FactPacket(
            task="pair_comparison",
            entities=list(pair),
            context_hashes=[context.manifest["artifacts"]["context.sqlite"]],
            policy_hash=policy.policy_hash,
            facts=[
                fact
                for entity in pair
                for fact in context.facts(entity, limit=100, policy=policy)["items"]
            ],
            missingness=[],
            selection={},
        )
        packets.append(packet)
        outputs.append(
            {
                "manifest": {
                    "packet_hash": canonical_hash(packet),
                    "visibility_policy_hash": policy.policy_hash,
                    "status": "validated",
                },
                "entities": [entity.model_dump() for entity in pair],
                "grounding_status": "validated",
                "claims": [
                    {
                        **claim.model_dump(),
                        "claim_id": canonical_hash(
                            {
                                "packet": canonical_hash(packet),
                                "claim": claim.model_dump(exclude={"claim_id"}),
                            }
                        ),
                    }
                    for claim in comparison_templates(packet)
                ],
            }
        )
    resource = build_explanation_resource(
        {context.ontology_version_id: context},
        entities,
        policy=policy,
        packets=packets,
        explanations=outputs,
        evidence={
            "candidate-child": [
                {
                    "evidence_id": "saved-shared-hierarchy",
                    "entity": entities[1].model_dump(),
                    "fact_ids": [
                        next(
                            fact["fact_id"]
                            for fact in context.facts(
                                entities[1], category="hierarchy", policy=policy
                            )["items"]
                        )
                    ],
                    "channel": "hierarchy",
                    "side": "target",
                    "values": {"support": 0.42},
                }
            ]
        },
    )
    return resource, policy


@pytest.mark.parametrize("shared_pair", [False, True])
def test_shared_axiom_preserves_each_subject_and_comparison_packet(tmp_path, shared_pair):
    resource, policy = comparison_resource(tmp_path, shared_pair=shared_pair)
    shared = [fact for fact in resource.facts if fact.category == "hierarchy"]
    assert len(shared) == 2 and shared[0].fact_id == shared[1].fact_id
    assert {fact.subject.iri for fact in shared} == {"urn:child", "urn:parent"}
    assert all(claim.packet_fact_subjects for claim in resource.pair_comparison)
    admitted = validate_explanation_resource(
        resource.model_dump_json(),
        {
            "policy_hash": policy.policy_hash.removeprefix("sha256:"),
            "visibility_policy": policy.model_dump(mode="json"),
        },
    )
    assert admitted == resource
    assert resource.evidence[0].scores[0].value == 0.42
    assert len(resource.pair_comparison) == (1 if shared_pair else 2)


@pytest.mark.parametrize(
    "violation",
    [
        "duplicate_subject",
        "conflicting_original",
        "packet_subject",
        "unaligned_packet",
        "ambiguous_legacy",
    ],
)
def test_shared_axiom_rejects_ambiguous_or_conflicting_provenance(tmp_path, violation):
    resource, _ = comparison_resource(tmp_path, shared_pair=True)
    payload = resource.model_dump(mode="json")
    shared = [fact for fact in payload["facts"] if fact["category"] == "hierarchy"]
    claim = payload["pair_comparison"][0]
    if violation == "duplicate_subject":
        payload["facts"].append(copy.deepcopy(shared[0]))
    elif violation == "conflicting_original":
        shared[1]["value"]["ast"]["super_class"]["iri"]["value"] = "urn:forged"
    elif violation == "packet_subject":
        index = claim["packet_fact_ids"].index(shared[0]["fact_id"])
        claim["packet_fact_subjects"][index] = payload["entities"][0]
    elif violation == "unaligned_packet":
        claim["packet_fact_subjects"].pop()
    else:
        claim.pop("packet_fact_subjects")
    with pytest.raises(ValueError):
        ExplanationResource.model_validate(payload)


def test_legacy_unambiguous_comparisons_remain_readable(tmp_path):
    resource, _ = comparison_resource(tmp_path)
    payload = resource.model_dump(mode="json")
    for claim in payload["pair_comparison"]:
        claim.pop("packet_fact_subjects")
    assert len(ExplanationResource.model_validate(payload).pair_comparison) == 2


def test_referenced_labels_preserve_originals_without_expanding_case_scope(tmp_path):
    from exact_inspect.context import build_context_package
    from exact_inspect.contracts import EntityRef, VisibilityPolicy
    from exact_inspect.study.builder import build_explanation_resource

    context = build_context_package(
        core.load_snapshot(
            b"""Ontology(
Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>)) Declaration(Class(<urn:C>)) Declaration(Class(<urn:unrelated>)) Declaration(ObjectProperty(<urn:property>))
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Focal class")
AnnotationAssertion(Annotation(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> "forbidden-label-qualifier") <http://www.w3.org/2000/01/rdf-schema#label> <urn:property> "has mechanism")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:B> "First filler"@en)
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:C> "Second filler")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:unrelated> "Unrelated label")
AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:B> "Do not fetch neighbor definitions")
SubClassOf(<urn:A> ObjectSomeValuesFrom(<urn:property> <urn:B>))
EquivalentClasses(<urn:A> ObjectIntersectionOf(<urn:B> ObjectAllValuesFrom(<urn:property> <urn:C>)))
)"""
        ),
        tmp_path / "referenced-context",
    )
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri="urn:A", kind="class")
    policy = VisibilityPolicy()
    resource = build_explanation_resource(
        {context.ontology_version_id: context}, [entity], policy=policy
    )
    labels = {fact.subject.iri: fact for fact in resource.referenced_labels}
    assert set(labels) == {"urn:property", "urn:B", "urn:C"}
    assert labels["urn:property"].value.lexical_form == "has mechanism"
    assert labels["urn:B"].value.language == "en"
    assert not labels["urn:property"].qualifiers
    assert all(fact.origins and fact.origins[0].document_sha256 for fact in labels.values())
    assert {fact.subject.iri for fact in resource.facts} == {"urn:A"}
    assert [ref.iri for ref in resource.entities] == ["urn:A"]
    assert not resource.referenced_labels_truncated
    assert "forbidden-label-qualifier" not in resource.model_dump_json()
    assert "Unrelated label" not in resource.model_dump_json()
    assert "Do not fetch neighbor definitions" not in resource.model_dump_json()
    study = {
        "policy_hash": policy.policy_hash.removeprefix("sha256:"),
        "visibility_policy": policy.model_dump(mode="json"),
    }
    assert validate_explanation_resource(resource.model_dump_json(), study) == resource

    unrestricted = VisibilityPolicy(
        categories=(*policy.categories, "xrefs"), allow_mapping_xrefs=True
    )
    property_label = labels["urn:property"]
    forged = resource.model_dump(mode="json")
    next(
        label for label in forged["referenced_labels"] if label["subject"]["iri"] == "urn:property"
    )["qualifiers"] = context.axiom(property_label.axiom_ref, policy=unrestricted)["ast"][
        "annotations"
    ]
    with pytest.raises(ValueError, match="qualifier"):
        validate_explanation_resource(json.dumps(forged), study)

    forged = resource.model_dump(mode="json")
    forged["referenced_labels"][0]["subject"]["iri"] = "urn:unrelated"
    with pytest.raises(ValueError, match="outside the admitted"):
        validate_explanation_resource(json.dumps(forged), study)
    forged = resource.model_dump(mode="json")
    forged["referenced_labels"][0]["category"] = "definitions"
    forged["referenced_labels"][0]["predicate_iri"] = "http://purl.obolibrary.org/obo/IAO_0000115"
    with pytest.raises(ValueError, match="original literal label"):
        validate_explanation_resource(json.dumps(forged), study)

    hidden = policy.model_copy(
        update={"categories": tuple(c for c in policy.categories if c != "labels")}
    )
    hidden_resource = build_explanation_resource(
        {context.ontology_version_id: context}, [entity], policy=hidden
    )
    assert not hidden_resource.referenced_labels
    assert "has mechanism" not in hidden_resource.model_dump_json()
    forged = resource.model_dump(mode="json")
    forged["facts"] = [fact for fact in forged["facts"] if fact["category"] != "labels"]
    forged["policy_hash"] = hidden.policy_hash.removeprefix("sha256:")
    with pytest.raises(ValueError, match="prohibited"):
        validate_explanation_resource(
            json.dumps(forged),
            {
                "policy_hash": forged["policy_hash"],
                "visibility_policy": hidden.model_dump(mode="json"),
            },
        )


def test_referenced_labels_have_count_and_byte_bounds(tmp_path):
    from exact_inspect.context import build_context_package
    from exact_inspect.contracts import EntityRef, VisibilityPolicy, canonical_json
    from exact_inspect.study.builder import build_explanation_resource
    from exact_inspect.study.resources import (
        REFERENCED_LABEL_BYTES,
        REFERENCED_LABEL_LIMIT,
    )

    iris = [f"urn:filler:{index}" for index in range(105)]
    source = (
        "Ontology(Declaration(Class(<urn:A>)) "
        + " ".join(
            f'Declaration(Class(<{iri}>)) AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <{iri}> "'
            + "long label " * 150
            + '")'
            for iri in iris
        )
        + " SubClassOf(<urn:A> ObjectIntersectionOf("
        + " ".join(f"<{iri}>" for iri in iris)
        + ")))"
    )
    context = build_context_package(
        core.load_snapshot(source.encode()), tmp_path / "bounded-labels"
    )
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri="urn:A", kind="class")
    resource = build_explanation_resource(
        {context.ontology_version_id: context}, [entity], policy=VisibilityPolicy()
    )
    assert 0 < len(resource.referenced_labels) <= REFERENCED_LABEL_LIMIT
    assert (
        len(canonical_json([fact.model_dump(mode="json") for fact in resource.referenced_labels]))
        <= REFERENCED_LABEL_BYTES
    )
    assert resource.referenced_labels_truncated
    assert "Referenced term labels are bounded" in " ".join(resource.limitations)
    oversized = resource.model_dump(mode="json")
    oversized["referenced_labels"][0]["value"]["lexical_form"] = "x" * REFERENCED_LABEL_BYTES
    with pytest.raises(ValueError, match="byte budget"):
        ExplanationResource.model_validate(oversized)


@pytest.mark.parametrize(
    "violation",
    [
        "incomplete_lessons",
        "duplicate_lesson",
        "duplicate_candidate",
        "scored_case_id",
        "scored_candidate_id",
        "score",
        "answer_key",
        "real_content_flag",
        "oversized_description",
    ],
)
def test_frozen_practice_rejects_unbounded_or_scored_inputs(resource, violation):
    from exact_inspect.study.models import StudyDefinition
    from tools.serve_explanation_ui_study import PRACTICE_CASES

    _, definition = resource
    definition["practice_cases"] = copy.deepcopy(PRACTICE_CASES)
    exercises = definition["practice_cases"]
    if violation == "incomplete_lessons":
        exercises.pop()
    elif violation == "duplicate_lesson":
        exercises[1]["practice_id"] = exercises[0]["practice_id"]
    elif violation == "duplicate_candidate":
        exercises[0]["candidates"][1]["candidate_id"] = exercises[0]["candidates"][0][
            "candidate_id"
        ]
    elif violation == "scored_case_id":
        exercises[0]["practice_id"] = definition["cases"][0]["case_id"]
    elif violation == "scored_candidate_id":
        exercises[0]["candidates"][0]["candidate_id"] = definition["cases"][0]["candidates"][0][
            "candidate_id"
        ]
    elif violation == "score":
        exercises[0]["candidates"][0]["score"] = 0.9
    elif violation == "answer_key":
        exercises[0]["acceptable_candidate_ids"] = [exercises[0]["candidates"][0]["candidate_id"]]
    elif violation == "real_content_flag":
        exercises[0]["synthetic"] = False
    else:
        exercises[0]["source"]["description"] = "x" * 4001
    with pytest.raises(ValueError):
        StudyDefinition.model_validate(definition)


def test_practice_content_is_frozen_public_state_and_legacy_hashes_stay_stable(tmp_path):
    from exact_inspect.study.models import Publish, StudyState
    from exact_inspect.study.store import StudyError, StudyStore
    from tools.serve_explanation_ui_study import PRACTICE_CASES

    legacy = publication(tmp_path)
    store = StudyStore(
        f"sqlite:///{tmp_path / 'practice.sqlite'}", tmp_path, allow_test_sqlite=True
    )
    first = store.publish(legacy)
    with store.transaction() as db:
        row = db.execute(
            "SELECT payload FROM studies WHERE revision = ?", (legacy.definition.study_revision,)
        ).fetchone()
        assert "practice_cases" not in json.loads(row["payload"])
    assert store.publish(legacy) == first
    token = store.issue(legacy.definition.study_revision)["invitations"][0]["invitation"].split(
        "=", 1
    )[1]
    _, _, state = store.exchange(token)
    assert StudyState.model_validate(state).practice_cases == []

    payload = legacy.model_dump(mode="json")
    payload["definition"]["study_revision"] += "-practice"
    payload["definition"]["practice_cases"] = copy.deepcopy(PRACTICE_CASES)
    frozen = Publish.model_validate(payload)
    store.publish(frozen)
    token = store.issue(frozen.definition.study_revision)["invitations"][0]["invitation"].split(
        "=", 1
    )[1]
    _, _, state = store.exchange(token)
    assert state["practice_cases"] == PRACTICE_CASES
    assert len(StudyState.model_validate(state).practice_cases) == 4
    assert "case_keys" not in state and "acceptable_candidate_ids" not in json.dumps(
        state["practice_cases"]
    )
    payload["definition"]["practice_cases"][0]["instructions"] = "Changed content"
    with pytest.raises(StudyError, match="immutable"):
        store.publish(Publish.model_validate(payload))


def test_claim_identity_cannot_collapse_different_scopes_but_exact_repeats_remain_valid(tmp_path):
    resource, _ = comparison_resource(tmp_path)
    original = resource.model_dump(mode="json")
    assert len(original["pair_comparison"]) >= 2
    duplicate = copy.deepcopy(original)
    duplicate["pair_comparison"].append(copy.deepcopy(duplicate["pair_comparison"][0]))
    ExplanationResource.model_validate(duplicate)
    conflicting = copy.deepcopy(original)
    conflicting["pair_comparison"][1]["claim_id"] = conflicting["pair_comparison"][0]["claim_id"]
    with pytest.raises(ValueError, match="Conflicting claim identity"):
        ExplanationResource.model_validate(conflicting)
