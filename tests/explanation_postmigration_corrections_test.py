"""Synthetic boundary regressions; real publication verification is recorded separately."""

import copy
import json
from pathlib import Path

import pyowl_core as core
import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import atomic_json, publish_bundle
from exact_inspect.context import OntologyContext, build_context_package
from exact_inspect.context_export import export_policy_context
from exact_inspect.context_resources import export_ontology_resource
from exact_inspect.contracts import (
    DomainError,
    EntityRef,
    VisibilityPolicy,
    canonical_hash,
    file_hash,
)
from exact_inspect.navigation import (
    PARTIAL_BASIS,
    EntryPointPage,
    PreparedHierarchyPage,
)
from exact_inspect.navigation_prepare import prepare_navigation
from exact_inspect.partial_elk import SEMANTICS, VERSIONS, admit, materialize
from exact_inspect.service import create_prepared_app
from tools.compose_review_publication import merge_evidence

OWL = b"""Ontology(Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
Declaration(Class(<urn:C>)) Declaration(Class(<urn:E>)) Declaration(Class(<urn:D>))
Declaration(Class(<urn:LabelOnly>)) Declaration(ObjectProperty(<urn:p>))
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Child")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:B> "Named parent")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:C> "Other parent")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:p> "has part")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:LabelOnly> "Only a label")
SubClassOf(<urn:A> <urn:B>) SubClassOf(<urn:A> <urn:C>)
SubClassOf(<urn:A> ObjectAllValuesFrom(<urn:p> <urn:B>))
SubClassOf(<urn:A> ObjectMinCardinality(2 <urn:p> <urn:B>))
EquivalentClasses(<urn:C> <urn:E>) SubClassOf(<urn:B> <urn:D>) SubClassOf(<urn:C> <urn:D>)
)"""
TOP = "http://www.w3.org/2002/07/owl#Thing"
BOTTOM = "http://www.w3.org/2002/07/owl#Nothing"


@pytest.fixture
def fixture(tmp_path):
    source = build_context_package(core.load_snapshot(OWL), tmp_path / "original")
    policy = VisibilityPolicy()
    filtered = export_policy_context(source, tmp_path / "filtered", policy)
    context = prepare_navigation(filtered, tmp_path / "ready", policy=policy, infer=False)
    resource = tmp_path / "admitted.ofn"
    receipt = export_ontology_resource(context, resource, policy)
    node = [[TOP], [BOTTOM], ["urn:A"], ["urn:B"], ["urn:C", "urn:E"], ["urn:D"], ["urn:LabelOnly"]]
    data = {
        "schema": "exact-cluster-elk-taxonomy/1",
        "semantics": SEMANTICS,
        "complete": False,
        "input_sha256": file_hash(resource),
        "policy_hash": policy.policy_hash,
        "ontology_version_id": context.ontology_version_id,
        "scope": context.manifest["scope"],
        "nodes": node,
        "direct_edges": [
            [node[2], node[3]],
            [node[2], node[4]],
            [node[3], node[5]],
            [node[4], node[5]],
            [node[5], node[0]],
            [node[6], node[0]],
            [node[1], node[2]],
        ],
        "top": node[0],
        "bottom": node[1],
    }
    tax = tmp_path / "taxonomy.json"
    atomic_json(tax, data)
    item = {
        "name": "fixture",
        "source": "admitted.ofn",
        "receipt": receipt.name,
        "source_sha256": file_hash(resource),
        "receipt_sha256": file_hash(receipt),
        "scope": context.manifest["scope"],
        "policy_hash": policy.policy_hash,
        "ontology_version_id": context.ontology_version_id,
    }
    manifest = {
        "schema": "exact-cluster-elk-inputs/1",
        "versions": VERSIONS,
        "inputs": [item],
        "options": {"workers": 1},
        "job_files": {"tools/cluster_elk_taxonomy.py": "sha256:" + "a" * 64},
    }
    mp = tmp_path / "manifest.json"
    atomic_json(mp, manifest)
    reasons = [
        {
            "constructors": ["ObjectAllValuesFrom"],
            "features": ["OBJECT_ALL_VALUES_FROM"],
            "polarities": ["POSITIVE"],
            "task": "class_taxonomy",
        }
    ]
    result = {
        "schema": "exact-cluster-elk-result/1",
        "versions": VERSIONS,
        "input": item,
        "manifest_sha256": file_hash(mp),
        "runner_sha256": manifest["job_files"]["tools/cluster_elk_taxonomy.py"],
        "status": "completed_incomplete",
        "eligible_for_integration_review": False,
        "parse": {"complete": True, "backend": "native", "documents": 1},
        "backend": {
            "name": "rust",
            "native_available": True,
            "implementation_version": "0.2.1",
            "fallback_reason": None,
        },
        "options": {
            "workers": 1,
            "backend": "rust",
            "require_native_pipeline": True,
            "unsupported": "ignore_with_explicit_incompleteness",
        },
        "consistency": {"value": True, "complete": False, "reasons": reasons},
        "classification": {"complete": False, "reasons": reasons},
        "phases": [
            {"phase": k, "status": "complete"}
            for k in [
                "native_parse",
                "native_admission",
                "consistency",
                "classification",
                "taxonomy_export",
            ]
        ],
        "taxonomy": {
            "sha256": file_hash(tax),
            "nodes": len(node),
            "direct_edges": len(data["direct_edges"]),
        },
        "source_documents": json.loads(receipt.read_text())["source_documents"],
    }
    rp = tmp_path / "result.json"
    atomic_json(rp, result)
    args = {
        "policy": policy,
        "manifest_path": mp,
        "result_path": rp,
        "taxonomy_path": tax,
        "resource_receipt_path": receipt,
        "resource_path": resource,
        "expected": {
            "manifest": file_hash(mp),
            "result": file_hash(rp),
            "taxonomy": file_hash(tax),
            "resource_receipt": file_hash(receipt),
        },
    }
    return context, policy, args


def ent(context, iri="urn:A", kind="class"):
    return EntityRef(ontology_version_id=context.ontology_version_id, iri=iri, kind=kind)


def test_partial_contract_pages_equivalents_empty_and_old_basis(fixture, tmp_path):
    context, policy, args = fixture
    data, binding = admit(context, **args)
    context = materialize(context, policy=policy, data=data, binding=binding)
    assert (
        context.hierarchy(ent(context), basis="reasoner_inferred", policy=policy)["status"]
        == "not_run"
    )
    first = context.hierarchy(ent(context), basis=PARTIAL_BASIS, policy=policy, limit=1)
    PreparedHierarchyPage.model_validate(first)
    assert first["status"] == "partial" and first["next_cursor"]
    allrows = []
    page = first
    while True:
        allrows += page["items"]
        if not page["next_cursor"]:
            break
        page = context.hierarchy(
            ent(context), basis=PARTIAL_BASIS, policy=policy, limit=1, cursor=page["next_cursor"]
        )
    assert {e["parent"]["iri"] for e in allrows} == {"urn:B", "urn:C", "urn:E"}
    assert "urn:D" not in {e["parent"]["iri"] for e in allrows}
    with pytest.raises(DomainError):
        context.hierarchy(
            ent(context), basis="literal_asserted", policy=policy, cursor=first["next_cursor"]
        )
    eq = context.equivalent_entities(
        ent(context, "urn:C"), basis=PARTIAL_BASIS, policy=policy, limit=1
    )
    nxt = context.equivalent_entities(
        ent(context, "urn:C"), basis=PARTIAL_BASIS, policy=policy, limit=1, cursor=eq["next_cursor"]
    )
    assert {e["iri"] for e in eq["items"] + nxt["items"]} == {"urn:C", "urn:E"}
    root = context.hierarchy(ent(context, "urn:D"), basis=PARTIAL_BASIS, policy=policy)
    assert (
        root["status"] == "partial"
        and root["items"] == []
        and root["inference"]["consistency"]["complete"] is False
    )
    assert (
        context.hierarchy(
            ent(context, "urn:p", "object_property"), basis=PARTIAL_BASIS, policy=policy
        )["status"]
        == "unsupported"
    )
    EntryPointPage.model_validate(context.entry_points(basis=PARTIAL_BASIS, policy=policy, limit=1))
    package = publish_bundle(
        tmp_path, policy=policy, ontologies={context.ontology_version_id: "ready"}
    )
    with TestClient(create_prepared_app(package)) as client:
        q = {**ent(context).model_dump(), "basis": PARTIAL_BASIS, "limit": 1}
        response = client.get("/api/v1/hierarchy", params=q)
        assert response.status_code == 200 and response.json()["status"] == "partial"
        response = client.get(
            f"/api/v1/ontologies/{context.ontology_version_id}/equivalents",
            params={"iri": "urn:C", "basis": PARTIAL_BASIS, "limit": 1},
        )
        assert (
            response.status_code == 200
            and response.json()["navigation_contract"] == "exact-partial-hierarchy/1"
        )
    restricted = export_policy_context(
        context, tmp_path / "narrow", VisibilityPolicy(categories=("labels",))
    )
    assert (
        not restricted.manifest.get("partial_navigation")
        and not (restricted.path / "partial-navigation.sqlite").exists()
    )


@pytest.mark.parametrize(
    "change",
    [
        "checksum",
        "policy",
        "scope",
        "input_identity",
        "resource_tamper",
        "consistency",
        "timeout",
        "missing_endpoint",
        "duplicate_group",
        "cycle",
        "truncated",
    ],
)
def test_partial_admission_fails_closed(fixture, change):
    context, policy, args = fixture
    if change == "checksum":
        Path(args["taxonomy_path"]).write_bytes(b"{}")
    elif change == "resource_tamper":
        Path(args["resource_path"]).write_bytes(b"Ontology()")
    else:
        key = "result" if change in {"consistency", "timeout"} else "taxonomy"
        path = args[key + "_path"]
        d = json.loads(path.read_bytes())
        if change == "policy":
            d["policy_hash"] = "sha256:" + "0" * 64
        elif change == "scope":
            d["scope"] = "another_scope"
        elif change == "input_identity":
            d["input_sha256"] = "sha256:" + "0" * 64
        elif change == "consistency":
            d["consistency"]["value"] = False
        elif change == "timeout":
            d["phases"][-1]["status"] = "failed"
        elif change == "missing_endpoint":
            d["direct_edges"][0][1] = ["urn:unknown"]
        elif change == "duplicate_group":
            d["nodes"].append(d["nodes"][2])
        elif change == "cycle":
            d["direct_edges"].append([d["nodes"][3], d["nodes"][2]])
        elif change == "truncated":
            d["direct_edges"].pop()
        atomic_json(path, d)
        args["expected"][key] = file_hash(path)
        if key == "taxonomy":
            rp = args["result_path"]
            result = json.loads(rp.read_bytes())
            result["taxonomy"]["sha256"] = file_hash(path)
            atomic_json(rp, result)
            args["expected"]["result"] = file_hash(rp)
    with pytest.raises((ValueError, KeyError)):
        admit(context, **args)


def test_readable_necessary_conditions_citations_and_frozen_resource(fixture):
    context, policy, _ = fixture
    from exact_inspect.descriptions import validate_prepared_support
    from exact_inspect.readable_descriptions import (
        enrich,
        presentation,
        profile_templates,
        revise,
    )
    from exact_inspect.study.builder import build_explanation_resource
    from tools.prepare_review_bundle import deterministic_result, packet_for

    contexts = {context.ontology_version_id: context}
    packet, _ = packet_for(ent(context), contexts, policy)
    updated = revise(deterministic_result(packet), contexts, policy)
    text = " ".join(c["text"] for c in updated["claims"])
    assert "Named parent" in text or "Other parent" in text
    assert "<urn:" not in text and updated["support_status"] == "typed_template_support"
    templates = profile_templates(enrich(packet, contexts, policy))
    alltext = " ".join(c.text for c in templates)
    assert "at least 2 distinct" in alltext and "without requiring any such value" in alltext
    resource = build_explanation_resource(
        contexts,
        [ent(context)],
        policy=policy,
        packets=[
            __import__(
                "exact_inspect.generation", fromlist=["FactPacket"]
            ).FactPacket.model_validate(updated["support_packet"])
        ],
        explanations=[updated],
    )
    assert set(c["claim_id"] for c in updated["claims"]) == {
        c.claim_id for c in resource.entity_profiles
    }
    corrupt = copy.deepcopy(updated)
    corrupt["support_packet"]["selection"]["referenced_labels"][0]["value"][
        "lexical_form"
    ] = "invented"
    with pytest.raises(ValueError):
        validate_prepared_support(corrupt, contexts, policy)
    label, _ = packet_for(ent(context, "urn:LabelOnly"), contexts, policy)
    fallback = revise(deterministic_result(label), contexts, policy)
    assert (
        fallback["preparation_kind"] == "factual_fallback"
        and presentation(fallback)["kind"] == "available_facts"
    )


def test_layer_composition_preserves_evidence_and_rejects_changed_scores(tmp_path):
    a = {
        "evidence_id": "e",
        "entity": {"iri": "a", "kind": "class"},
        "fact_ids": ["fact"],
        "feature_id": "feature",
        "values": {"weight": 0.2},
        "graph_projection": "not_recorded",
        "origin_resolution": "resolved",
    }
    b = {**a, "fact_ids": [], "feature_id": None, "origin_resolution": "unresolved"}
    assert merge_evidence([a], [b]) == [a]
    assert merge_evidence([b], [a]) == [a]
    with pytest.raises(ValueError):
        merge_evidence([a], [{**b, "values": {"weight": 0.3}}])
    from tools.review_run_cache import prepared_policy_run

    atomic_json(tmp_path / "decisions.manifest.json", {"literal_recovery": {"bound": True}})
    with pytest.raises(ValueError, match="discard enrichment"):
        prepared_policy_run(
            None,
            tmp_path,
            contexts={},
            policy=VisibilityPolicy(),
            checkpoint=tmp_path / "checkpoint",
            adopt_existing=True,
            exporter=None,
        )


def test_import_uses_canonical_recorded_qualifiers_without_guessing(tmp_path):
    from tools.review_private_run_cache import resolve_recorded_feature

    source = build_context_package(
        core.load_snapshot(
            b'Ontology(Declaration(Class(<urn:A>)) AnnotationAssertion(<urn:definition> <urn:A> "Meaning"@en) AnnotationAssertion(<urn:definition> <urn:A> "Ambiguous"@en) AnnotationAssertion(<urn:definition> <urn:A> "Ambiguous"@pt))'
        ),
        tmp_path / "context",
    )
    raw = {
        "entity_iri": "urn:A",
        "property_iri": "urn:definition",
        "value": "Meaning",
        "datatype": None,
        "language": None,
    }
    assert source.resolve_feature(raw, ent(source))["fact_ids"] == []  # former preparation boundary
    assert len(resolve_recorded_feature(source, raw, ent(source))["fact_ids"]) == 1
    assert (
        resolve_recorded_feature(source, {**raw, "language": "pt"}, ent(source))["fact_ids"] == []
    )
    assert (
        resolve_recorded_feature(source, {**raw, "value": "Ambiguous"}, ent(source))["fact_ids"]
        == []
    )
    assert (
        resolve_recorded_feature(source, {**raw, "entity_iri": "urn:Other"}, ent(source))[
            "fact_ids"
        ]
        == []
    )


def test_private_stage_resume_and_changed_source_fail_closed(tmp_path):
    from tools.compose_review_publication import private_copy, stage_tree

    source = tmp_path / "source"
    source.mkdir()
    atomic_json(source / "package.json", {"identity": "source"})
    (source / "context.sqlite").write_bytes(b"unchanged")
    target = tmp_path / "target"
    checkpoint = tmp_path / "staged.json"
    private_copy(source / "context.sqlite", target / "context.sqlite")
    before = (source / "context.sqlite").stat()
    stage_tree(source, target, checkpoint)
    stage_tree(source, target, checkpoint)
    assert before == (source / "context.sqlite").stat()
    assert (target / "context.sqlite").stat().st_nlink == 1
    atomic_json(source / "package.json", {"identity": "changed"})
    with pytest.raises(ValueError, match="identity changed"):
        stage_tree(source, target, checkpoint)


def test_partial_materialization_resume_published_and_scope_guard(fixture, tmp_path):
    context, policy, args = fixture
    data, binding = admit(context, **args)
    updated = materialize(context, policy=policy, data=data, binding=binding)
    digest = file_hash(updated.path / "manifest.json")
    assert materialize(updated, policy=policy, data=data, binding=binding) is updated
    assert file_hash(updated.path / "manifest.json") == digest
    stale = copy.deepcopy(binding)
    stale["inference"]["admission_identity"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="resume identity"):
        materialize(updated, policy=policy, data=data, binding=stale)
    context, policy, args = fixture
    altered = copy.deepcopy(args)
    altered["policy"] = VisibilityPolicy(categories=("labels",))
    with pytest.raises(ValueError):
        admit(context, **altered)


def test_frozen_description_presentation_cannot_invent_two_sided_scope(fixture):
    from exact_inspect.generation import FactPacket
    from exact_inspect.readable_descriptions import revise
    from exact_inspect.study.builder import build_explanation_resource
    from exact_inspect.study.resources import ExplanationResource
    from tools.prepare_review_bundle import deterministic_result, packet_for

    context, policy, _ = fixture
    contexts = {context.ontology_version_id: context}
    packet, _ = packet_for(ent(context), contexts, policy)
    result = revise(deterministic_result(packet), contexts, policy)
    resource = build_explanation_resource(
        contexts,
        [ent(context)],
        policy=policy,
        packets=[FactPacket.model_validate(result["support_packet"])],
        explanations=[result],
    )
    raw = resource.model_dump(mode="json")
    raw["entity_profiles"][0]["presentation"]["covered_sides"] = ["source", "target"]
    with pytest.raises(ValueError, match="presentation differs"):
        ExplanationResource.model_validate(raw)


def test_scoped_description_admits_only_bound_reference_labels(fixture, tmp_path):
    from exact_inspect.readable_descriptions import revise
    from exact_inspect.study.workspace import _admit_explanation
    from tools.prepare_review_bundle import deterministic_result, packet_for

    context, policy, _ = fixture
    contexts = {context.ontology_version_id: context}
    packet, _ = packet_for(ent(context), contexts, policy)
    result = revise(deterministic_result(packet), contexts, policy)
    case = {"source": ent(context).model_dump(), "candidates": []}
    _admit_explanation(result, case, policy, contexts)
    corrupt = copy.deepcopy(result)
    corrupt["claims"][0]["fact_ids"].append(
        context.facts(ent(context, "urn:LabelOnly"), category="labels", policy=policy)["items"][0][
            "fact_id"
        ]
    )
    with pytest.raises(ValueError):
        _admit_explanation(corrupt, case, policy, contexts)


def test_source_inventory_comparison_is_semantic_but_preserves_all_values():
    from tools.compose_review_publication import source_identity

    old = {"id": "saved", "iri": "urn:A", "kind": "class", "payload": '{"action":"emit","count":2}'}
    current = {**old, "payload": '{"count": 2, "action": "emit"}'}
    assert source_identity(old) == source_identity(current)
    assert source_identity(old) != source_identity(
        {**current, "payload": '{"count":1,"action":"emit"}'}
    )


def test_partial_cursor_cannot_reuse_unavailable_complete_only_basis(fixture):
    context, policy, args = fixture
    data, binding = admit(context, **args)
    context = materialize(context, policy=policy, data=data, binding=binding)
    page = context.hierarchy(ent(context), basis=PARTIAL_BASIS, policy=policy, limit=1)
    with pytest.raises(DomainError):
        context.hierarchy(
            ent(context), basis="reasoner_inferred", policy=policy, cursor=page["next_cursor"]
        )


@pytest.mark.parametrize("basis", ["literal_asserted", "elk_inferred_partial"])
def test_scoped_ancestor_graph_uses_same_bounded_artifact(fixture, tmp_path, basis):
    from tests.explanation_study_workspace_test import Store, client_for
    from exact_inspect.study.workspace import build_workspace_scope

    context, policy, args = fixture
    data, binding = admit(context, **args)
    context = materialize(context, policy=policy, data=data, binding=binding)
    case = {
        "case_id": "case-a",
        "source": ent(context).model_dump(),
        "candidates": [
            {"candidate_id": "candidate-b", "entity": ent(context, "urn:B").model_dump()}
        ],
        "explanation_refs": [],
    }
    scope = build_workspace_scope(
        tmp_path,
        scope_id="opaque-workspace",
        kind="case",
        case=case,
        contexts={context.ontology_version_id: context},
        policy=policy,
        reuse_prepared_contexts=True,
    )
    study = {
        "contract_version": "exact-study/1.0",
        "study_revision": "revision-a",
        "cases": [case],
        "policy_hash": policy.policy_hash.removeprefix("sha256:"),
        "visibility_policy": policy.model_dump(mode="json"),
        "workspace_scopes": [scope.model_dump(mode="json")],
    }
    store = Store(tmp_path, study)
    with client_for(store) as client:
        query = {**ent(context).model_dump(), "basis": basis, "direction": "ancestors", "limit": 5}
        response = client.get("/api/v1/study/workspace/opaque-workspace/hierarchy", params=query)
        assert response.status_code == 200, response.text
        page = response.json()
        original = context.hierarchy(
            ent(context), basis=basis, direction="ancestors", limit=5, policy=policy
        )
        assert page["items"] == original["items"] and page["nodes"] == original["nodes"]
        assert len(page["nodes"]) <= 5 and ent(context).model_dump() in page["nodes"]
        if basis == "elk_inferred_partial":
            assert page["status"] == "partial" and page["next_cursor"]
            rejected = client.get(
                "/api/v1/study/workspace/opaque-workspace/hierarchy",
                params={**query, "limit": 4, "cursor": page["next_cursor"]},
            )
            assert rejected.status_code == 409 and rejected.json()["code"] == "stale_cursor"
        store.current["condition"] = "ontology_baseline"
        assert (
            client.get(
                "/api/v1/study/workspace/opaque-workspace/hierarchy", params=query
            ).status_code
            == 403
        )


def test_composition_refuses_duplicate_latest_evidence_identity():
    saved = {"evidence_id": "same", "fact_ids": ["admitted"]}
    with pytest.raises(ValueError, match="identities"):
        merge_evidence([saved], [saved, saved])
