"""Regression boundaries from the independently reviewed S2/Luna integration."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pyowl_core as core
import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import atomic_json, publish_bundle
from exact_inspect.context import build_context_package
from exact_inspect.context_resources import (
    export_ontology_resource,
    validate_ontology_resource,
)
from exact_inspect.contracts import EntityRef, VisibilityPolicy, canonical_hash
from exact_inspect.generation import ExplanationJobs, GenerationProfile, entity_packet
from exact_inspect.navigation import CURATED_DEFAULTS, EntryPointPage
from exact_inspect.read_diagnostics import ReadDiagnostics
from exact_inspect.service import create_prepared_app
from tests.explanation_navigation_test import build, entity


def test_raw_context_export_has_complete_inventory_and_legacy_resume_is_explicit(tmp_path):
    snap = core.load_snapshot(
        b'Ontology(Declaration(Class(<urn:A>)) AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "A"))'
    )
    context = build_context_package(snap, tmp_path / "context")
    policy = VisibilityPolicy()
    resource = tmp_path / "ontology.ofn"
    receipt = export_ontology_resource(context, resource, policy)
    original = resource.read_bytes()
    validate_ontology_resource(resource, receipt, context=context)
    counts = context.manifest["completeness"]["category_counts"]
    assert sum(counts.values()) == context.manifest["completeness"]["axiom_count"] == 2
    context.manifest["completeness"].pop("category_counts")
    with pytest.raises(ValueError, match="complete retained category inventory"):
        validate_ontology_resource(resource, receipt, context=context)
    atomic_json(context.path / "manifest.json", context.manifest)
    with pytest.raises(ValueError, match="prepare a new revision"):
        build_context_package(snap, context.path)
    assert resource.read_bytes() == original


def test_curated_default_pages_keep_all_roots_and_cursor_binding(tmp_path, monkeypatch):
    _, context, policy = build(tmp_path, infer=False)
    original = context.entry_points(policy=policy, limit=100)
    choice = {
        "context_sha256": context.manifest["artifacts"]["context.sqlite"],
        "iri": "urn:Other",
        "rationale": "Explicit test curation, not an ontology edge",
        "revision": "test/1",
    }
    monkeypatch.setitem(CURATED_DEFAULTS, context.ontology_version_id, choice)
    first = context.entry_points(policy=policy, limit=1)
    assert first["default_entity"] == entity(context, "urn:Other")
    assert first["selection"]["kind"] == "curated_default"
    assert len(context.entry_points(policy=policy, limit=20)["items"]) == 1
    EntryPointPage.model_validate(first)
    rows, page = [], first
    while True:
        rows += page["items"]
        if not page["next_cursor"]:
            break
        page = context.entry_points(policy=policy, cursor=page["next_cursor"], limit=2)
    assert rows[0]["entity"]["iri"] == "urn:Other"
    assert len(rows) == len(original["items"]) == first["total_count"]
    assert sorted(x["entity"]["iri"] for x in rows) == sorted(
        x["entity"]["iri"] for x in original["items"]
    )
    from exact_inspect.contracts import DomainError

    choice["revision"] = "test/2"
    with pytest.raises(DomainError, match="another query"):
        context.entry_points(policy=policy, cursor=first["next_cursor"])
    choice["iri"] = "urn:not-admitted"
    assert context.entry_points(policy=policy)["default_entity"] == original["default_entity"]
    choice["context_sha256"] = "sha256:wrong"
    assert context.entry_points(policy=policy)["default_entity"] == original["default_entity"]


def test_curated_default_main_and_scoped_api_parity(tmp_path, monkeypatch):
    from exact_inspect.study.workspace import build_workspace_scope
    from tests.explanation_study_workspace_test import Store, client_for

    _, context, policy = build(tmp_path, infer=False)
    monkeypatch.setitem(
        CURATED_DEFAULTS,
        context.ontology_version_id,
        {
            "context_sha256": context.manifest["artifacts"]["context.sqlite"],
            "iri": "urn:Other",
            "revision": "test/1",
            "rationale": "Curated navigation",
        },
    )
    case = {
        "case_id": "case-a",
        "source": entity(context, "urn:A"),
        "candidates": [{"candidate_id": "candidate", "entity": entity(context, "urn:D")}],
        "explanation_refs": [],
    }
    scope = build_workspace_scope(
        tmp_path,
        scope_id="scope",
        kind="case",
        case=case,
        contexts={context.ontology_version_id: context},
        policy=policy,
        reuse_prepared_contexts=True,
    )
    study = {
        "contract_version": "exact-study/1.0",
        "study_revision": "r",
        "cases": [case],
        "policy_hash": policy.policy_hash[7:],
        "visibility_policy": policy.model_dump(),
        "workspace_scopes": [scope.model_dump(mode="json")],
    }
    store = Store(tmp_path, study)
    package = publish_bundle(
        tmp_path, ontologies={context.ontology_version_id: context.path.name}, policy=policy
    )
    with TestClient(create_prepared_app(package)) as main, client_for(store) as scoped:
        a = main.get(
            f"/api/v1/ontologies/{context.ontology_version_id}/entry-points", params={"limit": 1}
        )
        b = scoped.get(
            "/api/v1/study/workspace/scope/entry-points",
            params={"ontology_version_id": context.ontology_version_id, "limit": 1},
        )
        assert a.status_code == b.status_code == 200, b.text
        assert a.json()["items"] == b.json()["items"]
        assert a.json()["selection"] == b.json()["selection"]


def test_legacy_generation_retains_punned_and_shared_support_subjects(tmp_path):
    owl = b"""Ontology(Declaration(Class(<urn:A>)) Declaration(NamedIndividual(<urn:A>))
        Declaration(Class(<urn:B>)) EquivalentClasses(<urn:A> <urn:B>)
        AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Shared label"))"""
    context = build_context_package(core.load_snapshot(owl), tmp_path / "context")
    policy = VisibilityPolicy()
    a = EntityRef.model_validate(entity(context, "urn:A", "individual"))
    fact = context.facts(a, category="labels", policy=policy)["items"][0]
    packet = entity_packet(
        a, [fact], context_hash=context.manifest["artifacts"]["context.sqlite"], policy=policy
    )
    # The same source axiom has two authoritative typed focal views.
    other = EntityRef.model_validate(entity(context, "urn:A"))
    packet = packet.model_copy(
        update={
            "task": "pair_comparison",
            "entities": [a, other],
            "context_hashes": packet.context_hashes * 2,
            "facts": packet.facts
            + [
                {
                    k: v
                    for k, v in context.fact(fact["fact_id"], other, policy=policy).items()
                    if k in packet.facts[0]
                }
            ],
        }
    )
    from exact_inspect.descriptions import validate_prepared_support

    job = ExplanationJobs(tmp_path / "jobs")
    result = job.generate(
        packet,
        GenerationProfile(name="offline", model="fixture/label"),
        provider=lambda *args: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "claims": [
                                    {
                                        "text": "Shared label",
                                        "category": "meaning",
                                        "fact_ids": [fact["fact_id"]],
                                    }
                                ],
                                "limitations": [],
                                "relation": None,
                            }
                        )
                    }
                }
            ]
        },
    )
    assert [f["subject"]["kind"] for f in result["support_packet"]["facts"]] == [
        "individual",
        "class",
    ]
    validate_prepared_support(result, {context.ontology_version_id: context}, policy)
    from exact_inspect.study.builder import build_explanation_resource
    from exact_inspect.study.workspace import _admit_explanation

    resource = build_explanation_resource(
        {context.ontology_version_id: context},
        [a, other],
        policy=policy,
        packets=[packet],
        explanations=[result],
    )
    assert {f.subject.kind for f in resource.facts if f.fact_id == fact["fact_id"]} == {
        "class",
        "individual",
    }
    atomic_json(tmp_path / "explanation.json", result)
    package = publish_bundle(
        tmp_path,
        ontologies={context.ontology_version_id: "context"},
        explanations={result["explanation_id"]: "explanation.json"},
        policy=policy,
    )
    with TestClient(create_prepared_app(package)) as client:
        response = client.get("/api/v1/explanations/" + result["explanation_id"])
        assert response.status_code == 200, response.text
        assert response.json()["support_packet"]["facts"] == result["support_packet"]["facts"]
    case = {
        "source": a.model_dump(),
        "candidates": [{"candidate_id": "c", "entity": other.model_dump()}],
    }
    forged = copy.deepcopy(result)
    forged["support_packet"]["facts"][0]["value"]["lexical_form"] = "forged uncited extra detail"
    forged["manifest"]["packet_hash"] = canonical_hash(forged["support_packet"])
    with pytest.raises(ValueError, match="differs from admitted original"):
        _admit_explanation(forged, case, policy, {context.ontology_version_id: context})
    with pytest.raises(ValueError, match="fact differs from the original ontology"):
        build_explanation_resource(
            {context.ontology_version_id: context},
            [a, other],
            policy=policy,
            packets=[type(packet).model_validate(forged["support_packet"])],
            explanations=[forged],
        )
    corrupt = copy.deepcopy(result)
    corrupt["support_packet"]["facts"][0]["subject"]["kind"] = "object_property"
    corrupt["manifest"]["packet_hash"] = canonical_hash(corrupt["support_packet"])
    with pytest.raises((KeyError, ValueError)):
        validate_prepared_support(corrupt, {context.ontology_version_id: context}, policy)


def test_diagnostics_preserve_stream_cancellation_and_exclude_secrets():
    records = []
    secret = "invitation-cookie-participant-secret"

    async def inner(scope, receive, send):
        scope["route"] = SimpleNamespace(path="/api/v1/explanations/{explanation_id}")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-length", b"8"),
                    (b"content-security-policy", b"default-src 'self'"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b"abc", "more_body": True})
        assert (await receive())["type"] == "http.disconnect"
        raise asyncio.CancelledError(secret)

    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/explanations/" + secret,
        "query_string": secret.encode(),
        "headers": [(b"cookie", secret.encode())],
    }
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ReadDiagnostics(inner, emit=records.append)(scope, receive, send))
    assert secret not in json.dumps(records)
    assert records[-1]["error_type"] == "CancelledError"
    assert records[-1]["sent_bytes"] == 3 and records[-1]["declared_length"] == 8
    assert not records[-1]["complete"] and records[-1]["disconnect_seen"]
    assert messages[0]["headers"][:2] == [
        (b"content-length", b"8"),
        (b"content-security-policy", b"default-src 'self'"),
    ]
    assert messages[1]["body"] == b"abc"
