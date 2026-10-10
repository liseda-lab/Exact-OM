"""Prepared roots and real native taxonomy: policy, semantics and HTTP boundaries."""

import shutil
import subprocess
import sys

import pyowl_core as core
import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import publish_bundle
from exact_inspect.context import OntologyContext, build_context_package
from exact_inspect.context_export import export_policy_context
from exact_inspect.contracts import DomainError, VisibilityPolicy, file_hash
from exact_inspect.navigation import EntryPointPage, PreparedHierarchyPage
from exact_inspect.navigation_prepare import prepare_navigation
from exact_inspect.service import create_prepared_app
from exact_inspect.study.workspace import build_workspace_scope
from tests.explanation_study_workspace_test import Store, client_for

OWL = b"""Ontology(<urn:navigation>
Declaration(Class(<urn:A>)) Declaration(NamedIndividual(<urn:A>))
Declaration(Class(<urn:B>)) Declaration(Class(<urn:C>))
Declaration(Class(<urn:D>)) Declaration(Class(<urn:E>))
Declaration(Class(<urn:F>)) Declaration(Class(<urn:U>))
Declaration(Class(<urn:Cycle1>)) Declaration(Class(<urn:Cycle2>))
Declaration(Class(<urn:Other>))
Declaration(ObjectProperty(<urn:p>)) Declaration(ObjectProperty(<urn:q>))
Declaration(DataProperty(<urn:data>))
SubObjectPropertyOf(<urn:p> <urn:q>)
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:D> "Root")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:Other> "Root")
SubClassOf(<urn:A> <urn:B>) SubClassOf(<urn:B> <urn:D>)
SubClassOf(<urn:A> <urn:E>)
SubClassOf(<urn:A> <urn:D>)
EquivalentClasses(<urn:B> <urn:C>)
EquivalentClasses(<urn:F> ObjectIntersectionOf(<urn:B> <urn:E>))
SubClassOf(<urn:Cycle1> <urn:Cycle2>) SubClassOf(<urn:Cycle2> <urn:Cycle1>)
SubClassOf(<urn:U> <http://www.w3.org/2002/07/owl#Nothing>)
)"""


def build(tmp_path, owl=OWL, *, policy=None, infer=True):
    snapshot = core.load_snapshot(
        owl,
        options=core.LoadOptions(
            backend=core.BackendPreference.NATIVE, imports=core.ImportPolicy.IGNORE
        ),
    )
    original = build_context_package(snapshot, tmp_path / "original")
    policy = policy or VisibilityPolicy()
    filtered = export_policy_context(original, tmp_path / "filtered", policy)
    ready = prepare_navigation(
        filtered,
        tmp_path / "ready",
        infer=infer,
        policy=policy,
        source_snapshot=snapshot,
        timeout_seconds=30,
    )
    return original, ready, policy


def entity(context, iri, kind="class"):
    return {"ontology_version_id": context.ontology_version_id, "iri": iri, "kind": kind}


def test_roots_cycles_punning_properties_and_stable_pagination(tmp_path):
    _, context, policy = build(tmp_path, infer=False)
    first = context.entry_points(limit=1, policy=policy)
    EntryPointPage.model_validate(first)
    assert first["default_entity"]["iri"] == "urn:D"
    rows, page = [], first
    while True:
        rows += page["items"]
        if not page["next_cursor"]:
            break
        page = context.entry_points(limit=1, cursor=page["next_cursor"], policy=policy)
    assert len(rows) == first["total_count"]
    assert len({r["entity"]["iri"] for r in rows}) == len(rows)
    cycle = next(row for row in rows if row["entity"]["iri"] == "urn:Cycle1")
    assert cycle["entry_kind"] == "component_entry" and cycle["component_size"] == 2
    assert {r["entity"]["iri"] for r in rows if r["preferred_label"]["value"] == "Root"} == {
        "urn:D",
        "urn:Other",
    }
    assert (
        context.entry_points(kind="object_property", policy=policy)["items"][0]["entity"]["iri"]
        == "urn:q"
    )
    individuals = context.entry_points(kind="individual", policy=policy)["items"]
    assert individuals[0]["entity"] == entity(context, "urn:A", "individual")
    assert individuals[0]["entry_kind"] == "entity_listing"
    with pytest.raises(DomainError, match="another query"):
        context.entry_points(
            basis="structural_navigation", cursor=first["next_cursor"], policy=policy
        )


def test_native_directness_equivalence_bottom_and_structural_contrast(tmp_path):
    original, context, policy = build(tmp_path)
    info = context.navigation_metadata(policy)["inference"]
    assert info["status"] == "available", info
    assert info["reasoner"]["backend"]["effective"] == "native"
    assert info["consistency"] == "consistent"
    a = entity(context, "urn:A")
    asserted = context.hierarchy(a, policy=policy)
    inferred = context.hierarchy(a, basis="reasoner_inferred", policy=policy)
    PreparedHierarchyPage.model_validate(inferred)
    assert "urn:D" in {e["parent"]["iri"] for e in asserted["items"]}
    assert {e["parent"]["iri"] for e in inferred["items"]} == {"urn:F"}
    assert all(e["direct"] and e["axiom_id"] is None for e in inferred["items"])
    assert {e["interpretation"]["kind"] for e in inferred["items"]} == {"reasoner_inferred"}
    b = context.hierarchy(entity(context, "urn:B"), basis="reasoner_inferred", policy=policy)
    c = context.hierarchy(entity(context, "urn:C"), basis="reasoner_inferred", policy=policy)
    assert (
        b["entity_semantics"]["equivalence_group_id"]
        == c["entity_semantics"]["equivalence_group_id"]
    )
    assert b["entity_semantics"]["equivalent_entity_count"] == 2
    first = context.equivalent_entities(entity(context, "urn:B"), limit=1, policy=policy)
    second = context.equivalent_entities(
        entity(context, "urn:B"), limit=1, cursor=first["next_cursor"], policy=policy
    )
    assert first["total_count"] == 2
    assert {e["iri"] for e in first["items"] + second["items"]} == {"urn:B", "urn:C"}
    with pytest.raises(DomainError, match="another query"):
        context.equivalent_entities(
            entity(context, "urn:C"), cursor=first["next_cursor"], policy=policy
        )
    u = context.hierarchy(entity(context, "urn:U"), basis="reasoner_inferred", policy=policy)
    assert u["entity_semantics"]["is_bottom"] is True
    assert (
        context.hierarchy(
            entity(context, "urn:p", "object_property"), basis="reasoner_inferred", policy=policy
        )["status"]
        == "unsupported"
    )
    f = entity(context, "urn:F")
    assert context.hierarchy(f, basis="literal_asserted", policy=policy)["items"] == []
    assert {
        e["parent"]["iri"]
        for e in context.hierarchy(f, basis="structural_navigation", policy=policy)["items"]
    } == {"urn:B", "urn:E"}
    assert {
        e["parent"]["iri"]
        for e in context.hierarchy(f, basis="reasoner_inferred", policy=policy)["items"]
    } == {"urn:B", "urn:C", "urn:E"}
    assert "navigation" not in original.manifest


def test_policy_is_applied_before_native_inference_and_reuse_is_exact(tmp_path):
    policy = VisibilityPolicy(
        categories=tuple(c for c in VisibilityPolicy().categories if c != "restrictions")
    )
    _, context, _ = build(tmp_path, policy=policy)
    parents = context.hierarchy(entity(context, "urn:A"), basis="reasoner_inferred", policy=policy)
    assert {e["parent"]["iri"] for e in parents["items"]} == {"urn:B", "urn:C", "urn:E"}
    same = export_policy_context(context, tmp_path / "same-policy", policy)
    assert same.manifest == context.manifest
    narrower = VisibilityPolicy(categories=("labels", "hierarchy"))
    restricted = export_policy_context(context, tmp_path / "narrow", narrower)
    assert (
        restricted.hierarchy(
            entity(restricted, "urn:A"), basis="reasoner_inferred", policy=narrower
        )["status"]
        == "not_run"
    )
    assert not (restricted.path / "navigation.sqlite").exists()


@pytest.mark.parametrize(
    "owl, expected",
    [
        (
            b"Ontology(Declaration(Class(<urn:A>)) Declaration(NamedIndividual(<urn:i>)) SubClassOf(<urn:A> <http://www.w3.org/2002/07/owl#Nothing>) ClassAssertion(<urn:A> <urn:i>))",
            "failed",
        ),
        (
            b"Ontology(Declaration(Class(<urn:A>)) Declaration(ObjectProperty(<urn:p>)) TransitiveObjectProperty(<urn:p>) SubClassOf(<urn:A> ObjectMaxCardinality(1 <urn:p>)))",
            "unsupported",
        ),
    ],
)
def test_inconsistent_and_unsupported_are_not_empty_successes(tmp_path, owl, expected):
    _, context, policy = build(tmp_path, owl=owl)
    inferred = context.hierarchy(entity(context, "urn:A"), basis="reasoner_inferred", policy=policy)
    assert inferred["status"] == expected, inferred
    assert inferred["total_count"] is None and inferred["items"] == []
    assert context.entry_points(policy=policy)["items"]
    assert context.entry_points(basis="reasoner_inferred", policy=policy)["status"] == expected


def test_portable_actual_http_contract_and_lazy_reader(tmp_path):
    _, context, policy = build(tmp_path)
    package_dir = tmp_path / "package"
    shutil.copytree(context.path, package_dir / "ontology")
    package = publish_bundle(
        package_dir, policy=policy, ontologies={context.ontology_version_id: "ontology"}
    )
    before = {str(p): file_hash(p) for p in package_dir.rglob("*") if p.is_file()}
    shutil.rmtree(context.path)
    client = TestClient(create_prepared_app(package))
    listing = client.get("/api/v1/ontologies").json()["items"][0]
    assert listing["capabilities"]["navigation"]["contract_version"] == "exact-hierarchy/1"
    endpoint = f"/api/v1/ontologies/{context.ontology_version_id}/entry-points"
    response = client.get(endpoint, params={"limit": 1})
    assert response.status_code == 200, response.text
    first = response.json()
    assert first["navigation_contract"] == "exact-hierarchy/1"
    assert (
        client.get(
            endpoint, params={"kind": "individual", "cursor": first["next_cursor"]}
        ).status_code
        == 409
    )
    assert client.get(endpoint, params={"limit": 101}).status_code == 422
    response = client.get(
        "/api/v1/hierarchy", params={**entity(context, "urn:U"), "basis": "reasoner_inferred"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["entity_semantics"]["is_bottom"] is True
    response = client.get(
        f"/api/v1/ontologies/{context.ontology_version_id}/equivalents",
        params={"iri": "urn:B", "limit": 1},
    )
    assert response.status_code == 200 and response.json()["total_count"] == 2
    assert before == {str(p): file_hash(p) for p in package_dir.rglob("*") if p.is_file()}
    script = "from exact_inspect.context import OntologyContext; import sys; c=OntologyContext(sys.argv[1]); c.entry_points(); assert not any(n in sys.modules for n in ('pyowl_core','pyhermit','torch','exact.ontology.reasoning'))"
    subprocess.run([sys.executable, "-c", script, str(package_dir / "ontology")], check=True)


def test_workspace_rechecks_authorization_and_session_bound_entry_cursor(tmp_path):
    _, context, policy = build(tmp_path)
    version = context.ontology_version_id
    case = {
        "case_id": "case-a",
        "source": entity(context, "urn:A"),
        "candidates": [
            {"candidate_id": f"c{i}", "entity": entity(context, iri)}
            for i, iri in enumerate(("urn:B", "urn:C", "urn:D", "urn:E", "urn:F"))
        ],
        "explanation_refs": [],
    }
    scope = build_workspace_scope(
        tmp_path,
        scope_id="prepared/navigation",
        kind="case",
        case=case,
        contexts={version: context},
        policy=policy,
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
    client = client_for(store)
    base = "/api/v1/study/workspace/prepared%2Fnavigation"
    query = {"ontology_version_id": version, "limit": 1}
    capabilities = client.get(base + "/capabilities")
    assert capabilities.status_code == 200, capabilities.text
    assert capabilities.json()["reasoner"] == "available"
    first = client.get(base + "/entry-points", params=query)
    assert first.status_code == 200, first.text
    cursor = first.json()["next_cursor"]
    assert cursor
    assert client.get(base + "/entry-points", params={**query, "cursor": cursor}).status_code == 200
    store.current["presentation_id"] = "another"
    assert client.get(base + "/entry-points", params={**query, "cursor": cursor}).status_code == 409
    store.current["condition"] = "baseline"
    assert client.get(base + "/entry-points", params=query).status_code == 403
    store.current["condition"] = "explanation"
    store.state["stage"] = "paused"
    assert client.get(base + "/entry-points", params=query).status_code == 403


def test_navigation_checksum_and_policy_binding_are_enforced(tmp_path):
    _, context, policy = build(tmp_path, infer=False)
    (context.path / "navigation.sqlite").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        context.entry_points(policy=policy)
    with pytest.raises(ValueError, match="checksum"):
        OntologyContext(context.path)


def test_inference_deadline_keeps_asserted_roots_without_partial_taxonomy(tmp_path):
    original, context, policy = build(tmp_path, infer=False)
    with pytest.raises(ValueError, match="physically policy-filtered"):
        prepare_navigation(original, tmp_path / "unfiltered", policy=policy, infer=True)
    expired = prepare_navigation(context, tmp_path / "expired", infer=True, timeout_seconds=1e-9)
    info = expired.navigation_metadata(policy)["inference"]
    assert info["status"] == "failed" and info["failure_code"] == "TimeoutError"
    assert expired.entry_points(policy=policy)["items"]
    page = expired.hierarchy(entity(expired, "urn:A"), basis="reasoner_inferred", policy=policy)
    assert page["status"] == "failed" and page["items"] == [] and page["total_count"] is None
    with expired._navigation.connection() as db:
        assert db.execute("SELECT count(*) FROM edges").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM members").fetchone()[0] == 0


def test_imports_root_only_and_native_top_semantics(tmp_path):
    root = b"Ontology(<urn:r> Import(<urn:i>) SubClassOf(<urn:A> <urn:B>) EquivalentClasses(<urn:T> <http://www.w3.org/2002/07/owl#Thing>))"
    imported = b"Ontology(<urn:i> SubClassOf(<urn:B> <urn:C>))"
    snapshot = core.load_snapshot(
        root,
        options=core.LoadOptions(backend=core.BackendPreference.NATIVE),
        resolver=core.MappingResolver({"urn:i": imported}),
    )
    for scope in ("root", "closure"):
        original = build_context_package(snapshot, tmp_path / scope, scope=scope)
        prepared = prepare_navigation(
            original, tmp_path / (scope + "-ready"), infer=True, timeout_seconds=30
        )
        info = prepared.navigation_metadata()["inference"]
        assert info["status"] == "available", info
        b = prepared.hierarchy(entity(prepared, "urn:B"), basis="reasoner_inferred")
        assert {e["parent"]["iri"] for e in b["items"]} == (
            {"urn:C"} if scope == "closure" else {"urn:T"}
        )
        top = prepared.hierarchy(entity(prepared, "urn:T"), basis="reasoner_inferred")
        assert top["entity_semantics"]["is_top"] is True and top["items"] == []
        assert info["scope"] == scope


def test_redacted_logical_bytes_require_verified_source_not_unrestricted_inference(tmp_path):
    owl = b"""Ontology(Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
    SubClassOf(Annotation(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> "HIDDEN") <urn:A> <urn:B>))"""
    snapshot = core.load_snapshot(
        owl, options=core.LoadOptions(backend=core.BackendPreference.NATIVE)
    )
    context = build_context_package(snapshot, tmp_path / "original")
    policy = VisibilityPolicy()
    filtered = export_policy_context(context, tmp_path / "filtered", policy)
    unavailable = prepare_navigation(filtered, tmp_path / "without-source", infer=True)
    assert unavailable.navigation_metadata()["inference"]["status"] == "failed"
    ready = prepare_navigation(
        filtered, tmp_path / "with-source", infer=True, source_snapshot=snapshot
    )
    assert ready.navigation_metadata()["inference"]["status"] == "available"
    assert b"HIDDEN" not in (ready.path / "navigation.sqlite").read_bytes()
    assert b"HIDDEN" not in (ready.path / "manifest.json").read_bytes()


def test_shared_prepared_workspace_contexts_are_rooted_exact_policy_and_not_copied(tmp_path):
    _, context, policy = build(tmp_path)
    version = context.ontology_version_id
    case = {
        "case_id": "case-a",
        "source": entity(context, "urn:A"),
        "candidates": [
            {"candidate_id": str(i), "entity": entity(context, iri)}
            for i, iri in enumerate(("urn:B", "urn:C", "urn:D", "urn:E", "urn:F"))
        ],
        "explanation_refs": [],
    }

    def scope(root, identifier, selected_policy=policy):
        return build_workspace_scope(
            root,
            scope_id=identifier,
            kind="case",
            case=case,
            contexts={version: context},
            policy=selected_policy,
            reuse_prepared_contexts=True,
        )

    a, b = scope(tmp_path, "shared-a"), scope(tmp_path, "shared-b")
    assert a.contexts[0] == b.contexts[0]
    assert a.contexts[0].path == context.path.relative_to(tmp_path).as_posix()
    assert not list((tmp_path / "workspaces").rglob("context.sqlite"))
    with pytest.raises(ValueError, match="strictly inside"):
        scope(tmp_path / "other-root", "rejected")
    with pytest.raises(ValueError, match="exact prepared"):
        scope(tmp_path, "wrong-policy", VisibilityPolicy(categories=("labels",)))
