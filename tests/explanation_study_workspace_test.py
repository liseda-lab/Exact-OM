"""Participant authorization and continuation over complete prepared context indexes."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from contextlib import contextmanager

import pyowl_core as core
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from exact_inspect.context import build_context_package
from exact_inspect.contracts import DomainError, VisibilityPolicy
from exact_inspect.study.store import StudyError, StudyStore
from exact_inspect.study.workspace import (
    WorkspaceScope,
    authorized_workspace,
    build_workspace_scope,
    install_workspace_routes,
    validate_publication_workspaces,
)


class Store:
    _resource_path = StudyStore._resource_path
    _check_safe_json = StudyStore._check_safe_json

    def __init__(self, root, study):
        self.assets_dir = root.resolve()
        self.study = study
        self.state = {"consent": {"accepted": True}, "stage": "case"}
        self.current = {"condition": "explanation", "case_id": "case-a", "presentation_id": "p1"}

    @contextmanager
    def transaction(self):
        yield None

    def _session(self, db, sid, generation):
        if sid not in {"session-a", "session-b"} or generation != 1:
            raise StudyError(401, "Session unavailable")
        return {"study_revision": self.study["study_revision"]}, self.state, self.study

    def _current(self, state):
        return self.current


@pytest.fixture
def workspace(tmp_path):
    rows = ["Declaration(Class(<urn:A>))"]
    for i in range(70):
        rows += [f"Declaration(Class(<urn:B{i:02}>))", f"SubClassOf(<urn:A> <urn:B{i:02}>)"]
    rows += [
        'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Original source")',
        'AnnotationAssertion(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> <urn:A> "FORBIDDEN-XREF")',
    ]
    context = build_context_package(
        core.load_snapshot(("Ontology(" + " ".join(rows) + ")").encode()), tmp_path / "original"
    )
    version = context.ontology_version_id

    def entity(iri):
        return {"ontology_version_id": version, "iri": iri, "kind": "class"}

    case = {
        "case_id": "case-a",
        "source": entity("urn:A"),
        "candidates": [
            {"candidate_id": f"candidate-{i}", "entity": entity(f"urn:B{i:02}")} for i in range(5)
        ],
        "explanation_refs": [],
    }
    policy = VisibilityPolicy()
    scope = build_workspace_scope(
        tmp_path,
        scope_id="opaque-workspace",
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
    return Store(tmp_path, study), scope, version, case


def client_for(store):
    app = FastAPI()
    router = APIRouter(prefix="/api/v1")
    install_workspace_routes(router, store, lambda: ("session-a", 1))
    app.include_router(router)

    @app.exception_handler(StudyError)
    async def error(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    return TestClient(app)


def test_workspace_navigates_complete_index_and_resolves_late_fact(workspace):
    store, scope, version, case = workspace
    validate_publication_workspaces(store, store.study, {})
    client = client_for(store)
    base = "/api/v1/study/workspace/opaque-workspace"
    params = {"ontology_version_id": version, "iri": "urn:A", "kind": "class", "limit": 20}
    first = client.get(base + "/hierarchy", params=params).json()
    assert first["total_count"] == 70 and first["truncated"]
    rows = first["items"][:]
    while first["next_cursor"]:
        first = client.get(
            base + "/hierarchy", params={**params, "cursor": first["next_cursor"]}
        ).json()
        rows += first["items"]
    assert len(rows) == 70 and len({r["id"] for r in rows}) == 70
    late = client.get(base + "/facts/" + rows[-1]["axiom_id"], params=params)
    assert late.status_code == 200 and late.json()["subject"] == case["source"]
    assert late.json()["value"]["term_type"] == "expression_ref"
    search = client.get(
        base + "/entities", params={"ontology_version_id": version, "term": "urn:B69"}
    ).json()
    assert search["items"][0]["entity"]["iri"] == "urn:B69"
    # Whole ontology navigation does not authorize generated case data.
    assert (
        client.get(base + "/explanations", params={**params, "iri": "urn:B69"}).status_code == 403
    )
    assert (
        client.get(base + "/evidence", params={"candidate_id": "other-case-candidate"}).status_code
        == 403
    )
    context = client.get(base + "/entity-context", params=params).json()
    assert context["categories"]["hierarchy"]["total_count"] == 70
    continuation = client.get(
        base + "/entity-facts",
        params={
            **params,
            "category": "hierarchy",
            "cursor": context["categories"]["hierarchy"]["next_cursor"],
        },
    )
    assert continuation.status_code == 200 and continuation.json()["items"]
    assert "FORBIDDEN-XREF" not in json.dumps(context)
    database = store.assets_dir / scope.contexts[0].path / "context.sqlite"
    assert b"FORBIDDEN-XREF" not in database.read_bytes()


@pytest.mark.parametrize("change", ["baseline", "next_case", "paused", "no_consent"])
def test_workspace_authorizes_every_read_including_labels_and_cursors(workspace, change):
    store, scope, version, _ = workspace
    client = client_for(store)
    base = "/api/v1/study/workspace/opaque-workspace"
    params = {"ontology_version_id": version, "iri": "urn:A", "limit": 1}
    cursor = client.get(base + "/hierarchy", params=params).json()["next_cursor"]
    if change == "baseline":
        store.current["condition"] = "baseline"
    elif change == "next_case":
        store.current["case_id"] = "case-b"
    elif change == "paused":
        store.state["stage"] = "paused"
    else:
        store.state["consent"] = None
    for operation, query in [
        ("/labels", params),
        ("/hierarchy", {**params, "cursor": cursor}),
        ("/capabilities", {}),
    ]:
        assert client.get(base + operation, params=query).status_code == 403


def test_workspace_cursors_bind_query_session_generation_and_presentation(workspace):
    store, scope, version, case = workspace
    query = {"entity": case["source"], "direction": "parents", "basis": "literal_asserted"}

    def page(w, cursor=None):
        return w.page(
            "hierarchy",
            query,
            lambda c: w.context(version).hierarchy(
                case["source"], cursor=c, limit=1, policy=w.policy
            ),
            cursor,
        )

    with authorized_workspace(store, "session-a", 1, scope.scope_id) as w:
        cursor = page(w)["next_cursor"]
    with authorized_workspace(store, "session-b", 1, scope.scope_id) as w:
        with pytest.raises(DomainError, match="another query"):
            page(w, cursor)
    with pytest.raises(StudyError):
        with authorized_workspace(store, "session-a", 0, scope.scope_id):
            pass
    store.current["presentation_id"] = "new-presentation"
    with authorized_workspace(store, "session-a", 1, scope.scope_id) as w:
        with pytest.raises(DomainError, match="another query"):
            page(w, cursor)


def test_workspace_rejects_changed_index_and_unprepared_capability(workspace):
    store, scope, _, _ = workspace
    store.study["workspace_scopes"][0]["components"].append("profiles")
    with pytest.raises(ValueError, match="Promised workspace"):
        validate_publication_workspaces(store, store.study, {})
    store.study["workspace_scopes"][0]["components"].pop()
    with (store.assets_dir / scope.index_path).open("ab") as out:
        out.write(b"tampered")
    with pytest.raises(ValueError, match="index changed"):
        with authorized_workspace(store, "session-a", 1, scope.scope_id):
            pass


def test_workspace_rejects_path_escape_and_missing_v2_scopes(workspace):
    store, scope, _, _ = workspace
    payload = scope.model_dump(mode="json")
    payload["contexts"][0]["path"] = "../original"
    with pytest.raises(ValueError, match="safe relative"):
        WorkspaceScope.model_validate(payload)
    store.study["contract_version"] = "exact-study/2.0"
    with pytest.raises(ValueError, match="synthetic tutorial"):
        validate_publication_workspaces(store, store.study, {})


def test_workspace_serving_import_does_not_load_native_owl_or_matchers():
    script = """import sys
import exact_inspect.study.workspace
assert not any(name in sys.modules for name in ('pyowl_core', 'torch', 'transformers', 'pymatcha'))
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def add_tutorial(store, *, overlap=False):
    iris = [f"urn:practice:{i}" for i in range(6)] + (
        ["urn:B69"] if overlap is True or overlap == "entity" else []
    )
    reference = (
        " AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#comment> <urn:practice:0> <urn:B69>)"
        if overlap == "annotation"
        else ""
    )
    context = build_context_package(
        core.load_snapshot(
            (
                "Ontology("
                + " ".join(f"Declaration(Class(<{iri}>))" for iri in iris)
                + reference
                + ")"
            ).encode()
        ),
        store.assets_dir / "tutorial-original",
    )

    def entity(iri):
        return {"ontology_version_id": context.ontology_version_id, "iri": iri, "kind": "class"}

    case = {
        "case_id": "practice",
        "source": entity(iris[0]),
        "candidates": [
            {"candidate_id": f"practice-{i}", "entity": entity(iri)}
            for i, iri in enumerate(iris[1:6])
        ],
        "explanation_refs": [],
    }
    scope = build_workspace_scope(
        store.assets_dir,
        scope_id="tutorial-scope",
        kind="tutorial",
        case=case,
        contexts={context.ontology_version_id: context},
        policy=VisibilityPolicy(),
    )
    store.study["tutorial"] = {"case": case}
    store.study["workspace_scopes"].append(scope.model_dump(mode="json"))
    return scope


def test_synthetic_help_in_baseline_cannot_authorize_scored_workspace(workspace):
    store, _, _, _ = workspace
    add_tutorial(store)
    validate_publication_workspaces(store, store.study, {})
    store.current["condition"] = "baseline"
    client = client_for(store)
    response = client.get("/api/v1/study/workspace/tutorial-scope/capabilities")
    assert response.status_code == 200 and response.json()["synthetic"]
    assert client.get("/api/v1/study/workspace/opaque-workspace/capabilities").status_code == 403


@pytest.mark.parametrize("reference_kind", ["entity", "annotation"])
def test_tutorial_disjointness_covers_navigation_beyond_focal_case_entities(
    workspace, reference_kind
):
    store, _, _, _ = workspace
    add_tutorial(store, overlap=reference_kind)
    with pytest.raises(ValueError, match="intersects a scored ontology graph"):
        validate_publication_workspaces(store, store.study, {})
