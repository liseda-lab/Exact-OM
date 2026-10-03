"""Actual HTTP regressions for the backend-first study integration follow-up."""

import copy
import os
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from exact_inspect.context import OntologyContext
from exact_inspect.contracts import VisibilityPolicy
from exact_inspect.models import AxiomResponse, Fact
from exact_inspect.study import store as store_module
from exact_inspect.study import telemetry
from exact_inspect.study.api import create_study_app
from tests.explanation_corrective_http_test import corrected, mutate, reach_tutorial
from tests.explanation_study_test import ADMIN, ORIGIN, SECRET, invite
from tests.explanation_study_v2_test import prepare, train
from tools.build_explanation_v2_fixture import publication_v2


def allocate(corrected):
    client, store, publication = corrected
    sid = client.get("/api/v1/study/state").json()["session_id"]
    prepare(store, publication, (sid, 1))
    train(store, publication, (sid, 1))
    return client, store, publication, sid


def test_current_case_serializes_frozen_workspace_descriptor(corrected):
    client, store, publication, sid = allocate(corrected)
    response = client.get("/api/v1/study/cases/current")
    assert response.status_code == 200, response.text
    case = response.json()
    expected = next(
        s for s in publication.definition.workspace_scopes if s.case_id == case["case_id"]
    )
    assert case["workspace"] == {"scope_id": expected.scope_id}
    capabilities = client.get(f"/api/v1/study/workspace/{expected.scope_id}/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["study_revision"] == case["study_revision"]
    assert capabilities.json()["scope_id"] == expected.scope_id
    assert set(capabilities.json()["ontology_versions"]) == {
        case["source"]["ontology_version_id"],
        *(c["entity"]["ontology_version_id"] for c in case["candidates"]),
    }
    assert (
        client.get(
            "/api/v1/study/cases/current", headers={"X-Study-Session": "another-session"}
        ).status_code
        == 409
    )
    assert (
        client.get(
            "/api/v1/study/cases/current", cookies={"exact_study_session": "invalid"}
        ).status_code
        == 401
    )

    # Discovery remains available when paused; reads preserve their stricter rule.
    assert mutate(client, "POST", "pause").status_code == 200
    assert client.get("/api/v1/study/cases/current").status_code == 200
    assert (
        client.get(f"/api/v1/study/workspace/{expected.scope_id}/capabilities").status_code == 403
    )
    assert mutate(client, "POST", "resume", gap_activity="unknown").status_code == 200
    while case["condition"] == "explanation":
        assert (
            mutate(
                client,
                "POST",
                f"cases/{case['case_id']}/submit",
                presentation_id=case["presentation_id"],
                response_type="insufficient_evidence",
                ranked_candidate_ids=[],
            ).status_code
            == 200
        )
        # Consultation already allowed the current workspace; retain that permission.
        assert (
            client.get(
                f"/api/v1/study/workspace/{case['workspace']['scope_id']}/capabilities"
            ).status_code
            == 200
        )
        assert (
            mutate(
                client,
                "PUT",
                f"cases/{case['case_id']}/consultation",
                presentation_id=case["presentation_id"],
                form_version="exact-study-forms/2",
                consulted_external_ontologies=False,
            ).status_code
            == 200
        )
        case = client.get("/api/v1/study/cases/current").json()
    assert case["workspace"] is None and case["explanation_refs"] == []
    assert (
        client.get(f"/api/v1/study/workspace/{expected.scope_id}/capabilities").status_code == 403
    )
    tutorial = next(s for s in publication.definition.workspace_scopes if s.kind == "tutorial")
    assert (
        client.get(f"/api/v1/study/workspace/{tutorial.scope_id}/capabilities").status_code == 200
    )
    assert (
        client.get(f"/api/v1/study/workspace/{expected.scope_id}/capabilities").status_code == 403
    )
    store.reissue(sid, revoke=True)
    assert client.get("/api/v1/study/cases/current").status_code == 401


def test_encoded_scope_identifier_retains_authorization(corrected, monkeypatch):
    client, store, publication, sid = allocate(corrected)
    original = store._study
    case = client.get("/api/v1/study/cases/current").json()
    previous = case["workspace"]["scope_id"]
    scoped_id = "case/scope/nested"

    def scoped(db, revision, **kwargs):
        row, study = original(db, revision, **kwargs)
        study = copy.deepcopy(study)
        next(s for s in study["workspace_scopes"] if s["scope_id"] == previous)[
            "scope_id"
        ] = scoped_id
        return row, study

    monkeypatch.setattr(store, "_study", scoped)
    assert client.get("/api/v1/study/cases/current").json()["workspace"] == {"scope_id": scoped_id}
    base = "/api/v1/study/workspace/" + quote(scoped_id, safe="")
    capabilities = client.get(base + "/capabilities")
    assert capabilities.status_code == 200 and capabilities.json()["scope_id"] == scoped_id
    assert client.get(base + "/entity-context", params=case["source"]).status_code == 200
    assert mutate(client, "POST", "pause").status_code == 200
    assert client.get(base + "/capabilities").status_code == 403


@pytest.mark.parametrize("damage", ["missing", "invalid", "changed_index", "asset_root"])
def test_broken_required_workspace_is_typed_503(corrected, monkeypatch, damage):
    client, store, publication, sid = allocate(corrected)
    original = store._study
    current = client.get("/api/v1/study/cases/current").json()
    scope_id = current["workspace"]["scope_id"]
    if damage == "asset_root":
        monkeypatch.setattr(store, "assets_dir", None)

    def damaged(db, revision, **kwargs):
        row, study = original(db, revision, **kwargs)
        study = copy.deepcopy(study)
        scope = next(s for s in study["workspace_scopes"] if s["scope_id"] == scope_id)
        if damage == "missing":
            study["workspace_scopes"].remove(scope)
        elif damage == "invalid":
            scope["contexts"] = []
        elif damage == "changed_index":
            scope["index_sha256"] = "0" * 64
        return row, study

    monkeypatch.setattr(store, "_study", damaged)
    response = client.get("/api/v1/study/cases/current")
    assert response.status_code == 503 and "workspace" in response.json()["detail"]
    assert str(store.assets_dir) not in response.text
    if damage != "missing":
        assert client.get(f"/api/v1/study/workspace/{scope_id}/capabilities").status_code == 503


def test_full_frozen_context_http_paging_and_original_identity(tmp_path):
    publication = publication_v2(tmp_path, navigation_size=65)
    app = create_study_app(
        os.environ.get("EXACT_STUDY_TEST_DATABASE_URL", f'sqlite:///{tmp_path / "study.sqlite"}'),
        SECRET,
        ADMIN,
        ORIGIN,
        tmp_path,
        allow_test_sqlite=True,
    )
    store = app.state.study_store
    store.publish(publication)
    client = TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN})
    state = client.post("/api/v1/study/session", json={"secret": invite(store, publication)}).json()
    client.headers["X-Study-Session"] = state["session_id"]
    allocate((client, store, publication))
    case = client.get("/api/v1/study/cases/current").json()
    scope = next(
        s
        for s in publication.definition.workspace_scopes
        if s.scope_id == case["workspace"]["scope_id"]
    )
    base = "/api/v1/study/workspace/" + scope.scope_id
    entity = case["source"]
    ref = next(c for c in scope.contexts if c.ontology_version_id == entity["ontology_version_id"])
    shared = OntologyContext(tmp_path / ref.path)
    policy = VisibilityPolicy.model_validate(publication.definition.visibility_policy)
    search = client.get(
        base + "/entities",
        params={
            "ontology_version_id": entity["ontology_version_id"],
            "term": "Navigation source 064",
        },
    ).json()
    nonfocal = search["items"][0]["entity"]
    assert nonfocal["iri"] == "urn:source:navigation:064"
    assert (
        client.get(base + "/hierarchy", params={**nonfocal, "direction": "parents"}).json()[
            "total_count"
        ]
        == 5
    )
    assert client.get(base + "/explanations", params=nonfocal).status_code == 403

    def pages(route, params):
        result, cursor = [], None
        while True:
            response = client.get(
                base + route,
                params={**params, "limit": 7, **({"cursor": cursor} if cursor else {})},
            )
            assert response.status_code == 200, response.text
            page = response.json()
            result += page["items"]
            cursor = page["next_cursor"]
            if not cursor:
                break
        return result

    children = pages("/hierarchy", {**entity, "direction": "children"})
    assert len(children) == len({i["id"] for i in children}) == 65
    assert {i["id"] for i in children} == {
        i["id"]
        for i in shared.hierarchy(entity, direction="children", limit=100, policy=policy)["items"]
    }
    context = client.get(base + "/entity-context", params=entity).json()
    category = next(k for k, v in context["categories"].items() if (v["total_count"] or 0) >= 65)
    facts = pages("/entity-facts", {**entity, "category": category})
    assert (
        len(facts)
        == len({i["id"] for i in facts})
        == context["categories"][category]["total_count"]
    )
    for fact in (facts[0], facts[-1]):
        original = shared.fact(fact["id"], entity, policy=policy)
        wire = client.get(base + "/facts/" + fact["id"], params=entity).json()
        assert wire == Fact.model_validate(original).model_dump(mode="json")
        axiom = client.get(base + "/axioms/" + fact["axiom_id"], params=entity)
        assert axiom.status_code == 200
        assert axiom.json() == AxiomResponse.model_validate(
            shared.axiom(fact["axiom_id"], policy=policy)
        ).model_dump(mode="json")
    cursor = client.get(
        base + "/hierarchy", params={**entity, "direction": "children", "limit": 1}
    ).json()["next_cursor"]
    invalid = client.get(
        base + "/hierarchy", params={**entity, "direction": "parents", "limit": 1, "cursor": cursor}
    )
    assert invalid.status_code == 409 and invalid.json()["code"] == "stale_cursor"


def test_page_readiness_is_local_and_does_not_restart_case_clock(corrected, monkeypatch):
    client, store, publication, sid = allocate(corrected)
    case = client.get("/api/v1/study/cases/current").json()
    clock = {"now": "2026-01-01T00:00:00+00:00"}
    monkeypatch.setattr(telemetry, "utcnow", lambda: clock["now"])

    def ready(page, sequence=0, monotonic=1000):
        event = dict(
            event_id=uuid4().hex,
            page_instance_id=page,
            sequence=sequence,
            type="case_ready",
            case_id=case["case_id"],
            presentation_id=case["presentation_id"],
            client_monotonic_ms=monotonic,
            build_version=publication.definition.software_version,
        )
        response = client.post("/api/v1/study/events", json={"events": [event]})
        assert response.status_code == 200, response.text
        return event, response.json()

    first, receipt = ready("page-a")
    assert client.post("/api/v1/study/events", json={"events": [first]}).json() == receipt

    def segment(page, start, end):
        value = dict(
            segment_id=uuid4().hex,
            page_instance_id=page,
            stage="case",
            case_id=case["case_id"],
            presentation_id=case["presentation_id"],
            monotonic_start_ms=start,
            monotonic_end_ms=end,
        )
        response = client.post("/api/v1/study/timing", json=value)
        assert response.status_code == 200, response.text
        return value, response.json()

    _, before = segment("page-b", 0, 500)
    assert before["availability"] == "unavailable" and before["reason"] == "case_not_ready"
    interval, observed = segment("page-a", 1000, 2000)
    assert observed["availability"] == "observed"
    assert client.post("/api/v1/study/timing", json=interval).json() == observed
    clock["now"] = "2026-01-01T00:00:10+00:00"
    ready("page-b", monotonic=2000)
    segment("page-b", 2000, 4000)
    clock["now"] = "2026-01-01T00:00:15+00:00"
    ready("page-a", sequence=1, monotonic=3000)
    monkeypatch.setattr(store_module, "utcnow", lambda: "2026-01-01T00:01:40+00:00")
    assert (
        mutate(
            client,
            "POST",
            f"cases/{case['case_id']}/submit",
            presentation_id=case["presentation_id"],
            response_type="none_of_these",
            ranked_candidate_ids=[],
        ).status_code
        == 200
    )
    exported = store.export(publication.definition.study_revision, include_test=True)
    session = next(s for s in exported["data"]["sessions"] if s["session_id"] == sid)
    result = next(c for c in session["cases"] if c["case_id"] == case["case_id"])
    assert result["timing"]["raw_elapsed_seconds"] == 100
    assert result["timing"]["observed_segment_seconds"] == 3
    assert len([e for e in session["events"] if e["type"] == "case_ready"]) == 3
    assert len(session["timing_segments"]) == 3

    # Empty prepared selection is explicit terminal absence, not a transport error.
    base = "/api/v1/study/workspace/" + case["workspace"]["scope_id"]
    absent = client.get(
        base + "/explanations",
        params={
            **case["source"],
            "task": "entity_profile",
            "counterpart_iri": case["candidates"][0]["entity"]["iri"],
            "counterpart_ontology_version_id": case["candidates"][0]["entity"][
                "ontology_version_id"
            ],
        },
    )
    assert absent.status_code == 200 and absent.json()["status"] == "not_exported"
    assert absent.json()["items"] == [] and absent.json()["reason"]
