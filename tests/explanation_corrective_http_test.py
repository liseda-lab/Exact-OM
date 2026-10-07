"""Corrected HTTP contracts, optional timing and authenticated publication inventory."""

import json
import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from exact_inspect.study.api import create_study_app
from exact_inspect.study.store import StudyStore
from tests.explanation_study_test import (
    ADMIN,
    ORIGIN,
    SECRET,
    background_answers,
    invite,
)
from tests.explanation_study_v2_fixtures import publication_v2


@pytest.fixture
def corrected(tmp_path):
    publication = publication_v2(tmp_path)
    url = os.environ.get("EXACT_STUDY_TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'study.sqlite'}")
    app = create_study_app(url, SECRET, ADMIN, ORIGIN, tmp_path, allow_test_sqlite=True)
    store = app.state.study_store
    store.publish(publication)
    secret = invite(store, publication)
    client = TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN})
    state = client.post("/api/v1/study/session", json={"secret": secret}).json()
    client.headers["X-Study-Session"] = state["session_id"]
    return client, store, publication


def mutate(client, method, path, **payload):
    state = client.get("/api/v1/study/state").json()
    return client.request(
        method,
        "/api/v1/study/" + path,
        json={
            "idempotency_key": uuid4().hex,
            "expected_revision": state["revision"],
            **payload,
        },
    )


def reach_tutorial(client, publication):
    response = mutate(
        client,
        "PUT",
        "consent",
        accepted=True,
        information_version=publication.definition.information_version,
    )
    assert response.status_code == 200, response.text
    assert "protege_installed" not in (response.json()["setup"] or {})
    response = mutate(
        client,
        "PUT",
        "setup",
        setup_version="setup/2",
        instructions_acknowledged=True,
        external_inspection_optional_understood=True,
        resource_access="available",
        submitted=False,
    )
    assert response.status_code == 200 and response.json()["stage"] == "setup"
    response = mutate(
        client,
        "PUT",
        "setup",
        setup_version="setup/2",
        instructions_acknowledged=True,
        external_inspection_optional_understood=True,
        resource_access="available",
        submitted=True,
    )
    assert response.status_code == 200 and response.json()["stage"] == "background", response.text
    response = mutate(
        client,
        "PUT",
        "questionnaires/background",
        form_version="exact-study-forms/2",
        answers=background_answers(),
        submitted=True,
    )
    assert response.status_code == 200 and response.json()["stage"] == "tutorial", response.text
    return response.json()


def test_v2_http_tutorial_event_scope_and_late_timing_receipts(corrected):
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    assert "grading" not in state["tutorial"]
    event = dict(
        event_id=uuid4().hex,
        page_instance_id=uuid4().hex,
        sequence=0,
        scope="tutorial",
        tutorial_version=state["tutorial"]["version"],
        lesson_id="context",
        type="hierarchy_navigate",
        component_id="hierarchy_source",
        client_monotonic_ms=0,
        build_version=publication.definition.software_version,
    )
    response = client.post("/api/v1/study/events", json={"events": [event]})
    assert response.status_code == 200, response.text
    assert client.post("/api/v1/study/events", json={"events": [event]}).json() == response.json()
    bad = {
        **event,
        "event_id": uuid4().hex,
        "sequence": 1,
        "case_id": publication.definition.cases[0].case_id,
    }
    assert client.post("/api/v1/study/events", json={"events": [bad]}).status_code == 422
    bad = {**event, "event_id": uuid4().hex, "sequence": 1, "lesson_id": "not-a-lesson"}
    assert client.post("/api/v1/study/events", json={"events": [bad]}).status_code == 422
    segment = dict(
        segment_id=uuid4().hex,
        page_instance_id=event["page_instance_id"],
        stage="tutorial",
        monotonic_start_ms=0,
        monotonic_end_ms=1000,
    )
    assert client.post("/api/v1/study/timing", json=segment).json()["availability"] == "observed"
    assert mutate(client, "POST", "pause").status_code == 200
    late = {
        **segment,
        "segment_id": uuid4().hex,
        "monotonic_start_ms": 1000,
        "monotonic_end_ms": 2500,
    }
    receipt = client.post("/api/v1/study/timing", json=late)
    assert receipt.status_code == 200, receipt.text
    assert receipt.json()["availability"] == "unavailable"
    assert receipt.json()["reason"] == "stage_no_longer_current"
    assert client.post("/api/v1/study/timing", json=late).json() == receipt.json()
    restarted = StudyStore(store.database_url, store.assets_dir, allow_test_sqlite=True)
    exported = restarted.export(publication.definition.study_revision, include_test=True)
    session = exported["data"]["sessions"][0]
    assert session["tutorial_observed_seconds"] == 1
    assert len(session["timing_segments"]) == 2
    assert session["timing_segments"][1]["availability"] == "unavailable"
    assert not session["assignment"]


def test_corrected_schema_inventory_auth_and_scope_session_binding(corrected):
    client, store, publication = corrected
    assert client.get("/api/v1/admin/studies").status_code == 401
    response = client.get(
        "/api/v1/admin/studies", headers={"Authorization": f"Bearer {ADMIN}"}, params={"limit": 1}
    )
    assert response.status_code == 200, response.text
    seen = response.json()["items"]
    while response.json()["next_cursor"]:
        response = client.get(
            "/api/v1/admin/studies",
            headers={"Authorization": f"Bearer {ADMIN}"},
            params={"limit": 1, "cursor": response.json()["next_cursor"]},
        )
        seen += response.json()["items"]
    item = next(
        item for item in seen if item["study_revision"] == publication.definition.study_revision
    )
    assert item["contract_version"] == "exact-study/2.0" and item["test_sessions"] == 1
    assert item["setup_version"] == "setup/2"
    assert item["export_version"] == "exact-study-analysis/2"
    assert not any(
        key in json.dumps(item) for key in ("invite_digest", "case_keys", "consent_text")
    )
    state = reach_tutorial(client, publication)
    scope = next(
        scope for scope in publication.definition.workspace_scopes if scope.kind == "tutorial"
    )
    route = f"/api/v1/study/workspace/{scope.scope_id}/capabilities"
    assert client.get(route).status_code == 200
    assert client.get(route, headers={"X-Study-Session": "different-session"}).status_code == 409
    assert (
        mutate(
            client,
            "PUT",
            "tutorial/progress",
            tutorial_version=state["tutorial"]["version"],
            completed_requirements=["identity.inspect"],
        ).status_code
        == 422
    )
    store.reissue(state["session_id"], revoke=True)
    assert client.get(route).status_code == 401
