"""Actual HTTP study metadata isolation; PostgreSQL enabled by the standard test URL.

Invented ontology fixtures exercise admission and session boundaries. Tutorial timing
uses fixture segments; the separate HTTPS journey verifier waits actual elapsed time.
"""

import json
import os
import sqlite3
from collections import Counter

from fastapi.testclient import TestClient

from exact_inspect.context import OntologyContext
from exact_inspect.contracts import VisibilityPolicy, file_hash
from exact_inspect.evidence_metadata import pair_evidence_metadata
from exact_inspect.study.api import create_study_app
from exact_inspect.study.models import Publish
from exact_inspect.study.workspace import build_workspace_scope
from tests.explanation_corrective_http_test import mutate
from tests.explanation_review_forms_test import (
    answers_for,
    background,
    consent,
    consultation,
    ok,
    setup,
    submit_ranking,
    train,
)
from tests.explanation_review_publication_test import review_fixture
from tests.explanation_study_test import ADMIN, SECRET

ORIGIN = "https://localhost"


def modern_publication(root):
    raw = review_fixture(root)
    definition = raw["definition"]
    policy = VisibilityPolicy.model_validate(definition["visibility_policy"])
    cases = {case["case_id"]: case for case in definition["cases"]}
    for position, scope in enumerate(definition["workspace_scopes"]):
        if scope["kind"] == "tutorial":
            continue  # Retain the historical tutorial workspace bytes.
        case = cases[scope["case_id"]]
        with sqlite3.connect(root / scope["index_path"]) as db:
            explanations = [
                json.loads(row[0]) for row in db.execute("SELECT payload FROM explanations")
            ]
            evidence = {
                candidate["candidate_id"]: [
                    json.loads(row[0])
                    for row in db.execute(
                        "SELECT payload FROM evidence WHERE candidate_id=? ORDER BY id",
                        (candidate["candidate_id"],),
                    )
                ]
                for candidate in case["candidates"]
            }
        metadata = {
            candidate["candidate_id"]: dict(
                source=case["source"],
                target=candidate["entity"],
                evidence_metadata=pair_evidence_metadata(
                    {}, evidence[candidate["candidate_id"]], case["source"], candidate["entity"], []
                ),
            )
            for candidate in case["candidates"]
        }
        contexts = {
            ref["ontology_version_id"]: OntologyContext(root / ref["path"])
            for ref in scope["contexts"]
        }
        updated = build_workspace_scope(
            root,
            scope_id="pair-" + scope["scope_id"],
            kind="case",
            case=case,
            contexts=contexts,
            policy=policy,
            explanations=explanations,
            evidence=evidence,
            pair_metadata=metadata,
            components=scope["components"],
        )
        definition["workspace_scopes"][position] = updated.model_dump(mode="json")
    return Publish.model_validate(raw)


def test_pair_metadata_with_real_session_condition_pause_generation_and_case_boundaries(tmp_path):
    frozen = modern_publication(tmp_path)
    url = os.environ.get("EXACT_STUDY_TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'study.sqlite'}")
    app = create_study_app(url, SECRET, ADMIN, ORIGIN, tmp_path, allow_test_sqlite=True)
    store = app.state.study_store
    frozen_hashes = {
        scope.index_path: file_hash(tmp_path / scope.index_path)
        for scope in frozen.definition.workspace_scopes
    }
    scopes = {scope.case_id: scope for scope in frozen.definition.workspace_scopes}
    admin = {"Authorization": "Bearer " + ADMIN}
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN}) as client:
        ok(client.post("/api/v1/admin/studies", headers=admin, json=frozen.model_dump(mode="json")))
        invitation = ok(
            client.post(
                f"/api/v1/admin/studies/{frozen.definition.study_revision}/invitations",
                headers=admin,
                json={"test": True, "count": 1},
            )
        )["invitations"][0]
        state = ok(
            client.post(
                "/api/v1/study/session",
                json={
                    "secret": invitation["invitation"].split("=", 1)[1],
                    "client_contract": "study-review/1",
                },
            )
        )
        client.headers["X-Study-Session"] = state["session_id"]
        initial_route = f"/api/v1/study/workspace/{scopes[frozen.definition.cases[0].case_id].scope_id}/evidence-metadata"
        assert client.get(initial_route, params={"candidate_id": "candidate-1"}).status_code == 403
        state = consent(client, frozen)
        ok(setup(client, submitted=True))
        ok(background(client, answers_for(state["forms"])))
        train(client, store, frozen)
        conditions = Counter()
        previous_route = None
        checked_generation = False
        for _ in range(4):
            case = ok(client.get("/api/v1/study/cases/current"))
            conditions[case["condition"]] += 1
            route = f"/api/v1/study/workspace/{scopes[case['case_id']].scope_id}/evidence-metadata"
            if previous_route:
                assert (
                    client.get(previous_route, params={"candidate_id": "candidate-1"}).status_code
                    == 403
                )
            baseline = case["condition"] == "ontology_baseline"
            for candidate in case["candidates"]:
                response = client.get(route, params={"candidate_id": candidate["candidate_id"]})
                if baseline:
                    assert response.status_code == 403
                else:
                    metadata = ok(response)
                    assert metadata["candidate_id"] == candidate["candidate_id"]
                    assert (
                        metadata["source"] == case["source"]
                        and metadata["target"] == candidate["entity"]
                    )
                    assert metadata["evidence_metadata"]["counts"]["selected"] == 1
                    assert metadata["evidence_metadata"]["bridge_status"] == "not_recorded"
            assert client.get(route, params={"candidate_id": "foreign"}).status_code == 403
            if not baseline:
                ok(mutate(client, "POST", "pause"))
                assert client.get(route, params={"candidate_id": "candidate-1"}).status_code == 403
                ok(mutate(client, "POST", "resume"))
                assert client.get(route, params={"candidate_id": "candidate-1"}).status_code == 200
                if not checked_generation:
                    with TestClient(app, base_url=ORIGIN, headers=dict(client.headers)) as retired:
                        retired.cookies.update(client.cookies)
                        renewal = ok(
                            client.post(
                                f"/api/v1/admin/invitations/{state['session_id']}/reissue",
                                headers=admin,
                            )
                        )
                        assert (
                            retired.get(route, params={"candidate_id": "candidate-1"}).status_code
                            == 401
                        )
                        restored = ok(
                            client.post(
                                "/api/v1/study/session",
                                json={
                                    "secret": renewal["invitation"].split("=", 1)[1],
                                    "client_contract": "study-review/1",
                                },
                            )
                        )
                        assert restored["session_id"] == state["session_id"]
                        assert (
                            client.get(route, params={"candidate_id": "candidate-1"}).status_code
                            == 200
                        )
                        assert (
                            retired.get(route, params={"candidate_id": "candidate-1"}).status_code
                            == 401
                        )
                    checked_generation = True
            ranked = submit_ranking(client)
            ok(consultation(client, ranked, consulted_external_ontologies=False, methods=[]))
            previous_route = route
        assert conditions == {"explanation": 2, "ontology_baseline": 2}
        assert checked_generation
        assert client.get(previous_route, params={"candidate_id": "candidate-1"}).status_code == 403
        assert ok(client.get("/api/v1/study/state"))["stage"] == "final"
    assert {path: file_hash(tmp_path / path) for path in frozen_hashes} == frozen_hashes
