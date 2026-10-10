"""Actual HTTP admission and negotiation for real-data development purpose."""

from copy import deepcopy
import os

import pytest
from fastapi.testclient import TestClient

from exact_inspect.study.api import create_study_app
from exact_inspect.study.models import Publish
from tools.build_explanation_v2_fixture import publication_v2


def review_fixture(root):
    # Ontology contents are invented test fixtures; this only exercises admission.
    payload = publication_v2(root, forms_version="exact-study-forms/3").model_dump(mode="json")
    payload["definition"].update(
        purpose="development_review",
        synthetic=False,
        correctness_metrics="disabled",
        review_provenance=dict(
            run_manifest_sha256="a" * 64,
            selection_manifest_sha256="b" * 64,
            source_run_id="test-contract-only",
            limitation="Synthetic fixture verifies publication mechanics, not real-data coverage.",
        ),
    )
    for key in payload["case_keys"]:
        key.update(case_kind="unresolved", origin="unresolved", acceptable_candidate_ids=[])
    return payload


def with_catalogue(payload):
    from exact_inspect.review_metadata import ReviewBundleSummary

    definition = payload["definition"]
    definition["review_bundle"] = ReviewBundleSummary(
        title="Synthetic contract check",
        prepared_at="2026-10-10T00:00:00Z",
        source_run_id="test-contract-only",
        run_manifest_sha256="sha256:" + "a" * 64,
        selection_sha256="sha256:" + "b" * 64,
        policy_sha256="sha256:" + definition["policy_hash"],
        source_count=len(definition["cases"]),
        presented_pairs=sum(len(c["candidates"]) for c in definition["cases"]),
        coverage={},
        limitations=[],
        ontologies=[],
        cases=[
            {
                "case_id": c["case_id"],
                "source": c["source"],
                "targets": [t["entity"] for t in c["candidates"]],
            }
            for c in definition["cases"]
        ],
    ).model_dump()
    return payload


def test_review_purpose_is_explicit_and_test_only_at_actual_api(tmp_path):
    frozen = with_catalogue(review_fixture(tmp_path))
    app = create_study_app(
        database_url=os.environ.get(
            "EXACT_STUDY_TEST_DATABASE_URL", "sqlite:///" + str(tmp_path / "study.sqlite")
        ),
        signing_secret="s" * 48,
        researcher_token="r" * 48,
        origin="https://localhost",
        assets_dir=tmp_path,
        allow_test_sqlite=True,
    )
    with TestClient(
        app, base_url="https://localhost", headers={"Origin": "https://localhost"}
    ) as client:
        headers = {"Authorization": "Bearer " + "r" * 48}
        response = client.post("/api/v1/admin/studies", json=frozen, headers=headers)
        assert response.status_code == 200, response.text
        revision = frozen["definition"]["study_revision"]
        endpoint = f"/api/v1/admin/studies/{revision}/invitations"
        assert client.post(endpoint, json={"test": False}, headers=headers).status_code == 422
        invite = client.post(endpoint, json={"test": True}, headers=headers).json()["invitations"][
            0
        ]["invitation"]
        secret = invite.split("invite=")[1]
        assert client.post("/api/v1/study/session", json={"secret": secret}).status_code == 409
        response = client.post(
            "/api/v1/study/session", json={"secret": secret, "client_contract": "study-review/1"}
        )
        assert response.status_code == 200, response.text
        state = response.json()
        assert state["synthetic"] is False
        assert state["purpose"] == "development_review"
        assert state["correctness_metrics"] == "disabled"
        assert state["revision"] == 0
        public_about = {**frozen["definition"]["review_bundle"], "cases": []}
        assert state["review_bundle"] == public_about
        assert state["assignment_id"] is None
        assert client.get("/api/v1/study/state").json()["review_bundle"] == public_about
        export = client.post(
            f"/api/v1/admin/studies/{revision}/exports?include_test=true", headers=headers
        ).json()
        assert export["data"]["correctness_metrics"] == "disabled"
        assert export["data"]["summary"] == {"status": "disabled_development_review"}
        assert export["manifest"]["schema"] == "exact-study-analysis/4"
        before = response.json()
        assert client.post("/api/v1/admin/studies", json=frozen, headers=headers).status_code == 200
        assert client.get("/api/v1/study/state").json() == before
        # Project participant metadata without mutating the immutable publication.
        with app.state.study_store.transaction() as db:
            _, saved = app.state.study_store._study(db, revision)
        assert saved["review_bundle"] == frozen["definition"]["review_bundle"]
        assert len(saved["review_bundle"]["cases"]) == len(saved["cases"])

        consent = {
            "idempotency_key": "review-about-consent",
            "expected_revision": state["revision"],
            "information_version": frozen["definition"]["information_version"],
            "accepted": True,
        }
        response = client.put("/api/v1/study/consent", json=consent)
        assert response.status_code == 200, response.text
        acknowledged = response.json()
        assert acknowledged["review_bundle"] == public_about
        # Exercise a previously stored development receipt, not merely fresh saves.
        from exact_inspect.study.store import canonical

        historical = canonical(
            {**acknowledged, "review_bundle": frozen["definition"]["review_bundle"]}
        )
        with app.state.study_store.transaction() as db:
            db.execute(
                "UPDATE mutations SET result = ? WHERE session_id = ? AND key = ?",
                (historical, state["session_id"], consent["idempotency_key"]),
            )
        replay = client.put("/api/v1/study/consent", json=consent)
        assert replay.status_code == 200 and replay.json() == acknowledged
        assert client.get("/api/v1/study/state").json() == acknowledged
        with app.state.study_store.transaction() as db:
            retained = db.execute(
                "SELECT result FROM mutations WHERE session_id = ? AND key = ?",
                (state["session_id"], consent["idempotency_key"]),
            ).fetchone()
        assert retained["result"] == historical


def test_development_purpose_cannot_bypass_research_approval_or_adjudication(tmp_path):
    payload = review_fixture(tmp_path)
    old = deepcopy(payload)
    for key in ("purpose", "review_provenance", "correctness_metrics"):
        del old["definition"][key]
    with pytest.raises(ValueError, match="launch approvals"):
        Publish.model_validate(old)
    app = create_study_app(
        database_url=os.environ.get(
            "EXACT_STUDY_TEST_DATABASE_URL", "sqlite:///" + str(tmp_path / "study.sqlite")
        ),
        signing_secret="s" * 48,
        researcher_token="r" * 48,
        origin="https://localhost",
        assets_dir=tmp_path,
        allow_test_sqlite=True,
    )
    with TestClient(app, base_url="https://localhost") as client:
        payload["case_keys"][0].update(
            case_kind="answer_present",
            origin="natural",
            acceptable_candidate_ids=[
                payload["definition"]["cases"][0]["candidates"][0]["candidate_id"]
            ],
        )
        response = client.post(
            "/api/v1/admin/studies", json=payload, headers={"Authorization": "Bearer " + "r" * 48}
        )
        assert response.status_code == 422 and "unresolved" in response.text


def test_exact_frozen_publication_retry_does_not_apply_new_reachability(tmp_path, monkeypatch):
    from exact_inspect.study.store import StudyStore
    from exact_inspect.study import tutorial

    payload = publication_v2(tmp_path)
    store = StudyStore(
        "sqlite:///" + str(tmp_path / "frozen.sqlite"), assets_dir=tmp_path, allow_test_sqlite=True
    )
    store.migrate()
    original = store.publish(payload)

    def newly_unreachable(*args, **kwargs):
        raise ValueError("new admission rule must not reinterpret frozen history")

    monkeypatch.setattr(tutorial, "validate_publication", newly_unreachable)
    assert store.publish(payload) == original


def test_review_case_catalogue_cannot_claim_different_presented_entities(tmp_path):
    from exact_inspect.review_metadata import ReviewBundleSummary

    payload = review_fixture(tmp_path)
    definition = payload["definition"]
    about = ReviewBundleSummary(
        title="Synthetic contract check",
        prepared_at="2026-10-10T00:00:00Z",
        source_run_id="test-contract-only",
        run_manifest_sha256="sha256:" + "a" * 64,
        selection_sha256="sha256:" + "b" * 64,
        policy_sha256="sha256:" + definition["policy_hash"],
        source_count=len(definition["cases"]),
        presented_pairs=sum(len(c["candidates"]) for c in definition["cases"]),
        coverage={},
        limitations=[],
        ontologies=[],
        cases=[
            {
                "case_id": c["case_id"],
                "source": c["source"],
                "targets": [t["entity"] for t in c["candidates"]],
            }
            for c in definition["cases"]
        ],
    )
    definition["review_bundle"] = about.model_dump()
    Publish.model_validate(payload)
    definition["review_bundle"]["cases"][0]["targets"][0]["kind"] = "individual"
    with pytest.raises(ValueError, match="case identities differ"):
        Publish.model_validate(payload)
