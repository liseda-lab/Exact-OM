"""Real HTTP rejection safety, transaction invariance, and old mutation receipt replay."""

import copy
import json
from uuid import UUID, uuid4

import pytest

from exact_inspect.study.store import canonical, digest
from exact_inspect.study.v2_models import TutorialProgressMutation
from tests.explanation_corrective_http_test import (  # noqa: F401
    corrected,
    reach_tutorial,
)
from tests.explanation_study_v2_test import actions_for


def ledger(store, session_id):
    with store.transaction() as db:
        return {
            table: db.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE session_id=?", (session_id,)
            ).fetchone()["n"]
            for table in ("history", "mutations")
        }


@pytest.mark.parametrize("schema_failure", [False, True])
def test_safe_422_has_correlation_without_mutating_and_same_key_can_then_succeed(
    corrected, schema_failure
):  # noqa: F811
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    valid = next(
        a
        for a in actions_for(publication.definition, store.assets_dir)
        if a["action"] == "open_original_axiom"
    )
    key = uuid4().hex
    body = {
        "idempotency_key": key,
        "expected_revision": state["revision"],
        "tutorial_version": state["tutorial"]["version"],
        "actions": [copy.deepcopy(valid)],
    }
    marker = "PRIVATE_TOKEN_MUST_NOT_BE_ECHOED"
    if schema_failure:
        body["actions"][0]["unexpected_input"] = marker
    else:
        body["actions"][0]["fact_id"] = marker
    before = ledger(store, state["session_id"])
    response = client.put("/api/v1/study/tutorial/progress", json=body)
    assert response.status_code == 422, response.text
    rejected = response.json()
    assert len(rejected["detail"]) < 160
    assert marker not in json.dumps(rejected)
    assert rejected["validation"]["code"] == (
        "tutorial_action_payload_invalid" if schema_failure else "tutorial_fact_out_of_scope"
    )
    assert rejected["validation"]["action_ids"]
    UUID(rejected["correlation_id"])
    assert client.get("/api/v1/study/state").json() == state
    assert ledger(store, state["session_id"]) == before
    body["actions"] = [valid]
    accepted = client.put("/api/v1/study/tutorial/progress", json=body)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["revision"] == state["revision"] + 1
    assert client.put("/api/v1/study/tutorial/progress", json=body).json() == accepted.json()
    assert ledger(store, state["session_id"])["mutations"] == before["mutations"] + 1


def test_receipt_from_before_optional_evidence_identity_still_replays(corrected):  # noqa: F811
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    action = next(
        a
        for a in actions_for(publication.definition, store.assets_dir)
        if a["action"] == "open_original_axiom"
    )
    body = {
        "idempotency_key": uuid4().hex,
        "expected_revision": state["revision"],
        "tutorial_version": state["tutorial"]["version"],
        "actions": [action],
    }
    response = client.put("/api/v1/study/tutorial/progress", json=body)
    assert response.status_code == 200, response.text
    old_payload = TutorialProgressMutation.model_validate(body).model_dump(mode="json")
    old_payload.pop("position", None)
    # Old schema genuinely had no evidence_id field, including a null default.
    for old_action in old_payload["actions"]:
        old_action.pop("evidence_id", None)
    old_hash = digest(
        canonical({"operation": "tutorial_progress", "case_id": None, "payload": old_payload})
    )
    with store.transaction() as db:
        db.execute(
            "UPDATE mutations SET payload_hash=? WHERE session_id=? AND key=?",
            (old_hash, state["session_id"], body["idempotency_key"]),
        )
    before = ledger(store, state["session_id"])
    replay = client.put("/api/v1/study/tutorial/progress", json=body)
    assert replay.status_code == 200, replay.text
    assert replay.json() == response.json()
    assert ledger(store, state["session_id"]) == before
