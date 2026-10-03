"""Durable navigation, compatibility and immutable tutorial mutation receipts."""

import json
import subprocess
import sys
from uuid import uuid4

from exact_inspect.study.store import canonical
from tests.explanation_corrective_http_test import corrected, mutate, reach_tutorial
from tests.explanation_study_v2_test import train


def test_tutorial_position_only_mutation_survives_http_reload(corrected):
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    destination = {
        "view": "lesson",
        "lesson_id": state["tutorial"]["lessons"][1]["lesson_id"],
        "question_id": None,
    }
    response = mutate(
        client,
        "PUT",
        "tutorial/progress",
        tutorial_version=state["tutorial"]["version"],
        position=destination,
    )
    assert response.status_code == 200, response.text
    assert response.json()["tutorial_progress"]["position"] == destination
    assert client.get("/api/v1/study/state").json()["tutorial_progress"]["position"] == destination


def test_omitted_lesson_position_preserves_saved_position(corrected):
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    lesson = state["tutorial"]["lessons"][1]["lesson_id"]
    response = mutate(
        client,
        "PUT",
        "tutorial/progress",
        tutorial_version=state["tutorial"]["version"],
        current_lesson_id=lesson,
    )
    assert response.status_code == 200
    response = mutate(
        client,
        "PUT",
        "tutorial/progress",
        tutorial_version=state["tutorial"]["version"],
        help_opened=True,
    )
    assert response.status_code == 200
    assert response.json()["tutorial_progress"]["current_lesson_id"] == lesson


def test_position_shapes_assessment_drafts_conflicts_and_completed_help(corrected):
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    version = state["tutorial"]["version"]
    lesson = state["tutorial"]["lessons"][1]["lesson_id"]
    question = state["tutorial"]["assessment"][0]["question_id"]
    path = "/api/v1/study/tutorial/progress"

    def save(**values):
        return mutate(client, "PUT", "tutorial/progress", tutorial_version=version, **values)

    landing = {"view": "assessment", "lesson_id": None, "question_id": None}
    focused = {**landing, "question_id": question}
    assert save(position=landing).json()["tutorial_progress"]["position"] == landing
    assert save(position=focused).json()["tutorial_progress"]["position"] == focused
    choice = state["tutorial"]["assessment"][0]["options"][0]["code"]
    response = save(
        assessment_draft={"question_id": question, "response": {"choice": choice}},
        current_lesson_id=None,
    )
    assert response.status_code == 200, response.text
    assert response.json()["tutorial_progress"]["position"] == focused
    assert (
        save(lesson_id=lesson, help_opened=True).json()["tutorial_progress"]["position"] == focused
    )
    before = client.get("/api/v1/study/state").json()
    invalid = [
        None,
        {"view": "assessment", "lesson_id": lesson, "question_id": None},
        {"view": "lesson", "lesson_id": "unknown", "question_id": None},
        {"view": "assessment", "lesson_id": None, "question_id": "unknown"},
        {"view": "lesson", "lesson_id": lesson, "question_id": question},
        {"view": "assessment", "lesson_id": None, "question_id": None, "extra": True},
    ]
    for position in invalid:
        assert save(position=position, help_opened=True).status_code == 422
        assert client.get("/api/v1/study/state").json() == before
    assert (
        save(
            position={"view": "lesson", "lesson_id": lesson, "question_id": None},
            current_lesson_id=None,
        ).status_code
        == 422
    )
    assert save(position=landing, current_lesson_id=lesson).status_code == 422
    assert save(position=landing, current_lesson_id=None).status_code == 200
    assert (
        save(current_lesson_id=lesson).json()["tutorial_progress"]["position"]["lesson_id"]
        == lesson
    )
    original = client.get("/api/v1/study/state").json()
    payload = {
        "idempotency_key": uuid4().hex,
        "expected_revision": original["revision"],
        "tutorial_version": version,
        "position": landing,
    }
    receipt = client.put(path, json=payload)
    assert receipt.status_code == 200
    assert client.put(path, json=payload).json() == receipt.json()
    assert client.put(path, json={**payload, "position": focused}).status_code == 409
    assert client.put(path, json={**payload, "idempotency_key": uuid4().hex}).status_code == 409
    # A fresh process observes the exact acknowledged destination and partial draft.
    script = 'from exact_inspect.study.store import StudyStore; import json,sys; s=StudyStore(sys.argv[1],sys.argv[2],allow_test_sqlite=True); print(json.dumps(s.state(sys.argv[3],1)["tutorial_progress"]))'
    reopened = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                script,
                store.database_url,
                str(store.assets_dir),
                state["session_id"],
            ],
            text=True,
        )
    )
    assert reopened == receipt.json()["tutorial_progress"]
    train(store, publication, (state["session_id"], 1))
    completed = client.get("/api/v1/study/state").json()
    helped = save(position=focused, help_opened=True).json()
    assert helped["assignment_id"] == completed["assignment_id"]
    assert (
        helped["tutorial_progress"]["completed_at"]
        == completed["tutorial_progress"]["completed_at"]
    )
    assert helped["tutorial_progress"]["position"] == focused


def test_historical_progress_read_normalization_and_saved_receipt_replay(corrected):
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    sid = state["session_id"]
    version = state["tutorial"]["version"]
    first = state["tutorial"]["lessons"][0]["lesson_id"]
    second = state["tutorial"]["lessons"][1]["lesson_id"]
    question = state["tutorial"]["assessment"][0]["question_id"]

    def raw_state():
        with store.transaction() as db:
            return db.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()["state"]

    for old_lesson, drafts, expected in [
        (second, True, {"view": "lesson", "lesson_id": second, "question_id": None}),
        (None, True, {"view": "assessment", "lesson_id": None, "question_id": None}),
        (None, False, {"view": "lesson", "lesson_id": first, "question_id": None}),
    ]:
        if client.get("/api/v1/study/state").json()["stage"] == "paused":
            assert mutate(client, "POST", "resume", gap_activity="unknown").status_code == 200
        old = json.loads(raw_state())
        progress = old["tutorial_progress"]
        progress.pop("position", None)
        progress["current_lesson_id"] = old_lesson
        progress["assessment_drafts"] = {question: {}} if drafts else {}
        frozen = canonical(old)
        with store.transaction() as db:
            db.execute("UPDATE sessions SET state=? WHERE id=?", (frozen, sid))
        assert client.get("/api/v1/study/state").json()["tutorial_progress"]["position"] == expected
        assert raw_state() == frozen  # No migration on read.
        assert mutate(client, "POST", "pause").status_code == 200
        assert json.loads(raw_state())["tutorial_progress"]["position"] == expected
    # Seed a pre-extension saved receipt, keeping its legacy canonical request hash.
    assert mutate(client, "POST", "resume", gap_activity="unknown").status_code == 200
    state = client.get("/api/v1/study/state").json()
    body = {
        "idempotency_key": uuid4().hex,
        "expected_revision": state["revision"],
        "tutorial_version": version,
        "current_lesson_id": second,
    }
    receipt = client.put("/api/v1/study/tutorial/progress", json=body)
    assert receipt.status_code == 200
    legacy = receipt.json()
    legacy["tutorial_progress"].pop("position")
    legacy.pop("integration_contract", None)
    with store.transaction() as db:
        db.execute(
            "UPDATE mutations SET result=? WHERE session_id=? AND key=?",
            (canonical(legacy), sid, body["idempotency_key"]),
        )
        before = dict(
            db.execute(
                "SELECT * FROM mutations WHERE session_id=? AND key=?",
                (sid, body["idempotency_key"]),
            ).fetchone()
        )
    replay = client.put("/api/v1/study/tutorial/progress", json=body)
    assert replay.status_code == 200 and replay.json() == legacy
    with store.transaction() as db:
        after = dict(
            db.execute(
                "SELECT * FROM mutations WHERE session_id=? AND key=?",
                (sid, body["idempotency_key"]),
            ).fetchone()
        )
    assert after == before
    assert (
        client.get("/api/v1/study/state").json()["tutorial_progress"]["position"]["lesson_id"]
        == second
    )


def test_historical_consultation_receipt_replays_without_new_fields(corrected):
    from tests.explanation_integration_contract_test import allocate

    client, store, publication, sid = allocate(corrected)
    case = client.get("/api/v1/study/cases/current").json()
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
    body = {
        "idempotency_key": uuid4().hex,
        "expected_revision": client.get("/api/v1/study/state").json()["revision"],
        "presentation_id": case["presentation_id"],
        "form_version": "exact-study-forms/2",
        "consulted_external_ontologies": False,
    }
    path = f"/api/v1/study/cases/{case['case_id']}/consultation"
    saved = client.put(path, json=body)
    assert saved.status_code == 200, saved.text
    legacy = saved.json()
    legacy["tutorial_progress"].pop("position")
    legacy.pop("integration_contract", None)
    with store.transaction() as db:
        db.execute(
            "UPDATE mutations SET result=? WHERE session_id=? AND key=?",
            (canonical(legacy), sid, body["idempotency_key"]),
        )
    replay = client.put(path, json=body)
    assert replay.status_code == 200 and replay.json() == legacy
    assert client.get("/api/v1/study/state").json()["tutorial_progress"]["position"]
