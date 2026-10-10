"""Publication-selected B06 forms, method collection and restricted analysis exports."""

import csv
import io
import json
import os
import zipfile
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from exact_inspect.study.api import create_study_app
from exact_inspect.study.exports import csv_archive
from exact_inspect.study.forms import definitions, validate_answers
from exact_inspect.study.models import Publish
from exact_inspect.study.store import StudyStore, canonical, digest
from tests.explanation_corrective_http_test import mutate
from tests.explanation_study_test import ADMIN, ORIGIN, SECRET, invite
from tests.explanation_study_v2_test import actions_for
from tools.build_explanation_v2_fixture import publication_v2

FORMS = "exact-study-forms/3"
SCHEMA = "exact-study-analysis/4"
METHOD_NAME = "=Tool; Ónto 工具"
FAMILIAR_NAME = "+Familiar; Éditeur"
PRIVATE_ROLE = "private role detail"


def ok(response):
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def review(tmp_path):
    frozen = publication_v2(tmp_path, forms_version=FORMS)
    database = os.environ.get(
        "EXACT_STUDY_TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'forms.sqlite'}"
    )
    app = create_study_app(database, SECRET, ADMIN, ORIGIN, tmp_path, allow_test_sqlite=True)
    store = app.state.study_store
    store.publish(frozen)
    client = TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN})
    state = ok(
        client.post(
            "/api/v1/study/session",
            json={"secret": invite(store, frozen), "client_contract": "study-review/1"},
        )
    )
    client.headers["X-Study-Session"] = state["session_id"]
    return client, store, frozen


def setup(client, *, submitted=False, **values):
    return mutate(
        client,
        "PUT",
        "setup",
        setup_version="setup/3",
        instructions_acknowledged=True,
        external_inspection_optional_understood=True,
        resource_access="available",
        submitted=submitted,
        **values,
    )


def consent(client, frozen):
    return ok(
        mutate(
            client,
            "PUT",
            "consent",
            accepted=True,
            information_version=frozen.definition.information_version,
        )
    )


def answers_for(forms):
    answers = {}
    for question in forms["background"]:
        if not question["required"] or question["show_if"]:
            continue
        answer = question["option_order"][0]
        answers[question["id"]] = [answer] if question["multiple"] else answer
    return answers


def background(client, answers, submitted=True):
    return mutate(
        client,
        "PUT",
        "questionnaires/background",
        form_version=FORMS,
        answers=answers,
        submitted=submitted,
    )


def train(client, store, frozen):
    tutorial = frozen.definition.tutorial
    for _ in range(2):
        ok(
            client.post(
                "/api/v1/study/timing",
                json={
                    "segment_id": uuid4().hex,
                    "page_instance_id": uuid4().hex,
                    "stage": "tutorial",
                    "monotonic_start_ms": 0,
                    "monotonic_end_ms": 60000,
                },
            )
        )
    actions = actions_for(frozen.definition, store.assets_dir)
    method_action = next(a for a in actions if a["action"] == "report_multiple_methods")
    # These were valid for the historical tutorial but must fail on this publication.
    invalid = mutate(
        client,
        "PUT",
        "tutorial/progress",
        tutorial_version=tutorial.version,
        actions=[method_action],
    )
    assert invalid.status_code == 422
    method_action.update(methods=["plain_files", "other_method"], other_method="  ")
    assert (
        mutate(
            client,
            "PUT",
            "tutorial/progress",
            tutorial_version=tutorial.version,
            actions=[method_action],
        ).status_code
        == 422
    )
    method_action["other_method"] = "practice private tool"
    ok(
        mutate(
            client,
            "PUT",
            "tutorial/progress",
            tutorial_version=tutorial.version,
            actions=actions,
            completed_requirements=[a["requirement_id"] for a in actions],
        )
    )
    for question, rule in tutorial.grading.items():
        ok(
            mutate(
                client,
                "POST",
                "tutorial/assessment",
                tutorial_version=tutorial.version,
                question_id=question,
                attempt_id=uuid4().hex,
                response=rule.response.model_dump(exclude_none=True),
            )
        )
    return ok(mutate(client, "POST", "tutorial/complete", tutorial_version=tutorial.version))


def test_review_tutorial_position_only_request_preserves_omission_and_replay(review):
    client, store, frozen = review
    state = consent(client, frozen)
    ok(setup(client, submitted=True))
    state = ok(background(client, answers_for(state["forms"])))
    destination = {
        "view": "lesson",
        "lesson_id": state["tutorial"]["lessons"][1]["lesson_id"],
        "question_id": None,
    }
    path = "/api/v1/study/tutorial/progress"
    body = {
        "idempotency_key": uuid4().hex,
        "expected_revision": state["revision"],
        "tutorial_version": state["tutorial"]["version"],
        "position": destination,
    }
    accepted = ok(client.put(path, json=body))
    assert accepted["tutorial_progress"]["position"] == destination
    assert client.get("/api/v1/study/state").json() == accepted
    assert client.put(path, json=body).json() == accepted
    # Explicit null still disagrees with a lesson position; omission is distinct.
    rejected = client.put(
        path,
        json={
            **body,
            "idempotency_key": uuid4().hex,
            "expected_revision": accepted["revision"],
            "current_lesson_id": None,
        },
    )
    assert rejected.status_code == 422
    assert client.get("/api/v1/study/state").json() == accepted


def submit_ranking(client):
    state = ok(client.get("/api/v1/study/state"))
    return ok(
        mutate(
            client,
            "POST",
            f"cases/{state['current_case_id']}/submit",
            presentation_id=state["current_presentation_id"],
            response_type="none_of_these",
            ranked_candidate_ids=[],
        )
    )


def consultation(client, state, *, draft=False, **values):
    path = f"cases/{state['current_case_id']}/consultation" + ("/draft" if draft else "")
    return mutate(
        client,
        "PUT",
        path,
        form_version=FORMS,
        consultation_version="consultation/3",
        presentation_id=state["current_presentation_id"],
        **values,
    )


def export(client, frozen, *, named=False):
    route = f"/api/v1/admin/studies/{frozen.definition.study_revision}/exports"
    return ok(
        client.post(
            route,
            headers={"Authorization": f"Bearer {ADMIN}"},
            params={"include_test": True, "include_named_methods": named},
        )
    )


def test_frozen_old_codebooks_and_explicit_new_protocol(review):
    client, store, frozen = review
    hashes = {
        "exact-study-forms/1": "4e26923b9ba0272f4e9974edc10023b707400f004419ebaddd1d48399ccc1689",
        "exact-study-forms/2": "c492c1659215dee71d81df0fda7c6d4323529088da52c60c60288e6b4c168822",
    }
    for version, expected in hashes.items():
        assert digest(canonical(definitions(version=version))) == expected
    state = ok(client.get("/api/v1/study/state"))
    assert state["protocol_versions"]["methods"] == "study-methods/3"
    assert state["protocol_versions"]["export"] == SCHEMA
    forms = state["forms"]
    assert forms["method_dictionary"] == {
        "protege": "Protégé",
        "plain_files": "Text editor",
        "other_method": "Other method",
    }
    assert "prefer_not_to_say" not in canonical(forms["background"])
    bands = ["none", "less_than_1", "1_3", "4_5", "6_9", "10_plus"]
    by_id = {q["id"]: q for q in forms["background"]}
    assert by_id["cs_experience_years"]["option_order"] == bands
    assert (
        by_id["domain_specialty"]["label"]
        == "What is your main health or bioscience field, if any?"
    )
    for band in bands:
        answers, _ = validate_answers(
            forms["background"], {"cs_experience_years": band}, submitted=False, version=FORMS
        )
        assert answers["cs_experience_years"] == band
    raw = frozen.model_dump(mode="json")
    raw["definition"]["protocol_versions"].pop("methods")
    with pytest.raises(ValidationError):
        Publish.model_validate(raw)
    # Re-publication and a process-independent store read preserve frozen bytes.
    before = store.publish(frozen)
    restarted = StudyStore(store.database_url, store.assets_dir, allow_test_sqlite=True)
    assert restarted.publish(frozen) == before


def test_optional_familiarity_and_background_hidden_clearing_over_http(review):
    client, store, frozen = review
    consent(client, frozen)
    for methods in (
        ["none", "protege"],
        ["protege", "protege"],
        ["queries_scripts"],
        None,
        3,
        True,
        "protege",
        {},
    ):
        assert setup(client, familiar_methods=methods).status_code == 422
    for value in ("", "  ", "x" * 161):
        assert (
            setup(
                client,
                submitted=True,
                familiar_methods=["other_method"],
                familiar_other_method=value,
            ).status_code
            == 422
        )
    draft = ok(setup(client, familiar_methods=["other_method"], familiar_other_method="  "))
    assert draft["setup"]["familiar_other_method"] is None
    assert (
        export(client, frozen)["data"]["method_responses"][0]["named_detail_state"] == "unanswered"
    )
    ok(setup(client, familiar_methods=["none"], familiar_other_method="discard me"))
    assert (
        export(client, frozen)["data"]["method_responses"][0]["method_answer_state"]
        == "none_of_these"
    )
    state = ok(setup(client, submitted=True, familiar_methods=[]))
    assert state["stage"] == "background"
    assert (
        export(client, frozen)["data"]["method_responses"][0]["method_answer_state"] == "unanswered"
    )
    answers = answers_for(state["forms"])
    for parent in ("roles", "ontology_activities", "doid_contexts", "ncit_contexts"):
        answers[parent] = ["other"]
        answers[parent + "_other"] = "   "
    answers.update(doid_familiarity="used", ncit_familiarity="used")
    draft = ok(background(client, answers, submitted=False))
    assert draft["questionnaires"]["background"]["answer_states"]["roles_other"] == "not_answered"
    assert background(client, answers).status_code == 422
    for parent in ("roles", "ontology_activities", "doid_contexts", "ncit_contexts"):
        answers[parent + "_other"] = " other detail "
    answers["domain_specialty"] = "optional specialty"
    ok(background(client, answers, submitted=False))
    answers.update(
        doid_familiarity="unheard",
        ncit_familiarity="heard",
        roles=["researcher"],
        ontology_activities=["never_used"],
    )
    state = ok(background(client, answers))
    saved = state["questionnaires"]["background"]
    for key in (
        "roles_other",
        "ontology_activities_other",
        "doid_contexts",
        "doid_contexts_other",
        "ncit_contexts",
        "ncit_contexts_other",
    ):
        assert key not in saved["answers"]
        assert saved["answer_states"][key] == "skipped"
    restarted = TestClient(
        create_study_app(
            store.database_url, SECRET, ADMIN, ORIGIN, store.assets_dir, allow_test_sqlite=True
        ),
        base_url=ORIGIN,
        headers=dict(client.headers),
        cookies=client.cookies,
    )
    assert ok(restarted.get("/api/v1/study/state"))["questionnaires"] == state["questionnaires"]


def test_actual_consultation_export_opt_in_and_immutable_restart(review):
    client, store, frozen = review
    consent(client, frozen)
    state = ok(
        setup(
            client,
            submitted=True,
            familiar_methods=["protege", "other_method"],
            familiar_other_method="  " + FAMILIAR_NAME + "  ",
        )
    )
    answers = answers_for(state["forms"])
    answers.update(roles=["other"], roles_other=PRIVATE_ROLE, domain_specialty="private specialty")
    ok(background(client, answers))
    train(client, store, frozen)
    state = submit_ranking(client)
    assert state["consultation_draft"] is None
    assert state["consultation"] is None
    for malformed in (None, 3, True, "protege", {}):
        for draft in (True, False):
            assert (
                consultation(
                    client,
                    state,
                    draft=draft,
                    consulted_external_ontologies=True,
                    methods=malformed,
                ).status_code
                == 422
            )
    for values in (
        {},
        {"consulted_external_ontologies": True},
        {"consulted_external_ontologies": True, "methods": ["reasoner"]},
        {"consulted_external_ontologies": True, "methods": ["protege", "protege"]},
        {
            "consulted_external_ontologies": True,
            "methods": ["other_method"],
            "other_method": " ",
        },
        {
            "consulted_external_ontologies": True,
            "methods": ["other_method"],
            "other_method": "x" * 161,
        },
    ):
        assert consultation(client, state, **values).status_code == 422
    ok(
        consultation(
            client,
            state,
            draft=True,
            consulted_external_ontologies=True,
            methods=["other_method"],
        )
    )
    draft_export = export(client, frozen)
    assert draft_export["data"]["method_responses"][-1]["named_detail_state"] == "unanswered"
    ok(
        consultation(
            client,
            state,
            draft=True,
            consulted_external_ontologies=True,
            methods=["other_method"],
            other_method="historical hidden name",
        )
    )
    cleared = ok(
        consultation(
            client,
            state,
            draft=True,
            consulted_external_ontologies=False,
            methods=["other_method"],
            other_method="historical hidden name",
        )
    )
    assert cleared["consultation_draft"]["methods"] == []
    assert cleared["consultation_draft"]["other_method"] is None
    ok(
        consultation(
            client,
            state,
            draft=True,
            consulted_external_ontologies=True,
            methods=["plain_files"],
            other_method="deselected name",
        )
    )
    current = ok(client.get("/api/v1/study/state"))
    assert current["consultation_draft"]["other_method"] is None
    with store.transaction() as db:
        history = db.execute(
            "SELECT payload FROM history WHERE session_id = ? AND operation = ?",
            (current["session_id"], "consultation_draft"),
        ).fetchall()
        assert "historical hidden name" in canonical([row["payload"] for row in history])
    next_state = ok(
        consultation(
            client,
            state,
            consulted_external_ontologies=True,
            methods=["protege", "plain_files", "other_method"],
            other_method="  " + METHOD_NAME + "  ",
        )
    )
    assert next_state["stage"] == "case"
    assert next_state["consultation_draft"] is None
    assert next_state["consultation"] is None
    state = submit_ranking(client)
    assert state["consultation_draft"] is None
    ok(
        consultation(
            client,
            state,
            consulted_external_ontologies=False,
            methods=["other_method"],
            other_method="discarded final detail",
        )
    )
    normal = export(client, frozen)
    named = export(client, frozen, named=True)
    assert normal["manifest"]["schema"] == SCHEMA
    assert normal["manifest"]["includes_named_method_details"] is False
    assert named["manifest"]["includes_named_method_details"] is True
    assert (
        named["manifest"]["source_protocol_versions"]
        == frozen.definition.protocol_versions.model_dump()
    )
    for artifact in (normal, named):
        raw = canonical(artifact)
        for excluded in (
            PRIVATE_ROLE,
            "private specialty",
            "historical hidden name",
            "deselected name",
            "discarded final detail",
            "practice private tool",
        ):
            assert excluded not in raw
        session = artifact["data"]["sessions"][0]
        assert session["tutorial_timing"]["page_observation_seconds"] == 120
        assert session["tutorial_timing"]["page_instance_count"] == 2
        assert session["tutorial_observed_seconds"] == 120
        assert session["tutorial_timing"]["unique_elapsed_coverage_seconds"] is None
        timing = session["cases"][0]["timing"]
        assert timing["unique_elapsed_coverage_seconds"] is None
        assert timing["unobserved_elapsed_seconds"] is None
        assert timing["coverage_status"] == "not_established"
        assert timing["active_duration_known"] is False
    assert METHOD_NAME not in canonical(normal) and FAMILIAR_NAME not in canonical(normal)
    assert METHOD_NAME in canonical(named) and FAMILIAR_NAME in canonical(named)
    assert METHOD_NAME not in canonical(named["manifest"])
    rows = named["data"]["method_responses"]
    actual = [row for row in rows if row["stage"] == "consultation"]
    assert {row["method_answer_state"] for row in actual} == {"yes", "no"}
    assert all(row["case_id"] and row["presentation_id"] and row["condition"] for row in actual)
    assert {row["named_detail_state"] for row in actual} == {"included", "hidden"}
    for artifact in (normal, named):
        archive = csv_archive(artifact)
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            parsed = list(csv.DictReader(io.StringIO(zipped.read("method_responses.csv").decode())))
            manifest = json.loads(zipped.read("manifest.json"))
            assert manifest["archive_schema"] == "exact-study-csv/4"
            for path in zipped.namelist():
                assert PRIVATE_ROLE.encode() not in zipped.read(path)
            if artifact is normal:
                assert all("other_method" not in row for row in parsed)
            else:
                assert {row["other_method"] for row in parsed} >= {
                    "'" + METHOD_NAME,
                    "'" + FAMILIAR_NAME,
                }
        restarted = StudyStore(store.database_url, store.assets_dir, allow_test_sqlite=True)
        assert restarted.saved_export(artifact["manifest"]["export_id"]) == artifact
        assert csv_archive(restarted.saved_export(artifact["manifest"]["export_id"])) == archive
    route = f"/api/v1/admin/studies/{frozen.definition.study_revision}/exports"
    assert client.post(route, params={"include_named_methods": True}).status_code == 401
    assert (
        client.post(
            route,
            headers={"Authorization": f"Bearer {ADMIN}"},
            params={"analysis_schema": "exact-study-analysis/3"},
        ).status_code
        == 422
    )
    saved_route = "/api/v1/admin/exports/" + named["manifest"]["export_id"]
    assert client.get(saved_route).status_code == 401
    assert ok(client.get(saved_route, headers={"Authorization": f"Bearer {ADMIN}"})) == named
