"""Small synthetic study checks; optionally run unchanged against real PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from exact_inspect.study import StudyError, StudyStore, create_study_app
from exact_inspect.study.forms import definitions, validate_answers
from exact_inspect.study.models import (
    Consent,
    Consultation,
    EventBatch,
    Mutation,
    Publish,
    Questionnaire,
    Ranking,
    Resume,
    Setup,
    TimingSegment,
)
from exact_inspect.study.scoring import score_response, summarize

ORIGIN = "https://study.example.test"
SECRET = "test-signing-secret-never-deploy-00000000"
ADMIN = "test-researcher-secret-never-deploy-00000"


def publication(tmp_path):
    """Create explicit synthetic frozen resources and private adjudication."""
    revision = "synthetic-" + uuid4().hex
    import pyowl_core as core

    from exact_inspect.context import build_context_package
    from exact_inspect.context_resources import export_ontology_resource
    from exact_inspect.contracts import VisibilityPolicy

    contexts = {}
    for name, count in (("source", 4), ("target", 5)):
        identifiers = range(count) if name == "source" else range(1, count + 1)
        axioms = [f"Declaration(Class(<urn:{name}:{i}>))" for i in identifiers]
        if name == "target":
            axioms.append(
                'AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:target:1> "Synthetic frozen context")'
            )
        ontology = ("Ontology(<urn:synthetic:" + name + "> " + " ".join(axioms) + ")").encode()
        snapshot = core.load_snapshot(
            ontology,
            options=core.LoadOptions(
                backend=core.BackendPreference.NATIVE,
                imports=core.ImportPolicy.IGNORE,
                preserve_source_map=True,
            ),
        )
        contexts[name] = build_context_package(snapshot, tmp_path / (name + "-context"))
    policy = VisibilityPolicy(
        policy_id="synthetic-study-v1",
        ontology_ids=tuple(c.ontology_version_id for c in contexts.values()),
    )
    policy_hash = policy.policy_hash.removeprefix("sha256:")
    assets = []
    for name, context in contexts.items():
        receipt_path = export_ontology_resource(context, tmp_path / name, policy)
        content = (tmp_path / name).read_bytes()
        assets.append(
            {
                "asset_id": name,
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "kind": "ontology",
                "media_type": "application/owl-functional",
                "policy_hash": policy_hash,
                "admission_receipt_path": str(Path(receipt_path).relative_to(tmp_path)),
                "admission_receipt_sha256": hashlib.sha256(
                    Path(receipt_path).read_bytes()
                ).hexdigest(),
            }
        )
    from exact_inspect.contracts import EntityRef
    from exact_inspect.study.builder import build_explanation_resource

    target = EntityRef(
        ontology_version_id=contexts["target"].ontology_version_id,
        iri="urn:target:1",
        kind="class",
    )
    resource = build_explanation_resource(
        {context.ontology_version_id: context for context in contexts.values()},
        [target],
        policy=policy,
    )
    content = resource.model_dump_json().encode()
    (tmp_path / "context").write_bytes(content)
    assets.append(
        {
            "asset_id": "context",
            "path": "context",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "kind": "explanation",
            "media_type": "application/json",
            "policy_hash": policy_hash,
        }
    )
    cases, keys = [], []
    for n in range(4):
        candidates = [
            {
                "candidate_id": f"candidate-{i}",
                "entity": {
                    "ontology_version_id": contexts["target"].ontology_version_id,
                    "iri": f"urn:target:{i}",
                    "kind": "class",
                },
                "label": f"Synthetic candidate {i}",
                "score": 1 - i / 10,
                "score_meaning": "Saved synthetic score; not probability of correctness",
                "display_position": i,
            }
            for i in range(1, 6)
        ]
        cases.append(
            {
                "case_id": f"case-{n}",
                "source": {
                    "ontology_version_id": contexts["source"].ontology_version_id,
                    "iri": f"urn:source:{n}",
                    "kind": "class",
                },
                "source_label": f"Synthetic source {n}",
                "transfer_group": f"source-{n}",
                "package_version": "synthetic-package-v1",
                "candidates": candidates,
                "ontology_resource_ids": ["source", "target"],
                "explanation_refs": ["context"],
            }
        )
        keys.append(
            {
                "case_id": f"case-{n}",
                "case_kind": "answer_present" if n < 2 else "answer_absent",
                "acceptable_candidate_ids": [f"candidate-{n + 1}"] if n < 2 else [],
                "adjudication_version": "synthetic-key-v1",
                "criterion": "Synthetic equivalence only",
                "evidence": ["Synthetic oracle; no real adjudication"],
                "origin": "natural",
                "original_production_ranks": {f"candidate-{i}": i for i in range(1, 6)},
            }
        )
    schedules = []
    for form in range(2):
        blocks = [
            {
                "condition": "explanation",
                "case_ids": ["case-0", "case-2"] if form == 0 else ["case-1", "case-3"],
            },
            {
                "condition": "ontology_baseline",
                "case_ids": ["case-1", "case-3"] if form == 0 else ["case-0", "case-2"],
            },
        ]
        for order in range(2):
            schedules.append(
                {
                    "schedule_id": f"form-{form}-order-{order}",
                    "blocks": blocks if order == 0 else list(reversed(blocks)),
                }
            )
    return Publish.model_validate(
        {
            "definition": {
                "study_revision": revision,
                "software_version": "synthetic-build-v1",
                "form_version": "synthetic-forms-with-consultation-v1",
                "information_version": "synthetic-information-v1",
                "information_text": "Synthetic development session. No human study is launched.",
                "consent_text": "Synthetic consent flow test, not research consent wording.",
                "instructions": "Rank plausible equivalents, select none or insufficient information.",
                "setup_instructions": "Synthetic setup instructions for testing only.",
                "tutorial_steps": ["Simple", "Complex", "Partial ranking", "Explicit none"],
                "closes_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                "synthetic": True,
                "policy_hash": policy_hash,
                "visibility_policy": policy.model_dump(mode="json"),
                "analysis_plan": "Synthetic oracle; no participant analysis.",
                "cases": cases,
                "assets": assets,
                "schedules": schedules,
            },
            "case_keys": keys,
        }
    )


@pytest.fixture
def service(tmp_path):
    database_url = os.environ.get(
        "EXACT_STUDY_TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'study.sqlite'}"
    )
    app = create_study_app(database_url, SECRET, ADMIN, ORIGIN, tmp_path, allow_test_sqlite=True)
    store = app.state.study_store
    frozen = publication(tmp_path)
    store.publish(frozen)
    return app, store, frozen


def invite(store, frozen):
    value = store.issue(frozen.definition.study_revision)["invitations"][0]
    return value["invitation"].split("=", 1)[1]


def write(store, identity, operation, model, **values):
    state = store.state(*identity)
    return store.mutate(
        *identity,
        operation,
        model(idempotency_key=uuid4().hex, expected_revision=state["revision"], **values),
    )


def background_answers():
    return {
        q["id"]: (["prefer_not_to_say"] if q["multiple"] else "prefer_not_to_say")
        for q in definitions()["background"]
        if q["required"] and not q["show_if"]
    }


def ready_session(store, frozen):
    secret = invite(store, frozen)
    sid, generation, state = store.exchange(secret)
    identity = sid, generation
    write(
        store,
        identity,
        "consent",
        Consent,
        information_version=frozen.definition.information_version,
        accepted=True,
    )
    setup = {
        "protege_installed": True,
        "source_opened": True,
        "target_opened": True,
        "practice_source_located": True,
        "practice_definition_parents_inspected": True,
    }
    write(store, identity, "setup", Setup, **setup)
    write(
        store,
        identity,
        "questionnaire:background",
        Questionnaire,
        form_version=frozen.definition.form_version,
        answers=background_answers(),
        submitted=True,
    )
    write(store, identity, "setup", Setup, **setup, completed_tutorial_steps=[0, 1, 2, 3])
    return identity, secret


def ready_event(store, identity, *, event_id=None, sequence=0, page_id=None):
    case = store.current_case(*identity)
    event = {
        "event_id": event_id or uuid4().hex,
        "page_instance_id": page_id or uuid4().hex,
        "sequence": sequence,
        "case_id": case["case_id"],
        "presentation_id": case["presentation_id"],
        "type": "case_ready",
        "client_monotonic_ms": 100.0,
        "build_version": "synthetic-build-v1",
    }
    store.events(*identity, EventBatch(events=[event]))
    return event


def submit_case(store, identity, *, response_type="none_of_these", ranked=()):
    case = store.current_case(*identity)
    ready_event(store, identity)
    state = store.state(*identity)
    body = Ranking(
        idempotency_key=uuid4().hex,
        expected_revision=state["revision"],
        presentation_id=case["presentation_id"],
        response_type=response_type,
        ranked_candidate_ids=list(ranked),
    )
    return store.mutate(*identity, "submit", body, case["case_id"]), case


def test_reusable_invitation_cookie_loss_reissue_revocation_and_no_preview_allocation(service):
    app, store, frozen = service
    secret = invite(store, frozen)
    client = TestClient(app, base_url=ORIGIN)
    assert client.get("/participate").status_code == 404  # Separate frontend owns the shell.
    response = client.post(
        "/api/v1/study/session", json={"secret": secret}, headers={"Origin": ORIGIN}
    )
    assert response.status_code == 200
    first = response.json()
    assert first["stage"] == "welcome" and first["assignment_id"] is None
    cookie = client.cookies.get("exact_study_session")
    assert all(
        value in response.headers["set-cookie"].lower()
        for value in ["secure", "httponly", "samesite=strict"]
    )
    client.cookies.clear()
    repeated = client.post(
        "/api/v1/study/session", json={"secret": secret}, headers={"Origin": ORIGIN}
    ).json()
    assert repeated["session_id"] == first["session_id"] and repeated["revision"] == 0
    replacement = store.reissue(first["session_id"])
    assert client.get("/api/v1/study/state").status_code == 401
    client.cookies.set("exact_study_session", cookie, path="/api/v1/study")
    assert client.get("/api/v1/study/state").status_code == 401
    assert (
        client.post(
            "/api/v1/study/session", json={"secret": secret}, headers={"Origin": ORIGIN}
        ).status_code
        == 401
    )
    sid, generation, state = store.exchange(replacement["invitation"].split("=", 1)[1])
    assert sid == first["session_id"] and generation == 2 and state["revision"] == 0
    store.reissue(sid, revoke=True)
    with pytest.raises(StudyError, match="unavailable"):
        store.state(sid, generation)


def test_consent_setup_form_and_practice_gates(service):
    _, store, frozen = service
    sid, generation, state = store.exchange(invite(store, frozen))
    identity = sid, generation
    with pytest.raises(StudyError, match="Consent"):
        write(
            store,
            identity,
            "setup",
            Setup,
            protege_installed=False,
            source_opened=False,
            target_opened=False,
            practice_source_located=False,
            practice_definition_parents_inspected=False,
        )
    declined = write(
        store,
        identity,
        "consent",
        Consent,
        information_version=frozen.definition.information_version,
        accepted=False,
    )
    assert declined["stage"] == "closed"
    with pytest.raises(StudyError, match="closed"):
        write(store, identity, "complete", Mutation)
    second = store.exchange(invite(store, frozen))[:2]
    write(
        store,
        second,
        "consent",
        Consent,
        information_version=frozen.definition.information_version,
        accepted=True,
    )
    failure = write(
        store,
        second,
        "setup",
        Setup,
        protege_installed=False,
        source_opened=False,
        target_opened=False,
        practice_source_located=False,
        practice_definition_parents_inspected=False,
    )
    assert failure["stage"] == "setup" and failure["assignment_id"] is None
    with pytest.raises(StudyError, match="unavailable"):
        write(
            store,
            second,
            "questionnaire:background",
            Questionnaire,
            form_version=frozen.definition.form_version,
            answers=background_answers(),
            submitted=True,
        )


def test_idempotent_submit_draft_race_and_frozen_first_response(service):
    _, store, frozen = service
    identity, _ = ready_session(store, frozen)
    case = store.current_case(*identity)
    ready_event(store, identity)
    revision = store.state(*identity)["revision"]
    body = Ranking(
        idempotency_key=uuid4().hex,
        expected_revision=revision,
        presentation_id=case["presentation_id"],
        response_type="ranked_candidates",
        ranked_candidate_ids=["candidate-2", "candidate-4"],
    )
    draft = body.model_copy(
        update={
            "idempotency_key": uuid4().hex,
            "response_type": "none_of_these",
            "ranked_candidate_ids": [],
        }
    )

    def attempt(operation, payload):
        try:
            return store.mutate(*identity, operation, payload, case["case_id"])
        except StudyError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda values: attempt(*values), [("submit", body), ("draft", draft)])
        )
    assert sum(isinstance(x, StudyError) for x in results) == 1
    current = store.state(*identity)
    if current["stage"] == "case":
        body = body.model_copy(update={"expected_revision": current["revision"]})
        committed = store.mutate(*identity, "submit", body, case["case_id"])
    else:
        committed = results[0]
    assert committed["stage"] == "consultation"
    assert store.mutate(*identity, "submit", body, case["case_id"]) == committed
    with pytest.raises(StudyError, match="different content"):
        store.mutate(
            *identity,
            "submit",
            body.model_copy(update={"ranked_candidate_ids": ["candidate-1"]}),
            case["case_id"],
        )
    with pytest.raises(StudyError, match="read-only"):
        store.mutate(
            *identity,
            "draft",
            draft.model_copy(
                update={
                    "expected_revision": committed["revision"],
                    "idempotency_key": uuid4().hex,
                }
            ),
            case["case_id"],
        )
    assert store.state(*identity)["ranking"]["ranked_candidate_ids"] == [
        "candidate-2",
        "candidate-4",
    ]


def test_full_flow_baseline_bypass_consultation_resume_and_completion(service, monkeypatch):
    app, store, frozen = service
    import pyowl_core as core

    def runtime_parse_forbidden(*args, **kwargs):
        raise AssertionError("Participation must never parse an ontology")

    monkeypatch.setattr(core, "load_snapshot", runtime_parse_forbidden)
    identity, secret = ready_session(store, frozen)
    assigned = store.state(*identity)["assignment_id"]
    client = TestClient(app, base_url=ORIGIN)
    client.post("/api/v1/study/session", json={"secret": secret}, headers={"Origin": ORIGIN})
    seen, conditions = set(), set()
    for _ in range(4):
        case = store.current_case(*identity)
        assert case["case_id"] not in seen
        seen.add(case["case_id"])
        conditions.add(case["condition"])
        serialized = json.dumps(case)
        assert all(
            field not in serialized
            for field in [
                "case_kind",
                "acceptable_candidate_ids",
                "original_production_ranks",
                "transfer_group",
            ]
        )
        expected = 200 if case["condition"] == "explanation" else 403
        assert (
            client.get("/api/v1/study/resources/context?condition=explanation").status_code
            == expected
        )
        assert case["explanation_refs"] == (["context"] if expected == 200 else [])
        assert client.get("/api/v1/study/resources/source").status_code == 200
        assert client.get("/api/study/source?source=anything").status_code == 404
        state, case = submit_case(store, identity)
        assert state["stage"] == "consultation"
        resumed = store.exchange(secret)[2]
        assert resumed["stage"] == "consultation" and resumed["ranking"]["submitted_at"]
        assert resumed["assignment_id"] == assigned
        payload = Consultation(
            idempotency_key=uuid4().hex,
            expected_revision=resumed["revision"],
            consulted_external_ontologies=True,
            methods=["protege", "plain_files"],
        )
        progressed = store.mutate(*identity, "consultation", payload, case["case_id"])
        assert store.mutate(*identity, "consultation", payload, case["case_id"]) == progressed
    assert conditions == {"explanation", "ontology_baseline"}
    assert store.state(*identity)["stage"] == "final"
    with pytest.raises(StudyError, match="incomplete"):
        write(store, identity, "complete", Mutation)
    answers = {
        "component_usefulness": {c: "cannot_judge" for c in frozen.definition.components},
        "most_helpful_components": ["cannot_judge"],
        "workflow_preference": "no_preference",
        "mental_effort": {"explanation": "moderate", "ontology_baseline": "moderate"},
    }
    write(
        store,
        identity,
        "questionnaire:final",
        Questionnaire,
        form_version=frozen.definition.form_version,
        answers=answers,
        submitted=True,
    )
    completed = write(store, identity, "complete", Mutation)
    assert completed["stage"] == "completed" and completed["completed_cases"] == 4
    assert store.exchange(secret)[2]["stage"] == "completed"


def test_events_dedup_gaps_pause_visibility_and_timing(service):
    _, store, frozen = service
    identity, _ = ready_session(store, frozen)
    event = ready_event(store, identity, sequence=3)
    assert store.events(*identity, EventBatch(events=[event]))["acknowledged_event_ids"] == [
        event["event_id"]
    ]
    with pytest.raises(StudyError, match="different content"):
        store.events(*identity, EventBatch(events=[{**event, "type": "rank_add"}]))
    visibility = {
        **event,
        "event_id": uuid4().hex,
        "sequence": 4,
        "type": "visibility",
        "visibility": "hidden",
    }
    store.events(*identity, EventBatch(events=[visibility]))
    assert store.state(*identity)["stage"] == "case"
    segment = TimingSegment(
        segment_id=uuid4().hex,
        page_instance_id=event["page_instance_id"],
        case_id=event["case_id"],
        presentation_id=event["presentation_id"],
        stage="case",
        monotonic_start_ms=100,
        monotonic_end_ms=4000,
    )
    assert store.timing(*identity, segment) == store.timing(*identity, segment)
    with pytest.raises(StudyError, match="usable"):
        store.timing(
            *identity,
            segment.model_copy(update={"segment_id": uuid4().hex, "page_instance_id": uuid4().hex}),
        )
    with pytest.raises(StudyError, match="usable"):
        store.timing(
            *identity,
            segment.model_copy(update={"segment_id": uuid4().hex, "monotonic_start_ms": 0}),
        )
    with pytest.raises(StudyError, match="overlap"):
        store.timing(*identity, segment.model_copy(update={"segment_id": uuid4().hex}))
    paused = write(store, identity, "pause", Mutation)
    assert paused["stage"] == "paused"
    with pytest.raises(StudyError, match="no longer current"):
        store.timing(
            *identity,
            segment.model_copy(
                update={
                    "segment_id": uuid4().hex,
                    "monotonic_start_ms": 4000,
                    "monotonic_end_ms": 5000,
                }
            ),
        )
    resumed = write(store, identity, "resume", Resume, gap_activity="unknown")
    assert resumed["stage"] == "case"
    export = store.export(frozen.definition.study_revision, include_test=True)
    record = export["data"]["sessions"][0]
    assert record["events"][0]["sequence_gap_before"] == {
        "page_instance_id": event["page_instance_id"],
        "from": 0,
        "to": 2,
    }
    assert record["cases"][0]["timing"]["observed_segment_seconds"] == 3.9
    assert record["cases"][0]["timing"]["active_duration_known"] is False


def test_form_branching_exclusive_options_drafts_and_free_text_export(service):
    _, store, frozen = service
    answers = background_answers()
    answers.update(
        doid_familiarity="used",
        doid_contexts=["research", "clinical"],
        roles=["researcher", "clinician"],
    )
    normalized, states = validate_answers(definitions()["background"], answers, submitted=True)
    assert normalized["roles"] == ["researcher", "clinician"]
    answers["doid_familiarity"] = "heard"
    normalized, states = validate_answers(definitions()["background"], answers, submitted=True)
    assert "doid_contexts" not in normalized and states["doid_contexts"] == "skipped"
    answers["roles"] = ["researcher", "prefer_not_to_say"]
    with pytest.raises(ValueError, match="exclusive"):
        validate_answers(definitions()["background"], answers, submitted=True)
    normalized, states = validate_answers(definitions()["background"], {}, submitted=False)
    assert not normalized and states["roles"] == "not_answered"
    with pytest.raises(ValueError, match="Required"):
        validate_answers(definitions()["background"], {}, submitted=True)


def test_publication_immutability_hashes_and_frozen_questionnaires(service, tmp_path):
    _, store, frozen = service
    assert store.publish(frozen)["study_revision"] == frozen.definition.study_revision
    changed = frozen.model_copy(deep=True)
    changed.definition.instructions = "Changed instructions"
    with pytest.raises(StudyError, match="immutable"):
        store.publish(changed)
    changed = frozen.model_copy(deep=True)
    changed.case_keys[0].acceptable_candidate_ids = ["candidate-5"]
    with pytest.raises(StudyError, match="immutable"):
        store.publish(changed)
    identity, _ = ready_session(store, frozen)
    (tmp_path / "source").write_text("Tampered")
    with pytest.raises(StudyError, match="changed"):
        store.resource(*identity, "source")


def test_restarted_store_and_export_secrets_excluded(service):
    app, store, frozen = service
    identity, secret = ready_session(store, frozen)
    before, _ = submit_case(store, identity)
    restarted = StudyStore(store.database_url, store.assets_dir, allow_test_sqlite=True)
    assert restarted.exchange(secret)[2] == before
    assert restarted.state(*identity)["assignment_id"] == before["assignment_id"]
    assert restarted.export(frozen.definition.study_revision)["data"]["sessions"] == []
    exported = restarted.export(frozen.definition.study_revision, include_test=True)
    content = json.dumps(exported)
    assert secret not in content and SECRET not in content and ADMIN not in content
    assert "invite_digest" not in content and "researcher_case_keys" not in content
    assert restarted.saved_export(exported["manifest"]["export_id"]) == exported
    assert exported["data"]["summary"]["assigned"] == 4
    assert exported["data"]["summary"]["submitted"] == 1
    assert (
        "researcher_case_keys"
        in restarted.export(frozen.definition.study_revision, include_test=True, include_keys=True)[
            "data"
        ]
    )


def test_http_origin_cache_validation_admin_and_closed_states(service):
    app, store, frozen = service
    client = TestClient(app, base_url=ORIGIN)
    secret = invite(store, frozen)
    assert client.post("/api/v1/study/session", json={"secret": secret}).status_code == 403
    response = client.post(
        "/api/v1/study/session", json={"secret": secret}, headers={"Origin": ORIGIN}
    )
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert (
        client.post(
            "/api/v1/study/events",
            json={"events": [], "participant_id": "other"},
            headers={"Origin": ORIGIN},
        ).status_code
        == 422
    )
    assert client.post("/api/v1/admin/studies", json={}).status_code == 401
    assert (
        client.get(
            f"/api/v1/admin/studies/{frozen.definition.study_revision}/progress",
            headers={"Authorization": f"Bearer {ADMIN}"},
        ).status_code
        == 200
    )
    assert client.get("/api/ready").json() == {"ready": True}
    store.close(frozen.definition.study_revision)
    assert client.get("/api/v1/study/state").status_code == 410
    assert (
        client.post(
            "/api/v1/study/session", json={"secret": secret}, headers={"Origin": ORIGIN}
        ).status_code
        == 401
    )


def test_counterbalance_assignment_is_once_and_invites_do_not_consume_slots(service):
    _, store, frozen = service
    for _ in range(10):
        store.exchange(invite(store, frozen))
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(lambda _: ready_session(store, frozen)[0], range(8)))
    export = store.export(frozen.definition.study_revision, include_test=True)
    assigned = [s["assignment"] for s in export["data"]["sessions"] if s["assignment"]]
    counts = {schedule.schedule_id: 0 for schedule in frozen.definition.schedules}
    for allocation in assigned:
        counts[allocation["schedule_id"]] += 1
        assert len({p["case_id"] for p in allocation["presentations"]}) == 4
    assert set(counts.values()) == {2}
    for identity in identities:
        assert store.state(*identity)["assignment_id"] == store.state(*identity)["assignment_id"]


def test_scoring_all_checked_in_oracles_and_separate_denominators():
    fixtures = json.loads(
        (
            Path(__file__).parents[1]
            / "specs/explanation-framework/protocol/study-scoring-oracles.synthetic.json"
        ).read_text()
    )
    scores = []
    for oracle in fixtures["cases"]:
        result = score_response(
            oracle["case_kind"], oracle["acceptable_candidate_ids"], oracle["response"]
        )
        assert {k: result[k] for k in oracle["expected"]} == oracle["expected"], oracle["name"]
        scores.append(result)
    metrics = summarize(scores)
    assert metrics["positive_submitted"] == 5 and metrics["negative_submitted"] == 3
    assert metrics["submitted"] == 8 and metrics["assigned"] == 9
    assert metrics["mrr"] == 0.3 and metrics["correct_none_rate"] == pytest.approx(1 / 3)


def test_ranking_and_consultation_semantics_are_strict():
    base = {"idempotency_key": "retry-1", "expected_revision": 0, "presentation_id": "p-1"}
    for extra in [
        {"response_type": "ranked_candidates", "ranked_candidate_ids": []},
        {"response_type": "none_of_these", "ranked_candidate_ids": ["a"]},
        {"response_type": "ranked_candidates", "ranked_candidate_ids": ["a", "a"]},
        {"participant_id": "forged"},
    ]:
        with pytest.raises(ValidationError):
            Ranking(**base, **extra)
    assert Ranking(**base).response_type is None
    with pytest.raises(ValidationError):
        Consultation(
            idempotency_key="a",
            expected_revision=0,
            consulted_external_ontologies=False,
            methods=["protege"],
        )


def test_postgresql_real_restart_and_backup_restore(service, tmp_path):
    """Opt-in dedicated local PostgreSQL recovery; never restarts an unknown DB."""
    import subprocess
    from urllib.parse import urlsplit, urlunsplit

    app, store, frozen = service
    binaries = os.environ.get("EXACT_STUDY_TEST_PGBIN")
    pgdata = os.environ.get("EXACT_STUDY_TEST_PGDATA")
    if not store.postgres or not binaries or not pgdata:
        pytest.skip("Set dedicated PostgreSQL URL, PGBIN and PGDATA to exercise restart/restore")
    pgdata_path = Path(pgdata).resolve()
    dedicated_root = Path(__file__).resolve().parents[1] / "data/explanation-framework/postgres"
    if (
        not (pgdata_path.is_relative_to(Path("/tmp")) or pgdata_path.is_relative_to(dedicated_root))
        or not (pgdata_path / ".exact-synthetic-test-cluster").is_file()
    ):
        pytest.fail("Recovery requires an explicitly marked dedicated synthetic PostgreSQL cluster")
    identity, secret = ready_session(store, frozen)
    event = ready_event(store, identity)
    store.timing(
        *identity,
        TimingSegment(
            segment_id=uuid4().hex,
            page_instance_id=event["page_instance_id"],
            case_id=event["case_id"],
            presentation_id=event["presentation_id"],
            stage="case",
            monotonic_start_ms=100,
            monotonic_end_ms=250,
        ),
    )
    store.mutate(
        *identity,
        "draft",
        Ranking(
            idempotency_key=uuid4().hex,
            expected_revision=store.state(*identity)["revision"],
            presentation_id=event["presentation_id"],
            response_type="ranked_candidates",
            ranked_candidate_ids=["candidate-1"],
        ),
        event["case_id"],
    )
    submitted, _ = submit_case(
        store, identity, response_type="ranked_candidates", ranked=["candidate-2"]
    )
    exported = store.export(frozen.definition.study_revision, include_test=True)

    def durable_rows(instance):
        with instance.transaction() as db:
            result = {
                "sessions": [
                    dict(row)
                    for row in db.execute(
                        "SELECT * FROM sessions WHERE id = ?", (identity[0],)
                    ).fetchall()
                ]
            }
            for table in ("mutations", "history", "events", "timing_segments"):
                result[table] = sorted(
                    (
                        dict(row)
                        for row in db.execute(
                            f"SELECT * FROM {table} WHERE session_id = ?", (identity[0],)
                        ).fetchall()
                    ),
                    key=lambda row: json.dumps(row, sort_keys=True),
                )
            return result

    expected_rows = durable_rows(store)
    url = urlsplit(store.database_url)
    env = {
        **os.environ,
        "PGHOST": url.hostname or "127.0.0.1",
        "PGPORT": str(url.port or 5432),
        "PGUSER": url.username or "",
        "PGPASSWORD": url.password or "",
    }
    backup = tmp_path / "study.dump"

    def run(program, *args):
        subprocess.run(
            [str(Path(binaries) / program), *map(str, args)],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )

    run(
        "pg_dump",
        "--format=custom",
        "--compress=0",
        "--file",
        backup,
        "--dbname",
        url.path.lstrip("/"),
    )
    run(
        "pg_ctl",
        "-D",
        pgdata_path,
        "-l",
        tmp_path / "postgres-restart.log",
        "restart",
        "-m",
        "fast",
        "-w",
    )
    restarted = StudyStore(store.database_url, store.assets_dir)
    assert restarted.exchange(secret)[2] == submitted
    assert durable_rows(restarted) == expected_rows
    restored_name = "exact_restore_" + uuid4().hex
    run("createdb", restored_name)
    try:
        run("pg_restore", "--no-owner", "--dbname", restored_name, backup)
        restored_url = urlunsplit(
            (url.scheme, url.netloc, "/" + restored_name, url.query, url.fragment)
        )
        restored = StudyStore(restored_url, store.assets_dir)
        assert restored.exchange(secret)[2] == submitted
        assert restored.state(*identity)["ranking"]["ranked_candidate_ids"] == ["candidate-2"]
        assert restored.saved_export(exported["manifest"]["export_id"]) == exported
        assert durable_rows(restored) == expected_rows
        evidence_dir = os.environ.get("EXACT_STUDY_TEST_RECOVERY_DIR")
        if evidence_dir:
            import shutil

            destination = Path(evidence_dir).resolve()
            if not destination.is_relative_to(dedicated_root):
                pytest.fail("Recovery evidence must stay in the dedicated synthetic workspace")
            destination.mkdir(parents=True, exist_ok=True)
            retained_backup = destination / "study-recovery.dump"
            shutil.copyfile(backup, retained_backup)
            retained_backup.chmod(0o600)
            receipt = {
                "artifact_type": "study_recovery_verification",
                "schema_version": 1,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "synthetic_only": True,
                "database": subprocess.run(
                    [str(Path(binaries) / "postgres"), "--version"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "verified": [
                    "actual_server_restart",
                    "reusable_invitation",
                    "same_assignment_and_submission",
                    "same_acknowledgements_history_and_events",
                    "pg_dump_and_pg_restore_into_separate_database",
                    "same_immutable_export",
                ],
                "verified_row_counts": {table: len(rows) for table, rows in expected_rows.items()},
                "backup_sha256": hashlib.sha256(retained_backup.read_bytes()).hexdigest(),
                "backup_size_bytes": retained_backup.stat().st_size,
                "immutable_export_sha256": exported["manifest"]["content_sha256"],
                "retention": "Private ignored local test artifacts; no production retention claim",
            }
            (destination / "recovery-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        run("dropdb", restored_name)


def test_answer_save_survives_telemetry_loss_and_validation_never_echoes_secrets(service):
    app, store, frozen = service
    identity, secret = ready_session(store, frozen)
    case = store.current_case(*identity)
    state = store.state(*identity)
    payload = Ranking(
        idempotency_key=uuid4().hex,
        expected_revision=state["revision"],
        presentation_id=case["presentation_id"],
        response_type="insufficient_evidence",
    )
    submitted = store.mutate(*identity, "submit", payload, case["case_id"])
    assert submitted["stage"] == "consultation"
    export = store.export(frozen.definition.study_revision, include_test=True)
    timing = export["data"]["sessions"][0]["cases"][0]["timing"]
    assert timing["raw_elapsed_seconds"] is None and timing["active_duration_known"] is False
    client = TestClient(app, base_url=ORIGIN)
    leaked_value = secret * 10
    failure = client.post(
        "/api/v1/study/session", json={"secret": leaked_value}, headers={"Origin": ORIGIN}
    )
    assert failure.status_code == 422 and secret not in failure.text and "input" not in failure.text
    oversized = client.post(
        "/api/v1/study/session",
        content=b" " * 1_000_001,
        headers={"Origin": ORIGIN, "Content-Type": "application/json"},
    )
    assert oversized.status_code == 413


def test_csv_exports_are_reproducible_safe_and_authenticated(service):
    import io
    import zipfile

    from exact_inspect.study.exports import csv_archive

    app, store, frozen = service
    identity, _ = ready_session(store, frozen)
    submit_case(store, identity)
    exported = store.export(frozen.definition.study_revision, include_test=True)
    archive = csv_archive(exported)
    assert archive == csv_archive(exported)
    with zipfile.ZipFile(io.BytesIO(archive)) as files:
        manifest = json.loads(files.read("manifest.json"))
        for name, record in manifest["files"].items():
            assert hashlib.sha256(files.read(name)).hexdigest() == record["sha256"]
        assert {"cases.csv", "questionnaires.csv", "data-dictionary.json"} <= set(files.namelist())
        assert "researcher-case-keys.json" not in files.namelist()
    client = TestClient(app, base_url=ORIGIN)
    response = client.get(
        f"/api/v1/admin/exports/{exported['manifest']['export_id']}?format=csv",
        headers={"Authorization": f"Bearer {ADMIN}"},
    )
    assert response.status_code == 200 and response.content == archive
    assert response.headers["cache-control"] == "private, no-store"


def test_expired_invitation_and_exposed_case_id_remain_closed(service):
    _, store, frozen = service
    secret = invite(store, frozen)
    sid, generation, _ = store.exchange(secret)
    # An expired publication is tested with a controlled DB clock boundary.
    with store.transaction() as db:
        row = db.execute(
            "SELECT payload FROM studies WHERE revision = ?",
            (frozen.definition.study_revision,),
        ).fetchone()
        definition = json.loads(row["payload"])
        definition["closes_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        db.execute(
            "UPDATE studies SET payload = ? WHERE revision = ?",
            (json.dumps(definition), frozen.definition.study_revision),
        )
    with pytest.raises(StudyError, match="unavailable"):
        store.exchange(secret)
    with pytest.raises(StudyError, match="closed"):
        store.state(sid, generation)


@pytest.mark.parametrize("preset", ["20", "24"])
def test_published_synthetic_presets_run_through_real_allocator(service, preset):
    _, store, frozen = service
    blueprint = json.loads(
        (
            Path(__file__).parents[1]
            / "specs/explanation-framework/protocol/study-schedules.synthetic.json"
        ).read_text()
    )["profiles"][preset]
    raw = frozen.model_dump(mode="json")
    template = raw["definition"]["cases"][0]
    raw["definition"]["study_revision"] = "preset-" + preset + "-" + uuid4().hex
    cases, keys = [], []
    for item in blueprint["case_inventory"]:
        case = json.loads(json.dumps(template))
        case["case_id"] = item["case_id"]
        case["transfer_group"] = item["case_id"]
        case["source"]["iri"] = "urn:synthetic:" + item["case_id"]
        cases.append(case)
        keys.append(
            {
                "case_id": item["case_id"],
                "case_kind": item["kind"],
                "acceptable_candidate_ids": (
                    ["candidate-" + str(item["system_accepted_rank"])]
                    if item["kind"] == "answer_present"
                    else []
                ),
                "adjudication_version": "synthetic-preset-v1",
                "criterion": "Synthetic equivalence",
                "evidence": ["Synthetic schedule oracle"],
                "origin": "natural",
                "original_production_ranks": {"candidate-" + str(i): i for i in range(1, 6)},
            }
        )
    raw["definition"]["cases"] = cases
    raw["case_keys"] = keys
    raw["definition"]["schedules"] = [
        {
            "schedule_id": s["schedule_id"],
            "blocks": [
                {"condition": condition, "case_ids": s["case_allocation"][condition]}
                for condition in s["block_order"]
            ],
        }
        for s in blueprint["schedules"]
    ]
    prepared = Publish.model_validate(raw)
    store.publish(prepared)
    for _ in range(4):
        identity, _ = ready_session(store, prepared)
        assert store.state(*identity)["assigned_case_count"] == int(preset)
    exported = store.export(prepared.definition.study_revision, include_test=True)
    for session in exported["data"]["sessions"]:
        presentations = session["assignment"]["presentations"]
        by_condition = {
            condition: [p["case_id"] for p in presentations if p["condition"] == condition]
            for condition in ("explanation", "ontology_baseline")
        }
        assert {len(ids) for ids in by_condition.values()} == {int(preset) // 2}
        for ids in by_condition.values():
            selected = [k for k in keys if k["case_id"] in ids]
            assert sum(k["case_kind"] == "answer_absent" for k in selected) == 2


def test_complete_synthetic_http_client_flow(service):
    """Exercise the complete frontend contract through first-party HTTP only."""
    app, _, frozen = service
    client = TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN})
    revision = frozen.definition.study_revision
    issued = client.post(
        f"/api/v1/admin/studies/{revision}/invitations",
        headers={"Authorization": f"Bearer {ADMIN}"},
        json={"count": 1, "test": True},
    )
    assert issued.status_code == 200
    secret = issued.json()["invitations"][0]["invitation"].split("=", 1)[1]
    response = client.post("/api/v1/study/session", json={"secret": secret})
    assert response.status_code == 200
    state = response.json()

    def mutate(method, route, **values):
        nonlocal state
        body = {
            "idempotency_key": uuid4().hex,
            "expected_revision": state["revision"],
            **values,
        }
        result = client.request(method, "/api/v1/study/" + route, json=body)
        assert result.status_code == 200, result.text
        state = result.json()
        retry = client.request(method, "/api/v1/study/" + route, json=body)
        assert retry.json() == state
        return state

    mutate(
        "PUT",
        "consent",
        information_version=frozen.definition.information_version,
        accepted=True,
    )
    setup = {
        key: True
        for key in (
            "protege_installed",
            "source_opened",
            "target_opened",
            "practice_source_located",
            "practice_definition_parents_inspected",
        )
    }
    mutate("PUT", "setup", **setup)
    mutate(
        "PUT",
        "questionnaires/background",
        form_version=frozen.definition.form_version,
        answers=background_answers(),
        submitted=True,
    )
    mutate("PUT", "setup", **setup, completed_tutorial_steps=[0, 1, 2, 3])
    assignment = state["assignment_id"]
    for i in range(4):
        case = client.get("/api/v1/study/cases/current").json()
        page = uuid4().hex
        event = {
            "event_id": uuid4().hex,
            "page_instance_id": page,
            "sequence": 0,
            "case_id": case["case_id"],
            "presentation_id": case["presentation_id"],
            "type": "case_ready",
            "client_monotonic_ms": 0,
            "build_version": frozen.definition.software_version,
        }
        assert client.post("/api/v1/study/events", json={"events": [event]}).status_code == 200
        segment = {
            "segment_id": uuid4().hex,
            "page_instance_id": page,
            "case_id": case["case_id"],
            "presentation_id": case["presentation_id"],
            "stage": "case",
            "monotonic_start_ms": 0,
            "monotonic_end_ms": 100,
        }
        assert client.post("/api/v1/study/timing", json=segment).status_code == 200
        mutate(
            "PUT",
            f"cases/{case['case_id']}/draft",
            presentation_id=case["presentation_id"],
            response_type="ranked_candidates",
            ranked_candidate_ids=["candidate-2", "candidate-1"],
        )
        client.cookies.clear()
        recovered = client.post("/api/v1/study/session", json={"secret": secret}).json()
        assert recovered == state and recovered["assignment_id"] == assignment
        mutate("POST", "pause")
        mutate("POST", "resume", gap_activity="external_work")
        mutate(
            "POST",
            f"cases/{case['case_id']}/submit",
            presentation_id=case["presentation_id"],
            response_type="ranked_candidates" if i < 2 else "none_of_these",
            ranked_candidate_ids=["candidate-2"] if i < 2 else [],
        )
        assert state["stage"] == "consultation"
        mutate(
            "PUT",
            f"cases/{case['case_id']}/consultation",
            consulted_external_ontologies=False,
            methods=[],
        )
    mutate(
        "PUT",
        "questionnaires/final",
        form_version=frozen.definition.form_version,
        answers={
            "component_usefulness": {
                component: "cannot_judge" for component in frozen.definition.components
            },
            "most_helpful_components": ["cannot_judge"],
            "workflow_preference": "no_preference",
            "mental_effort": {"explanation": "moderate", "ontology_baseline": "low"},
        },
        submitted=True,
    )
    mutate("POST", "complete")
    assert state["stage"] == "completed" and state["completed_cases"] == 4


def test_publication_rejects_untyped_or_policy_mismatched_resources(service, tmp_path):
    _, store, frozen = service
    original = (tmp_path / "context").read_bytes()
    for payload in [
        {"definitions": ["Unadmitted arbitrary scorer blob"]},
        {**json.loads(original), "answer_key": "private"},
        {**json.loads(original), "policy_hash": "f" * 64},
    ]:
        changed = frozen.model_copy(deep=True)
        content = json.dumps(payload).encode()
        (tmp_path / "context").write_bytes(content)
        asset = next(asset for asset in changed.definition.assets if asset.asset_id == "context")
        asset.sha256, asset.size_bytes = hashlib.sha256(content).hexdigest(), len(content)
        with pytest.raises(StudyError):
            store.publish(changed)
    (tmp_path / "context").write_bytes(original)
    changed = frozen.model_copy(deep=True)
    ontology = next(asset for asset in changed.definition.assets if asset.kind == "ontology")
    (tmp_path / ontology.admission_receipt_path).write_text("{}")
    ontology.admission_receipt_sha256 = hashlib.sha256(b"{}").hexdigest()
    with pytest.raises(StudyError, match="admission"):
        store.publish(changed)


@pytest.mark.parametrize("violation", ["entity", "claim", "candidate"])
def test_publication_rejects_explanation_resources_from_another_case(service, tmp_path, violation):
    _, store, frozen = service
    payload = json.loads((tmp_path / "context").read_bytes())
    other_source = frozen.definition.cases[1].source.model_dump(mode="json")
    if violation == "entity":
        payload["entities"].append(other_source)
    elif violation == "claim":
        fact = next(fact for fact in payload["facts"] if fact["category"] == "definitions")
        payload["entity_profiles"] = [
            {
                "claim_id": "out-of-case-claim",
                "text": fact["value"]["lexical_form"],
                "fact_ids": [fact["fact_id"]],
                "grounding": "exact_extract",
                "scoped_entities": [other_source],
                "generation_manifest_sha256": "1" * 64,
            }
        ]
    else:
        payload["evidence"] = [
            {
                "evidence_id": "out-of-case-evidence",
                "candidate_id": "candidate-from-another-case",
                "fact_ids": [],
                "channel": "synthetic",
                "role": "target",
                "interpretation": "projected",
                "status": "unresolved",
            }
        ]
    content = json.dumps(payload).encode()
    (tmp_path / "context").write_bytes(content)
    changed = frozen.model_copy(deep=True)
    asset = next(asset for asset in changed.definition.assets if asset.asset_id == "context")
    asset.sha256, asset.size_bytes = hashlib.sha256(content).hexdigest(), len(content)
    with pytest.raises(StudyError, match="case scope"):
        store.publish(changed)


def test_scoring_and_exports_preserve_missingness_and_condition_denominators(service):
    _, store, frozen = service
    identity, _ = ready_session(store, frozen)
    case = store.current_case(*identity)
    state = store.state(*identity)
    store.mutate(
        *identity,
        "draft",
        Ranking(
            idempotency_key=uuid4().hex,
            expected_revision=state["revision"],
            presentation_id=case["presentation_id"],
            response_type="ranked_candidates",
            ranked_candidate_ids=["candidate-1"],
        ),
        case["case_id"],
    )
    export = store.export(frozen.definition.study_revision, include_test=True)["data"]
    current = next(
        row for row in export["sessions"][0]["cases"] if row["case_id"] == case["case_id"]
    )
    assert current["first_response"] is None and current["final_response"] is None
    assert current["draft_response"]["workflow_state"] == "draft" and current["score"]["missing"]
    assert sum(summary["assigned"] for summary in export["condition_summaries"].values()) == 4
    response = {
        "workflow_state": "submitted",
        "response_type": "ranked_candidates",
        "ranked_candidate_ids": ["wrong", "right"],
    }
    score = score_response("answer_present", ["right"], response, ["right", "wrong"])
    assert score["rr"] == 0.5 and score["system_rr"] == 1 and score["delta_rr"] == -0.5
    assert score["correct_to_wrong"] == 1 and score["wrong_to_correct"] == 0
    assert "correct_to_wrong" not in score_response(
        "answer_present", ["right"], response, ["right", "wrong"], origin="constructed"
    )


def test_consultation_wording_and_component_forms_are_frozen(service):
    _, store, frozen = service
    identity, _ = ready_session(store, frozen)
    form = store.state(*identity)["forms"]["consultation"]
    assert [question["id"] for question in form] == [
        "consulted_external_ontologies",
        "methods",
        "other_editor",
        "other_resource",
    ]
    assert form[0]["options"] == {"true": "Yes", "false": "No"}
    assert form[1]["show_if"] == {"consulted_external_ontologies": True}
    changed = frozen.model_copy(deep=True)
    changed.definition.study_revision = "modified-form-" + uuid4().hex
    changed.definition.components = ["original_context"]
    with pytest.raises(StudyError, match="Form version"):
        store.publish(changed)
    changed.definition.form_version = "changed-components-" + uuid4().hex
    assert store.publish(changed)["study_revision"] == changed.definition.study_revision


def test_asset_paths_cannot_alias_or_overwrite_publication_files(service):
    _, _, frozen = service
    for path in (
        "../outside",
        "/absolute",
        "study-definition.json",
        "./study-definition.json",
        "",
    ):
        raw = frozen.model_dump(mode="json")
        raw["definition"]["assets"][2]["path"] = path
        with pytest.raises(ValidationError):
            Publish.model_validate(raw)
    raw = frozen.model_dump(mode="json")
    raw["definition"]["assets"][2]["path"] = "./source"
    with pytest.raises(ValidationError, match="unique"):
        Publish.model_validate(raw)


def test_study_openapi_has_strict_participant_outputs(service):
    app, _, _ = service
    schema = app.openapi()
    for route in ("/api/v1/study/session", "/api/v1/study/cases/{case_id}/submit"):
        assert schema["paths"][route]["post"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ] == {"$ref": "#/components/schemas/StudyState"}
    case = schema["components"]["schemas"]["StudyCase"]
    assert case["additionalProperties"] is False
    assert (
        not {"case_kind", "acceptable_candidate_ids", "original_production_ranks"}
        & case["properties"].keys()
    )
    assert schema["components"]["schemas"]["RankingResponse"]["properties"]["workflow_state"][
        "enum"
    ] == ["draft", "submitted"]
