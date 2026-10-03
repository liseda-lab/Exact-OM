"""Corrected study protocol gates, preparation durability and legacy coexistence."""

from __future__ import annotations

import copy
import io
import json
import os
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from exact_inspect.study import StudyError, StudyStore
from exact_inspect.study.exports import csv_archive
from exact_inspect.study.forms import definitions, validate_ordered_forms
from exact_inspect.study.models import Consent, Publish, Questionnaire, Ranking, Setup
from exact_inspect.study.store import canonical, digest
from exact_inspect.study.v2_models import (
    ConsultationDraftV2,
    ConsultationV2,
    SetupV2,
    StudyStateV2,
    TutorialAssessment,
    TutorialComplete,
    TutorialDefinition,
    TutorialProgressMutation,
)
from tests.explanation_study_test import publication, write
from tests.explanation_study_v2_fixtures import publication_v2


@pytest.fixture
def corrected(tmp_path):
    frozen = publication_v2(tmp_path)
    store = StudyStore(
        os.environ.get("EXACT_STUDY_TEST_DATABASE_URL", f'sqlite:///{tmp_path / "study.sqlite"}'),
        tmp_path,
        allow_test_sqlite=True,
    )
    store.publish(frozen)
    invite = store.issue(frozen.definition.study_revision)["invitations"][0]
    sid, generation, _ = store.exchange(invite["invitation"].split("=", 1)[1])
    return store, frozen, (sid, generation)


def prepare(store, frozen, identity):
    write(
        store,
        identity,
        "consent",
        Consent,
        information_version=frozen.definition.information_version,
        accepted=True,
    )
    state = write(
        store,
        identity,
        "setup",
        SetupV2,
        setup_version="setup/2",
        instructions_acknowledged=True,
        external_inspection_optional_understood=True,
        resource_access="available",
        submitted=True,
    )
    answers = {}
    for q in state["forms"]["background"]:
        if not q["required"] or q["show_if"]:
            continue
        choices = q["option_order"]
        answer = "prefer_not_to_say" if "prefer_not_to_say" in choices else choices[0]
        answers[q["id"]] = [answer] if q["multiple"] else answer
    return write(
        store,
        identity,
        "questionnaire:background",
        Questionnaire,
        form_version=state["forms"]["version"],
        answers=answers,
        submitted=True,
    )


def actions_for(study, assets_dir):
    tutorial = study.tutorial.model_dump(mode="json")
    resource = json.loads((Path(assets_dir) / "practice-explanations.json").read_text())
    candidate = tutorial["case"]["candidates"][1]["candidate_id"]
    fact = resource["facts"][0]["fact_id"]
    initial = [c["candidate_id"] for c in tutorial["case"]["candidates"]]
    actions = []
    for lesson in tutorial["lessons"]:
        for requirement in lesson["requirements"]:
            action = {
                "requirement_id": requirement["requirement_id"],
                "action": requirement["action"],
            }
            kind = action["action"]
            if kind in {"inspect_other_candidate", "return_to_candidate"}:
                action["candidate_id"] = candidate
            elif kind in {
                "search_entity",
                "navigate_parent",
                "navigate_child",
                "return_to_compared",
                "copy_iri",
            }:
                action["entity"] = (
                    resource["hierarchy"][0]["parent" if kind == "navigate_parent" else "child"]
                    if kind in {"navigate_parent", "navigate_child"}
                    else tutorial["case"]["source"]
                )
            elif kind in {
                "open_citation",
                "open_original_axiom",
                "locate_in_evidence_list",
                "inspect_graph_or_list",
            }:
                action["fact_id"] = (
                    resource["evidence"][0]["fact_ids"][0]
                    if kind in {"locate_in_evidence_list", "inspect_graph_or_list"}
                    else fact
                )
            elif kind in {
                "add_rank",
                "move_rank",
                "remove_rank",
                "undo_rank",
                "keep_initial_order",
                "check_partial_ranking",
                "choose_none",
                "choose_insufficient",
                "rank_with_details_open",
            }:
                action.update(
                    response_type="ranked_candidates",
                    ranked_candidate_ids=initial if kind == "keep_initial_order" else [candidate],
                )
                if kind in {"choose_none", "choose_insufficient"}:
                    action.update(
                        response_type=(
                            "none_of_these" if kind == "choose_none" else "insufficient_evidence"
                        ),
                        ranked_candidate_ids=[],
                    )
            elif kind in {"report_multiple_methods", "report_no_methods"}:
                action["methods"] = (
                    ["queries_scripts", "reasoner"] if kind == "report_multiple_methods" else []
                )
            elif kind == "locate_downloads":
                action["asset_ids"] = [
                    a["asset_id"] for a in tutorial["case"]["ontology_resources"]
                ]
            actions.append(action)
    return actions


def train(store, frozen, identity):
    tutorial = frozen.definition.tutorial
    actions = actions_for(frozen.definition, store.assets_dir)
    write(
        store,
        identity,
        "tutorial_progress",
        TutorialProgressMutation,
        tutorial_version=tutorial.version,
        current_lesson_id="baseline",
        lesson_id="baseline",
        actions=actions,
        completed_requirements=[a["requirement_id"] for a in actions],
    )
    for question, rule in tutorial.grading.items():
        write(
            store,
            identity,
            "tutorial_assessment",
            TutorialAssessment,
            tutorial_version=tutorial.version,
            question_id=question,
            attempt_id=uuid4().hex,
            response=rule.response.model_dump(exclude_none=True),
        )
    return write(
        store, identity, "tutorial_complete", TutorialComplete, tutorial_version=tutorial.version
    )


def test_v2_setup_explicit_submit_and_legacy_payload_denial(corrected):
    store, frozen, identity = corrected
    state = write(
        store,
        identity,
        "consent",
        Consent,
        information_version=frozen.definition.information_version,
        accepted=True,
    )
    assert not state["tutorial_progress"]["attempts"]
    assert "grading" not in state["tutorial"]
    resource = json.loads((store.assets_dir / "practice-explanations.json").read_text())
    claims = resource["entity_profiles"] + resource["pair_comparison"]
    assert len({claim["claim_id"] for claim in claims}) == len(claims)
    focal = [
        state["tutorial"]["case"]["source"],
        *(c["entity"] for c in state["tutorial"]["case"]["candidates"]),
    ]
    assert all(
        any(
            entity in claim["scoped_entities"] and claim["fact_ids"]
            for claim in resource["entity_profiles"]
        )
        for entity in focal
    )

    assert state["protocol_versions"]["export"] == "exact-study-analysis/2"
    draft = write(
        store,
        identity,
        "setup",
        SetupV2,
        setup_version="setup/2",
        instructions_acknowledged=True,
        external_inspection_optional_understood=True,
        resource_access="available",
        submitted=False,
    )
    assert draft["stage"] == "setup" and draft["assignment_id"] is None
    needs_help = write(
        store,
        identity,
        "setup",
        SetupV2,
        setup_version="setup/2",
        instructions_acknowledged=True,
        external_inspection_optional_understood=True,
        resource_access="needs_help",
        submitted=False,
    )
    assert needs_help["setup"]["resource_access"] == "needs_help"
    with pytest.raises(ValidationError):
        SetupV2(
            idempotency_key="invalid",
            expected_revision=needs_help["revision"],
            setup_version="setup/2",
            instructions_acknowledged=True,
            external_inspection_optional_understood=True,
            resource_access="needs_help",
            submitted=True,
        )
    with pytest.raises(StudyError, match="Legacy setup"):
        write(
            store,
            identity,
            "setup",
            Setup,
            protege_installed=True,
            source_opened=True,
            target_opened=True,
            practice_source_located=True,
            practice_definition_parents_inspected=True,
            completed_tutorial_steps=list(range(6)),
        )
    for asset in state["ontology_resources"]:
        assert asset["role"] in {"source", "target"} and asset["ontology_version_id"]
        assert not asset["asset_id"].startswith("practice-")
    StudyStateV2.model_validate(store.state(*identity))


def test_tutorial_progress_validation_drafts_and_restart(corrected):
    store, frozen, identity = corrected
    state = prepare(store, frozen, identity)
    assert state["stage"] == "tutorial" and state["assignment_id"] is None
    version = frozen.definition.tutorial.version
    with pytest.raises(StudyError, match="typed action"):
        write(
            store,
            identity,
            "tutorial_progress",
            TutorialProgressMutation,
            tutorial_version=version,
            completed_requirements=["identity.inspect"],
        )
    with pytest.raises(StudyError, match="outside the synthetic"):
        write(
            store,
            identity,
            "tutorial_progress",
            TutorialProgressMutation,
            tutorial_version=version,
            actions=[
                {
                    "requirement_id": "identity.inspect",
                    "action": "inspect_other_candidate",
                    "candidate_id": "candidate-1",
                }
            ],
        )
    with pytest.raises(StudyError, match="incomplete"):
        write(store, identity, "tutorial_complete", TutorialComplete, tutorial_version=version)
    state = write(
        store,
        identity,
        "tutorial_progress",
        TutorialProgressMutation,
        tutorial_version=version,
        current_lesson_id="identity",
        actions=[
            {
                "requirement_id": "identity.inspect",
                "action": "inspect_other_candidate",
                "candidate_id": "practice-c2",
            }
        ],
        completed_requirements=["identity.inspect"],
        practice={
            "key": "practice-case",
            "response_type": "ranked_candidates",
            "ranked_candidate_ids": ["practice-c2"],
        },
        assessment_draft={"question_id": "score_meaning", "response": {"choice": "advice"}},
    )
    assert state["tutorial_progress"]["assessment_drafts"]["score_meaning"] == {"choice": "advice"}
    assert (
        not state["tutorial_progress"]["attempts"]
        and not state["tutorial_progress"]["passed_items"]
    )
    restarted = StudyStore(store.database_url, store.assets_dir, allow_test_sqlite=True)
    assert restarted.state(*identity)["tutorial_progress"] == state["tutorial_progress"]
    invitation = restarted.reissue(identity[0])
    new_id, new_gen, resumed = restarted.exchange(invitation["invitation"].split("=", 1)[1])
    assert resumed["tutorial_progress"] == state["tutorial_progress"]
    assert new_id == identity[0] and new_gen != identity[1]


def test_attempts_are_server_graded_append_only_and_allocation_once(corrected):
    store, frozen, identity = corrected
    state = prepare(store, frozen, identity)
    version = frozen.definition.tutorial.version
    request = TutorialAssessment(
        idempotency_key="first-attempt",
        expected_revision=state["revision"],
        tutorial_version=version,
        question_id="score_meaning",
        attempt_id="attempt-1",
        response={"choice": "certain"},
    )
    first = store.mutate(*identity, "tutorial_assessment", request)
    assert first["tutorial_progress"]["attempts"][0]["correct"] is False
    assert first["assignment_id"] is None
    assert store.mutate(*identity, "tutorial_assessment", request) == first
    trained = train(store, frozen, identity)
    progress = trained["tutorial_progress"]
    assert trained["stage"] == "case" and progress["completed_at"] and not progress["outstanding"]
    active_case = store.current_case(*identity)
    assert {a["asset_id"] for a in trained["ontology_resources"]} == set(
        active_case["ontology_resource_ids"]
    )
    assert len(progress["attempts"]) == 6
    assert progress["attempts"][0]["correct"] is False
    assignment = trained["assignment_id"]
    repeated = write(
        store, identity, "tutorial_complete", TutorialComplete, tutorial_version=version
    )
    assert repeated["assignment_id"] == assignment
    help_state = write(
        store,
        identity,
        "tutorial_progress",
        TutorialProgressMutation,
        tutorial_version=version,
        current_lesson_id="evidence",
        help_opened=True,
    )
    assert help_state["tutorial_progress"]["completed_at"] == progress["completed_at"]
    assert (
        help_state["assignment_id"] == assignment
        and help_state["tutorial_progress"]["help_opened"] == 1
    )
    with store.transaction() as db:
        assert (
            db.execute(
                "SELECT allocation_count FROM studies WHERE revision = ?",
                (frozen.definition.study_revision,),
            ).fetchone()["allocation_count"]
            == 1
        )


def test_consultation_drafts_are_presentation_bound_and_report_not_inferred(corrected):
    store, frozen, identity = corrected
    prepare(store, frozen, identity)
    train(store, frozen, identity)
    case = store.current_case(*identity)
    state = store.state(*identity)
    rank = Ranking(
        idempotency_key=uuid4().hex,
        expected_revision=state["revision"],
        presentation_id=case["presentation_id"],
        response_type="none_of_these",
    )
    state = store.mutate(*identity, "submit", rank, case["case_id"])
    assert state["consultation_draft"] is None and state["previous_consultation"] is None
    draft = ConsultationDraftV2(
        idempotency_key=uuid4().hex,
        expected_revision=state["revision"],
        presentation_id=case["presentation_id"],
        form_version="exact-study-forms/2",
        consulted_external_ontologies=True,
    )
    saved = store.mutate(*identity, "consultation_draft", draft, case["case_id"])
    assert saved["stage"] == "consultation" and saved["consultation_draft"]["methods"] == []
    assert saved["ranking"] == state["ranking"] and saved["consultation"] is None
    with pytest.raises(StudyError):
        store.mutate(
            *identity,
            "consultation_draft",
            draft.model_copy(
                update={
                    "idempotency_key": uuid4().hex,
                    "expected_revision": saved["revision"],
                    "presentation_id": "another-presentation",
                }
            ),
            case["case_id"],
        )
    with pytest.raises(ValidationError):
        ConsultationV2(
            idempotency_key="bad-no",
            expected_revision=saved["revision"],
            presentation_id=case["presentation_id"],
            form_version="exact-study-forms/2",
            consulted_external_ontologies=False,
            methods=["reasoner"],
        )
    final = ConsultationV2(
        idempotency_key="consult-final",
        expected_revision=saved["revision"],
        presentation_id=case["presentation_id"],
        form_version="exact-study-forms/2",
        consulted_external_ontologies=True,
        methods=["queries_scripts", "reasoner", "other_method"],
        other_method="local tool name",
        resource_scope=None,
    )
    done = store.mutate(*identity, "consultation", final, case["case_id"])
    assert done["completed_cases"] == 1 and done["consultation_draft"] is None
    assert done["previous_consultation"]["resource_scope"] is None
    assert store.mutate(*identity, "consultation", final, case["case_id"]) == done
    exported = store.export(frozen.definition.study_revision, include_test=True)
    assert exported["manifest"]["schema"] == "exact-study-analysis/2"
    text = canonical(exported)
    assert "local tool name" not in text and "other_method" in text
    assert exported["data"]["sessions"][0]["tutorial_outcomes"]["score_meaning"]["first"]["correct"]
    archive = zipfile.ZipFile(io.BytesIO(csv_archive(exported)))
    assert {"tutorial_attempts.csv", "tutorial_outcomes.csv", "consultation_drafts.csv"} <= set(
        archive.namelist()
    )


def test_publication_rejects_missing_training_wrong_hash_roles_and_order(corrected):
    store, frozen, _ = corrected
    raw = frozen.model_dump(mode="json")
    for mutate in (
        lambda d: d["definition"].pop("tutorial"),
        lambda d: d["definition"]["tutorial"].update(hash="sha256:" + "0" * 64),
        lambda d: d["definition"]["tutorial"]["lessons"].pop(),
        lambda d: d["definition"]["tutorial"]["case"]["source"].update(
            iri=d["definition"]["cases"][0]["source"]["iri"]
        ),
    ):
        damaged = copy.deepcopy(raw)
        mutate(damaged)
        with pytest.raises(ValidationError):
            Publish.model_validate(damaged)
    damaged = copy.deepcopy(raw)
    damaged["definition"]["assets"][0]["role"] = "target"
    with pytest.raises(StudyError, match="role"):
        store.publish(Publish.model_validate(damaged))
    form = definitions(version="exact-study-forms/2")
    transported = json.loads(canonical(form))
    assert transported["background"][0]["option_order"] == [
        "none",
        "less_than_1",
        "1_2",
        "3_4",
        "5_9",
        "10_plus",
        "prefer_not_to_say",
    ]
    validate_ordered_forms(transported)
    transported["background"][0]["option_order"].pop()
    with pytest.raises(ValueError, match="option_order"):
        validate_ordered_forms(transported)


def test_v1_canonical_publication_bytes_and_state_unchanged(tmp_path):
    frozen = publication(tmp_path)
    store = StudyStore(f'sqlite:///{tmp_path / "legacy.sqlite"}', tmp_path, allow_test_sqlite=True)
    raw = frozen.definition.model_dump(mode="json")
    raw.pop("practice_cases")
    raw["questionnaires"] = definitions(raw["components"], raw["form_version"])
    published = store.publish(frozen)
    assert published["frozen_hash"] == digest(canonical(raw))
    assert store.publish(Publish.model_validate(frozen.model_dump(mode="json"))) == published
    invited = store.issue(frozen.definition.study_revision)["invitations"][0]
    _, _, state = store.exchange(invited["invitation"].split("=", 1)[1])
    assert state["contract_version"] == "exact-study/1.0"
    assert "tutorial_progress" not in state and "protocol_versions" not in state
    assert "option_order" not in state["forms"]["background"][0]


def test_new_live_v1_is_rejected_but_historical_republish_preserves_bytes(tmp_path):
    frozen = publication(tmp_path)
    data = frozen.model_dump(mode="json")
    data["definition"].update(
        synthetic=False, launch_approvals=["synthetic-test-of-historical-approval"]
    )
    frozen = Publish.model_validate(data)
    store = StudyStore(
        f'sqlite:///{tmp_path / "legacy-live.sqlite"}', tmp_path, allow_test_sqlite=True
    )
    with pytest.raises(StudyError, match="New live publications require"):
        store.publish(frozen)
    # Install an already-frozen historical revision, without exercising new publication.
    historical = frozen.definition.model_dump(mode="json")
    historical.pop("practice_cases")
    historical["questionnaires"] = definitions(historical["components"], historical["form_version"])
    payload = canonical(historical)
    with store.transaction() as db:
        db.execute(
            "INSERT INTO studies(revision,payload,frozen_hash) VALUES (?,?,?)",
            (historical["study_revision"], payload, digest(payload)),
        )
        db.execute(
            "INSERT INTO researcher_case_keys(study_revision,payload) VALUES (?,?)",
            (
                historical["study_revision"],
                canonical({k.case_id: k.model_dump(mode="json") for k in frozen.case_keys}),
            ),
        )
    assert store.publish(frozen)["frozen_hash"] == digest(payload)
    invitation = store.issue(frozen.definition.study_revision, test=False)["invitations"][0]
    assert (
        store.exchange(invitation["invitation"].split("=", 1)[1])[2]["contract_version"]
        == "exact-study/1.0"
    )


def test_v2_competing_tutorial_writes_and_crash_atomicity(corrected, tmp_path):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    store, frozen, identity = corrected
    state = prepare(store, frozen, identity)
    version = frozen.definition.tutorial.version
    first = TutorialAssessment(
        idempotency_key="race-first",
        expected_revision=state["revision"],
        tutorial_version=version,
        question_id="score_meaning",
        attempt_id="race-attempt",
        response={"choice": "certain"},
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(
            pool.map(lambda _: store.mutate(*identity, "tutorial_assessment", first), range(2))
        )
    assert receipts[0] == receipts[1]
    assert len(receipts[0]["tutorial_progress"]["attempts"]) == 1
    before = store.state(*identity)
    request = TutorialProgressMutation(
        idempotency_key="crash-progress",
        expected_revision=before["revision"],
        tutorial_version=version,
        current_lesson_id="context",
        help_opened=True,
    )
    script = """
import json, os, sys
from exact_inspect.study import StudyStore
from exact_inspect.study.store import Connection
from exact_inspect.study.v2_models import TutorialProgressMutation
config=json.loads(open(sys.argv[1]).read())
store=StudyStore(config['database'],config['assets'],allow_test_sqlite=True)
mode=sys.argv[2]
if mode == 'before':
    execute=Connection.execute
    def crash(self, sql, args=()):
        result=execute(self,sql,args)
        if sql.startswith('UPDATE sessions SET revision'):
            os._exit(91)
        return result
    Connection.execute=crash
store.mutate(*config['identity'],'tutorial_progress',TutorialProgressMutation.model_validate(config['request']))
os._exit(92)
"""
    config = tmp_path / "crash-config.json"
    config.write_text(
        json.dumps(
            {
                "database": store.database_url,
                "assets": str(store.assets_dir),
                "identity": identity,
                "request": request.model_dump(mode="json", exclude_unset=True),
            }
        )
    )
    config.chmod(0o600)
    for phase, expected in (("before", 91), ("after", 92)):
        result = subprocess.run(
            [sys.executable, "-c", script, str(config), phase],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stderr
        restarted = StudyStore(store.database_url, store.assets_dir, allow_test_sqlite=True)
        resumed = restarted.state(*identity)
        if phase == "before":
            assert resumed == before
        else:
            assert resumed["revision"] == before["revision"] + 1
            assert resumed["tutorial_progress"]["help_opened"] == 1
            assert restarted.mutate(*identity, "tutorial_progress", request) == resumed


def test_v2_postgresql_backup_restore_preserves_training_and_draft(corrected, tmp_path):
    import subprocess
    from urllib.parse import urlsplit, urlunsplit

    store, frozen, identity = corrected
    binaries = os.environ.get("EXACT_STUDY_TEST_PGBIN")
    pgdata = os.environ.get("EXACT_STUDY_TEST_PGDATA")
    if not store.postgres or not binaries or not pgdata:
        pytest.skip("Dedicated PostgreSQL URL, PGBIN and PGDATA required for v2 backup/restore")
    root = Path(pgdata).resolve()
    dedicated = Path(__file__).resolve().parents[1] / "data/explanation-framework/postgres"
    if (
        not (root.is_relative_to(Path("/tmp").resolve()) or root.is_relative_to(dedicated))
        or not (root / ".exact-synthetic-test-cluster").is_file()
    ):
        pytest.fail("V2 recovery requires an explicitly marked dedicated synthetic database")
    prepare(store, frozen, identity)
    train(store, frozen, identity)
    case = store.current_case(*identity)
    state = store.state(*identity)
    store.mutate(
        *identity,
        "submit",
        Ranking(
            idempotency_key=uuid4().hex,
            expected_revision=state["revision"],
            presentation_id=case["presentation_id"],
            response_type="none_of_these",
        ),
        case["case_id"],
    )
    state = store.state(*identity)
    before = store.mutate(
        *identity,
        "consultation_draft",
        ConsultationDraftV2(
            idempotency_key=uuid4().hex,
            expected_revision=state["revision"],
            presentation_id=case["presentation_id"],
            form_version="exact-study-forms/2",
            consulted_external_ontologies=True,
            methods=["reasoner"],
        ),
        case["case_id"],
    )
    position = {
        "view": "assessment",
        "lesson_id": None,
        "question_id": frozen.definition.tutorial.assessment[0].question_id,
    }
    before = store.mutate(
        *identity,
        "tutorial_progress",
        TutorialProgressMutation(
            idempotency_key=uuid4().hex,
            expected_revision=before["revision"],
            tutorial_version=frozen.definition.tutorial.version,
            position=position,
            help_opened=True,
        ),
    )
    exported = store.export(frozen.definition.study_revision, include_test=True)
    url = urlsplit(store.database_url)
    env = {
        **os.environ,
        "PGHOST": url.hostname or "127.0.0.1",
        "PGPORT": str(url.port or 5432),
        "PGUSER": url.username or "",
        "PGPASSWORD": url.password or "",
    }

    def run(program, *args):
        subprocess.run(
            [str(Path(binaries) / program), *map(str, args)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    backup = tmp_path / "study-v2.dump"
    run(
        "pg_dump",
        "--format=custom",
        "--compress=0",
        "--file",
        backup,
        "--dbname",
        url.path.lstrip("/"),
    )
    restored_name = "exact_v2_restore_" + uuid4().hex
    run("createdb", restored_name)
    try:
        run("pg_restore", "--no-owner", "--dbname", restored_name, backup)
        restored = StudyStore(
            urlunsplit((url.scheme, url.netloc, "/" + restored_name, url.query, url.fragment)),
            store.assets_dir,
        )
        assert restored.state(*identity) == before
        assert restored.saved_export(exported["manifest"]["export_id"]) == exported
        assert restored.state(*identity)["tutorial_progress"]["completed_at"]
        assert restored.state(*identity)["tutorial_progress"]["position"] == position
        assert restored.state(*identity)["consultation_draft"]["methods"] == ["reasoner"]
    finally:
        run("dropdb", restored_name)
