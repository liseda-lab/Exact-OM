"""V2 optional events honor frozen component, lesson and target scopes."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from exact_inspect.study.store import StudyError
from exact_inspect.study.telemetry import _scope
from tests.explanation_corrective_http_test import (  # noqa: F401
    corrected,
    reach_tutorial,
)


def case(identity, *, synthetic=False):
    result = {
        "case_id": identity,
        "source": {"iri": f"urn:{identity}:source", "ontology_version_id": f"{identity}-source"},
        "candidates": [
            {
                "candidate_id": identity + "-candidate",
                "entity": {
                    "iri": f"urn:{identity}:target",
                    "ontology_version_id": f"{identity}-target",
                },
            }
        ],
        "explanation_refs": [identity + "-explanation"],
    }
    if synthetic:
        result["ontology_resources"] = [{"asset_id": identity + "-download"}]
    else:
        result["ontology_resource_ids"] = [identity + "-download"]
    return result


@pytest.fixture
def policy():
    scored, practice = case("scored"), case("practice", synthetic=True)
    presentation = {"case_id": "scored", "presentation_id": "present", "condition": "explanation"}
    state = {"assignment": {"presentations": [presentation]}, "case_index": 0, "stage": "case"}
    study = {
        "cases": [scored],
        "components": [
            "original_context",
            "entity_description",
            "hierarchy",
            "evidence_table",
            "evidence_graph",
            "pair_comparison",
        ],
        "workspace_scopes": [
            {
                "case_id": identity,
                "kind": kind,
                "components": ["context", "profiles", "hierarchy", "evidence", "comparison"],
            }
            for identity, kind in [("scored", "case"), ("practice", "tutorial")]
        ],
        "tutorial": {
            "version": "tutorial/2",
            "compatible_builds": ["ui/2"],
            "case": practice,
            "lessons": [
                {"lesson_id": "context", "view": "explanation"},
                {"lesson_id": "baseline", "view": "baseline"},
            ],
        },
    }
    return SimpleNamespace(_current=lambda ignored: presentation), state, study


def event(**overrides):
    return {
        "scope": "case",
        "case_id": "scored",
        "presentation_id": "present",
        "type": "hierarchy_expand",
        "component_id": "hierarchy_source",
        "build_version": "ui/2",
        **overrides,
    }


def synthetic_event(**overrides):
    return event(
        **{
            "scope": "tutorial",
            "case_id": None,
            "presentation_id": None,
            "tutorial_version": "tutorial/2",
            "lesson_id": "context",
            **overrides,
        }
    )


def accepted(policy, payload):
    _scope(*policy, payload)


@pytest.mark.parametrize(
    "action,component,element",
    [
        ("resources_open", "downloads", None),
        ("external_resource_link", "downloads", "scored-download"),
        ("candidate_inspected", "ranking", "scored-candidate"),
        ("rank_add", "ranking", "scored-candidate"),
        ("rank_remove", "ranking", "scored-candidate"),
        ("rank_move", "ranking", "scored-candidate"),
        ("revision", "ranking", "undo"),
        ("response_type_change", "ranking", "none_of_these"),
        ("copy_iri", "workspace", "source"),
        ("tab_open", "details", "hierarchy"),
        ("table_open", "details", None),
        ("graph_open", "details", None),
        ("evidence_open", "graph", "evidence-id"),
        ("axiom_open", "citation", "sha256:fact"),
    ],
)
def test_actual_frontend_case_action_shapes_are_admitted(policy, action, component, element):
    accepted(policy, event(type=action, component_id=component, element_id=element))


@pytest.mark.parametrize(
    "changes",
    [
        {"component_id": "downloads"},
        {"component_id": "context"},
        {"type": "resources_open", "component_id": "graph"},
        {
            "type": "external_resource_link",
            "component_id": "downloads",
            "element_id": "practice-download",
        },
        {"type": "rank_add", "component_id": "ranking", "element_id": "practice-candidate"},
        {"type": "rank_move", "component_id": "ranking", "element_id": "guessed-candidate"},
        {"type": "copy_iri", "component_id": "workspace", "element_id": "my-private-note"},
        {"type": "tab_open", "component_id": "details", "element_id": "unknown-tab"},
    ],
)
def test_case_action_component_and_typed_target_mismatch_denied(policy, changes):
    with pytest.raises(StudyError):
        accepted(policy, event(**changes))


def test_declared_case_components_and_prepared_scope_both_bound_events(policy):
    store, state, study = policy
    study["components"].remove("hierarchy")
    with pytest.raises(StudyError, match="publication"):
        accepted(policy, event())
    study["components"].append("hierarchy")
    study["workspace_scopes"][0]["components"].remove("hierarchy")
    with pytest.raises(StudyError, match="workspace"):
        accepted(policy, event())
    with pytest.raises(StudyError):
        accepted(policy, event(type="tab_open", component_id="details", element_id="hierarchy"))


@pytest.mark.parametrize("action", ["rank_add", "rank_remove", "rank_move", "submit"])
def test_practice_rank_and_submit_observations_are_synthetic(policy, action):
    payload = synthetic_event(type=action, component_id="ranking")
    if action != "submit":
        payload["element_id"] = "practice-candidate"
    accepted(policy, payload)
    with pytest.raises(StudyError):
        accepted(policy, {**payload, "element_id": "scored-candidate"})


@pytest.mark.parametrize(
    "element",
    [
        "scored",
        "scored-candidate",
        "scored-download",
        "scored-explanation",
        "urn:scored:source",
        "private-free-text",
        "child:urn:scored:source",
    ],
)
def test_synthetic_events_cannot_carry_scored_or_unknown_identifiers(policy, element):
    with pytest.raises(StudyError):
        accepted(policy, synthetic_event(element_id=element))


def test_lesson_targets_and_baseline_views_are_bound(policy):
    accepted(policy, synthetic_event(element_id="urn:practice:source"))
    with pytest.raises(StudyError, match="lesson"):
        accepted(policy, synthetic_event(lesson_id=None))
    with pytest.raises(StudyError, match="different lesson"):
        accepted(policy, synthetic_event(component_id="baseline"))
    with pytest.raises(StudyError, match="baseline practice"):
        accepted(policy, synthetic_event(lesson_id="baseline"))
    accepted(
        policy,
        synthetic_event(
            lesson_id="baseline",
            type="rank_add",
            component_id="ranking",
            element_id="practice-candidate",
        ),
    )
    accepted(
        policy,
        synthetic_event(lesson_id="baseline", type="resources_open", component_id="downloads"),
    )
    accepted(
        policy,
        synthetic_event(
            scope="help", lesson_id=None, type="help_open", component_id="tutorial_help"
        ),
    )
    # Synthetic help remains available during a scored baseline assignment.
    policy[1]["assignment"]["presentations"][0]["condition"] = "ontology_baseline"
    accepted(policy, synthetic_event())
    with pytest.raises(StudyError, match="baseline"):
        accepted(policy, event())


def test_rejected_observation_batch_is_atomic_and_downloads_do_not_poison_queue(
    corrected,
):  # noqa: F811
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    first = {
        "event_id": uuid4().hex,
        "page_instance_id": uuid4().hex,
        "sequence": 0,
        "scope": "tutorial",
        "tutorial_version": state["tutorial"]["version"],
        "lesson_id": "baseline",
        "type": "resources_open",
        "component_id": "downloads",
        "client_monotonic_ms": 0,
        "build_version": publication.definition.software_version,
    }
    bad = {
        **first,
        "event_id": uuid4().hex,
        "sequence": 1,
        "type": "rank_add",
        "component_id": "ranking",
        "element_id": publication.definition.cases[0].candidates[0].candidate_id,
    }
    assert client.post("/api/v1/study/events", json={"events": [first, bad]}).status_code == 403
    with store.transaction() as db:
        assert store._records(db, "events", state["session_id"]) == []
    response = client.post("/api/v1/study/events", json={"events": [first]})
    assert response.status_code == 200, response.text
    assert response.json()["acknowledged_event_ids"] == [first["event_id"]]
    assert client.post("/api/v1/study/events", json={"events": [first]}).json() == response.json()
