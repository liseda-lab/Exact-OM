"""Finite tutorial action predicates over frozen resources, without browser trust."""

import copy
import json
from types import SimpleNamespace

import pytest

from exact_inspect.study import tutorial
from exact_inspect.study.tutorial_validation import (
    TutorialValidationError,
    validate_reachability,
    validate_resource_action,
)
from exact_inspect.study.v2_models import TutorialAction


def entity(iri, kind="class"):
    return {"ontology_version_id": "practice", "iri": "urn:" + iri, "kind": kind}


def practice():
    source = entity("s")
    candidates = [
        {"candidate_id": f"c{i}", "entity": entity(f"t{i}"), "display_position": i + 1}
        for i in range(5)
    ]
    case = {
        "source": source,
        "candidates": candidates,
        "explanation_refs": ["practice"],
        "ontology_resources": [],
    }
    resources = [
        {
            "entities": [
                source,
                *(c["entity"] for c in candidates),
                entity("parent"),
                entity("child"),
            ],
            "facts": [
                {"fact_id": f"f{i}", "subject": c["entity"]} for i, c in enumerate(candidates)
            ]
            + [{"fact_id": "fs", "subject": source}],
            "referenced_labels": [],
            "entity_profiles": [
                {"fact_ids": [f"f{i}"], "scoped_entities": [c["entity"]]}
                for i, c in enumerate(candidates)
            ],
            "pair_comparison": [],
            "hierarchy": [
                {"child": source, "parent": entity("parent"), "fact_ids": ["fs"]},
                {"child": entity("child"), "parent": source, "fact_ids": ["fs"]},
            ],
            "evidence": [
                {
                    "evidence_id": f"e{i}",
                    "candidate_id": f"c{i}",
                    "fact_ids": [f"f{i}"],
                    "status": "available",
                }
                for i in range(5)
            ],
        }
    ]
    lessons = [
        {
            "lesson_id": "context",
            "optional": False,
            "requirements": [
                {"requirement_id": kind, "action": kind}
                for kind in (
                    "search_entity",
                    "navigate_parent",
                    "navigate_child",
                    "return_to_compared",
                    "open_citation",
                    "open_original_axiom",
                    "locate_in_evidence_list",
                )
            ],
        }
    ]
    return {"version": "tutorial/2", "case": case, "lessons": lessons, "assessment": []}, resources


def action(kind, **kwargs):
    return {"requirement_id": kind, "action": kind, **kwargs}


@pytest.mark.parametrize(
    "kind",
    ["open_citation", "open_original_axiom", "locate_in_evidence_list", "inspect_graph_or_list"],
)
def test_factless_graph_bridge_or_context_action_is_never_completion(kind):
    definition, resources = practice()
    with pytest.raises(TutorialValidationError) as error:
        validate_resource_action(action(kind), resources, definition["case"], {}, set())
    assert error.value.as_dict() == {
        "code": "tutorial_fact_required",
        "requirement_ids": [kind],
        "action_ids": [kind],
    }


def test_optional_selected_identity_is_scoped_and_legacy_fact_only_still_valid():
    definition, resources = practice()

    def check(**fields):
        validate_resource_action(
            action("locate_in_evidence_list", **fields), resources, definition["case"], {}, set()
        )

    check(fact_id="f0")
    check(fact_id="f0", evidence_id="e0", candidate_id="c0")
    for fields in (
        {"fact_id": "f0", "evidence_id": "e1", "candidate_id": "c0"},
        {"fact_id": "f0", "evidence_id": "e0", "candidate_id": "c1"},
    ):
        with pytest.raises(TutorialValidationError, match="linked"):
            check(**fields)
    resources[0]["evidence"][0]["status"] = "unresolved"
    with pytest.raises(TutorialValidationError):
        check(fact_id="f0")


def test_ambiguous_feature_identity_requires_candidate_scope():
    definition, resources = practice()
    resources[0]["evidence"].append(
        {"evidence_id": "e0", "candidate_id": "c1", "fact_ids": ["f0"], "status": "available"}
    )
    with pytest.raises(TutorialValidationError) as error:
        validate_resource_action(
            action("locate_in_evidence_list", fact_id="f0", evidence_id="e0"),
            resources,
            definition["case"],
            {},
            set(),
        )
    assert error.value.code == "tutorial_evidence_scope_required"
    validate_resource_action(
        action("locate_in_evidence_list", fact_id="f0", evidence_id="e0", candidate_id="c0"),
        resources,
        definition["case"],
        {},
        set(),
    )


def test_citation_requires_prepared_claim_and_optional_candidate_scope():
    definition, resources = practice()
    with pytest.raises(TutorialValidationError) as error:
        validate_resource_action(
            action("open_citation", fact_id="fs"), resources, definition["case"], {}, set()
        )
    assert error.value.code == "tutorial_citation_unrecorded"
    with pytest.raises(TutorialValidationError):
        validate_resource_action(
            action("open_citation", fact_id="f0", candidate_id="c1"),
            resources,
            definition["case"],
            {},
            set(),
        )
    validate_resource_action(
        action("open_citation", fact_id="f0", candidate_id="c0"),
        resources,
        definition["case"],
        {},
        set(),
    )


def test_return_requires_real_away_action_and_entity_kind_is_not_erased():
    definition, resources = practice()
    state = {}
    with pytest.raises(TutorialValidationError) as error:
        validate_resource_action(
            action("return_to_compared", entity=entity("s")),
            resources,
            definition["case"],
            state,
            set(),
        )
    assert error.value.code == "tutorial_return_requires_navigation"
    with pytest.raises(TutorialValidationError):
        validate_resource_action(
            action("navigate_parent", entity=entity("parent", "individual")),
            resources,
            definition["case"],
            state,
            set(),
        )
    validate_resource_action(
        action("navigate_parent", entity=entity("parent")),
        resources,
        definition["case"],
        state,
        set(),
    )
    validate_resource_action(
        action("return_to_compared", entity=entity("s")),
        resources,
        definition["case"],
        state,
        set(),
    )
    assert state["navigated_away"] is False
    # Cumulative retransmission of already-acknowledged completion remains accepted.
    validate_resource_action(
        action("return_to_compared", entity=entity("s")),
        resources,
        definition["case"],
        state,
        {"return_to_compared"},
    )


def test_reachability_checks_every_candidate_and_incident_hierarchy():
    definition, resources = practice()
    validate_reachability(definition, resources)
    missing = copy.deepcopy(resources)
    missing[0]["evidence"] = missing[0]["evidence"][:-1]
    with pytest.raises(TutorialValidationError) as error:
        validate_reachability(definition, missing)
    assert error.value.action_ids == ["locate_in_evidence_list"]
    unrelated = copy.deepcopy(resources)
    unrelated[0]["hierarchy"] = [
        {"child": entity("parent"), "parent": entity("child"), "fact_ids": ["fs"]}
    ]
    with pytest.raises(TutorialValidationError) as error:
        validate_reachability(definition, unrelated)
    assert set(error.value.action_ids) == {"navigate_child", "navigate_parent"}


def test_rejection_is_atomic_and_next_valid_action_succeeds(monkeypatch):
    definition, resources = practice()
    from exact_inspect.study import resources as resource_module

    monkeypatch.setattr(
        resource_module,
        "ExplanationResource",
        SimpleNamespace(
            model_validate_json=lambda data: SimpleNamespace(
                model_dump=lambda **_: json.loads(data)
            )
        ),
    )
    store = SimpleNamespace(_asset_bytes=lambda _: json.dumps(resources[0]).encode())
    study = {"tutorial": definition, "assets": [{"asset_id": "practice"}]}
    progress = tutorial.initial_progress(definition)
    interaction = {}
    before = copy.deepcopy(progress)
    bad = {
        "completed_requirements": [],
        "actions": [
            action("navigate_parent", entity=entity("parent")),
            action("open_original_axiom"),
        ],
    }
    with pytest.raises(TutorialValidationError):
        tutorial.apply_progress(store, study, progress, bad, interaction)
    assert progress == before and interaction == {}
    good = {
        "completed_requirements": ["open_original_axiom"],
        "actions": [action("open_original_axiom", fact_id="f0")],
    }
    tutorial.apply_progress(store, study, progress, good, interaction)
    assert progress["completed_requirements"] == ["open_original_axiom"]
    assert progress["passed_items"] == []  # Successful actions cannot client-pass assessment.


def test_checkbox_completion_rejected_without_leaking_participant_text():
    definition, _ = practice()
    progress = tutorial.initial_progress(definition)
    with pytest.raises(TutorialValidationError) as error:
        tutorial.apply_progress(
            None,
            {"tutorial": definition},
            progress,
            {"completed_requirements": ["open_citation"], "actions": []},
        )
    assert "typed action evidence" in str(error.value)
    assert progress["completed_requirements"] == []
    assert (
        TutorialAction.model_validate(action("locate_in_evidence_list", fact_id="f0")).evidence_id
        is None
    )
