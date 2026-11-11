"""Balanced allocation and honest tutorial-skip receipts."""

import random
from types import SimpleNamespace
from uuid import uuid4

import pytest

from exact_inspect.study import StudyError, StudyStore
from exact_inspect.study.models import Publish, Ranking
from exact_inspect.study.sampling import (
    sample_blocks,
    validate_pool,
    validate_reference_strata,
)
from exact_inspect.study.v2_models import (
    ConsultationV2,
    RankBalancedSampling,
    TutorialComplete,
)
from tests.explanation_study_test import write
from tests.explanation_study_v2_test import prepare
from tools.build_explanation_v2_fixture import publication_v2


def plan():
    return {
        "version": "rank-balanced/1",
        "selection_manifest_sha256": "a" * 64,
        "reference_alignment_sha256": "b" * 64,
        "strata": [
            {"reference_rank": rank, "case_ids": [f"case-{rank}-{n}" for n in range(8)]}
            for rank in range(1, 6)
        ],
    }


def test_each_assignment_has_one_per_rank_per_condition_and_no_repeat():
    sampling = plan()
    cases = {cid: s["reference_rank"] for s in sampling["strata"] for cid in s["case_ids"]}
    seen = set()
    for seed in range(1000):
        result = sample_blocks(sampling, seed, random.Random(seed))
        assert result == sample_blocks(sampling, seed, random.Random(seed))
        assert result["blocks"][0]["condition"] == (
            "ontology_baseline" if seed % 2 else "explanation"
        )
        assigned = [cid for block in result["blocks"] for cid in block["case_ids"]]
        assert len(set(assigned)) == 10
        seen.update(assigned)
        for block in result["blocks"]:
            assert sorted(cases[cid] for cid in block["case_ids"]) == [1, 2, 3, 4, 5]
    assert seen == set(cases)


def test_pool_rejects_duplicates_and_stratum_key_mismatch():
    sampling = RankBalancedSampling.model_validate(plan())
    cases = [
        SimpleNamespace(
            case_id=cid,
            transfer_group=cid,
            source=SimpleNamespace(ontology_version_id="source", iri=cid, kind="class"),
        )
        for s in sampling.strata
        for cid in s.case_ids
    ]
    validate_pool(cases, sampling)
    cases[-1].transfer_group = cases[0].transfer_group
    with pytest.raises(ValueError, match="near-duplicate"):
        validate_pool(cases, sampling)
    keys = {
        c.case_id: {
            "case_kind": "answer_present",
            "origin": "natural",
            "acceptable_candidate_ids": ["answer"],
            "original_production_ranks": {"answer": int(c.case_id.split("-")[1])},
        }
        for c in cases
    }
    validate_reference_strata({"sampling": plan()}, keys)
    keys[cases[0].case_id]["original_production_ranks"]["answer"] = 5
    with pytest.raises(ValueError, match="original rank"):
        validate_reference_strata({"sampling": plan()}, keys)


@pytest.mark.parametrize("allowed", [False, True])
def test_skip_requires_explicit_publication_and_preserves_incomplete_training(tmp_path, allowed):
    frozen = publication_v2(tmp_path)
    raw = frozen.model_dump(mode="json")
    raw["definition"]["tutorial_skip_allowed"] = allowed
    frozen = Publish.model_validate(raw)
    store = StudyStore(f"sqlite:///{tmp_path / 'study.sqlite'}", tmp_path, allow_test_sqlite=True)
    store.publish(frozen)
    invite = store.issue(frozen.definition.study_revision)["invitations"][0]
    sid, generation, state = store.exchange(invite["invitation"].split("=", 1)[1])
    identity = sid, generation
    state = prepare(store, frozen, identity)
    assert state.get("tutorial_skip_allowed", False) is allowed
    assert state["tutorial_progress"]["outstanding"]
    with pytest.raises(StudyError, match="incomplete"):
        write(
            store,
            identity,
            "tutorial_complete",
            TutorialComplete,
            tutorial_version=frozen.definition.tutorial.version,
        )
    if not allowed:
        with pytest.raises(StudyError, match="not enabled"):
            write(
                store,
                identity,
                "tutorial_complete",
                TutorialComplete,
                tutorial_version=frozen.definition.tutorial.version,
                skip_confirmed=True,
            )
        assert store.state(*identity)["stage"] == "tutorial"
        return
    state = write(
        store,
        identity,
        "tutorial_complete",
        TutorialComplete,
        tutorial_version=frozen.definition.tutorial.version,
        skip_confirmed=True,
    )
    assert state["stage"] == "case"
    progress = state["tutorial_progress"]
    assert progress["skipped_at"] and progress["completed_at"] is None
    assert progress["outstanding"] and not progress["passed_items"]
    assert store.state(*identity)["assignment_id"] == state["assignment_id"]
    again = write(
        store,
        identity,
        "tutorial_complete",
        TutorialComplete,
        tutorial_version=frozen.definition.tutorial.version,
        skip_confirmed=True,
    )
    assert again["assignment_id"] == state["assignment_id"]
    export = store.export(frozen.definition.study_revision, include_test=True)
    session = export["data"]["sessions"][0]
    assert session["tutorial_status"] == "skipped"
    assert session["tutorial_skipped_at"] == progress["skipped_at"]


def test_real_allocator_freezes_ten_balanced_cases_and_baseline_only_reports(tmp_path):
    frozen = publication_v2(tmp_path, case_count=40)
    raw = frozen.model_dump(mode="json")
    sampling = plan()
    for s in sampling["strata"]:
        s["case_ids"] = [f"case-{i}" for i in range(40) if i % 5 + 1 == s["reference_rank"]]
    raw["definition"].update(
        sampling=sampling,
        schedules=[],
        tutorial_skip_allowed=True,
        consultation_scope="baseline_cases",
    )
    frozen = Publish.model_validate(raw)
    store = StudyStore(f"sqlite:///{tmp_path / 'sampled.sqlite'}", tmp_path, allow_test_sqlite=True)
    result = store.publish(frozen)
    assert store.publish(frozen) == result
    first_conditions = []
    for _ in range(2):
        invited = store.issue(frozen.definition.study_revision)["invitations"][0]
        sid, generation, state = store.exchange(invited["invitation"].split("=", 1)[1])
        identity = sid, generation
        assert state["planned_case_count"] == 10
        assert "sampling" not in state and "reference_rank" not in str(state)
        prepare(store, frozen, identity)
        state = write(
            store,
            identity,
            "tutorial_complete",
            TutorialComplete,
            tutorial_version=frozen.definition.tutorial.version,
            skip_confirmed=True,
        )
        assignment = state["assignment_id"]
        seen, ranks = set(), {"explanation": [], "ontology_baseline": []}
        for n in range(10):
            case = store.current_case(*identity)
            cid = case["case_id"]
            assert cid not in seen
            seen.add(cid)
            if n == 0:
                first_conditions.append(case["condition"])
            ranks[case["condition"]].append(int(cid.split("-")[1]) % 5 + 1)
            assert store.state(*identity)["assignment_id"] == assignment
            request = Ranking(
                idempotency_key=uuid4().hex,
                expected_revision=state["revision"],
                presentation_id=case["presentation_id"],
                response_type="ranked_candidates",
                ranked_candidate_ids=[c["candidate_id"] for c in case["candidates"]],
            )
            state = store.mutate(*identity, "submit", request, cid)
            assert store.mutate(*identity, "submit", request, cid) == state
            if case["condition"] == "ontology_baseline":
                assert state["stage"] == "consultation"
                assert case["explanation_refs"] == []
                request = ConsultationV2(
                    idempotency_key=uuid4().hex,
                    expected_revision=state["revision"],
                    presentation_id=case["presentation_id"],
                    form_version=frozen.definition.form_version,
                    consulted_external_ontologies=False,
                )
                state = store.mutate(*identity, "consultation", request, cid)
            else:
                assert state["stage"] != "consultation"
            assert state["completed_cases"] == n + 1
        assert state["stage"] == "final"
        assert all(sorted(values) == [1, 2, 3, 4, 5] for values in ranks.values())
    assert first_conditions == ["explanation", "ontology_baseline"]
    exported = store.export(frozen.definition.study_revision, include_test=True)["data"]
    assert exported["sampling_design"] == sampling
    for session in exported["sessions"]:
        assert len(session["cases"]) == 10
        for case in session["cases"]:
            if case["condition"] == "explanation":
                assert case["consultation"] is None
                assert case["consultation_collection_status"] == "not_requested"
            else:
                assert case["consultation"]["consulted_external_ontologies"] is False
                assert case["consultation_collection_status"] == "submitted"
