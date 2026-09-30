"""Second review: coherent weak targets and reusable collector provenance."""

import copy
import json

import pytest

from exact.repair.learning import RepairLabel, collect_sampled_repairs
from exact.repair.records import canonical_hash
from exact.repair.semantic_fidelity import (
    TEACHER,
    ValidatedFidelityAggregateV3,
    aggregate_comparisons,
    offline_plan_label,
    validate_fidelity_training_records,
)
from tests.repair_semantic_fidelity_test import (
    _offline_lookup_fixture,
    adapter,
    judgment,
    response,
    run,
)


@pytest.mark.parametrize(
    "observations, decision, scores, reason",
    [
        ([("A", 0.8, 0.2), ("B", 0.2, 0.8)], "abstain", (None, None), "tied_vote_counts"),
        (
            [("A", 0.2, 0.1), ("A", 0.9, 0.8), ("B", 0.0, 1.0)],
            "abstain",
            (None, None),
            "preference_numeric_conflict",
        ),
        ([("A", 0.8, 0.2), ("A", 0.7, 0.3), ("B", 0.1, 0.9)], "A", (0.7, 0.3), None),
        ([("tie", 0.5, 0.5), ("tie", 0.5, 0.5)], "tie", (0.5, 0.5), None),
        (
            [("tie", 0.8, 0.8), ("tie", 0.2, 0.2), ("A", 1.0, 0.0)],
            "abstain",
            (None, None),
            "preference_numeric_conflict",
        ),
    ],
)
def test_review2_aggregate_votes_and_numeric_labels_agree(
    tmp_path, monkeypatch, observations, decision, scores, reason
):
    case, packet, _ = _offline_lookup_fixture()
    configuration = run(
        packet_hashes=(packet.content_hash,),
        evidence_manifest_hash=canonical_hash((packet.content_hash,)),
        repetitions=len(observations),
    )

    def transport(**kwargs):
        payload = json.loads(kwargs["content"])
        index = json.loads(payload["messages"][1]["content"])["repetition"]
        preference, a, b = observations[index]
        return response(judgment(packet, decision=preference, a=a, b=b))

    annotator, calls = adapter(tmp_path, monkeypatch, configuration, handler=transport)
    schedule = annotator.schedule(
        packet,
        role=TEACHER,
        quorum=len(observations),
        repetitions=tuple(range(len(observations))),
        max_disagreement=1.0,
    )
    rows = tuple(
        annotator.annotate(packet, role=TEACHER, repetition=i) for i in range(len(observations))
    )
    result = aggregate_comparisons(rows + rows[:1], schedule=schedule)
    assert result["decision"] == decision
    assert result["abstention_reason"] == reason
    assert (result["overall_score_a"], result["overall_score_b"]) == scores
    assert len(result["unique_observations"]) == len(observations)
    assert len(calls) == len(observations)
    assert result["audit"][0]["reason"] == "duplicate"
    aggregate = ValidatedFidelityAggregateV3(schedule, rows + rows[:1], result)
    eligible = decision != "abstain"
    assert aggregate.global_target_eligible is eligible
    validated = validate_fidelity_training_records({case.case_id: [(packet, aggregate)]}, "train")
    for assignment, score in zip(((0,), (1,)), scores):
        label = offline_plan_label(case, assignment, (), validated[case.case_id])
        if eligible:
            assert label.usable and label.benefit == score
        else:
            assert label is None


def dependency_hashes():
    return {
        name: canonical_hash((name, "original"))
        for name in (
            "input",
            "patch",
            "policy",
            "query",
            "inventory",
            "backend",
            "profile",
            "semantic_target",
            "annotation_aggregate",
        )
    }


def collector_options():
    return dict(
        case_id="case",
        parent_group_id="parent",
        split="train",
        hashes=dependency_hashes(),
        model_hash="model",
        round_id="round",
        max_assignments=4,
        deadline_seconds=5,
        seed=19,
        plan_quotas=dict(utility=0, proposal=0, diversity=4, quartet=0, uniform=0),
        object_candidate_ids=(("left", ("l0", "l1", "l2")), ("right", ("r0", "r1", "r2"))),
    )


def partial_collection():
    state = {}
    calls = []

    def label(assignment):
        calls.append(assignment)
        return RepairLabel(assignment, True, 0.75, 0.1)

    def save(value):
        state.update(copy.deepcopy(value))
        if len(value["attempts"]) == 1:
            raise InterruptedError("one durable label")

    with pytest.raises(InterruptedError):
        collect_sampled_repairs((3, 3), label, progress=save, **collector_options())
    return state, calls, label


@pytest.mark.parametrize("dependency", tuple(dependency_hashes()))
def test_review2_collector_rejects_changed_label_dependencies_before_reuse(dependency):
    state, calls, label = partial_collection()
    options = collector_options()
    options["hashes"][dependency] = canonical_hash((dependency, "changed"))
    with pytest.raises(ValueError, match="collection.*(dependencies|identity)"):
        collect_sampled_repairs((3, 3), label, resume_state=state, **options)
    assert len(calls) == 1


@pytest.mark.parametrize("dependency", ["case_id", "parent_group_id", "round_id", "model_hash"])
def test_review2_collector_binds_case_and_round_provenance(dependency):
    state, calls, label = partial_collection()
    options = collector_options()
    options[dependency] = "replacement"
    with pytest.raises(ValueError, match="collection.*(dependencies|identity)"):
        collect_sampled_repairs((3, 3), label, resume_state=state, **options)
    assert len(calls) == 1


def test_review2_collector_valid_resume_preserves_source_identity_and_costs():
    state, calls, label = partial_collection()
    frozen = copy.deepcopy(state)
    resumed = collect_sampled_repairs(
        (3, 3), label, resume_state=json.loads(json.dumps(state)), **collector_options()
    )
    baseline = collect_sampled_repairs(
        (3, 3), lambda a: RepairLabel(a, True, 0.75, 0.1), **collector_options()
    )
    assert len(calls) == len(set(calls)) == 4
    assert resumed.attempts == baseline.attempts
    assert resumed.cache.labels == baseline.cache.labels
    assert (
        resumed.collection_identity == baseline.collection_identity == state["collection_identity"]
    )
    assert resumed.collection_dependencies == state["collection_dependencies"]
    assert dict(resumed.cache.hashes) == collector_options()["hashes"]
    assert resumed.cache.elapsed_seconds >= state["elapsed_seconds"]
    assert state == frozen


@pytest.mark.parametrize("missing", ["policy", "semantic_target", "backend", "profile"])
def test_review2_collector_requires_complete_label_identity(missing):
    options = collector_options()
    del options["hashes"][missing]
    with pytest.raises(ValueError, match="supervision dependencies"):
        collect_sampled_repairs((3, 3), lambda a: None, **options)


def test_review2_collector_rejects_legacy_state_without_provenance():
    state, calls, label = partial_collection()
    legacy = {
        key: value
        for key, value in state.items()
        if key not in {"schema", "collection_dependencies", "collection_identity"}
    }
    with pytest.raises(ValueError, match="collection.*(dependencies|identity|schema)"):
        collect_sampled_repairs((3, 3), label, resume_state=legacy, **collector_options())
    assert len(calls) == 1


def test_review2_collector_binds_candidate_map_and_checks_saved_descriptor():
    state, calls, label = partial_collection()
    options = collector_options()
    options["object_candidate_ids"] = (("left", ("l1", "l0", "l2")), ("right", ("r0", "r1", "r2")))
    with pytest.raises(ValueError, match="collection.*dependencies"):
        collect_sampled_repairs((3, 3), label, resume_state=state, **options)
    state["collection_dependencies"]["case_id"] = "corrupt"
    with pytest.raises(ValueError, match="collection.*dependencies"):
        collect_sampled_repairs((3, 3), label, resume_state=state, **collector_options())
    assert len(calls) == 1
