"""REV-11 regressions: unique durable judgments and explicit training receipts."""

import dataclasses
import json

import pytest

from exact.repair.records import canonical_hash, read_record
from exact.repair.semantic_fidelity import (
    AGGREGATION_REVISION,
    FIDELITY_TRAINING_SCHEMA,
    PARSER_VERSION,
    TEACHER,
    ValidatedFidelityAggregateV3,
    aggregate_comparisons,
    offline_plan_label,
    read_fidelity_training_artifact,
)
from tests.repair_semantic_fidelity_test import (
    _offline_lookup_fixture,
    adapter,
    judgment,
    packet,
    response,
    run,
)


def test_replayed_and_reparsed_response_never_supplies_missing_quorum(tmp_path, monkeypatch):
    annotator, calls = adapter(tmp_path, monkeypatch)
    p = packet()
    schedule = annotator.schedule(
        p, role=TEACHER, quorum=2, parser_versions=(PARSER_VERSION, "parser-revision-2")
    )
    first = annotator.annotate(p, role=TEACHER)
    replay = annotator.annotate(p, role=TEACHER)
    reparsed = annotator.annotate(p, role=TEACHER, parser_version="parser-revision-2")
    assert len(calls) == 1
    assert first == replay and reparsed.comparison_id != first.comparison_id
    for observations in ([first, first], [first, reparsed], [reparsed, replay, first]):
        result = aggregate_comparisons(observations, schedule=schedule)
        assert result["decision"] == "abstain"
        assert result["scheduled"] == 2 and result["eligible"] == 1 and result["missing"] == 1
        assert len(result["unique_observations"]) == 1
        assert len(result["audit"]) == len(observations) - 1
    assert len(calls) == 1  # Aggregation only inspects retained evidence.


def test_distinct_scheduled_identical_judgments_keep_both_provenances(tmp_path, monkeypatch):
    annotator, calls = adapter(tmp_path, monkeypatch)
    p = packet()
    schedule = annotator.schedule(p, role=TEACHER, quorum=2)
    observations = tuple(annotator.annotate(p, role=TEACHER, repetition=i) for i in range(2))
    result = aggregate_comparisons(observations, schedule=schedule)
    assert result["decision"] == "A" and result["eligible"] == 2
    assert len({o["request_id"] for o in result["unique_observations"]}) == 2
    assert result["missing"] == 0 and len(calls) == 2
    assert "not independent human experts" in result["claim_scope"]


def test_swaps_dissent_and_abstention_obey_frozen_denominator(tmp_path, monkeypatch):
    p = packet()

    def handler(**kwargs):
        payload = json.loads(kwargs["content"])
        context = json.loads(payload["messages"][1]["content"])
        swapped = context["swapped"]
        return response(
            judgment(
                p,
                swapped=swapped,
                decision="B" if swapped else "A",
                a=0.25 if swapped else 0.75,
                b=0.75 if swapped else 0.25,
            )
        )

    annotator, calls = adapter(tmp_path, monkeypatch, handler=handler)
    schedule = annotator.schedule(
        p, role=TEACHER, quorum=2, repetitions=(0,), presentation_orders=(False, True)
    )
    rows = [annotator.annotate(p, role=TEACHER, swapped=v) for v in (False, True)]
    result = aggregate_comparisons(rows, schedule=schedule)
    assert result["decision"] == "A" and result["overall_score_a"] == 0.75
    assert result["votes"] == {"A": 2, "B": 0, "tie": 0} and len(calls) == 2

    def disagreement(**kwargs):
        payload = json.loads(kwargs["content"])
        repetition = json.loads(payload["messages"][1]["content"])["repetition"]
        return response(
            judgment(
                p,
                decision="B" if repetition else "A",
                a=0.25 if repetition else 0.75,
                b=0.75 if repetition else 0.25,
            )
        )

    other, _ = adapter(tmp_path / "dissent", monkeypatch, handler=disagreement)
    schedule = other.schedule(p, role=TEACHER, quorum=2)
    rows = [other.annotate(p, role=TEACHER, repetition=i) for i in range(2)]
    result = aggregate_comparisons(rows, schedule=schedule)
    assert result["decision"] == "abstain" and result["disagreement"] == 0.5

    def abstention(**kwargs):
        value = judgment(p)
        value.update(decision="abstain", abstention_reason="Insufficient evidence")
        for criterion in value["criteria"]:
            criterion.update(status="unknown", preference="abstain", a_score=None, b_score=None)
        return response(value)

    masked, _ = adapter(tmp_path / "abstain", monkeypatch, handler=abstention)
    schedule = masked.schedule(p, role=TEACHER, quorum=1)
    result = aggregate_comparisons([masked.annotate(p, role=TEACHER)], schedule=schedule)
    assert result["decision"] == "abstain" and result["invalid_or_abstained"] == 1
    assert result["eligible"] == 0 and result["missing"] == 1


def test_correction_is_a_chargeable_response_but_only_one_scheduled_vote(tmp_path, monkeypatch):
    p = packet()

    def correcting(**kwargs):
        payload = json.loads(kwargs["content"])
        context = json.loads(payload["messages"][1]["content"])
        value = judgment(p)
        if not context["correction"]:
            value["criteria"][0]["evidence_ids"].append("invented")
        return response(value)

    annotator, calls = adapter(tmp_path, monkeypatch, handler=correcting)
    schedule = annotator.schedule(p, role=TEACHER, quorum=2)
    with pytest.raises(ValueError, match="Unknown evidence citation"):
        annotator.annotate(p, role=TEACHER)
    corrected = annotator.annotate(
        p, role=TEACHER, correction=1, correction_errors=("Unknown evidence citation",)
    )
    result = aggregate_comparisons([corrected, corrected], schedule=schedule)
    assert result["decision"] == "abstain" and result["eligible"] == 1
    assert result["scheduled"] == 2 and result["missing"] == 1 and len(calls) == 2
    assert annotator.summary()["reserved_requests"] == 2
    bad = dataclasses.replace(
        corrected, annotator={**dict(corrected.annotator), "correction_parent": {}}
    )
    with pytest.raises(ValueError, match="receipt"):
        aggregate_comparisons([bad], schedule=schedule)


def test_frozen_schedule_and_receipts_reject_forged_or_unscheduled_votes(tmp_path, monkeypatch):
    annotator, _ = adapter(tmp_path, monkeypatch)
    p = packet()
    schedule = annotator.schedule(p, role=TEACHER, quorum=1, repetitions=(0,))
    c = annotator.annotate(p, role=TEACHER)
    with pytest.raises(ValueError, match="Duplicate scheduled"):
        dataclasses.replace(schedule, slots=schedule.slots + schedule.slots)
    with pytest.raises(ValueError, match="outside the frozen schedule"):
        aggregate_comparisons(
            [annotator.annotate(p, role=TEACHER, repetition=1)], schedule=schedule
        )
    bad = dataclasses.replace(c, annotator={**dict(c.annotator), "parameters_hash": "invented"})
    with pytest.raises(ValueError, match="canonical request"):
        aggregate_comparisons([bad], schedule=schedule)
    receipt = dict(c.annotator["wire_receipt"])
    receipt["raw_response"] += " "
    bad = dataclasses.replace(c, annotator={**dict(c.annotator), "wire_receipt": receipt})
    with pytest.raises(ValueError, match="integrity"):
        aggregate_comparisons([bad], schedule=schedule)
    with pytest.raises(ValueError, match="frozen schedule"):
        aggregate_comparisons([c], schedule=schedule, scheduled_count=2)
    with pytest.raises(ValueError, match="frozen annotation schedule"):
        aggregate_comparisons([c, c], quorum=2)


def test_training_and_evaluation_reference_validated_aggregate_and_unique_observations(
    tmp_path, monkeypatch
):
    case, p, raw = _offline_lookup_fixture()
    config = run(
        packet_hashes=(p.content_hash,), evidence_manifest_hash=canonical_hash((p.content_hash,))
    )
    annotator, calls = adapter(
        tmp_path / "ledger", monkeypatch, config, handler=lambda **kwargs: response(judgment(p))
    )
    schedule = annotator.schedule(p, role=TEACHER, quorum=1, repetitions=(0,))
    c = annotator.annotate(p, role=TEACHER)
    result = aggregate_comparisons([c, c], schedule=schedule)
    aggregate = ValidatedFidelityAggregateV3(schedule, (c, c), result)
    assert read_record(aggregate.to_dict()) == aggregate
    assert aggregate.annotator["aggregation_revision"] == AGGREGATION_REVISION
    assert len(aggregate.annotator["unique_observations"]) == 1
    path = tmp_path / "labels.json"
    artifact = {
        "schema": FIDELITY_TRAINING_SCHEMA,
        "aggregation_revision": AGGREGATION_REVISION,
        "comparisons": [{"packet": p.to_dict(), "comparison": aggregate.to_dict()}],
    }
    path.write_text(json.dumps(artifact))
    labels = read_fidelity_training_artifact(path, "train")
    label = offline_plan_label(case, (0,), (), labels[case.case_id])
    assert label.benefit == 0.75 and len(calls) == 1
    assert (
        labels[case.case_id][0][1].result["unique_observations"]
        == aggregate.result["unique_observations"]
    )
    with pytest.raises(ValueError, match="validated aggregation"):
        offline_plan_label(case, (0,), (), [(p, raw)])
    with pytest.raises(ValueError, match="split"):
        read_fidelity_training_artifact(path, "development")
    artifact["schema"] = "exact-repair/fidelity-training/v3"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="aggregation revision"):
        read_fidelity_training_artifact(path, "train")


def qualified_offline_fixture(tmp_path, monkeypatch, split="train"):
    from exact.repair.semantic_fidelity import EVALUATOR

    case, p, _ = _offline_lookup_fixture(split)
    parents = {p.parent_group_id: split}
    config = run(
        packet_hashes=(p.content_hash,),
        evidence_manifest_hash=canonical_hash((p.content_hash,)),
        parent_splits=parents,
        split_manifest_hash=canonical_hash(parents),
        evaluator_split=split if split != "train" else "test",
    )
    role = TEACHER if split == "train" else EVALUATOR
    model = "vendor/teacher" if split == "train" else "vendor/evaluator"
    annotator, _ = adapter(
        tmp_path, monkeypatch, config, handler=lambda **kwargs: response(judgment(p), model=model)
    )
    schedule = annotator.schedule(p, role=role, quorum=1, repetitions=(0,))
    comparison = annotator.annotate(p, role=role)
    aggregate = ValidatedFidelityAggregateV3(
        schedule, (comparison,), aggregate_comparisons([comparison], schedule=schedule)
    )
    return case, p, aggregate


def test_training_deduplicates_identical_aggregates_and_rejects_overlapping_receipts(
    tmp_path, monkeypatch
):
    from exact.repair.semantic_fidelity import validate_fidelity_training_records

    _, p, aggregate = qualified_offline_fixture(tmp_path, monkeypatch)
    rows = {p.case_id: [(p, aggregate), (p, aggregate)]}
    assert len(validate_fidelity_training_records(rows, "train")[p.case_id]) == 1
    replayed = ValidatedFidelityAggregateV3(
        aggregate.schedule,
        aggregate.observations + aggregate.observations,
        aggregate_comparisons(
            aggregate.observations + aggregate.observations, schedule=aggregate.schedule
        ),
    )
    with pytest.raises(ValueError, match="reuse one durable observation"):
        validate_fidelity_training_records({p.case_id: [(p, aggregate), (p, replayed)]}, "train")
