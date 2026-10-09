"""Additive supervision and exact plan/role admission with mock HTTP ledger receipts."""

import dataclasses
import json
from functools import partial

import pytest
import torch

from exact.repair.learning import SemanticTargetSpec, TeacherCache
from exact.repair.records import canonical_hash
from exact.repair.semantic_fidelity import (
    TEACHER,
    EVALUATOR,
    SelectionAnnotationRunV1,
    ValidatedFidelityAggregateV3,
    aggregate_comparisons,
    offline_plan_label,
)
from tests.repair_semantic_fidelity_test import (
    _offline_lookup_fixture,
    adapter,
    judgment,
    response,
    run,
)
from tools.repair import train
from tools.repair.corpus import GeneratedCase

POLICY = dict(revision="per_criterion_median/v1", minimum_ratings=1, max_criterion_range=1.0)


def native_case(split="train"):
    case, packet, _ = _offline_lookup_fixture(split)
    case.case_id += ":" + split
    packet = dataclasses.replace(packet, case_id=case.case_id)
    result = GeneratedCase(
        **vars(case),
        family="semantic",
        intended_assignment=(0,),
        generation_seed=13,
        mirrored=False,
        schema_revision="v3",
        intended_theory=case.problem.objects[0].original_axioms
    )
    labels = tuple(train._assignment_label(result, (i,), train.DEFAULT_PROFILE) for i in range(3))
    hashes = dict(
        input=result.problem.content_hash,
        policy=result.problem.policy.content_hash,
        query=canonical_hash(result.probes),
        inventory=canonical_hash(tuple(o.candidates for o in result.problem.objects)),
        profile=canonical_hash(train.DEFAULT_PROFILE),
        semantic_target=SemanticTargetSpec(canonical_hash(result.probes)).content_hash,
    )
    return (
        result,
        packet,
        TeacherCache(
            (3,),
            labels,
            True,
            "complete",
            tuple(sorted(hashes.items())),
            0,
            schema="exact-repair/teacher-cache/v3",
        ),
    )


def aggregate(tmp_path, monkeypatch, packet, *, decision="A", a=0.75, b=0.25):
    config = run(
        packet_hashes=(packet.content_hash,),
        evidence_manifest_hash=canonical_hash((packet.content_hash,)),
        repetitions=1,
    )
    client, calls = adapter(
        tmp_path,
        monkeypatch,
        config,
        handler=lambda **kw: response(judgment(packet, decision=decision, a=a, b=b)),
    )
    schedule = client.schedule(packet, role=TEACHER, quorum=1, repetitions=(0,))
    rows = (client.annotate(packet, role=TEACHER),)
    value = ValidatedFidelityAggregateV3(
        schedule, rows, aggregate_comparisons(rows, schedule=schedule)
    )
    assert len(calls) == 1
    return value


def test_declared_same_teacher_dev_selection_and_test_model_exclusion():
    lookup = partial(offline_plan_label, require_aggregate=False, role="evaluator")
    case, packet, comparison = _offline_lookup_fixture("development")
    shared = dataclasses.replace(
        comparison,
        annotator={
            **dict(comparison.annotator),
            "actual_model": "teacher",
            "independent_evaluator": False,
            "development_use_policy": "development_selection/v1",
        },
    )
    with pytest.raises(ValueError, match="independence"):
        lookup(case, (0,), (), [(packet, shared)])
    assert (
        lookup(case, (0,), (), [(packet, shared)], use_policy="development_selection/v1").benefit
        == 0.75
    )
    # This approval cannot be reused as an independence declaration at TEST.
    case, packet, comparison = _offline_lookup_fixture("test")
    contaminated = dataclasses.replace(
        comparison, annotator={**dict(comparison.annotator), "selection_model_ids": ["evaluator"]}
    )
    with pytest.raises(ValueError, match="independence"):
        lookup(case, (0,), (), [(packet, contaminated)])
    with pytest.raises(ValueError, match="TEST"):
        lookup(case, (0,), (), [(packet, comparison)], use_policy="development_selection/v1")


@pytest.mark.parametrize("decision,a,b", [("A", 0.75, 0.25), ("tie", 0.5, 0.5)])
def test_combined_losses_keep_symbolic_terms_and_reach_parameters(
    tmp_path, monkeypatch, decision, a, b
):
    fitting_case, packet, cache = native_case()
    dev, _, dev_cache = native_case("development")
    weak = aggregate(tmp_path / "ledger", monkeypatch, packet, decision=decision, a=a, b=b)
    records = {fitting_case.case_id: [(packet, weak)]}
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: dict(exact_regret=0.0))
    monkeypatch.setattr(
        train,
        "generated_development",
        lambda *a, **k: dict(decoded=dict(status="verified", selected_utility=0.5, checks=1)),
    )
    options = dict(
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        epochs=1,
        batch_cases=1,
        proposal_arm="bounded_enumeration",
        sampled_assignments=0,
        deadline_seconds=60,
    )
    baseline, baseline_report = train.train_cases(
        [(fitting_case, cache)], [(dev, dev_cache)], **options
    )
    masked, masked_report = train.train_cases(
        [(fitting_case, cache)],
        [(dev, dev_cache)],
        **options,
        target_basis="symbolic_plus_llm",
        loss_contract="provenance_additive/v1",
        weak_anchor_weight=1,
        weak_comparison_weight=1
    )
    assert all(torch.equal(v, masked.state_dict()[k]) for k, v in baseline.state_dict().items())
    combined, combined_report = train.train_cases(
        [(fitting_case, cache)],
        [(dev, dev_cache)],
        **options,
        target_basis="symbolic_plus_llm",
        loss_contract="provenance_additive/v1",
        weak_anchor_weight=1,
        weak_comparison_weight=1,
        fidelity_labels=records,
        plan_rating_aggregation=POLICY,
        checkpoint_path=tmp_path / "fit" / "state.pt"
    )
    row = combined_report["loss_eligibility"][fitting_case.case_id]
    symbolic = baseline_report["loss_eligibility"][fitting_case.case_id]
    assert row["symbolic_value_loss"] == symbolic["symbolic_value_loss"]
    assert row["symbolic_rank_loss"] == symbolic["symbolic_rank_loss"]
    assert row["risk"] == symbolic["risk"] == 3
    assert row["weak_anchor_count"] == 2 and row["llm_weak_comparisons"] == 1
    assert combined_report["parameter_gradient_updates"]["value_head"] == 1
    assert any(
        not torch.equal(v, combined.state_dict()[k]) for k, v in baseline.state_dict().items()
    )
    frozen = json.loads((tmp_path / "fit" / "frozen-plan-ratings.json").read_text())
    assert len(frozen["aggregates"]) == 2
    assert all(r["ratings"] for r in frozen["aggregates"].values())


def test_repeated_plan_scores_are_aggregated_not_required_equal(tmp_path, monkeypatch):
    case, packet, _ = _offline_lookup_fixture()
    first = aggregate(tmp_path / "first", monkeypatch, packet, a=0.6, b=0.1)
    # A separate committed annotation run retains distinct request/response provenance.
    config = run(
        run_id="second-run",
        packet_hashes=(packet.content_hash,),
        evidence_manifest_hash=canonical_hash((packet.content_hash,)),
        repetitions=1,
    )
    client, _ = adapter(
        tmp_path / "second",
        monkeypatch,
        config,
        handler=lambda **kw: response(judgment(packet, a=0.8, b=0.2)),
    )
    schedule = client.schedule(packet, role=TEACHER, quorum=1, repetitions=(0,))
    rows = (client.annotate(packet, role=TEACHER),)
    second = ValidatedFidelityAggregateV3(
        schedule, rows, aggregate_comparisons(rows, schedule=schedule)
    )
    records = [(packet, first), (packet, second)]
    with pytest.raises(ValueError, match="disagree"):
        offline_plan_label(case, (0,), (), records)
    label = offline_plan_label(
        case, (0,), (), records, rating_aggregation={**POLICY, "minimum_ratings": 2}
    )
    assert label.benefit == pytest.approx(0.7)
    assert offline_plan_label(case, (2,), (), records, rating_aggregation=POLICY) is None
    assert (
        offline_plan_label(
            case, (0,), (), records, rating_aggregation={**POLICY, "max_criterion_range": 0.1}
        )
        is None
    )


def test_post_decode_provider_attaches_new_exact_plan_after_committed_output(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from exact.repair import kernel
    from exact.repair.graph import build_observable_graph
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import FrozenNeuralRound, repair_neural_round
    from exact.repair.records import ObjectiveV3
    from exact.repair.semantic_fidelity import (
        AGGREGATION_REVISION,
        FIDELITY_TRAINING_SCHEMA,
        SemanticConsequenceReportV3,
        consequence_basis_from_probes,
        semantic_plan_from_verification,
    )
    from exact.repair.workers import CallResult

    case, old_packet, cache = native_case("development")
    # Candidate 2 had no offline annotation. Its exact new output is committed first.
    verified = kernel.verify_assignment(case.problem, (2,))
    label = train._assignment_label(case, (2,), train.DEFAULT_PROFILE)
    basis = consequence_basis_from_probes(case.probes)
    consequence = SemanticConsequenceReportV3(
        verified.theory_hash,
        verified.policy_hash,
        canonical_hash(basis),
        {
            o.probe_id: dict(
                status="true" if o.entailed else "false",
                complete=o.entailed is not None and o.nonvacuous is not None,
                nonvacuity="pass" if o.nonvacuous else "fail",
            )
            for o in label.semantic_vector
        },
        verified.backend,
    )
    new_plan = semantic_plan_from_verification(
        case.problem, (2,), verified, consequence_basis=basis, consequence_report=consequence
    )
    packet = dataclasses.replace(old_packet, plan_a=new_plan)
    parents = {case.structural_parent: "development"}
    config = run(
        packet_hashes=(packet.content_hash,),
        evidence_manifest_hash=canonical_hash((packet.content_hash,)),
        parent_splits=parents,
        split_manifest_hash=canonical_hash(parents),
        evaluator_split="development",
        repetitions=1,
    )
    client, _ = adapter(
        tmp_path / "ledger",
        monkeypatch,
        config,
        handler=lambda **kw: response(judgment(packet), model="vendor/evaluator"),
    )
    schedule = client.schedule(packet, role=EVALUATOR, quorum=1, repetitions=(0,))
    observations = (client.annotate(packet, role=EVALUATOR),)
    aggregate_value = ValidatedFidelityAggregateV3(
        schedule, observations, aggregate_comparisons(observations, schedule=schedule)
    )
    artifact = tmp_path / "new-exact-labels.json"
    artifact.write_text(
        json.dumps(
            dict(
                schema=FIDELITY_TRAINING_SCHEMA,
                aggregation_revision=AGGREGATION_REVISION,
                comparisons=[dict(packet=packet.to_dict(), comparison=aggregate_value.to_dict())],
            )
        )
    )
    manifest = tmp_path / "annotation-manifest.json"
    selection_slot = dict(
        seed=13, supervision_condition="symbolic_plus_llm", epoch=1, case_id=case.case_id
    )
    manifest.write_text(
        json.dumps(
            dict(
                scope="local fixture only",
                frozen_annotation_slots={
                    canonical_hash(selection_slot): dict(
                        selection_slot=selection_slot, swapped=False
                    )
                },
            )
        )
    )
    invoked = []

    def provider(request_path, manifest_path, *, seconds):
        saved = json.loads(request_path.read_text())
        assert (
            saved["assignment"] == [2]
            and saved["verification"]["hash"] == solved.verification.content_hash
        )
        assert saved["case"]["case"]["case_id"] == case.case_id
        assert manifest_path == manifest and seconds > 0
        invoked.append(request_path)
        return artifact

    monkeypatch.setitem(
        sys.modules, "tools.repair.corrective_semantics", SimpleNamespace(annotate_decoded=provider)
    )
    objective = ObjectiveV3(((0, 0, 100),), pool_hash=canonical_hash(case.problem.objects))
    frozen = FrozenNeuralRound(case.problem, objective, "fixture-graph", "fixture-model", ())
    solved = repair_neural_round(frozen, preserve_verified_input=False)
    assert solved.assignment == (2,)
    monkeypatch.setattr(
        train, "_freeze_training_model", lambda *a, **k: CallResult("complete", frozen)
    )

    def direct(function, *args, **kwargs):
        kwargs.pop("timeout", None)
        kwargs.pop("cpu_seconds", None)
        return CallResult(
            "complete", solved if function is repair_neural_round else function(*args, **kwargs)
        )

    monkeypatch.setattr(train, "bounded_call", direct)
    graph = build_observable_graph(case.problem.objects, fixed_axioms=case.problem.fixed_axioms)
    model = RepairModel(
        graph.metadata, encoder="none", hidden_dim=8, heads=2, layers=0, dropout=0, revision="v3"
    )
    report = train.generated_development(
        model,
        case,
        cache,
        seconds=300,
        target_basis="symbolic_plus_llm",
        post_decode_annotation_manifest=str(manifest),
        annotation_directory=tmp_path / "requests",
        plan_rating_aggregation=POLICY,
        selection_slot=selection_slot,
    )
    assert invoked and report["decoded"]["weak_benefit"] == 0.75
    assert report["decoded"]["status"] == "verified"
