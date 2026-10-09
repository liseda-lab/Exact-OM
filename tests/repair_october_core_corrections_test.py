"""October native/core corrective contracts with complete-bundle fixtures."""

from dataclasses import replace
from types import SimpleNamespace
import operator
import time

import pyowl_core as owl
import pytest

from exact.repair.candidates import (
    make_candidate,
    materialize_retrieved_endpoints,
    mapping_candidates,
    normalise_axioms,
    replacement_cost_features,
)
from exact.repair.records import (
    RepairInputV2,
    PolicyV2,
    RevisionObjectV2,
    BudgetsV2,
    canonical_hash,
    freeze_public_policy,
)
from exact.repair.retrieval import ObjectMenus


def cls(name):
    return owl.Class(owl.IRI("urn:october:" + name))


def composite():
    s, t, e, a = map(cls, ("S", "T", "E", "A"))
    active = owl.ObjectIntersectionOf(owl.CanonicalSet((s, e)))
    axioms = normalise_axioms((owl.SubClassOf(active, t), owl.SubClassOf(t, s)))
    keep = make_candidate("m", axioms, ("keep",), active_expressions=(active,))
    delete = make_candidate("m", (), ("delete",))
    obj = RevisionObjectV2("m", "mapping", axioms, (keep, delete), source_entity=s, target_entity=t)
    problem = RepairInputV2((), (obj,), PolicyV2((s, t, e, a)))
    return problem, s, t, e, a, active


@pytest.mark.parametrize("side", ["source", "target"])
def test_complete_composite_substitution_updates_every_bound_occurrence_and_activation(side):
    problem, s, t, e, a, active = composite()
    menu = ObjectMenus("m", endpoint_alternatives=((side, a),))
    result = materialize_retrieved_endpoints(problem, SimpleNamespace(for_object=lambda _: menu))
    candidate = next(c for c in result.objects[0].candidates if "replace_endpoint" in c.action_tags)
    left = owl.ObjectIntersectionOf(owl.CanonicalSet((a, e))) if side == "source" else active
    right = a if side == "target" else t
    expected = normalise_axioms(
        (owl.SubClassOf(left, right), owl.SubClassOf(right, a if side == "source" else s))
    )
    assert candidate.axioms == expected
    assert candidate.active_expressions == (left,)
    costs = dict(candidate.cost_features)
    assert costs["endpoint_change"] == 1
    assert (
        costs["new_constructor"] == costs["removed_direction"] == costs["necessary_condition"] == 0
    )
    assert candidate.cost_features == replacement_cost_features(
        problem.objects[0].original_axioms,
        candidate.axioms,
        active_expressions=candidate.active_expressions,
    )
    assert (
        materialize_retrieved_endpoints(result, SimpleNamespace(for_object=lambda _: menu))
        == result
    )


def test_ambiguous_binding_and_invalid_alternative_do_not_remove_other_actions():
    problem, s, t, e, a, _ = composite()
    obj = replace(problem.objects[0], target_entity=s)
    result = materialize_retrieved_endpoints(
        replace(problem, objects=(obj,)),
        SimpleNamespace(
            for_object=lambda _: ObjectMenus("m", endpoint_alternatives=(("target", a),))
        ),
    )
    assert [c.candidate_id for c in result.objects[0].candidates] == [
        c.candidate_id for c in obj.candidates
    ]
    assert any(
        k == "endpoint_unavailable" and "ambiguous" in v
        for c in result.objects[0].candidates
        for k, v in c.provenance
    )


@pytest.mark.parametrize(
    "entity_type,relation",
    [(owl.ObjectProperty, "<"), (owl.DataProperty, "<"), (owl.NamedIndividual, "=")],
)
def test_elementary_nonclass_endpoint_replacement_remains_supported(entity_type, relation):
    s, t, a = [entity_type(owl.IRI("urn:typed:" + x)) for x in ("s", "t", "a")]
    kind = {
        owl.ObjectProperty: "object_property",
        owl.DataProperty: "data_property",
        owl.NamedIndividual: "individual",
    }[entity_type]
    candidates = mapping_candidates("m", s, t, relation, entity_kind=kind)
    obj = RevisionObjectV2(
        "m", "mapping", candidates[0].axioms, candidates, source_entity=s, target_entity=t
    )
    problem = RepairInputV2((), (obj,), PolicyV2())
    result = materialize_retrieved_endpoints(
        problem,
        SimpleNamespace(
            for_object=lambda _: ObjectMenus("m", endpoint_alternatives=(("target", a),))
        ),
    )
    changed = next(c for c in result.objects[0].candidates if "replace_endpoint" in c.action_tags)
    assert a in {e for ax in changed.axioms for e in owl.signature(ax)}
    assert t not in {e for ax in changed.axioms for e in owl.signature(ax)}
    assert dict(changed.cost_features)["endpoint_change"] == 1


def test_context_deduplicates_activations_preserving_check_budget_and_proofs():
    from exact.repair.grammar import mapping_grammar, with_immutable_context
    from exact.repair.detection import detect_violations, validate_proof

    s, t, a = map(cls, ("S", "T", "A"))
    pool = mapping_candidates("m", s, t, "=")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    fixed = (owl.DisjointClasses(owl.CanonicalSet((s, a))),)
    policy = PolicyV2((s, t, a))
    encoding = mapping_grammar(obj, (a,), max_depth=1, max_constructors=1)
    checked = with_immutable_context(encoding, fixed, policy, max_checks=100)
    assert checked.contextual_checks > checked.contextual_expensive_calls > 0
    assert (
        checked.contextual_cache_hits
        == checked.contextual_checks - checked.contextual_expensive_calls
    )
    for proof in checked.context_proofs:
        active = (proof.activation_expression,)
        assert validate_proof(proof, fixed, active, policy)
        assert proof in detect_violations(fixed, active, policy)
    changed = with_immutable_context(encoding, (), policy, max_checks=100)
    assert not changed.forbidden_assignments and not changed.context_proofs
    assert with_immutable_context(encoding, fixed, policy, max_checks=1).contextual_checks == 1


def test_nested_cleanup_cannot_be_admitted_by_compiler_or_enumerator(monkeypatch):
    from exact.repair import compilation, proposals
    from exact.repair.workers import CallResult
    from exact.repair.grammar import mapping_grammar

    s, t = map(cls, ("S", "T"))
    pool = mapping_candidates("m", s, t)
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool)
    encoding = mapping_grammar(obj, (s, t), max_depth=0, max_constructors=0)
    monkeypatch.setattr(
        compilation,
        "bounded_call",
        lambda *a, **k: CallResult("complete", None, cleanup_complete=False),
    )
    compilation._compile_bounded_cached.cache_clear()
    with pytest.raises(RuntimeError) as caught:
        compilation.compile_bounded(encoding, seconds=1)
    assert caught.value.measurement_receipt["supervised_call"]["cleanup_complete"] is False
    monkeypatch.setattr(
        proposals,
        "bounded_call",
        lambda *a, **k: CallResult("complete", (), cleanup_complete=False),
    )
    with pytest.raises(ValueError, match="enumeration failed"):
        proposals.enumerate_grammar(encoding, deadline=time.monotonic() + 1)


def test_native_worker_quarantines_after_cleanup_failure(monkeypatch):
    import exact.repair.workers as workers

    stop = workers._stop

    def incomplete(pid, own_group):
        stop(pid, own_group)
        return False

    monkeypatch.setattr(workers, "_stop", incomplete)
    monkeypatch.setattr(workers, "_NESTED_CLEANUP_COMPLETE", True)
    outcome = workers.bounded_call(operator.add, 1, 2, timeout=5)
    assert outcome.status == "cleanup_incomplete" and not outcome.cleanup_complete
    refused = workers.bounded_call(operator.add, 3, 4, timeout=5)
    assert refused.status == "cleanup_incomplete" and "quarantined" in refused.detail


def test_native_feasibility_early_stop_keeps_unchecked_queries_unknown():
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms

    a, b, c = map(cls, ("A", "B", "C"))
    snap = snapshot_from_axioms((owl.SubClassOf(a, b), owl.Declaration(c)))
    report = OwlVerifier("auto").check_theory(
        snap,
        (a, b, c),
        prohibited=(owl.SubClassOf(a, b), owl.SubClassOf(b, c)),
        feasibility_only=True,
    )
    assert report.logical_status == "VERIFIED_INFEASIBLE", report.support.issues
    assert report.obligations[-2].satisfied is False
    assert report.obligations[-1].verdict is None and not report.obligations[-1].complete
    full = OwlVerifier("auto").check_theory(
        snap, (a, b, c), prohibited=(owl.SubClassOf(a, b), owl.SubClassOf(b, c))
    )
    assert full.logical_status == report.logical_status and full.obligations[-1].complete


def _stall():
    time.sleep(20)


def test_stalled_rich_generator_retains_native_verified_elementary_incumbent(monkeypatch):
    from exact.repair import pipeline
    from exact.repair.model import RepairModel
    from exact.repair.graph_schema import declared_metadata, generic_graph_schema
    from exact.repair.workers import bounded_call

    s, t = map(cls, ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2(
        (owl.DisjointClasses(owl.CanonicalSet((s, t))),),
        (obj,),
        PolicyV2((s, t)),
        budgets=BudgetsV2(total_seconds=30, verification_seconds=5, solver_seconds=5),
    )
    model = RepairModel(
        declared_metadata(generic_graph_schema()),
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
    )
    native = pipeline.bounded_freeze_neural_round

    def freeze(problem, model, **options):
        if options.get("enabled_actions") is not None:
            return native(problem, model, **options)
        return bounded_call(_stall, timeout=0.1)

    monkeypatch.setattr(pipeline, "bounded_freeze_neural_round", freeze)
    result = pipeline.staged_verified_repair(
        problem,
        model,
        total_seconds=30,
        elementary_seconds=22,
        final_verification_seconds=5,
        repair_options={"diagnose": False},
        generation_options={
            "draws_per_object": 1,
            "max_depth": 1,
            "max_constructors": 1,
            "proposal_arm": "grammar_uniform",
        },
    )
    assert result.result is not None, str(result.stages)
    assert result.result.verification.authorizes
    assert not result.result.selected[0].axioms
    assert result.result.input_hash == result.frozen.problem.content_hash
    assert result.generation_status == "ELEMENTARY_INCUMBENT_PARTIAL_GENERATION"
    assert any(x["phase"] == "rich_generation" and x["status"] == "timeout" for x in result.stages)


def test_coherent_staged_input_bypasses_model_loading_and_generation(monkeypatch):
    from exact.repair import pipeline

    s, t = map(cls, ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2((), (obj,), PolicyV2((s, t)), budgets=BudgetsV2(total_seconds=20))

    def forbidden(*args, **kwargs):
        raise AssertionError("coherent input must not load a model")

    monkeypatch.setattr(pipeline, "bounded_freeze_checkpoint", forbidden)
    result = pipeline.staged_verified_repair(
        problem,
        checkpoint_path="must-not-be-read.pt",
        total_seconds=20,
        elementary_seconds=12,
        repair_options={"diagnose": False},
    )
    assert result.result.verification.authorizes, result.stages
    assert result.result.optimization_bypassed
    assert result.result.selected[0].axioms == obj.original_axioms
    assert result.generation_status == "INITIAL_COHERENT_NO_GENERATION"


def test_native_batched_compilation_matches_isolated_support_and_keeps_completed_prefix(tmp_path):
    import torch
    from exact.repair.compilation import compile_bounded, compile_bounded_batch
    from exact.repair.circuit import ProposalEncoding, ConditionedMixture

    encoding = ProposalEncoding((("x", ("a", "b")),), ((True, False), (False, True)), ("a", "b"))
    isolated = compile_bounded(encoding, seconds=5, cache_directory=tmp_path / "isolated")
    started = time.monotonic()
    results = compile_bounded_batch(
        (encoding, encoding, encoding),
        seconds=5,
        call_seconds=2,
        memory_mb=1024,
        cache_directory=str(tmp_path / "batch"),
        max_tasks_per_worker=2,
    )
    assert all(row.status == "complete" and row.cleanup_complete for row in results), results
    assert time.monotonic() - started < 5.5
    expected = ConditionedMixture(isolated, torch.zeros((1, 2)))
    for row in results:
        actual = ConditionedMixture(row.circuit, torch.zeros((1, 2)))
        assert row.circuit.root.model_count() == isolated.root.model_count()
        assert actual.log_probability((True, False)).item() == pytest.approx(
            expected.log_probability((True, False)).item()
        )
        assert (
            dict(row.circuit.telemetry)["measurement_receipt"]["supervised_batch"][
                "max_tasks_per_worker"
            ]
            == 2
        )
    assert dict(results[1].circuit.telemetry)["persistent_cache_hit"]
    limited = compile_bounded_batch(
        (encoding, encoding), seconds=0.001, call_seconds=0.001, max_tasks_per_worker=2
    )
    assert all(row.status == "not_attempted" for row in limited)


def test_substitution_collapses_duplicate_conjuncts_without_inventing_cost():
    from exact.repair.candidates import substitute_bound_endpoint

    a, b, c = map(cls, ("A", "B", "C"))
    original = (owl.SubClassOf(owl.ObjectIntersectionOf(owl.CanonicalSet((a, b))), c),)
    emitted = (owl.SubClassOf(b, c),)
    assert substitute_bound_endpoint(original[0], a, b) == emitted[0]
    costs = dict(replacement_cost_features(original, emitted))
    assert costs["endpoint_change"] == 1 and costs["new_constructor"] == 0


def test_direct_deletion_runner_never_calls_rich_generator_and_keeps_denominator(
    tmp_path, monkeypatch
):
    from tools.repair import corrective_study as runner
    from tools.repair.prepare import case_to_dict
    from tools.repair.corpus import GeneratedCase
    from exact.repair.learning import TeacherProbe
    from tools.repair import fresh_evaluation

    s, t = map(cls, ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2(
        (owl.DisjointClasses(owl.CanonicalSet((s, t))),),
        (obj,),
        PolicyV2((s, t)),
        budgets=BudgetsV2(total_seconds=20, verification_seconds=5, solver_seconds=5),
    )
    case = GeneratedCase(
        "corrupt",
        "parent",
        "overlap",
        "development",
        problem,
        (TeacherProbe("q", owl.SubClassOf(s, t), "desired", False),),
        (0,),
        13,
        False,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("native deletion must not invoke rich generation")

    monkeypatch.setattr(fresh_evaluation, "generate_control", forbidden)
    protocol = {
        "resources": {"case_wall_seconds": 20, "verification_seconds": 5, "case_rss_mb": 2048},
        "objective": {"edit_weights": {"mapping_deletion": 0.1}, "integer_scale": 1000},
    }
    result = runner.evaluate_row(
        case_to_dict(case),
        dict(arm="native_deletion", protocol=protocol, source_revision="fixture"),
        tmp_path,
    )
    assert result["logical_status"] == "VERIFIED_FEASIBLE", result
    assert result["generation_status"] == "ELEMENTARY_DIRECT_NO_RICH_GENERATION"
    assert result["semantic_benefit"] is not None
    assert (tmp_path / "selection.json").exists()
    assert list((tmp_path / "phases").glob("*.json"))


def _committed_then_stalled_task():
    from exact.repair.workers import emit_event

    emit_event(("bounded_task_started", 0, 1.0))
    emit_event(("completed_fixture", "already-committed"))
    emit_event(("bounded_task_finished", 0))
    emit_event(("bounded_task_started", 1, 0.1))
    time.sleep(10)


def test_external_task_deadline_retains_prior_batch_commit_and_accounts_cleanup():
    from exact.repair.workers import bounded_call

    started = time.monotonic()
    outcome = bounded_call(_committed_then_stalled_task, timeout=5, memory_mb=1024)
    assert outcome.status == "timeout" and outcome.cleanup_complete
    assert "task deadline" in outcome.detail
    assert time.monotonic() - started < 3
    assert ("completed_fixture", "already-committed") in tuple(outcome.events)


def test_complete_plan_observable_control_scores_full_plans_not_independent_factors(tmp_path):
    from tools.repair.corrective_study import _whole_plan_search
    from exact.repair.learning import TeacherProbe

    s, t = map(cls, ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2(
        (owl.DisjointClasses(owl.CanonicalSet((s, t))),),
        (obj,),
        PolicyV2((s, t)),
        budgets=BudgetsV2(total_seconds=20, verification_seconds=5, solver_seconds=5),
    )
    probes = (TeacherProbe("unwanted", owl.SubClassOf(s, t), "observable", False),)
    result = _whole_plan_search(problem, probes, (), tmp_path, 20)
    assert result["attempted"] == result["qualified"] == 2, result
    assert result["scored"] == 1 and result["best"] is not None
    assert result["search_status"] == "OPTIMAL_IN_SHARED_POOL"
    assert result["best"]["assignment_ids"] == [pool[1].candidate_id]
    assert result["query_basis_hash"] == canonical_hash(probes)


def test_cuda_owning_parent_uses_clean_spawn_broker_without_fork(monkeypatch):
    import exact.repair.workers as workers

    def forbidden():
        raise AssertionError("must not fork a live CUDA parent")

    monkeypatch.setattr(workers, "_cuda_context_is_live", lambda: True)
    monkeypatch.setattr(workers.os, "fork", forbidden)
    outcome = workers.bounded_call(operator.add, 2, 3, timeout=5, memory_mb=1024)
    assert outcome.status == "complete" and outcome.value == 5 and outcome.cleanup_complete, outcome
