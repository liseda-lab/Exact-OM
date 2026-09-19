"""Finite exact repair conformance with small exhaustive reference spaces."""

import dataclasses
import itertools
import os
import pickle
import random
import struct
import time

import pyowl_core as owl
import pytest

import exact.repair.kernel as kernel
import exact.repair.workers as workers
from exact.repair.maxsat import PresenceCut, solve_master
from exact.repair.records import (
    BudgetsV2,
    ObjectiveV2,
    ObligationV2,
    PolicyV2,
    RepairInputV2,
    ReplacementCandidateV2,
    RevisionObjectV2,
    VerificationReportV2,
    canonical_hash,
    make_objective,
    read_record,
)
from exact.repair.workers import CallResult, bounded_call


def cls(name):
    return owl.Class(owl.IRI(f"urn:repair:kernel:{name}"))


def inventory(widths=(2, 2)):
    objects = []
    for i, width in enumerate(widths):
        axiom = owl.SubClassOf(cls(f"a{i}"), cls(f"b{i}"))
        candidates = tuple(
            ReplacementCandidateV2(
                f"object-{i}",
                f"candidate-{i}-{a}",
                (axiom,) if a == 0 else (),
                ("keep",) if a == 0 else ("delete",),
            )
            for a in range(width)
        )
        objects.append(RevisionObjectV2(f"object-{i}", "mapping", (axiom,), candidates))
    return tuple(objects)


def synchronous(function, *args, timeout, **kwargs):
    """Use real master logic without process startup in exhaustive kernel tests."""
    if timeout <= 0:
        return CallResult("timeout")
    try:
        return CallResult("complete", function(*args, **kwargs))
    except Exception as error:
        return CallResult("error", detail=str(error))


def controlled_verifier(problem, assignment):
    verdict = dict(problem.evidence)["verdicts"][str(assignment)]
    complete = verdict != "UNKNOWN"
    return VerificationReportV2(
        canonical_hash(assignment),
        canonical_hash(kernel.materialize(problem, assignment)),
        problem.policy.content_hash,
        verdict,
        "complete_supported_fragment" if complete else "partial_detection",
        (
            ObligationV2(
                "controlled",
                "unknown" if not complete else "pass" if verdict == "VERIFIED_FEASIBLE" else "fail",
                complete,
            ),
        ),
    )


def controlled_problem(verdicts, widths=(2, 2), **kwargs):
    return RepairInputV2(
        (), inventory(widths), PolicyV2(), evidence=(("verdicts", verdicts),), **kwargs
    )


@pytest.fixture
def local_supervisor(monkeypatch):
    pytest.importorskip("pysat")
    monkeypatch.setattr(kernel, "bounded_call", synchronous)


def test_maxsat_matches_exhaustive_signed_integer_objectives():
    pytest.importorskip("pysat")
    rng = random.Random(17)
    assignments = tuple(itertools.product(range(3), repeat=3))
    for _ in range(20):
        objective = ObjectiveV2(
            tuple(tuple(rng.randrange(-20, 21) for _ in range(3)) for _ in range(3)),
            tuple(
                (i, a, j, b, rng.randrange(-50, 51))
                for i in range(3)
                for j in range(i + 1, 3)
                for a in range(3)
                for b in range(3)
            ),
        )
        excluded = tuple(rng.sample(assignments, rng.randrange(8)))
        result = solve_master(objective, excluded)
        expected = max(objective.score(a) for a in assignments if a not in excluded)
        assert result.assignment not in excluded
        assert result.upper_bound == expected == objective.score(result.assignment)


def test_maxsat_large_coefficients_empty_inventory_and_exhaustion():
    pytest.importorskip("pysat")
    enormous = 10**30
    objective = ObjectiveV2(
        ((enormous, -enormous), (0, 1)), ((0, 0, 1, 0, -3 * enormous), (0, 1, 1, 1, 5 * enormous))
    )
    result = solve_master(objective)
    assert result.assignment == (1, 1)
    assert result.upper_bound == 4 * enormous + 1
    assert solve_master(ObjectiveV2(())).assignment == ()
    assignments = tuple(itertools.product(range(2), repeat=2))
    assert solve_master(objective, assignments).status == "empty"


def test_presence_cuts_track_every_emitter_fixed_occurrence_and_activation():
    pytest.importorskip("pysat")
    axiom = owl.SubClassOf(cls("A"), cls("B"))
    objects = tuple(
        RevisionObjectV2(
            f"o{i}",
            "mapping",
            (axiom,),
            (
                ReplacementCandidateV2(f"o{i}", f"k{i}", (axiom,)),
                ReplacementCandidateV2(f"o{i}", f"d{i}", (), ("delete",)),
            ),
        )
        for i in range(2)
    )
    problem = RepairInputV2((), objects, PolicyV2())
    objective = ObjectiveV2(((1, 0), (1, 0)))
    cut = PresenceCut((axiom,))
    assert solve_master(objective, problem=problem, presence_cuts=(cut,)).assignment == (1, 1)
    fixed = dataclasses.replace(problem, fixed_axioms=(axiom,))
    assert solve_master(objective, problem=fixed, presence_cuts=(cut,)).status == "empty"
    conditional = PresenceCut((axiom,), ((0, 0),))
    assert solve_master(objective, problem=fixed, presence_cuts=(conditional,)).assignment == (1, 0)
    absent = PresenceCut((owl.SubClassOf(cls("Absent"), cls("B")),))
    assert solve_master(objective, problem=problem, presence_cuts=(absent,)).assignment == (0, 0)


def test_costs_are_subtracted_once_before_quantization():
    objects = inventory((2,))
    changed = dataclasses.replace(objects[0].candidates[1], cost_features=(("delete", 0.25),))
    objects = (dataclasses.replace(objects[0], candidates=(objects[0].candidates[0], changed)),)
    objective = make_objective(objects, ((0.0, 1.005),), profile=(("delete", 2.0),), scale=100)
    assert objective.costs == ((0.0, 0.5),)
    assert objective.unary == ((0, 50),)


def test_kernel_optimum_equals_exhaustive_feasible_pool(local_supervisor):
    rng = random.Random(23)
    assignments = tuple(itertools.product(range(2), repeat=3))
    for _ in range(12):
        values = {
            str(a): rng.choice(("VERIFIED_FEASIBLE", "VERIFIED_INFEASIBLE")) for a in assignments
        }
        problem = controlled_problem(values, (2, 2, 2))
        objective = ObjectiveV2(
            tuple((rng.randrange(-7, 8), rng.randrange(-7, 8)) for _ in range(3)),
            ((0, 0, 2, 1, -4), (1, 1, 2, 0, 6)),
        )
        result = kernel.repair(
            problem,
            objective,
            verifier=controlled_verifier,
            diagnose=False,
            preserve_verified_input=False,
        )
        feasible = [a for a in assignments if values[str(a)] == "VERIFIED_FEASIBLE"]
        if feasible:
            assert result.search_status == "OPTIMAL_IN_POOL"
            assert result.lower_bound == result.upper_bound == max(map(objective.score, feasible))
            assert result.assignment in feasible
        else:
            assert result.search_status == "NO_FEASIBLE_IN_POOL"
            assert result.assignment is result.alignment is result.lower_bound is result.gap is None


def test_unknown_high_value_retains_bound_and_verified_lower_value(local_supervisor):
    verdicts = {str((0,)): "UNKNOWN", str((1,)): "VERIFIED_FEASIBLE"}
    problem = controlled_problem(verdicts, (2,))
    result = kernel.repair(
        problem, ObjectiveV2(((10, 4),)), verifier=controlled_verifier, diagnose=False
    )
    assert result.search_status == "INCUMBENT_WITH_GAP"
    assert result.assignment == (1,)
    assert (result.lower_bound, result.upper_bound, result.gap) == (4, 10, 6)
    assert [p.assignment for p in result.pending] == [(0,)]
    assert result.exclusions == ()


def test_unknown_without_incumbent_has_no_authorizing_alignment(local_supervisor):
    verdicts = {str((0,)): "UNKNOWN", str((1,)): "VERIFIED_INFEASIBLE"}
    result = kernel.repair(
        controlled_problem(verdicts, (2,)),
        ObjectiveV2(((10, 4),)),
        verifier=controlled_verifier,
        diagnose=False,
    )
    assert result.search_status == "UNRESOLVED"
    assert result.assignment is result.alignment is result.lower_bound is result.gap is None
    assert result.upper_bound == 10
    assert len(result.pending) == len(result.exclusions) == 1


def test_original_fully_verified_input_is_returned_unchanged(local_supervisor):
    verdicts = {str((0,)): "VERIFIED_FEASIBLE", str((1,)): "VERIFIED_FEASIBLE"}
    result = kernel.repair(
        controlled_problem(verdicts, (2,)),
        ObjectiveV2(((0, 100),)),
        verifier=controlled_verifier,
        diagnose=False,
    )
    assert result.assignment == (0,)
    assert result.solves == 0
    assert result.search_status == "INCUMBENT_WITH_GAP"


def test_forged_report_cannot_create_incumbent_or_sound_exclusion(local_supervisor):
    def forged(problem, assignment):
        valid = controlled_verifier(problem, assignment)
        return dataclasses.replace(valid, theory_hash="different theory")

    problem = controlled_problem(
        {str((0,)): "VERIFIED_FEASIBLE", str((1,)): "VERIFIED_INFEASIBLE"}, (2,)
    )
    result = kernel.repair(problem, ObjectiveV2(((2, 1),)), verifier=forged, diagnose=False)
    assert result.assignment is None
    assert len(result.pending) == 2
    assert result.exclusions == ()


def test_full_safety_and_optimality_replay_reconstructs_asserted_theory(local_supervisor):
    pytest.importorskip("pyhermit")
    a, b = cls("A"), cls("B")
    mapping = owl.SubClassOf(a, b)
    objects = (
        RevisionObjectV2(
            "mapping",
            "mapping",
            (mapping,),
            (
                ReplacementCandidateV2("mapping", "keep", (mapping,)),
                ReplacementCandidateV2("mapping", "delete", (), ("delete",)),
            ),
        ),
    )
    source = (owl.Declaration(a),)
    target = (owl.SubClassOf(b, owl.OWL_NOTHING),)
    policy = PolicyV2(
        (a, b),
        exceptions=(b,),
        exception_evidence=(kernel.source_exception_evidence(target, side="target"),),
    )
    problem = RepairInputV2(
        (*source, *target), objects, policy, source_axioms=source, target_axioms=target
    )
    objective = ObjectiveV2(((1, 0),))
    result = kernel.repair(problem, objective, diagnose=True)
    assert result.assignment == (1,)
    assert result.search_status == "OPTIMAL_IN_POOL"
    assert len(result.baseline) == 4
    assert result.verification.support
    assert any(
        q.name.startswith("source_exception:target") for q in result.verification.obligations
    )
    assert kernel.replay_safety(problem, result)
    assert kernel.replay_optimality(problem, objective, result)
    assert not kernel.replay_safety(problem, dataclasses.replace(result, assignment=(99,)))
    assert not kernel.replay_safety(problem, dataclasses.replace(result, selected=()))
    assert not kernel.replay_safety(problem, dataclasses.replace(result, alignment=(mapping,)))
    assert result.elapsed_seconds >= sum(seconds for _, seconds in result.stage_seconds)


def test_exception_hash_cannot_invent_a_source_proof():
    pytest.importorskip("pyhermit")
    a = cls("A")
    source = (owl.Declaration(a),)
    policy = PolicyV2(
        (a,), exceptions=(a,), exception_evidence=(kernel.source_exception_evidence(source),)
    )
    problem = RepairInputV2((owl.SubClassOf(a, owl.OWL_NOTHING),), (), policy, source_axioms=source)
    report = kernel.verify_assignment(problem, ())
    assert report.verdict == "UNKNOWN"
    assert not report.authorizes
    assert report.obligations[0].verdict == "fail"
    forged = dataclasses.replace(policy, exception_evidence=(("source", "wrong hash"),))
    assert not kernel.verify_assignment(dataclasses.replace(problem, policy=forged), ()).authorizes


def test_records_roundtrip_immutable_evidence_and_reject_legacy_corruption():
    external = {"features": [1, 2], "nested": {"matcher": "test"}}
    problem = RepairInputV2((), (), PolicyV2(), evidence=(("all_features", external),))
    original_hash = problem.content_hash
    external["features"].append(3)
    assert problem.content_hash == original_hash
    stored = dict(problem.evidence)["all_features"]
    with pytest.raises(TypeError):
        stored["features"] = ()
    assert read_record(problem.to_dict()) == problem
    assert pickle.loads(pickle.dumps(problem)) == problem
    with pytest.raises(ValueError, match="schema"):
        read_record({**problem.to_dict(), "schema": "exact-repair/records/v1"})
    with pytest.raises(ValueError, match="hash"):
        read_record({**problem.to_dict(), "hash": "wrong"})


def worker_sleep():
    time.sleep(5)


def worker_crash():
    os._exit(3)


def worker_identity(value):
    return value


def test_supervisor_completion_timeout_and_crash_are_distinct():
    assert bounded_call(worker_identity, 7, timeout=3).value == 7
    started = time.monotonic()
    assert bounded_call(worker_sleep, timeout=0.1).status == "timeout"
    assert time.monotonic() - started < 1.5
    crashed = bounded_call(worker_crash, timeout=3)
    assert crashed.status == "error"
    assert "without a result" in crashed.detail


def worker_partial_frame(connection, function, args, kwargs):
    # Simulate a crashed/stalled transport after the frame becomes readable.
    os.write(connection.fileno(), struct.pack("!i", 2000) + b"incomplete")
    time.sleep(5)


def test_supervisor_deadline_covers_partially_received_result(monkeypatch):
    monkeypatch.setattr(workers, "_execute", worker_partial_frame)
    started = time.monotonic()
    assert bounded_call(worker_identity, 1, timeout=0.3).status == "timeout"
    assert time.monotonic() - started < 1.5


def test_unknown_retry_never_becomes_a_logical_cut(local_supervisor):
    attempts = {}
    problem = controlled_problem(
        {str((0,)): "VERIFIED_FEASIBLE", str((1,)): "VERIFIED_FEASIBLE"},
        (2,),
        budgets=BudgetsV2(retries=1),
    )

    def transient(problem, assignment):
        attempts[assignment] = attempts.get(assignment, 0) + 1
        report = controlled_verifier(problem, assignment)
        if assignment == (0,) and attempts[assignment] == 1:
            return dataclasses.replace(
                report,
                verdict="UNKNOWN",
                scope="partial_detection",
                obligations=(ObligationV2("transient", "unknown", False),),
            )
        return report

    result = kernel.repair(problem, ObjectiveV2(((10, 4),)), verifier=transient, diagnose=False)
    assert result.assignment == (0,)
    assert result.search_status == "OPTIMAL_IN_POOL"
    assert result.pending == result.exclusions == ()
    assert attempts[(0,)] == 2


def test_zero_check_budget_does_not_start_a_verification(local_supervisor):
    problem = controlled_problem(
        {str((0,)): "VERIFIED_FEASIBLE", str((1,)): "VERIFIED_FEASIBLE"},
        (2,),
        budgets=BudgetsV2(max_checks=0),
    )
    result = kernel.repair(
        problem, ObjectiveV2(((10, 4),)), verifier=controlled_verifier, diagnose=False
    )
    assert result.checks == result.solves == 0
    assert result.assignment is result.lower_bound is result.gap is None
    assert result.search_status == "UNRESOLVED"
