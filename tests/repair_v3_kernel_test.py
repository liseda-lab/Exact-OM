"""XR-2.1 sound supports, complete signatures and risk-frontier conformance."""

import dataclasses
import time

import pyowl_core as owl
import pytest

from exact.repair import kernel
from exact.repair.detection import detect_violations, validate_proof
from exact.repair.records import (
    BudgetsV2,
    ObjectiveV2,
    ObligationV2,
    PolicyV2,
    PolicyV3,
    RepairInputV2,
    RepairInputV3,
    ReplacementCandidateV2,
    RevisionObjectV2,
    VerificationEventV3,
    VerificationReportV2,
    canonical_hash,
    promote_input_v3,
    read_record,
)
from exact.repair.workers import CallResult, bounded_call, emit_event


def cls(name):
    return owl.Class(owl.IRI("urn:xr21:" + name))


def test_separate_existentials_do_not_invent_witness_equality():
    a, b, c = map(cls, "ABC")
    r = owl.ObjectProperty(owl.IRI("urn:xr21:r"))
    exists_b = owl.ObjectSomeValuesFrom(r, b)
    disjoint = owl.DisjointClasses((b, c))
    positive = (
        owl.SubClassOf(a, exists_b),
        owl.SubClassOf(a, owl.ObjectSomeValuesFrom(r, c)),
        disjoint,
    )
    policy = PolicyV2((a, b, c))
    assert detect_violations(positive, (), policy) == ()
    impossible = (positive[0], owl.SubClassOf(a, owl.ObjectAllValuesFrom(r, c)), disjoint)
    proofs = detect_violations(impossible, (), policy)
    assert len(proofs) == 1 and proofs[0].query == a
    assert set(proofs[0].asserted_support) == set(impossible)
    assert validate_proof(proofs[0], impossible, (), policy)
    assert not validate_proof(
        dataclasses.replace(proofs[0], asserted_support=impossible[:2]), impossible, (), policy
    )


def test_domain_range_and_nested_witness_proofs_have_asserted_premises():
    a, b, c, d = map(cls, "ABCD")
    r = owl.ObjectProperty(owl.IRI("urn:xr21:r"))
    axioms = (
        owl.SubClassOf(a, owl.ObjectSomeValuesFrom(r, b)),
        owl.ObjectPropertyDomain(r, c),
        owl.ObjectPropertyRange(r, d),
        owl.DisjointClasses((b, d)),
    )
    policy = PolicyV2((a,), prohibited=(owl.SubClassOf(a, c),))
    proofs = detect_violations(axioms, (), policy)
    assert {p.kind for p in proofs} == {"class_satisfiability", "prohibited_entailment"}
    assert all(set(p.asserted_support) <= set(axioms) for p in proofs)
    assert all(validate_proof(p, axioms, (), policy) for p in proofs)


def test_v3_signature_includes_new_candidate_and_deleted_original_and_read_is_strict():
    a, b, new = map(cls, ("A", "B", "NEW"))
    original = owl.SubClassOf(a, b)
    candidate = ReplacementCandidateV2("m", "new", (owl.SubClassOf(new, owl.OWL_NOTHING),))
    obj = RevisionObjectV2("m", "mapping", (original,), (candidate,))
    with pytest.raises(ValueError, match="public classes"):
        RepairInputV3((), (obj,), PolicyV3((a, b)))
    legacy = RepairInputV2((), (obj,), PolicyV2((a, b)))
    modern = promote_input_v3(legacy)
    assert set(modern.policy.monitored_classes) == {a, b, new}
    assert modern.migration_source_hash == legacy.content_hash
    assert read_record(modern.to_dict()) == modern
    assert read_record(legacy.to_dict()) == legacy
    with pytest.raises(ValueError, match="hash"):
        read_record({**modern.to_dict(), "schema": "exact-repair/records/v2"})


def _controlled(problem, assignment):
    verdict = dict(problem.evidence)["verdicts"][str(assignment)]
    return VerificationReportV2(
        canonical_hash(assignment),
        canonical_hash(kernel.materialize(problem, assignment)),
        problem.policy.content_hash,
        verdict,
        "complete_supported_fragment",
        (
            ObligationV2(
                "test",
                "pass" if verdict == "VERIFIED_FEASIBLE" else "unknown",
                verdict != "UNKNOWN",
            ),
        ),
    )


def _sync(function, *args, timeout, **kwargs):
    return CallResult("complete", function(*args, **kwargs))


def _problem(max_checks=3):
    obj = RevisionObjectV2(
        "m",
        "mapping",
        (),
        tuple(
            ReplacementCandidateV2("m", str(i), (), active_expressions=(cls(str(i)),))
            for i in range(3)
        ),
    )
    return RepairInputV2(
        (),
        (obj,),
        PolicyV2(),
        budgets=BudgetsV2(max_checks=max_checks),
        evidence=(
            (
                "verdicts",
                {"(0,)": "VERIFIED_FEASIBLE", "(1,)": "UNKNOWN", "(2,)": "VERIFIED_FEASIBLE"},
            ),
        ),
    )


def prefer_low(assignment):
    return -assignment[0]


def test_risk_keeps_higher_untested_plan_in_upper_bound(monkeypatch):
    monkeypatch.setattr(kernel, "bounded_call", _sync)
    result = kernel.repair(
        _problem(1),
        ObjectiveV2(((10, 9, 8),)),
        verifier=_controlled,
        diagnose=False,
        preserve_verified_input=False,
        shortlist_size=3,
        utility_window=2,
        risk_order=prefer_low,
        risk_identity="test-risk",
    )
    assert result.assignment == (2,) and result.lower_bound == 8 and result.upper_bound == 10
    assert result.search_status == "INCUMBENT_WITH_GAP"
    assert {x.assignment for x in result.ledger.deferred} == {(0,), (1,)}


def test_first_feasible_full_master_optimum_certifies_without_exhaustion(monkeypatch):
    monkeypatch.setattr(kernel, "bounded_call", _sync)
    result = kernel.repair(
        _problem(),
        ObjectiveV2(((10, 9, 8),)),
        verifier=_controlled,
        diagnose=False,
        preserve_verified_input=False,
    )
    assert result.search_status == "OPTIMAL_IN_POOL" and result.checks == result.solves == 1


def test_restart_keeps_costs_bounds_and_pending_and_rejects_changed_epoch(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel, "bounded_call", _sync)
    path = tmp_path / "ledger.json"
    problem, objective = _problem(1), ObjectiveV2(((10, 9, 8),))
    first = kernel.repair(
        problem,
        objective,
        verifier=_controlled,
        diagnose=False,
        preserve_verified_input=False,
        shortlist_size=3,
        utility_window=2,
        risk_order=prefer_low,
        risk_identity="test-risk",
        ledger_path=path,
    )
    resumed = kernel.repair(
        problem,
        objective,
        verifier=_controlled,
        diagnose=False,
        preserve_verified_input=False,
        shortlist_size=3,
        utility_window=2,
        risk_order=prefer_low,
        risk_identity="test-risk",
        ledger_path=path,
        resume=True,
    )
    assert resumed.checks == first.checks and resumed.elapsed_seconds >= first.elapsed_seconds
    assert (
        resumed.upper_bound == first.upper_bound
        and resumed.ledger.deferred == first.ledger.deferred
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        kernel.repair(
            problem, ObjectiveV2(((11, 9, 8),)), verifier=_controlled, ledger_path=path, resume=True
        )

    with pytest.raises(ValueError, match="scheduling"):
        kernel.repair(
            problem,
            ObjectiveV2(((10, 9, 8),)),
            verifier=_controlled,
            ledger_path=path,
            resume=True,
            shortlist_seconds=0.5,
        )


class SlowSerialization:
    def __reduce__(self):
        time.sleep(30)
        return int, (1,)


def _identity(x):
    return x


def _event_then_hang():
    emit_event(("proved", 1))
    time.sleep(30)


def test_startup_serialization_and_completed_events_are_supervised():
    before = time.monotonic()
    result = bounded_call(_identity, SlowSerialization(), timeout=0.15)
    assert result.status == "timeout" and result.cleanup_complete
    assert time.monotonic() - before < 2
    result = bounded_call(_event_then_hang, timeout=2)
    assert result.status == "timeout" and result.events == (("proved", 1),)


def test_parent_passes_duplicate_emitter_proof_to_master(monkeypatch):
    monkeypatch.setattr(kernel, "bounded_call", _sync)
    a, b, c = map(cls, "ABC")
    emitted = owl.SubClassOf(a, b)
    objects = tuple(
        RevisionObjectV2(
            str(i),
            "mapping",
            (emitted,),
            (
                ReplacementCandidateV2(str(i), "keep", (emitted,)),
                ReplacementCandidateV2(str(i), "off", (), ("delete",)),
            ),
        )
        for i in range(2)
    )
    problem = RepairInputV2(
        (owl.SubClassOf(a, c), owl.DisjointClasses((b, c))), objects, PolicyV2((a, b, c))
    )
    result = kernel.repair(
        problem, ObjectiveV2(((10, 0), (10, 0))), diagnose=False, preserve_verified_input=False
    )
    assert result.assignment == (1, 1) and result.search_status == "OPTIMAL_IN_POOL"
    assert result.checks == 2 and result.ledger.proofs


def test_fixed_only_conditional_support_preserves_inactive_plan(monkeypatch):
    monkeypatch.setattr(kernel, "bounded_call", _sync)
    b, c = map(cls, "BC")
    expression = owl.ObjectIntersectionOf((b, c))
    obj = RevisionObjectV2(
        "m",
        "mapping",
        (),
        (
            ReplacementCandidateV2("m", "off", ()),
            ReplacementCandidateV2("m", "on", (), active_expressions=(expression,)),
        ),
    )
    problem = RepairInputV2((owl.DisjointClasses((b, c)),), (obj,), PolicyV2((b, c)))
    result = kernel.repair(
        problem, ObjectiveV2(((0, 10),)), diagnose=False, preserve_verified_input=False
    )
    assert result.assignment == (0,) and result.search_status == "OPTIMAL_IN_POOL"
    assert result.ledger.proofs[0].activation_expression == expression


def _proof_then_hang(problem, assignment):
    axioms, active = kernel.materialize(problem, assignment)
    proof = detect_violations(axioms, active, problem.policy)[0]
    from exact.repair.owl import _query_id

    emit_event(
        VerificationEventV3(
            canonical_hash(assignment),
            canonical_hash((axioms, active)),
            problem.policy.content_hash,
            0,
            ObligationV2(f"{proof.kind}:{_query_id(proof.query)}", "fail", True),
            proof.rule_version,
            canonical_hash(proof.rule_version),
            proof,
        )
    )
    time.sleep(30)


def test_streamed_failure_survives_later_timeout_and_is_durable(tmp_path):
    a = cls("A")
    axiom = owl.SubClassOf(a, owl.OWL_NOTHING)
    obj = RevisionObjectV2(
        "m", "mapping", (axiom,), (ReplacementCandidateV2("m", "keep", (axiom,)),)
    )
    problem = RepairInputV2(
        (),
        (obj,),
        PolicyV2((a,)),
        budgets=BudgetsV2(total_seconds=6, verification_seconds=2, max_checks=1),
    )
    path = tmp_path / "ledger.json"
    result = kernel.repair(
        problem, ObjectiveV2(((10,),)), verifier=_proof_then_hang, diagnose=False, ledger_path=path
    )
    assert result.exclusions and not result.pending and result.ledger.events
    assert result.exclusions[0][1].verdict == "VERIFIED_INFEASIBLE"
    assert read_record(__import__("json").loads(path.read_text())).events


class SlowReturn:
    def __reduce__(self):
        return time.sleep, (30,)


def _unsafe_return():
    return SlowReturn()


def test_output_cannot_run_user_deserialization_in_coordinator():
    before = time.monotonic()
    outcome = bounded_call(_unsafe_return, timeout=2)
    assert outcome.status == "error" and "unsupported worker result constructor" in outcome.detail
    assert time.monotonic() - before < 3
