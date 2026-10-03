"""Second review: compact exception composition and independent negative replay."""

import dataclasses
import json
import pickle
import sqlite3
import time
from pathlib import Path

import pyowl_core as owl
import pytest

from exact.repair import kernel, study, workers
from exact.repair.detection import detect_violations
from exact.repair.owl import _query_id
from exact.repair.records import (
    BudgetsV2,
    CompactVerificationReportV3,
    ObjectiveV3,
    ObligationV2,
    PendingAssignmentV2,
    PolicyV3,
    RecoverySearchLedgerV3,
    RepairInputV3,
    VerificationEventV3,
    canonical_hash,
    compact_verification_report,
)


def large_exception_problem():
    classes = tuple(owl.Class(owl.IRI(f"urn:review2:public:{i}")) for i in range(130))
    exempt = owl.Class(owl.IRI("urn:review2:exception"))
    source = (owl.SubClassOf(exempt, owl.OWL_NOTHING),)
    policy = PolicyV3(
        (*classes, exempt),
        exceptions=(exempt,),
        exception_evidence=(kernel.source_exception_evidence(source),),
    )
    return RepairInputV3(
        source,
        (),
        policy,
        source_axioms=source,
        budgets=BudgetsV2(total_seconds=30, verification_seconds=5, max_checks=5),
    )


@pytest.fixture
def native_baseline():
    problem = large_exception_problem()
    raw = kernel.verify_theory(*kernel.materialize(problem, ()), problem.policy, canonical_hash(()))
    assert raw.authorizes and len(raw.obligations) > 128
    compact = compact_verification_report(raw)
    assert isinstance(compact, CompactVerificationReportV3)
    artifact = kernel._verify_exception_artifact(problem)
    assert all(q.verdict == "pass" for q in artifact.exception_checks)
    return problem, raw, compact, artifact


def test_review2_05_kernel_preserves_compact_baseline_and_exception_proof(
    native_baseline, monkeypatch
):
    from tests.repair_kernel_test import synchronous

    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    problem, raw, compact, artifact = native_baseline
    outcomes = []
    for report in (raw, compact):
        baseline = dataclasses.replace(artifact, reports=(("alignment", report),))
        result = kernel.repair(problem, ObjectiveV3(()), baseline_evidence=baseline)
        assert result.optimization_bypassed and result.assignment == (), result.failures
        assert result.checks == 0 and result.verification.authorizes
        assert kernel._valid_report(problem, (), result.verification)
        support = dict(result.verification.support)
        assert support["source_exception_proof_hashes"] == tuple(
            p.content_hash for p in artifact.exception_proofs
        )
        outcomes.append(result.verification)
    assert outcomes[0].authorizes == outcomes[1].authorizes
    assert outcomes[1].coverage_count == len(outcomes[0].expected_obligations)
    assert len(pickle.dumps(outcomes[1])) < 8192


def test_review2_05_greedy_reuses_compact_baseline(native_baseline):
    problem, raw, compact, artifact = native_baseline
    baseline = dataclasses.replace(artifact, reports=(("alignment", compact),))

    def forbidden(*args):
        raise AssertionError("complete shared baseline must not be reverified")

    result = study._greedy(problem, ObjectiveV3(()), {}, forbidden, baseline)
    assert result.assignment == () and result.verification.authorizes
    assert result.checks == 0 and kernel._valid_report(problem, (), result.verification)


def test_review2_05_missing_compact_coverage_does_not_become_complete(native_baseline, monkeypatch):
    from tests.repair_kernel_test import synchronous

    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    problem, _, compact, artifact = native_baseline
    compact = dataclasses.replace(
        compact, coverage_count=compact.coverage_count - 1, passed_count=compact.passed_count - 1
    )
    baseline = dataclasses.replace(artifact, reports=(("alignment", compact),))
    problem = dataclasses.replace(
        problem, budgets=dataclasses.replace(problem.budgets, max_checks=0)
    )
    result = kernel.repair(problem, ObjectiveV3(()), baseline_evidence=baseline)
    assert result.assignment is None and not result.optimization_bypassed


def conflict_problem():
    cls = owl.Class(owl.IRI("urn:review2:conflict"))
    return RepairInputV3(
        (owl.SubClassOf(cls, owl.OWL_NOTHING),),
        (),
        PolicyV3((cls,)),
        budgets=BudgetsV2(total_seconds=20, verification_seconds=3, max_checks=1),
    )


def conflict_event(problem):
    axioms, active = kernel.materialize(problem, ())
    proof = detect_violations(axioms, active, problem.policy)[0]
    return VerificationEventV3(
        canonical_hash(()),
        canonical_hash((axioms, active)),
        problem.policy.content_hash,
        0,
        ObligationV2(f"{proof.kind}:{_query_id(proof.query)}", "fail", True),
        proof.rule_version,
        canonical_hash(proof.rule_version),
        proof,
    )


def journal(directory, event):
    directory = str(directory)
    receipt = workers.CommittedEvents(directory, 0, 0, "0" * 64)
    workers._commit_event_batch(directory, receipt, (event,))
    workers._commit_result(directory, kernel._event_failure_report(event, "completed conflict"))
    return directory


def save_pending(path, problem, directory):
    objective = ObjectiveV3(())
    record = RecoverySearchLedgerV3(
        problem.content_hash,
        objective.content_hash,
        problem.policy.content_hash,
        pending=(PendingAssignmentV2((), 0, kernel._unknown(problem, (), "interrupted")),),
        elapsed_seconds=0.1,
        checks=1,
        upper_bound=0,
        work_upper_bound=0,
        event_journals=(((), directory),),
    )
    path.write_text(json.dumps(record.to_dict()))
    return objective


def test_review2_06_corrupt_completion_keeps_qualified_conflict_on_resume(tmp_path):
    problem = conflict_problem()
    directory = journal(tmp_path / "events", conflict_event(problem))
    with sqlite3.connect(Path(directory) / "events.sqlite") as connection:
        connection.execute("UPDATE completion SET digest=? WHERE id=1", ("0" * 64,))
    path = tmp_path / "search.json"
    objective = save_pending(path, problem, directory)
    result = kernel.repair(problem, objective, ledger_path=path, resume=True, diagnose=False)
    assert result.assignment is None and not result.pending
    assert len(result.exclusions) == 1 and result.exclusions[0][1].verdict == "VERIFIED_INFEASIBLE"
    assert result.ledger.proofs and result.checks == 1


def replay_with_blocked_completion(problem, directory):
    def blocked(_):
        time.sleep(30)

    kernel.committed_result = blocked
    return kernel._replay_journal(problem, (), directory)


def test_review2_06_negative_replay_does_not_wait_for_completion(
    tmp_path, monkeypatch, record_property
):
    problem = conflict_problem()
    directory = journal(tmp_path / "events", conflict_event(problem))

    def forbidden_completion(_):
        raise AssertionError("qualified negative replay must not read final completion")

    # Assert independence directly, without conflating process startup/storage
    # throughput with the negative-replay contract.
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "committed_result", forbidden_completion)
        failure, report = kernel._replay_journal(problem, (), directory)
    assert failure.proof is not None and not report.authorizes

    # Retain the actual spawned-worker/stalled-completion regression. Five
    # seconds allows OWL imports and durable result transport on the allocated
    # node, but cannot conceal the 30-second blocked completion above. Separate
    # REV-02 tests continue to check subsecond deadline responsiveness.
    timeout = 5
    result = workers.bounded_call(replay_with_blocked_completion, problem, directory, timeout=timeout)
    record_property("qualification_timeout_seconds", timeout)
    record_property("replay_status", result.status)
    for key, value in result.resource_usage:
        record_property(key, value)
    assert result.status == "complete", result.detail
    assert result.value[0].proof is not None and not result.value[1].authorizes


@pytest.mark.parametrize("corruption", ["policy", "assignment", "proof", "checksum"])
def test_review2_06_unqualified_events_never_become_solver_constraints(tmp_path, corruption):
    problem = conflict_problem()
    event = conflict_event(problem)
    if corruption == "policy":
        event = dataclasses.replace(event, policy_hash="0" * 64)
    elif corruption == "assignment":
        event = dataclasses.replace(event, assignment_hash="0" * 64)
    elif corruption == "proof":
        event = dataclasses.replace(
            event, proof=dataclasses.replace(event.proof, asserted_support=())
        )
    directory = journal(tmp_path / "events", event)
    if corruption == "checksum":
        with sqlite3.connect(Path(directory) / "events.sqlite") as connection:
            connection.execute("UPDATE events SET digest=? WHERE ordinal=0", ("0" * 64,))
    path = tmp_path / "search.json"
    objective = save_pending(path, problem, directory)
    result = kernel.repair(problem, objective, ledger_path=path, resume=True, diagnose=False)
    assert result.assignment is None and result.pending
    assert not result.exclusions and not result.ledger.proofs


@pytest.mark.parametrize("compact", [False, True])
def test_review2_05_composed_report_reuse_is_idempotent(native_baseline, monkeypatch, compact):
    from tests.repair_kernel_test import synchronous

    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    problem, raw, small, artifact = native_baseline
    report = kernel._compose_exceptions(
        problem, (), small if compact else raw, artifact.exception_checks, artifact
    )
    assert report.authorizes
    baseline = dataclasses.replace(artifact, reports=(("alignment", report),))
    result = kernel.repair(problem, ObjectiveV3(()), baseline_evidence=baseline)
    assert result.optimization_bypassed and result.verification == report
    greedy = study._greedy(problem, ObjectiveV3(()), {}, kernel.verify_assignment, baseline)
    assert greedy.assignment == () and greedy.verification.authorizes
    assert (
        kernel._compose_exceptions(problem, (), report, artifact.exception_checks, artifact)
        == report
    )
    tampered = dataclasses.replace(
        report,
        support=tuple(
            (k, ("0" * 64,) if k == "source_exception_proof_hashes" else v)
            for k, v in report.support
        ),
    )
    assert not kernel._compose_exceptions(
        problem, (), tampered, artifact.exception_checks, artifact
    ).authorizes


def test_review2_05_greedy_rejects_invalid_exception_proof_as_pending(native_baseline):
    problem, _, compact, artifact = native_baseline
    baseline = dataclasses.replace(
        artifact, reports=(("alignment", compact),), qualification_hash="0" * 64
    )
    result = study._greedy(problem, ObjectiveV3(()), {}, kernel.verify_assignment, baseline)
    assert result.assignment is None and result.pending and not result.exclusions


def test_review2_06_qualified_prefix_survives_later_corrupt_batch(tmp_path):
    problem = conflict_problem()
    event = conflict_event(problem)
    directory = journal(tmp_path / "events", event)
    receipt = workers.committed_events(directory)
    workers._commit_event_batch(directory, receipt, (dataclasses.replace(event, sequence=1),))
    with sqlite3.connect(Path(directory) / "events.sqlite") as connection:
        connection.execute("UPDATE events SET digest=? WHERE ordinal=1", ("0" * 64,))
    failure, report = kernel._replay_journal(problem, (), directory)
    assert failure == event and report.verdict == "VERIFIED_INFEASIBLE"
