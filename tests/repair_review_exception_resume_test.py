"""REV-10: standalone recovery preserves complete source-exception evidence."""

import dataclasses
import json

import pyowl_core as owl
import pytest

from exact.repair import kernel
from exact.repair.records import (
    BaselineReportV3,
    BudgetsV2,
    ObjectiveV3,
    PolicyV3,
    QualifiedBaselineReportV3,
    RecoverySearchLedgerV3,
    RepairInputV3,
    ReplacementCandidateV2,
    RevisionObjectV2,
    canonical_hash,
    read_record,
)


def fixture():
    a = owl.Class(owl.IRI("urn:review:exception:A"))
    b = owl.Class(owl.IRI("urn:review:exception:B"))
    mapping = owl.SubClassOf(a, b)
    source, target = (owl.Declaration(a),), (owl.SubClassOf(b, owl.OWL_NOTHING),)
    objects = (
        RevisionObjectV2(
            "map",
            "mapping",
            (mapping,),
            (
                ReplacementCandidateV2("map", "keep", (mapping,), ("keep",)),
                ReplacementCandidateV2("map", "delete", (), ("delete",)),
            ),
        ),
    )
    policy = PolicyV3(
        (a, b),
        exceptions=(b,),
        exception_evidence=(kernel.source_exception_evidence(target, side="target"),),
    )
    problem = RepairInputV3(
        (*source, *target),
        objects,
        policy,
        source_axioms=source,
        target_axioms=target,
        source_documents=(("source.owl", "source-closure", "hash-a"),),
        target_documents=(("target.owl", "target-closure", "hash-b"),),
        budgets=BudgetsV2(total_seconds=60, verification_seconds=5, max_checks=20),
    )
    return problem, ObjectiveV3(((10, 0),), pool_hash=canonical_hash(objects))


def test_native_standalone_resume_restores_exception_proofs_and_gets_new_incumbent(
    tmp_path, monkeypatch
):
    problem, objective = fixture()
    path = tmp_path / "search.json"
    original_worker = kernel.bounded_call
    interrupted = []

    def stop_after_proofs(function, *args, **kwargs):
        outcome = original_worker(function, *args, **kwargs)
        if function is kernel._atomic_record and outcome.status == "complete":
            record = args[1]
            if (
                isinstance(record, RecoverySearchLedgerV3)
                and isinstance(record.baseline_artifact, QualifiedBaselineReportV3)
                and not interrupted
            ):
                interrupted.append(record)
                raise InterruptedError("after durable exception acquisition")
        return outcome

    monkeypatch.setattr(kernel, "bounded_call", stop_after_proofs)
    monkeypatch.setattr(kernel, "_NATIVE_BOUNDED_CALL", stop_after_proofs)
    with pytest.raises(InterruptedError, match="exception acquisition"):
        kernel.repair(problem, objective, ledger_path=path)
    saved = read_record(json.loads(path.read_text()))
    proof = saved.baseline_artifact.exception_proofs[0]
    assert proof.side == "target" and proof.documents == problem.target_documents
    assert proof.asserted_source_hash == canonical_hash(
        tuple(sorted(set(problem.target_axioms), key=canonical_hash))
    )
    assert proof.report["obligations"][0]["verdict"] is True
    assert proof.report["obligations"][1]["verdict"] is False
    assert proof.report["support"]["input_supported"] is True
    assert saved.feasible == ()
    spent, checks = saved.elapsed_seconds, saved.checks

    def no_reacquisition(function, *args, **kwargs):
        assert function not in {kernel.collect_baselines, kernel._verify_exception_artifact}
        return original_worker(function, *args, **kwargs)

    monkeypatch.setattr(kernel, "bounded_call", no_reacquisition)
    monkeypatch.setattr(kernel, "_NATIVE_BOUNDED_CALL", no_reacquisition)
    result = kernel.repair(problem, objective, ledger_path=path, resume=True)
    assert result.assignment == (1,), result.failures
    assert result.logical_status == "VERIFIED_FEASIBLE" and result.verification.authorizes
    assert result.elapsed_seconds >= spent and result.checks > checks
    assert result.ledger.baseline_artifact.content_hash == saved.baseline_artifact.content_hash
    assert result.ledger.pending == () and result.ledger.deferred == ()


def test_qualified_artifact_rejects_changed_source_import_policy_query_and_backend():
    problem, _ = fixture()
    artifact = kernel._verify_exception_artifact(problem)
    assert all(
        q.verdict == "pass" for q in kernel._validated_baseline_exceptions(problem, artifact)
    )
    proof = artifact.exception_proofs[0]
    for changed in (
        dataclasses.replace(proof, asserted_source_hash=canonical_hash(())),
        dataclasses.replace(proof, documents=(("changed", "closure", "hash"),)),
        dataclasses.replace(proof, policy_hash="f" * 64),
        dataclasses.replace(proof, exceptions=()),
        dataclasses.replace(proof, qualification_hash="a" * 64),
    ):
        with pytest.raises(ValueError, match="mismatch"):
            kernel._validated_baseline_exceptions(
                problem, dataclasses.replace(artifact, exception_proofs=(changed,))
            )
    report = dict(proof.report)
    report["support"] = {**dict(report["support"]), "input_supported": False}
    corrupted = dataclasses.replace(
        artifact, exception_proofs=(dataclasses.replace(proof, report=report),)
    )
    with pytest.raises(ValueError, match="disagree"):
        kernel._validated_baseline_exceptions(problem, corrupted)
    with pytest.raises(ValueError, match="missing"):
        kernel._validated_baseline_exceptions(
            problem, dataclasses.replace(artifact, exception_proofs=())
        )


def test_legacy_standalone_receipt_revalidates_and_exhausted_budget_stays_exhausted(
    tmp_path, monkeypatch
):
    from tests.repair_kernel_test import synchronous

    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    problem, objective = fixture()
    path = tmp_path / "legacy-search.json"
    bare = BaselineReportV3(kernel.baseline_identity(problem), (), 0, 0)
    saved = RecoverySearchLedgerV3(
        problem.content_hash,
        objective.content_hash,
        problem.policy.content_hash,
        elapsed_seconds=1.0,
        checks=2,
        upper_bound=objective.upper_cap,
        work_upper_bound=objective.upper_cap,
        baseline_artifact=bare,
        baseline_qualification_hash=kernel._baseline_qualification(),
    )
    path.write_text(json.dumps(saved.to_dict()))
    result = kernel.repair(problem, objective, ledger_path=path, resume=True)
    assert result.assignment == (1,) and result.checks >= 4
    assert isinstance(result.ledger.baseline_artifact, QualifiedBaselineReportV3)
    assert any("missing or incompatible" in failure for failure in result.failures)
    spent = dataclasses.replace(saved, elapsed_seconds=problem.budgets.total_seconds)
    path.write_text(json.dumps(spent.to_dict()))
    result = kernel.repair(problem, objective, ledger_path=path, resume=True)
    assert result.assignment is None and result.checks == 2
    assert result.elapsed_seconds >= problem.budgets.total_seconds
    assert result.upper_bound == objective.upper_cap


def test_shared_and_standalone_exception_projections_agree(monkeypatch):
    from tests.repair_kernel_test import synchronous

    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    problem, _ = fixture()
    shared = kernel.collect_baselines(problem)
    standalone = kernel._verify_exception_artifact(problem)
    assert isinstance(shared, QualifiedBaselineReportV3)
    assert shared.exception_checks == standalone.exception_checks
    left, right = shared.exception_proofs[0], standalone.exception_proofs[0]
    assert left.asserted_source_hash == right.asserted_source_hash
    assert left.documents == right.documents and left.qualification_hash == right.qualification_hash
    assert left.report["obligations"] == right.report["obligations"]
    # Native diagnostics retain actual measured durations, not fabricated identical timings.
    assert read_record(shared.to_dict()) == shared


def test_exhausted_recovery_preserves_pending_deferred_counts_and_bounds(tmp_path, monkeypatch):
    from exact.repair.records import DeferredAssignmentV3, PendingAssignmentV2
    from tests.repair_kernel_test import synchronous

    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    problem, objective = fixture()
    artifact = kernel._verify_exception_artifact(problem)
    pending = PendingAssignmentV2(
        (0,), objective.score((0,)), kernel._unknown(problem, (0,), "retained timeout"), attempts=2
    )
    deferred = DeferredAssignmentV3((1,), objective.score((1,)))
    saved = RecoverySearchLedgerV3(
        problem.content_hash,
        objective.content_hash,
        problem.policy.content_hash,
        pending=(pending,),
        deferred=(deferred,),
        solves=4,
        checks=7,
        elapsed_seconds=problem.budgets.total_seconds,
        upper_bound=objective.upper_cap,
        work_upper_bound=objective.upper_cap,
        baseline_artifact=artifact,
        baseline_qualification_hash=kernel._baseline_qualification(),
    )
    path = tmp_path / "exhausted.json"
    path.write_text(json.dumps(saved.to_dict()))
    before = path.read_bytes()
    result = kernel.repair(problem, objective, ledger_path=path, resume=True)
    assert result.ledger.pending == (pending,) and result.ledger.deferred == (deferred,)
    assert result.ledger.checks == 7 and result.ledger.solves == 4
    assert result.ledger.elapsed_seconds >= saved.elapsed_seconds
    assert result.ledger.upper_bound == saved.upper_bound
    assert result.ledger.work_upper_bound == saved.work_upper_bound
    assert result.ledger.baseline_artifact == artifact
    assert path.read_bytes() == before  # No write or fresh allocation after exhaustion.


def test_recursive_record_reader_preserves_nested_corrective_receipts():
    from exact.repair.records import CompactVerificationReportV3, ObligationV2

    problem, objective = fixture()
    artifact = kernel._verify_exception_artifact(problem)
    report = CompactVerificationReportV3(
        canonical_hash((1,)),
        canonical_hash(kernel.materialize(problem, (1,))),
        problem.policy.content_hash,
        "VERIFIED_FEASIBLE",
        "complete_supported_fragment",
        (ObligationV2("consistency:consistency", "pass", True),),
        coverage_hash=canonical_hash(("consistency:consistency",)),
        coverage_count=1,
        passed_count=1,
        exact_coverage=True,
    )
    saved = RecoverySearchLedgerV3(
        problem.content_hash,
        objective.content_hash,
        problem.policy.content_hash,
        feasible=(((1,), report),),
        baseline_artifact=artifact,
        baseline_qualification_hash=kernel._baseline_qualification(),
    )
    restored = read_record(saved.to_dict())
    assert restored == saved
    assert isinstance(restored.feasible[0][1], CompactVerificationReportV3)
    assert isinstance(restored.baseline_artifact, QualifiedBaselineReportV3)
    assert restored.baseline_artifact.exception_proofs[0] == artifact.exception_proofs[0]


def test_exception_class_failure_needs_a_complete_consistent_original_source():
    problem, _ = fixture()
    artifact = kernel._verify_exception_artifact(problem)
    proof = artifact.exception_proofs[0]
    report = dict(proof.report)
    obligations = list(report["obligations"])
    obligations[0] = {**dict(obligations[0]), "verdict": False}
    report["obligations"] = tuple(obligations)
    changed = dataclasses.replace(
        artifact, exception_proofs=(dataclasses.replace(proof, report=report),)
    )
    with pytest.raises(ValueError, match="disagree"):
        kernel._validated_baseline_exceptions(problem, changed)


def test_intact_exception_backed_incumbent_survives_exhausted_budget(tmp_path, monkeypatch):
    from tests.repair_kernel_test import synchronous

    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    problem, objective = fixture()
    artifact = kernel._verify_exception_artifact(problem)
    report = kernel._verify_with_exceptions(problem, (1,), artifact.exception_checks)
    assert report.authorizes
    saved = RecoverySearchLedgerV3(
        problem.content_hash,
        objective.content_hash,
        problem.policy.content_hash,
        feasible=(((1,), report),),
        elapsed_seconds=problem.budgets.total_seconds,
        checks=9,
        solves=3,
        upper_bound=objective.upper_cap,
        work_upper_bound=objective.upper_cap,
        work_empty=False,
        baseline_artifact=artifact,
        baseline_qualification_hash=kernel._baseline_qualification(),
    )
    path = tmp_path / "exhausted-incumbent.json"
    path.write_text(json.dumps(saved.to_dict()))
    original = kernel.bounded_call

    def no_reasoner_or_work(function, *args, **kwargs):
        assert function not in {
            kernel._verify_exception_artifact,
            kernel.collect_baselines,
            kernel._verify_with_exceptions,
            kernel.verify_assignment,
            kernel.solve_master,
        }
        return original(function, *args, **kwargs)

    monkeypatch.setattr(kernel, "bounded_call", no_reasoner_or_work)
    result = kernel.repair(problem, objective, ledger_path=path, resume=True)
    assert result.assignment == (1,) and result.verification.authorizes
    assert result.ledger.checks == 9 and result.ledger.solves == 3
    assert result.ledger.elapsed_seconds >= saved.elapsed_seconds
    assert result.lower_bound == 0 and result.upper_bound == objective.upper_cap
    assert result.ledger.pending == () and result.ledger.deferred == ()
