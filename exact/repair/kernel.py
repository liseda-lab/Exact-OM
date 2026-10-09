"""Small finite-pool kernel: complete replacements, verified incumbents and bounds."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, cast

from .maxsat import PresenceCut, solve_master
from .records import (
    BaselineReportV3,
    CompactVerificationReportV3,
    DeferredAssignmentV3,
    ObjectiveV2,
    ObjectiveV3,
    ObligationV2,
    PendingAssignmentV2,
    PolicyV2,
    ProofSupportV3,
    QualifiedBaselineReportV3,
    RecoverySearchLedgerV3,
    RepairInputV2,
    RepairInputV3,
    RepairResultV2,
    RepairResultV3,
    SearchLedgerV3,
    SourceExceptionProofV3,
    VerificationEventV3,
    VerificationReportV2,
    VerificationReportV3,
    canonical_hash,
    compose_verification_report,
    freeze_public_policy,
    read_record,
)
from .workers import (
    SUPERVISION_GRACE_SECONDS,
    CallResult,
    bounded_call,
    committed_events,
    committed_result,
    emit_event,
    emit_events,
)

_NATIVE_BOUNDED_CALL = bounded_call


def _memory_options(memory_mb: float | None) -> dict[str, Any]:
    return {"memory_mb": memory_mb} if memory_mb is not None else {}


def materialize(
    problem: RepairInputV2, assignment: tuple[int, ...]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Reconstruct asserted axioms; never retain stale originals or inferred closure."""
    if len(assignment) != len(problem.objects) or any(
        type(a) is not int or not 0 <= a < len(obj.candidates)
        for obj, a in zip(problem.objects, assignment)
    ):
        raise ValueError("invalid replacement assignment")
    axioms = set(problem.fixed_axioms)
    active: set[Any] = set()
    for obj, a in zip(problem.objects, assignment):
        candidate = obj.candidates[a]
        axioms.update(candidate.axioms)
        active.update(candidate.active_expressions)
    return tuple(sorted(axioms, key=canonical_hash)), tuple(sorted(active, key=canonical_hash))


def expected_queries(active: tuple[Any, ...], policy: PolicyV2) -> tuple[str, ...]:
    """Reconstruct exact coverage independently of a worker's claimed query list."""
    import pyowl_core as owl

    from .owl import _query_id

    classes = {owl.Class(owl.IRI(c)) if isinstance(c, str) else c for c in policy.monitored_classes}
    exempt = {owl.Class(owl.IRI(c)) if isinstance(c, str) else c for c in policy.exceptions}
    names = ["consistency:consistency"]
    names += [
        f"class_satisfiability:{_query_id(c)}"
        for c in sorted(classes - exempt - {owl.OWL_NOTHING}, key=canonical_hash)
    ]
    names += [f"required_entailment:{_query_id(q)}" for q in policy.required]
    names += [f"prohibited_entailment:{_query_id(q)}" for q in policy.prohibited]
    names += [
        f"active_satisfiability:{_query_id(q)}" for q in sorted(set(active), key=canonical_hash)
    ]
    return tuple(dict.fromkeys(names))


def verify_theory(
    axioms: tuple[Any, ...],
    active: tuple[Any, ...],
    policy: PolicyV2,
    assignment_hash: str,
    *,
    backend: str = "auto",
) -> VerificationReportV3:
    """Run sound rejection first, then full capability-qualified acceptance."""
    import pyowl_core as owl

    from .detection import detect_violations
    from .owl import OwlVerifier, _query_id, snapshot_from_axioms

    theory_hash = canonical_hash((axioms, active))
    expected = expected_queries(active, policy)
    events: list[VerificationEventV3] = []
    buffer: list[VerificationEventV3] = []
    sequence = 0
    proofs = detect_violations(axioms, active, policy)
    for proof in proofs:
        name = f"{proof.kind}:{'consistency' if proof.kind == 'consistency' else _query_id(proof.query)}"
        event = VerificationEventV3(
            assignment_hash,
            theory_hash,
            policy.content_hash,
            len(events),
            ObligationV2(name, "fail", True),
            proof.rule_version,
            canonical_hash(proof.rule_version),
            proof,
        )
        emit_event(event)
        events.append(event)
    if proofs:
        by_name = {e.obligation.name: e.obligation for e in events}
        return VerificationReportV3(
            assignment_hash,
            theory_hash,
            policy.content_hash,
            "VERIFIED_INFEASIBLE",
            "partial_detection",
            tuple(
                by_name.get(
                    name, ObligationV2(name, "unknown", False, "stopped after proved failure")
                )
                for name in expected
            ),
            backend="repair-horn/v3",
            expected_obligations=expected,
            events=tuple(events),
            proofs=proofs,
        )

    def completed(query: Any, support: Any) -> None:
        nonlocal sequence
        obligation = ObligationV2(
            f"{query.kind}:{query.query_id}",
            "pass" if query.verdict == query.expected else "fail",
            True,
            query.reason or "",
        )
        event = VerificationEventV3(
            assignment_hash,
            theory_hash,
            policy.content_hash,
            sequence,
            obligation,
            f"{support.reasoner}/{support.package_version}:{support.backend}",
            canonical_hash(dataclasses.asdict(support)),
            capability=tuple(sorted(dataclasses.asdict(support).items())),
        )
        sequence += 1
        if len(events) < 128 or event.obligation.verdict == "fail":
            events.append(event)
        buffer.append(event)
        if len(buffer) >= 128 or event.obligation.verdict == "fail":
            emit_events(buffer)
            buffer.clear()

    # Inert public declarations give disappeared classes qualified fresh-entity semantics.
    declarations = tuple(
        owl.Declaration(owl.Class(owl.IRI(c)) if isinstance(c, str) else c)
        for c in policy.monitored_classes
    )
    snapshot = snapshot_from_axioms((*axioms, *declarations))
    report = OwlVerifier(reasoner=backend).check_theory(
        snapshot,
        policy.monitored_classes,
        required=policy.required,
        prohibited=policy.prohibited,
        activated=active,
        exceptions=policy.exceptions,
        on_complete=completed,
        feasibility_only=True,
    )
    emit_events(buffer)
    obligations = tuple(
        ObligationV2(
            f"{q.kind}:{q.query_id}",
            "unknown" if q.verdict is None else "pass" if q.verdict == q.expected else "fail",
            q.complete,
            q.reason or "",
        )
        for q in report.obligations
    )
    return VerificationReportV3(
        assignment_hash,
        theory_hash,
        policy.content_hash,
        report.logical_status,
        report.verification_scope,
        obligations,
        backend=f"{report.support.reasoner}/{report.support.package_version}:{report.support.backend}",
        support=tuple(sorted(dataclasses.asdict(report.support).items())),
        expected_obligations=expected,
        events=tuple(events),
    )


def verify_assignment(problem: RepairInputV2, assignment: tuple[int, ...]) -> VerificationReportV2:
    """Fresh full-theory check before incumbent promotion."""
    if set(freeze_public_policy(problem).monitored_classes) != set(
        problem.policy.monitored_classes
    ):
        raise ValueError("incomplete public policy; explicitly migrate the input before execution")
    axioms, active = materialize(problem, assignment)
    artifact = _verify_exception_artifact(problem) if problem.policy.exceptions else None
    exception_checks = artifact.exception_checks if artifact is not None else ()
    if any(not query.complete or query.verdict != "pass" for query in exception_checks):
        return VerificationReportV2(
            canonical_hash(assignment),
            canonical_hash((axioms, active)),
            problem.policy.content_hash,
            "UNKNOWN",
            "partial_detection",
            exception_checks,
            detail="a frozen source exception could not be proved",
        )
    report = verify_theory(axioms, active, problem.policy, canonical_hash(assignment))
    return _compose_exceptions(problem, assignment, report, exception_checks, artifact)


def source_exception_evidence(
    axioms: tuple[Any, ...],
    *,
    side: str = "source",
) -> tuple[str, str]:
    """Bind an exception request to a named original source theory.

    Policy evidence entries align positionally with policy.exceptions. This hash
    identifies premises; verify_assignment must still prove unsatisfiability.
    """
    if side not in {"source", "target"}:
        raise ValueError("exceptions require a source-only or target-only theory")
    return side, canonical_hash(tuple(sorted(set(axioms), key=canonical_hash)))


def _exception_groups(problem: RepairInputV2) -> dict[str, tuple[Any, ...]]:
    groups: dict[str, list[Any]] = {"source": [], "target": []}
    sources = {"source": problem.source_axioms, "target": problem.target_axioms}
    for expression, (side, digest) in zip(
        problem.policy.exceptions, problem.policy.exception_evidence
    ):
        if side not in sources or (side, digest) != source_exception_evidence(
            sources[side], side=side
        ):
            raise ValueError("exception premises do not match a frozen source theory")
        groups[side].append(expression)
    return {
        side: tuple(sorted(set(values), key=canonical_hash))
        for side, values in groups.items()
        if values
    }


def _exception_checks(
    problem: RepairInputV2, proofs: tuple[SourceExceptionProofV3, ...]
) -> tuple[ObligationV2, ...]:
    """Validate every source/report dependency; never infer proof from a bare pass."""
    from importlib.metadata import version

    from .owl import _query_id, snapshot_from_axioms

    groups = _exception_groups(problem)
    if {proof.side for proof in proofs} != set(groups) or len(proofs) != len(groups):
        raise ValueError("source-exception evidence has missing or duplicate source reports")
    checks = []
    for proof in proofs:
        axioms = problem.source_axioms if proof.side == "source" else problem.target_axioms
        documents = problem.source_documents if proof.side == "source" else problem.target_documents
        if (
            proof.asserted_source_hash
            != canonical_hash(tuple(sorted(set(axioms), key=canonical_hash)))
            or proof.documents != documents
            or proof.policy_hash != problem.policy.content_hash
            or proof.exceptions != groups[proof.side]
            or proof.qualification_hash != _baseline_qualification()
            or proof.theory_hash != snapshot_from_axioms(axioms).logical_fingerprint.hex
            or proof.report.get("theory_hash") != proof.theory_hash
        ):
            raise ValueError("source-exception source/import/policy/query/qualification mismatch")
        support = proof.report.get("support", {})
        reasoner = support.get("reasoner")
        supported = (
            reasoner in {"elk", "hermit"}
            and support.get("input_supported") is True
            and support.get("complete_imports") is True
            and support.get("implementation_version") not in {None, "", "unknown"}
            and support.get("package_version")
            == version("pyhermit" if reasoner == "hermit" else "pyelk-reasoner")
            and str(support.get("package_version")).startswith("0.2.")
        )
        obligations = proof.report.get("obligations", ())
        expected = {
            ("consistency", "consistency"),
            *(("class_satisfiability", _query_id(q)) for q in proof.exceptions),
        }
        indexed = {(row.get("kind"), row.get("query_id")): row for row in obligations}
        if len(indexed) != len(obligations) or set(indexed) != expected:
            raise ValueError("source-exception receipt does not cover exactly the declared queries")
        coverage = {
            (kind, query): complete for kind, query, complete in support.get("query_support", ())
        }
        if set(coverage) != expected or any(
            coverage[key] != indexed[key].get("complete") for key in expected
        ):
            raise ValueError("source-exception backend coverage differs from query results")
        consistency = indexed[("consistency", "consistency")]
        consistent = (
            supported and consistency.get("complete") is True and consistency.get("verdict") is True
        )
        if not consistent:
            checks.append(
                ObligationV2(
                    f"source_exception:{proof.side}:consistency",
                    "unknown",
                    False,
                    "completed supported source consistency is required before an exception",
                )
            )
        for expression in proof.exceptions:
            query = _query_id(expression)
            row = indexed[("class_satisfiability", query)]
            complete = consistent and row.get("complete") is True
            checks.append(
                ObligationV2(
                    f"source_exception:{proof.side}:{query}",
                    (
                        "unknown"
                        if not complete
                        else "pass" if row.get("verdict") is False else "fail"
                    ),
                    complete,
                    row.get("reason") or "",
                )
            )
    return tuple(checks)


def _verify_exception_artifact(problem: RepairInputV2) -> QualifiedBaselineReportV3:
    from .owl import OwlVerifier, snapshot_from_axioms

    started = time.monotonic()
    proofs = []
    try:
        groups = _exception_groups(problem)
        for side, expressions in groups.items():
            axioms = problem.source_axioms if side == "source" else problem.target_axioms
            documents = problem.source_documents if side == "source" else problem.target_documents
            snapshot = snapshot_from_axioms(axioms)
            report = OwlVerifier().check_theory(snapshot, expressions)
            proofs.append(
                SourceExceptionProofV3(
                    side,
                    canonical_hash(tuple(sorted(set(axioms), key=canonical_hash))),
                    documents,
                    problem.policy.content_hash,
                    expressions,
                    snapshot.logical_fingerprint.hex,
                    _baseline_qualification(),
                    report.to_dict(),
                )
            )
        checks = _exception_checks(problem, tuple(proofs))
    except (ValueError, LookupError) as error:
        checks = (ObligationV2("source_exception_evidence", "unknown", False, str(error)),)
    return QualifiedBaselineReportV3(
        baseline_identity(problem),
        (),
        time.monotonic() - started,
        int(bool(problem.policy.exceptions)),
        exception_checks=checks,
        exception_proofs=tuple(proofs),
        qualification_hash=_baseline_qualification(),
    )


def _verify_exceptions(problem: RepairInputV2) -> tuple[ObligationV2, ...]:
    """Independently reprove exceptions; durable acquisition keeps the full artifact."""
    return _verify_exception_artifact(problem).exception_checks if problem.policy.exceptions else ()


def _validated_baseline_exceptions(
    problem: RepairInputV2, artifact: BaselineReportV3
) -> tuple[ObligationV2, ...]:
    if artifact.identity != baseline_identity(problem):
        raise ValueError("shared baseline does not match this input/policy")
    if not problem.policy.exceptions:
        return ()
    if (
        not isinstance(artifact, QualifiedBaselineReportV3)
        or artifact.qualification_hash != _baseline_qualification()
        or artifact.evidence_revision != "source-exception-evidence/v3.1"
    ):
        raise ValueError("missing or incompatible qualified source-exception artifact")
    checks = _exception_checks(problem, artifact.exception_proofs)
    if checks != artifact.exception_checks:
        raise ValueError("source-exception checks disagree with their complete proof receipt")
    return checks


def _unknown(
    problem: RepairInputV2, assignment: tuple[int, ...], detail: str
) -> VerificationReportV2:
    return VerificationReportV2(
        canonical_hash(assignment),
        canonical_hash(materialize(problem, assignment)),
        problem.policy.content_hash,
        "UNKNOWN",
        "partial_detection",
        (ObligationV2("verification", "unknown", False, detail),),
        detail=detail,
    )


def _valid_report(problem: RepairInputV2, assignment: tuple[int, ...], report: Any) -> bool:
    if not (
        isinstance(report, VerificationReportV2)
        and report.assignment_hash == canonical_hash(assignment)
        and report.theory_hash == canonical_hash(materialize(problem, assignment))
        and report.policy_hash == problem.policy.content_hash
    ):
        return False
    if isinstance(report, VerificationReportV3) and report.authorizes:
        from .owl import _query_id

        expected = set(expected_queries(materialize(problem, assignment)[1], problem.policy))
        expected.update(
            f"source_exception:{side}:{_query_id(expression)}"
            for expression, (side, _) in zip(
                problem.policy.exceptions, problem.policy.exception_evidence
            )
        )
        if isinstance(report, CompactVerificationReportV3):
            if report.coverage_hash != canonical_hash(
                tuple(sorted(expected))
            ) or report.coverage_count != len(expected):
                return False
        elif set(report.expected_obligations) != expected:
            return False
    if (
        isinstance(problem, RepairInputV3)
        and report.authorizes
        and not isinstance(report, VerificationReportV3)
    ):
        return False
    return True


def _baseline_qualification() -> str:
    from importlib.metadata import PackageNotFoundError, version

    versions = []
    for package in ("pyowl-core", "pyelk-reasoner", "pyhermit"):
        try:
            versions.append((package, version(package)))
        except PackageNotFoundError:
            versions.append((package, "unavailable"))
    return canonical_hash(("source-exception-qualification/r2", tuple(versions)))


def baseline_identity(problem: RepairInputV2) -> str:
    """Match common diagnosis across inventory-filtered arms with the same policy."""
    return canonical_hash(
        (
            problem.source_axioms,
            problem.target_axioms,
            problem.source_documents,
            problem.target_documents,
            tuple((o.object_id, o.original_axioms) for o in problem.objects),
            problem.policy,
        )
    )


def collect_baselines(problem: RepairInputV2) -> BaselineReportV3:
    """Capture four scopes once under a single declared preparation budget."""
    import pyowl_core as owl

    started = time.monotonic()
    reports, failures = [], []
    checks = 0
    exception_checks: tuple[ObligationV2, ...] = ()
    exception_proofs: tuple[SourceExceptionProofV3, ...] = ()
    if problem.policy.exceptions and checks < problem.budgets.max_checks:
        checks += 1
        outcome = bounded_call(
            _verify_exception_artifact,
            problem,
            timeout=min(problem.budgets.total_seconds, problem.budgets.verification_seconds),
            **_memory_options(problem.budgets.memory_mb),
        )
        if outcome.status == "complete" and isinstance(outcome.value, QualifiedBaselineReportV3):
            exception_checks, exception_proofs = (
                outcome.value.exception_checks,
                outcome.value.exception_proofs,
            )
        else:
            exception_checks = (ObligationV2("source_exception", "unknown", False, outcome.detail),)
    theories = (
        ("source", problem.source_axioms),
        ("target", problem.target_axioms),
        ("union", (*problem.source_axioms, *problem.target_axioms)),
        (
            "alignment",
            (*problem.fixed_axioms, *(a for o in problem.objects for a in o.original_axioms)),
        ),
    )
    for name, axioms in theories:
        remaining = problem.budgets.total_seconds - (time.monotonic() - started)
        if remaining <= 0 or checks >= problem.budgets.max_checks:
            failures.append(f"baseline {name}: budget exhausted")
            continue
        selected = tuple(sorted(set(axioms), key=canonical_hash))
        policy = (
            problem.policy
            if name == "alignment"
            else PolicyV2(
                monitored_classes=tuple(
                    sorted(
                        {
                            e
                            for ax in selected
                            for e in owl.signature(ax)
                            if isinstance(e, owl.Class) and e != owl.OWL_NOTHING
                        },
                        key=canonical_hash,
                    )
                )
            )
        )
        checks += 1
        outcome = bounded_call(
            verify_theory,
            selected,
            (),
            policy,
            canonical_hash(name),
            timeout=min(remaining, problem.budgets.verification_seconds),
            **_memory_options(problem.budgets.memory_mb),
        )
        if outcome.status == "complete" and isinstance(outcome.value, VerificationReportV2):
            reports.append((name, outcome.value))
        else:
            failures.append(f"baseline {name}: {outcome.status}: {outcome.detail}")
    return QualifiedBaselineReportV3(
        baseline_identity(problem),
        tuple(reports),
        time.monotonic() - started,
        checks,
        tuple(failures),
        exception_checks=exception_checks,
        exception_proofs=exception_proofs,
        qualification_hash=_baseline_qualification(),
    )


def _compose_exceptions(
    problem: RepairInputV2,
    assignment: tuple[int, ...],
    report: VerificationReportV2,
    exceptions: tuple[ObligationV2, ...],
    artifact: BaselineReportV3 | None,
) -> VerificationReportV2:
    if not problem.policy.exceptions:
        return report
    if not isinstance(artifact, QualifiedBaselineReportV3) or not isinstance(
        report, VerificationReportV3
    ):
        return _unknown(problem, assignment, "missing qualified exception composition evidence")
    if exceptions != artifact.exception_checks or any(
        not q.complete or q.verdict != "pass" for q in exceptions
    ):
        return _unknown(problem, assignment, "unresolved exception composition evidence")
    _, active = materialize(problem, assignment)
    return compose_verification_report(
        report,
        expected_queries(active, problem.policy),
        exceptions,
        source_exception_proof_hashes=tuple(
            proof.content_hash for proof in artifact.exception_proofs
        ),
    )


def _verify_with_exceptions(
    problem: RepairInputV2,
    assignment: tuple[int, ...],
    exceptions: tuple[ObligationV2, ...],
    artifact: BaselineReportV3 | None = None,
) -> VerificationReportV2:
    if any(not q.complete or q.verdict != "pass" for q in exceptions):
        return _unknown(problem, assignment, "frozen exception proof is unresolved")
    if problem.policy.exceptions:
        artifact = artifact or _verify_exception_artifact(problem)
        if _validated_baseline_exceptions(problem, artifact) != exceptions:
            return _unknown(problem, assignment, "exception checks lack matching source proof")
    axioms, active = materialize(problem, assignment)
    report = verify_theory(axioms, active, problem.policy, canonical_hash(assignment))
    return _compose_exceptions(problem, assignment, report, exceptions, artifact)


def _atomic_record(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record.to_dict(), stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_search_ledger(
    path: Path,
    problem: RepairInputV2,
    objective: ObjectiveV2,
    shortlist_size: int,
    utility_window: int,
    risk_identity: str,
    shortlist_seconds: float | None,
) -> tuple[SearchLedgerV3, tuple[ObligationV2, ...] | None]:
    """Read, decode and qualify recovery evidence inside a supervised process."""
    saved = read_record(json.loads(path.read_text()))
    if not isinstance(saved, SearchLedgerV3) or (
        saved.input_hash,
        saved.objective_hash,
        saved.policy_hash,
        saved.shortlist_size,
        saved.utility_window,
        saved.risk_identity,
        saved.shortlist_seconds,
    ) != (
        problem.content_hash,
        objective.content_hash,
        problem.policy.content_hash,
        shortlist_size,
        utility_window,
        risk_identity,
        shortlist_seconds,
    ):
        raise ValueError("ledger epoch or scheduling identity mismatch")
    entries: tuple[PendingAssignmentV2 | DeferredAssignmentV3, ...] = (
        *saved.pending,
        *saved.deferred,
    )
    for entry in entries:
        if entry.value != objective.score(entry.assignment):
            raise ValueError("ledger utility mismatch")
    for assignment, report in saved.feasible:
        if not _valid_report(problem, assignment, report) or not report.authorizes:
            raise ValueError("invalid ledger incumbent")
    for assignment, report in saved.logical_exclusions:
        if (
            not _valid_report(problem, assignment, report)
            or report.verdict != "VERIFIED_INFEASIBLE"
            or not any(q.complete and q.verdict == "fail" for q in report.obligations)
        ):
            raise ValueError("invalid ledger logical exclusion")
    restored_exceptions = None
    if (
        isinstance(saved, RecoverySearchLedgerV3)
        and saved.baseline_artifact is not None
        and saved.baseline_qualification_hash == _baseline_qualification()
    ):
        try:
            restored_exceptions = _validated_baseline_exceptions(problem, saved.baseline_artifact)
        except ValueError:
            pass  # Old/incompatible receipts require the explicit budgeted recovery path.
    return saved, restored_exceptions


def _rank_shortlist(
    scorer: Callable[[tuple[int, ...]], float], assignments: tuple[tuple[int, ...], ...]
) -> tuple[float, ...]:
    """One model load per bounded shortlist, never one per candidate worker."""
    values = tuple(float(scorer(assignment)) for assignment in assignments)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("risk scores must be finite")
    return values


@dataclasses.dataclass
class _VerificationStream:
    """One killable validator with immutable indexes and bounded backend attempts."""

    problem: RepairInputV2
    assignment: tuple[int, ...]
    require_complete_stream: bool = True

    def _initialize(self) -> None:
        if hasattr(self, "expected"):
            return
        self.axioms, self.active = materialize(self.problem, self.assignment)
        self.theory_hash = canonical_hash((self.axioms, self.active))
        self.assignment_hash = canonical_hash(self.assignment)
        self.policy_hash = self.problem.policy.content_hash
        self.expected = set(expected_queries(self.active, self.problem.policy))
        self.sequence = 0
        self.observed: dict[tuple[str, str], str] = {}
        self.passed: set[str] = set()
        self.failure: VerificationEventV3 | None = None

    def __call__(self, event: Any) -> bool:
        self._initialize()
        if (
            not isinstance(event, VerificationEventV3)
            or event.sequence != self.sequence
            or not event.obligation.complete
            or event.obligation.verdict not in {"pass", "fail"}
            or event.assignment_hash != self.assignment_hash
            or event.theory_hash != self.theory_hash
            or event.policy_hash != self.policy_hash
            or event.obligation.name not in self.expected
        ):
            return False
        if event.proof is not None:
            from .detection import validate_proof

            if event.obligation.verdict != "fail" or not validate_proof(
                event.proof, self.axioms, self.active, self.problem.policy
            ):
                return False
            attempt = "detector"
        else:
            capability = dict(event.capability)
            attempt = str(capability.get("reasoner", ""))
            if not (
                capability.get("input_supported") is True
                and capability.get("complete_imports") is True
                and attempt in {"elk", "hermit"}
                and str(capability.get("package_version", "")).startswith("0.2.")
                and canonical_hash(capability) == event.capability_hash
                and event.backend
                == f"{attempt}/{capability['package_version']}:{capability.get('backend')}"
            ):
                return False
        key = (attempt, event.obligation.name)
        if key in self.observed:
            return False  # one completed result per frozen obligation/backend route
        self.observed[key] = event.obligation.verdict
        self.sequence += 1
        if event.obligation.verdict == "fail" and self.failure is None:
            self.failure = event
        if event.obligation.verdict == "pass":
            self.passed.add(event.obligation.name)
        return True

    def finish(self, report: Any) -> VerificationReportV2:
        self._initialize()
        if self.failure is not None:
            detail = "qualified completed failure retained"
            if getattr(report, "authorizes", False):
                detail += "; integrity discrepancy: positive final report contradicts failure"
            return _event_failure_report(self.failure, detail)
        if not _valid_report(self.problem, self.assignment, report):
            return _unknown(
                self.problem, self.assignment, "final verification identity/coverage mismatch"
            )
        if report.authorizes and self.require_complete_stream and self.passed != self.expected:
            return _unknown(
                self.problem,
                self.assignment,
                "final report lacks committed complete event coverage",
            )
        return cast(VerificationReportV2, report)


def _event_failure_report(event: VerificationEventV3, detail: str) -> VerificationReportV3:
    return VerificationReportV3(
        event.assignment_hash,
        event.theory_hash,
        event.policy_hash,
        "VERIFIED_INFEASIBLE",
        "partial_detection",
        (event.obligation,),
        backend=event.backend,
        detail=detail,
        events=(event,),
        proofs=(event.proof,) if event.proof else (),
    )


def _replay_journal(problem: RepairInputV2, assignment: tuple[int, ...], directory: str):
    validator = _VerificationStream(problem, assignment)
    receipt = committed_events(directory)
    for event in receipt:
        if not validator(event):
            raise ValueError("invalid committed verification event")
        if validator.failure is not None:
            # A qualified counterexample is sufficient independently of any later
            # query or completion record. Positive acceptance still needs all of them.
            return validator.failure, _event_failure_report(
                validator.failure, "qualified committed conflict; completion not required"
            )
    validator._initialize()
    report = committed_result(directory)
    return validator.failure, validator.finish(report) if report is not None else None


def repair(
    problem: RepairInputV2,
    objective: ObjectiveV2,
    *,
    verifier: Callable[[RepairInputV2, tuple[int, ...]], VerificationReportV2] = verify_assignment,
    diagnose: bool = True,
    preserve_verified_input: bool = True,
    shortlist_size: int = 1,
    utility_window: int = 0,
    shortlist_seconds: float | None = None,
    risk_order: Callable[[tuple[int, ...]], float] | None = None,
    risk_identity: str = "none",
    baseline_evidence: BaselineReportV3 | None = None,
    initial_assignment: tuple[int, ...] | None = None,
    ledger_path: str | Path | None = None,
    resume: bool = False,
) -> RepairResultV3:
    """Proof-guided exact selection with a durable, conservatively bounded frontier.

    Each enumerated plan stays deferred until verified, rejected, or made pending.
    Risk changes order only. Restart retains used time/calls, not a fresh budget.
    """
    if len(objective.unary) != len(problem.objects) or any(
        len(row) != len(obj.candidates) for row, obj in zip(objective.unary, problem.objects)
    ):
        raise ValueError("objective shape does not match the frozen inventory")
    if verifier is verify_assignment and set(
        freeze_public_policy(problem).monitored_classes
    ) != set(problem.policy.monitored_classes):
        raise ValueError("incomplete public policy; explicitly migrate the input before execution")
    if (
        isinstance(objective, ObjectiveV3)
        and objective.pool_hash
        and objective.pool_hash != canonical_hash(problem.objects)
    ):
        raise ValueError("objective pool identity differs from the frozen inventory")
    if (
        type(shortlist_size) is not int
        or shortlist_size < 1
        or type(utility_window) is not int
        or utility_window < 0
    ):
        raise ValueError("shortlist size/window must be positive/nonnegative integers")
    if shortlist_seconds is not None and (
        not math.isfinite(shortlist_seconds) or shortlist_seconds <= 0
    ):
        raise ValueError("shortlist construction budget must be finite and positive")
    if risk_order is not None and risk_identity == "none":
        raise ValueError("a risk ordering callback requires a frozen identity")
    if initial_assignment is not None:
        materialize(problem, initial_assignment)  # validate before starting any work
        if resume:
            raise ValueError("initial assignment cannot alter a resumed search epoch")
    budgets, started = problem.budgets, time.monotonic()
    prior_elapsed = 0.0
    reserved_until = 0.0
    upper: int | None = objective.upper_cap
    work_bound: int | None = objective.upper_cap
    lower = None
    incumbent = incumbent_report = None
    first_verified_seconds = None
    cuts: list[tuple[tuple[int, ...], VerificationReportV2]] = []
    proofs: list[ProofSupportV3] = []
    presence_cuts: list[PresenceCut] = []
    pending: dict[tuple[int, ...], PendingAssignmentV2] = {}
    deferred: dict[tuple[int, ...], DeferredAssignmentV3] = {}
    feasible: dict[tuple[int, ...], VerificationReportV2] = {}
    events: list[VerificationEventV3] = []
    event_journals: list[tuple[tuple[int, ...], str]] = []
    retained_baseline = baseline_evidence
    restored_exception_checks = None
    verification_cache: dict[str, VerificationReportV2] = {}
    validated_proofs: set[str] = set()
    proof_deadline = float("inf")
    cache_hits = 0
    persistence_failure = ""
    failures: list[str] = []
    baseline: list[tuple[str, VerificationReportV2]] = []
    solves = checks = revision = 0
    work_empty = False
    timings = {"baseline": 0.0, "verification": 0.0, "master": 0.0, "risk": 0.0, "persistence": 0.0}
    exceptions: tuple[ObligationV2, ...] = ()
    path = Path(ledger_path) if ledger_path is not None else None

    def remaining(stage: float) -> float:
        return min(
            stage, max(0.0, budgets.total_seconds - prior_elapsed - (time.monotonic() - started))
        )

    def snapshot() -> RecoverySearchLedgerV3:
        return RecoverySearchLedgerV3(
            problem.content_hash,
            objective.content_hash,
            problem.policy.content_hash,
            tuple(cuts),
            tuple(proofs),
            tuple(deferred.values()),
            tuple(pending.values()),
            tuple(feasible.items()),
            work_bound,
            upper,
            work_empty,
            revision,
            solves,
            checks,
            prior_elapsed + max(time.monotonic(), reserved_until) - started,
            shortlist_size,
            utility_window,
            risk_identity,
            tuple(failures),
            tuple(baseline),
            tuple(events),
            shortlist_seconds=shortlist_seconds,
            baseline_artifact=retained_baseline,
            baseline_qualification_hash=_baseline_qualification(),
            event_journals=tuple(event_journals),
        )

    def persist() -> bool:
        nonlocal revision, persistence_failure
        revision += 1
        if path is None:
            return True
        if persistence_failure:
            return False
        allowance = remaining(budgets.verification_seconds)
        if allowance <= 0:
            persistence_failure = (
                "ledger persistence: total budget exhausted; last committed state retained"
            )
            failures.append(persistence_failure)
            return False
        before = time.monotonic()
        # The write itself can be interrupted after publication. Its durable
        # receipt reserves the bounded operation, including cleanup; a later
        # successful publication reconciles earlier reservations to elapsed work.
        record = snapshot()
        persistence_reservation = allowance + SUPERVISION_GRACE_SECONDS
        record = dataclasses.replace(
            record,
            elapsed_seconds=max(
                record.elapsed_seconds, prior_elapsed + before - started + persistence_reservation
            ),
            persistence_reservation_seconds=persistence_reservation,
        )
        result = bounded_call(
            _atomic_record, path, record, timeout=allowance, **_memory_options(budgets.memory_mb)
        )
        timings["persistence"] += time.monotonic() - before
        if result.status != "complete":
            persistence_failure = f"ledger persistence: {result.status}: {result.detail}; last committed state retained"
            failures.append(persistence_failure)
            return False
        return True

    def refresh_bound() -> None:
        nonlocal upper
        values = (
            ([lower] if lower is not None else [])
            + [x.value for x in pending.values()]
            + [x.value for x in deferred.values()]
        )
        if not work_empty and work_bound is not None:
            values.append(work_bound)
        upper = min(upper, max(values)) if values and upper is not None else None

    def applies(proof: ProofSupportV3, assignment: tuple[int, ...]) -> bool:
        axioms, active = materialize(problem, assignment)
        return set(proof.asserted_support) <= set(axioms) and (
            proof.activation_expression is None or proof.activation_expression in active
        )

    def qualify_proof(proof: ProofSupportV3, assignment: tuple[int, ...]) -> bool:
        from .detection import validate_proof

        if proof.content_hash in validated_proofs:
            return True
        allowance = min(remaining(budgets.verification_seconds), proof_deadline - time.monotonic())
        if allowance <= 0:
            return False
        outcome = bounded_call(
            validate_proof,
            proof,
            *materialize(problem, assignment),
            problem.policy,
            timeout=allowance,
            **_memory_options(budgets.memory_mb),
        )
        if outcome.status == "complete" and outcome.value is True:
            validated_proofs.add(proof.content_hash)
            return True
        return False

    def register_proof(proof: ProofSupportV3, assignment: tuple[int, ...]) -> None:
        if not qualify_proof(proof, assignment):
            failures.append("rejected unqualified proof support")
            return
        if proof in proofs:
            return
        proofs.append(proof)
        if proof.activation_expression is None:
            presence_cuts.append(PresenceCut(proof.asserted_support))
        else:
            # OR activation: one guarded clause for every activating emitter.
            for i, obj in enumerate(problem.objects):
                for a, candidate in enumerate(obj.candidates):
                    if proof.activation_expression in candidate.active_expressions:
                        presence_cuts.append(PresenceCut(proof.asserted_support, ((i, a),)))
        for selected in feasible:
            if applies(proof, selected):
                raise RuntimeError("proof contradicts a completely verified incumbent")
        for selected in tuple(dict.fromkeys((*pending, *deferred))):
            if applies(proof, selected):
                rebound = dataclasses.replace(
                    proof, theory_hash=canonical_hash(materialize(problem, selected))
                )
                report = VerificationReportV3(
                    canonical_hash(selected),
                    rebound.theory_hash,
                    problem.policy.content_hash,
                    "VERIFIED_INFEASIBLE",
                    "partial_detection",
                    (ObligationV2(f"proof:{rebound.content_hash}", "fail", True),),
                    backend=rebound.rule_version,
                    proofs=(rebound,),
                )
                cuts.append((selected, report))
                pending.pop(selected, None)
                deferred.pop(selected, None)

    def accept_report(
        assignment: tuple[int, ...], report: VerificationReportV2, attempts: int = 1
    ) -> None:
        nonlocal incumbent, incumbent_report, lower, first_verified_seconds
        value = objective.score(assignment)
        deferred.pop(assignment, None)
        if report.authorizes:
            pending.pop(assignment, None)
            feasible[assignment] = report
            if lower is None or value > lower:
                if incumbent is None:
                    first_verified_seconds = prior_elapsed + time.monotonic() - started
                incumbent, incumbent_report, lower = assignment, report, value
        elif report.verdict == "VERIFIED_INFEASIBLE" and any(
            q.complete and q.verdict == "fail" for q in report.obligations
        ):
            pending.pop(assignment, None)
            if not any(a == assignment for a, _ in cuts):
                cuts.append((assignment, report))
            for proof in getattr(report, "proofs", ()):
                register_proof(proof, assignment)
        else:
            pending[assignment] = PendingAssignmentV2(assignment, value, report, attempts)
        refresh_bound()
        persist()

    def event_valid(event: Any, assignment: tuple[int, ...]) -> bool:
        if not isinstance(event, VerificationEventV3) or not event.obligation.complete:
            return False
        if (
            event.assignment_hash != canonical_hash(assignment)
            or event.theory_hash != canonical_hash(materialize(problem, assignment))
            or event.policy_hash != problem.policy.content_hash
            or event.obligation.name
            not in expected_queries(materialize(problem, assignment)[1], problem.policy)
        ):
            return False
        if event.proof is not None:
            return qualify_proof(event.proof, assignment)
        capability = dict(event.capability)
        return (
            capability.get("input_supported") is True
            and capability.get("complete_imports") is True
            and capability.get("reasoner") in {"elk", "hermit"}
            and str(capability.get("package_version", "")).startswith("0.2.")
            and canonical_hash(capability) == event.capability_hash
            and event.backend
            == f"{capability['reasoner']}/{capability['package_version']}:{capability.get('backend')}"
        )

    def stage_call(stage: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal reserved_until
        before = time.monotonic()
        allowance = max(0.0, float(kwargs.get("timeout", 0.0)))
        reserved_until = before + allowance + SUPERVISION_GRACE_SECONDS
        try:
            if not persist():
                return CallResult("persistence_error", detail=persistence_failure)
            kwargs["timeout"] = min(
                remaining(allowance), max(0.0, allowance - (time.monotonic() - before))
            )
            return bounded_call(function, *args, **kwargs, **_memory_options(budgets.memory_mb))
        finally:
            timings[stage] += time.monotonic() - before
            reserved_until = 0.0
            persist()

    def check(assignment: tuple[int, ...]) -> VerificationReportV2:
        nonlocal checks, cache_hits, proof_deadline
        checks += 1
        theory_key = canonical_hash((materialize(problem, assignment), problem.policy, exceptions))
        if theory_key in verification_cache:
            cached = verification_cache[theory_key]
            cache_hits += 1
            extras: dict[str, Any] = (
                {
                    "events": tuple(
                        dataclasses.replace(e, assignment_hash=canonical_hash(assignment))
                        for e in cached.events
                    )
                }
                if isinstance(cached, VerificationReportV3)
                else {}
            )
            return dataclasses.replace(cached, assignment_hash=canonical_hash(assignment), **extras)
        if assignment in pending:
            pending[assignment] = dataclasses.replace(
                pending[assignment], attempts=pending[assignment].attempts + 1
            )
        else:
            deferred.pop(assignment, None)
            pending[assignment] = PendingAssignmentV2(
                assignment,
                objective.score(assignment),
                _unknown(problem, assignment, "verification dispatched; not yet completed"),
                1,
            )
        persist()
        event_offset = len(events)
        directory = str(
            (path.parent / (path.name + ".events") if path else Path(tempfile.gettempdir()))
            / ("verification-" + uuid.uuid4().hex)
        )
        event_journals.append((assignment, directory))
        persist()
        sink = _VerificationStream(problem, assignment, verifier is verify_assignment)

        function: Callable[..., VerificationReportV2] = verifier
        args: tuple[Any, ...] = (problem, assignment)
        if verifier is verify_assignment:
            function, args = _verify_with_exceptions, (problem, assignment, exceptions)
            if problem.policy.exceptions:
                args += (retained_baseline,)
        allowance = remaining(budgets.verification_seconds)
        proof_deadline = time.monotonic() + allowance
        outcome = stage_call(
            "verification",
            function,
            *args,
            timeout=allowance,
            **(
                {"event_handler": sink, "event_directory": directory}
                if bounded_call is _NATIVE_BOUNDED_CALL
                else {}
            ),
        )
        proof_deadline = float("inf")
        if outcome.event_failure is not None:
            events.append(dataclasses.replace(outcome.event_failure, receipt_index=len(events)))
            persist()
        validated = (
            isinstance(outcome.value, VerificationReportV2)
            if bounded_call is _NATIVE_BOUNDED_CALL
            else _valid_report(problem, assignment, outcome.value)
        )
        if outcome.status == "complete" and validated:
            report = cast(VerificationReportV2, outcome.value)
            if isinstance(report, VerificationReportV3) and bounded_call is _NATIVE_BOUNDED_CALL:
                report = dataclasses.replace(report, events=tuple(events[event_offset:]))
            if report.verdict in {"VERIFIED_FEASIBLE", "VERIFIED_INFEASIBLE"}:
                verification_cache[theory_key] = report
            return report
        completed = (
            events[event_offset:]
            if bounded_call is _NATIVE_BOUNDED_CALL
            else [event for event in outcome.events if event_valid(event, assignment)]
        )
        rejected = [event for event in completed if event.obligation.verdict == "fail"]
        if rejected:
            return VerificationReportV3(
                canonical_hash(assignment),
                canonical_hash(materialize(problem, assignment)),
                problem.policy.content_hash,
                "VERIFIED_INFEASIBLE",
                "partial_detection",
                tuple(e.obligation for e in rejected),
                backend=rejected[0].backend,
                detail=f"retained completed failure after {outcome.status}",
                events=tuple(completed),
                proofs=tuple(e.proof for e in rejected if e.proof is not None),
            )
        detail = outcome.detail or "verifier returned a report for a different theory/policy"
        failures.append(f"verification: {outcome.status}: {detail}")
        return _unknown(problem, assignment, detail)

    if resume:
        if path is None:
            raise ValueError("resume requires an existing search ledger")
        restored_ledger = bounded_call(
            _read_search_ledger,
            path,
            problem,
            objective,
            shortlist_size,
            utility_window,
            risk_identity,
            shortlist_seconds,
            timeout=remaining(budgets.verification_seconds),
            **_memory_options(budgets.memory_mb),
        )
        if restored_ledger.status != "complete":
            raise ValueError(
                "search ledger recovery: " + restored_ledger.status + ": " + restored_ledger.detail
            )
        saved, restored_exception_checks = restored_ledger.value
        prior_elapsed, solves, checks, revision = (
            saved.elapsed_seconds,
            saved.solves,
            saved.checks,
            saved.revision,
        )
        upper, work_bound, work_empty = saved.upper_bound, saved.work_upper_bound, saved.work_empty
        cuts, failures, baseline, events = (
            list(saved.logical_exclusions),
            list(saved.failures),
            list(saved.baseline),
            list(saved.events),
        )
        if isinstance(saved, RecoverySearchLedgerV3):
            event_journals = list(saved.event_journals)
            if retained_baseline is None:
                retained_baseline = saved.baseline_artifact
        pending = {p.assignment: p for p in saved.pending}
        deferred = {p.assignment: p for p in saved.deferred}
        for assignment, report in saved.feasible:
            feasible[assignment] = report
            if lower is None or objective.score(assignment) > lower:
                incumbent, incumbent_report, lower = assignment, report, objective.score(assignment)
        # A stored scalar is not a solver certificate. Rebuild conservative bounds
        # on restart; this may lose tightness, never unresolved alternatives.
        upper = objective.upper_cap
        work_empty, work_bound = False, objective.upper_cap
        if lower is not None and saved.upper_bound is not None and saved.upper_bound < lower:
            raise ValueError("ledger upper bound is below its incumbent")
        for proof in saved.proofs:
            origin = next(
                (a for a, report in cuts if report.theory_hash == proof.theory_hash), None
            )
            if origin is None:
                raise ValueError("proof has no retained verified origin")
            register_proof(proof, origin)
        for assignment in tuple(dict.fromkeys((*pending, *deferred))):
            for event in saved.events:
                if event_valid(event, assignment) and event.obligation.verdict == "fail":
                    report = VerificationReportV3(
                        event.assignment_hash,
                        event.theory_hash,
                        event.policy_hash,
                        "VERIFIED_INFEASIBLE",
                        "partial_detection",
                        (event.obligation,),
                        backend=event.backend,
                        events=(event,),
                        proofs=(event.proof,) if event.proof else (),
                    )
                    accept_report(assignment, report)
                    break

    if resume:
        for assignment, directory in event_journals:
            if assignment in feasible or any(a == assignment for a, _ in cuts):
                continue
            if remaining(budgets.verification_seconds) <= 0:
                break
            restored = stage_call(
                "verification",
                _replay_journal,
                problem,
                assignment,
                directory,
                timeout=remaining(budgets.verification_seconds),
            )
            if restored.status != "complete":
                failures.append(f"event replay: {restored.status}: {restored.detail}")
                continue
            failure, completed_report = restored.value
            if failure is not None:
                events.append(dataclasses.replace(failure, receipt_index=len(events)))
                accept_report(
                    assignment, _event_failure_report(failure, "replayed committed failure")
                )
            elif completed_report is not None:
                accept_report(assignment, completed_report)

    # REV-10: restore the complete immutable baseline receipt, not its projected booleans.
    if baseline_evidence is None and resume and isinstance(saved, RecoverySearchLedgerV3):
        if (
            saved.baseline_artifact is not None
            and saved.baseline_qualification_hash == _baseline_qualification()
        ):
            baseline_evidence = saved.baseline_artifact
        elif problem.policy.exceptions:
            failures.append(
                "source exception evidence missing or qualification changed; revalidation required"
            )
    if baseline_evidence is not None and baseline_evidence.identity != baseline_identity(problem):
        raise ValueError("shared baseline does not match this input/policy")
    if (
        baseline_evidence is None
        and diagnose
        and verifier is verify_assignment
        and not resume
        and remaining(1) > 0
    ):
        remaining_problem = dataclasses.replace(
            problem,
            budgets=dataclasses.replace(
                budgets,
                total_seconds=remaining(budgets.total_seconds),
                max_checks=max(0, budgets.max_checks - checks),
            ),
        )
        capture = stage_call(
            "baseline",
            collect_baselines,
            remaining_problem,
            timeout=remaining(budgets.total_seconds),
        )
        if capture.status == "complete" and isinstance(capture.value, BaselineReportV3):
            baseline_evidence = capture.value
            checks += baseline_evidence.checks
            failures.extend(baseline_evidence.failures)
        else:
            failures.append("baseline acquisition: " + capture.status + ": " + capture.detail)
    qualified = not problem.policy.exceptions
    if baseline_evidence is not None:
        retained_baseline = baseline_evidence
        baseline = list(baseline_evidence.reports)
        if (
            resume
            and restored_exception_checks is not None
            and isinstance(saved, RecoverySearchLedgerV3)
            and baseline_evidence == saved.baseline_artifact
        ):
            exceptions = restored_exception_checks
            qualified = True
        elif problem.policy.exceptions and remaining(1) > 0:
            validation = stage_call(
                "baseline",
                _validated_baseline_exceptions,
                problem,
                baseline_evidence,
                timeout=remaining(budgets.verification_seconds),
            )
            if validation.status == "complete":
                exceptions = validation.value
                qualified = True
            else:
                failures.append("source exception receipt unavailable: " + validation.detail)
    if (
        not qualified
        and verifier is verify_assignment
        and remaining(1) > 0
        and checks < budgets.max_checks
    ):
        checks += 1
        acquisition = stage_call(
            "baseline",
            _verify_exception_artifact,
            problem,
            timeout=remaining(budgets.verification_seconds),
        )
        if acquisition.status == "complete" and isinstance(
            acquisition.value, QualifiedBaselineReportV3
        ):
            fresh = acquisition.value
            retained_baseline = dataclasses.replace(
                fresh,
                reports=tuple(baseline),
                elapsed_seconds=fresh.elapsed_seconds
                + (baseline_evidence.elapsed_seconds if baseline_evidence else 0.0),
                checks=fresh.checks + (baseline_evidence.checks if baseline_evidence else 0),
            )
            exceptions = fresh.exception_checks
            qualified = True
        else:
            exceptions = (
                ObligationV2(
                    "source_exception",
                    "unknown",
                    False,
                    "source exception revalidation "
                    + acquisition.status
                    + ": "
                    + acquisition.detail,
                ),
            )
    elif not qualified:
        exceptions = (
            ObligationV2(
                "source_exception",
                "unknown",
                False,
                "qualified source exception evidence unavailable; revalidation resource budget exhausted",
            ),
        )
    if problem.policy.exceptions and (
        not qualified or any(not q.complete or q.verdict != "pass" for q in exceptions)
    ):
        # Old scalar pass records cannot keep an incumbent authorized without its source proof.
        for assignment in feasible:
            pending[assignment] = PendingAssignmentV2(
                assignment,
                objective.score(assignment),
                _unknown(problem, assignment, "source exception evidence unavailable on resume"),
            )
        feasible.clear()
        incumbent = incumbent_report = lower = None
    persist()  # proof acquisition is durable before any candidate verification begins.

    original = tuple(
        next(
            (
                a
                for a, c in enumerate(o.candidates)
                if set(c.axioms) == set(o.original_axioms) and not c.active_expressions
            ),
            -1,
        )
        for o in problem.objects
    )
    bypass = False
    if not resume and -1 not in original:
        existing = dict(baseline).get("alignment")
        if existing is not None and existing.theory_hash == canonical_hash(
            materialize(problem, original)
        ):
            existing = dataclasses.replace(existing, assignment_hash=canonical_hash(original))
            if exceptions and existing.authorizes and isinstance(existing, VerificationReportV3):
                existing = _compose_exceptions(
                    problem, original, existing, exceptions, retained_baseline
                )
            if _valid_report(problem, original, existing):
                accept_report(original, existing)
                bypass = existing.authorizes and preserve_verified_input
            else:
                failures.append(
                    "shared alignment baseline lacks complete matching verification evidence"
                )
        elif checks < budgets.max_checks and remaining(1) > 0:
            deferred[original] = DeferredAssignmentV3(original, objective.score(original))
            report = check(original)
            baseline.append(("alignment", report))
            accept_report(original, report)
            bypass = report.authorizes and preserve_verified_input

    if (
        not bypass
        and initial_assignment is not None
        and initial_assignment not in feasible
        and checks < budgets.max_checks
        and remaining(1) > 0
    ):
        # New inventory/objective epoch: candidate IDs were remapped by the
        # scheduler. Recheck all context and score it in this objective; no old
        # certificate or scalar bound authorizes this initial assignment.
        deferred[initial_assignment] = DeferredAssignmentV3(
            initial_assignment, objective.score(initial_assignment)
        )
        accept_report(initial_assignment, check(initial_assignment))

    while (
        not bypass and not persistence_failure and remaining(1) > 0 and checks < budgets.max_checks
    ):
        refresh_bound()
        if lower is not None and upper == lower:
            break
        if (
            not deferred
            and not work_empty
            and (lower is None or work_bound is None or work_bound > lower)
        ):
            construction_started, floor = time.monotonic(), None
            while (
                solves < budgets.max_solves and len(deferred) < shortlist_size and remaining(1) > 0
            ):
                allowance = remaining(budgets.solver_seconds)
                if shortlist_seconds is not None:
                    allowance = min(
                        allowance, shortlist_seconds - (time.monotonic() - construction_started)
                    )
                if allowance <= 0:
                    break
                blocked = tuple(
                    dict.fromkeys((*[a for a, _ in cuts], *pending, *feasible, *deferred))
                )
                solves += 1
                persist()
                outcome = stage_call(
                    "master",
                    solve_master,
                    objective,
                    blocked,
                    problem=problem,
                    presence_cuts=tuple(presence_cuts),
                    timeout=allowance,
                )
                if outcome.status != "complete":
                    failures.append(f"master: {outcome.status}: {outcome.detail}")
                    break
                master = outcome.value
                if master.status == "empty" and master.assignment is None:
                    work_empty, work_bound = True, None
                    persist()
                    break
                if (
                    master.status != "optimal"
                    or master.assignment is None
                    or objective.score(master.assignment) != master.upper_bound
                    or master.assignment in blocked
                ):
                    failures.append("master returned an unqualified result")
                    break
                work_bound = master.upper_bound
                if floor is None:
                    floor = master.upper_bound - utility_window
                if master.upper_bound < floor or (
                    lower is not None and master.upper_bound <= lower
                ):
                    # Leave lower strata in the unrestricted work region, with its exact bound.
                    persist()
                    break
                deferred[master.assignment] = DeferredAssignmentV3(
                    master.assignment, master.upper_bound
                )
                persist()  # exclusion and region transfer are the same parent transaction
            if risk_order is not None and len(deferred) > 1 and remaining(1) > 0:
                assignments = tuple(deferred)
                ranked = stage_call(
                    "risk",
                    _rank_shortlist,
                    risk_order,
                    assignments,
                    timeout=remaining(budgets.solver_seconds),
                )
                if ranked.status == "complete" and len(ranked.value) == len(assignments):
                    for assignment, risk in zip(assignments, ranked.value):
                        deferred[assignment] = dataclasses.replace(deferred[assignment], risk=risk)
                else:
                    failures.append(
                        "risk ordering unavailable; deterministic utility tie order used"
                    )
                persist()
            refresh_bound()
        if lower is not None and upper == lower:
            break
        eligible = [d for d in deferred.values() if lower is None or d.value > lower]
        if eligible:
            selected = min(eligible, key=lambda d: (d.risk, -d.value, d.assignment))
            accept_report(selected.assignment, check(selected.assignment))
            continue
        retry = next(
            (
                p
                for p in sorted(pending.values(), key=lambda p: (-p.value, p.assignment))
                if p.attempts <= budgets.retries and (lower is None or p.value > lower)
            ),
            None,
        )
        if retry is not None:
            claimed_attempts = retry.attempts + 1
            accept_report(retry.assignment, check(retry.assignment), claimed_attempts)
            continue
        break
    refresh_bound()
    status = (
        "OPTIMAL_IN_POOL"
        if lower is not None and upper == lower
        else (
            "INCUMBENT_WITH_GAP"
            if lower is not None
            else (
                "NO_FEASIBLE_IN_POOL"
                if work_empty and not pending and not deferred
                else "UNRESOLVED"
            )
        )
    )
    persist()
    result = _result(
        problem,
        objective,
        status,
        incumbent,
        incumbent_report,
        lower,
        upper,
        pending,
        cuts,
        failures,
        solves,
        checks,
        baseline,
        elapsed_seconds=prior_elapsed + time.monotonic() - started,
        stage_seconds=tuple(sorted(timings.items())),
        first_verified_seconds=first_verified_seconds,
    )
    generation = {
        "bounded_enumerated": "COMPLETE_DECLARED_ENUMERATION",
        "sampled": "SAMPLED",
        "bounded_sampled": "SAMPLED",
    }.get(problem.candidate_coverage, "PARTIAL_RESOURCE_LIMIT")
    statuses = [
        report.get("generation_status")
        for report in problem.proposal_provenance
        if hasattr(report, "get") and report.get("generation_status")
    ]
    if statuses:
        order = (
            "ERROR",
            "INVALID_LANGUAGE",
            "PARTIAL_RESOURCE_LIMIT",
            "SAMPLED",
            "COMPLETE_DECLARED_ENUMERATION",
        )
        generation = next(s for s in order if s in statuses)
    values = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    if result.assignment is None:
        values["ontology_patch"] = None
    final_ledger = snapshot()
    values["elapsed_seconds"] = final_ledger.elapsed_seconds
    return RepairResultV3(
        **values,
        ledger=final_ledger,
        generation_status=generation,
        optimization_bypassed=bypass,
        resource_counters=(("verification_cache_hits", cache_hits),),
    )


def _result(
    problem,
    objective,
    status,
    assignment,
    report,
    lower,
    upper,
    pending,
    cuts,
    failures,
    solves,
    checks,
    baseline,
    *,
    elapsed_seconds: float = 0.0,
    stage_seconds: tuple[tuple[str, float], ...] = (),
    first_verified_seconds: float | None = None,
) -> RepairResultV2:
    selected = (
        ()
        if assignment is None
        else tuple(obj.candidates[a] for obj, a in zip(problem.objects, assignment))
    )
    alignment = (
        None
        if assignment is None
        else tuple(
            ax
            for obj, c in zip(problem.objects, selected)
            if obj.kind == "mapping"
            for ax in c.axioms
        )
    )
    patch = tuple(
        (obj.occurrence_id, obj.original_axioms, c.axioms)
        for obj, c in zip(problem.objects, selected)
        if obj.kind == "ontology_axiom" and set(obj.original_axioms) != set(c.axioms)
    )
    return RepairResultV2(
        problem.content_hash,
        objective.content_hash,
        "VERIFIED_FEASIBLE" if assignment is not None else "UNKNOWN",
        status,
        report.scope if report else "partial_detection",
        problem.candidate_coverage,
        assignment,
        selected,
        alignment,
        patch,
        lower,
        upper,
        report,
        tuple(pending.values()),
        tuple(cuts),
        tuple(failures),
        solves,
        checks,
        tuple(baseline),
        model_status=problem.model_status,
        elapsed_seconds=elapsed_seconds,
        stage_seconds=stage_seconds,
        first_verified_seconds=first_verified_seconds,
    )


def replay_safety(problem: RepairInputV2, result: RepairResultV2, *, timeout: float = 30.0) -> bool:
    """Recheck a recorded repair from asserted inputs without model or optimiser."""
    if result.input_hash != problem.content_hash or result.assignment is None:
        return False
    try:
        materialize(problem, result.assignment)
    except (ValueError, TypeError):
        return False
    expected = tuple(obj.candidates[a] for obj, a in zip(problem.objects, result.assignment))
    expected_alignment = tuple(
        axiom
        for obj, candidate in zip(problem.objects, expected)
        if obj.kind == "mapping"
        for axiom in candidate.axioms
    )
    expected_patch = tuple(
        (obj.occurrence_id, obj.original_axioms, candidate.axioms)
        for obj, candidate in zip(problem.objects, expected)
        if obj.kind == "ontology_axiom" and set(obj.original_axioms) != set(candidate.axioms)
    )
    if (
        expected != result.selected
        or result.alignment != expected_alignment
        or result.ontology_patch != expected_patch
    ):
        return False
    outcome = bounded_call(verify_assignment, problem, result.assignment, timeout=timeout)
    return (
        outcome.status == "complete"
        and _valid_report(problem, result.assignment, outcome.value)
        and outcome.value.authorizes
    )


def replay_optimality(
    problem: RepairInputV2, objective: ObjectiveV2, result: RepairResultV2
) -> bool:
    """Independently redo bounded exact search, without trusting exported cuts/gaps."""
    if (
        result.objective_hash != objective.content_hash
        or result.assignment is None
        or result.search_status != "OPTIMAL_IN_POOL"
        or not replay_safety(problem, result)
    ):
        return False
    replay = repair(problem, objective, diagnose=False, preserve_verified_input=False)
    return (
        replay.search_status == "OPTIMAL_IN_POOL"
        and replay.lower_bound == result.lower_bound
        and result.lower_bound == objective.score(result.assignment) == result.upper_bound
    )
