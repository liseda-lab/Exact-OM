"""Small finite-pool kernel: complete replacements, verified incumbents and bounds."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, cast

from .maxsat import PresenceCut, solve_master
from .records import (
    BaselineReportV3,
    DeferredAssignmentV3,
    ObjectiveV2,
    ObjectiveV3,
    ObligationV2,
    PendingAssignmentV2,
    PolicyV2,
    ProofSupportV3,
    RepairInputV2,
    RepairInputV3,
    RepairResultV2,
    RepairResultV3,
    SearchLedgerV3,
    VerificationEventV3,
    VerificationReportV2,
    VerificationReportV3,
    canonical_hash,
    freeze_public_policy,
    read_record,
)
from .workers import bounded_call, emit_event

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
            len(events),
            obligation,
            f"{support.reasoner}/{support.package_version}:{support.backend}",
            canonical_hash(dataclasses.asdict(support)),
            capability=tuple(sorted(dataclasses.asdict(support).items())),
        )
        emit_event(event)
        events.append(event)

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
    )
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
    exception_checks = _verify_exceptions(problem)
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
    return dataclasses.replace(
        report,
        obligations=exception_checks + report.obligations,
        expected_obligations=tuple(q.name for q in exception_checks) + report.expected_obligations,
    )


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


def _verify_exceptions(problem: RepairInputV2) -> tuple[ObligationV2, ...]:
    """Reprove frozen exception obligations, also during independent safety replay."""
    if not problem.policy.exceptions:
        return ()
    from .owl import OwlVerifier, snapshot_from_axioms

    evidence = problem.policy.exception_evidence
    groups: dict[str, list[Any]] = {"source": [], "target": []}
    sources = {"source": problem.source_axioms, "target": problem.target_axioms}
    for expression, (side, digest) in zip(problem.policy.exceptions, evidence):
        if side not in sources or (side, digest) != source_exception_evidence(
            sources[side], side=side
        ):
            return (
                ObligationV2(
                    "source_exception_evidence",
                    "unknown",
                    False,
                    "exception premises do not match a frozen source theory",
                ),
            )
        groups[side].append(expression)
    checks = []
    for side, expressions in groups.items():
        if not expressions:
            continue
        report = OwlVerifier().check_theory(snapshot_from_axioms(sources[side]), expressions)
        consistent = report.obligations[0].complete and report.obligations[0].verdict is True
        if not consistent:
            checks.append(
                ObligationV2(
                    f"source_exception:{side}:consistency",
                    "unknown",
                    False,
                    "source consistency is required before an incoherence exception",
                )
            )
        for result in report.obligations:
            if result.kind != "class_satisfiability":
                continue
            checks.append(
                ObligationV2(
                    f"source_exception:{side}:{result.query_id}",
                    (
                        "unknown"
                        if not result.complete
                        else "pass" if result.verdict is False else "fail"
                    ),
                    result.complete,
                    result.reason or "",
                )
            )
    return tuple(checks)


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
        if set(report.expected_obligations) != expected:
            return False
    if (
        isinstance(problem, RepairInputV3)
        and report.authorizes
        and not isinstance(report, VerificationReportV3)
    ):
        return False
    return True


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
    if problem.policy.exceptions and checks < problem.budgets.max_checks:
        checks += 1
        outcome = bounded_call(
            _verify_exceptions,
            problem,
            timeout=min(problem.budgets.total_seconds, problem.budgets.verification_seconds),
            **_memory_options(problem.budgets.memory_mb),
        )
        exception_checks = (
            outcome.value
            if outcome.status == "complete"
            else (ObligationV2("source_exception", "unknown", False, outcome.detail),)
        )
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
    return BaselineReportV3(
        baseline_identity(problem),
        tuple(reports),
        time.monotonic() - started,
        checks,
        tuple(failures),
        exception_checks=exception_checks,
    )


def _verify_with_exceptions(
    problem: RepairInputV2, assignment: tuple[int, ...], exceptions: tuple[ObligationV2, ...]
) -> VerificationReportV2:
    if any(not q.complete or q.verdict != "pass" for q in exceptions):
        return _unknown(problem, assignment, "frozen exception proof is unresolved")
    axioms, active = materialize(problem, assignment)
    report = verify_theory(axioms, active, problem.policy, canonical_hash(assignment))
    return dataclasses.replace(
        report,
        obligations=exceptions + report.obligations,
        expected_obligations=tuple(q.name for q in exceptions) + report.expected_obligations,
    )


def _atomic_record(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record.to_dict(), stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _rank_shortlist(
    scorer: Callable[[tuple[int, ...]], float], assignments: tuple[tuple[int, ...], ...]
) -> tuple[float, ...]:
    """One model load per bounded shortlist, never one per candidate worker."""
    values = tuple(float(scorer(assignment)) for assignment in assignments)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("risk scores must be finite")
    return values


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
    verification_cache: dict[str, VerificationReportV2] = {}
    validated_proofs: set[str] = set()
    proof_deadline = float("inf")
    cache_hits = 0
    failures: list[str] = []
    baseline: list[tuple[str, VerificationReportV2]] = []
    solves = checks = revision = 0
    work_empty = False
    timings = {"baseline": 0.0, "verification": 0.0, "master": 0.0, "risk": 0.0}
    exceptions: tuple[ObligationV2, ...] = ()
    path = Path(ledger_path) if ledger_path is not None else None

    def remaining(stage: float) -> float:
        return min(
            stage, max(0.0, budgets.total_seconds - prior_elapsed - (time.monotonic() - started))
        )

    def snapshot() -> SearchLedgerV3:
        return SearchLedgerV3(
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
        )

    def persist() -> None:
        nonlocal revision
        revision += 1
        if path is not None:
            _atomic_record(path, snapshot())

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
        reserved_until = before + max(0.0, float(kwargs.get("timeout", 0.0)))
        persist()  # a lost process is charged this operation's full reservation
        try:
            return bounded_call(
                function,
                *args,
                **kwargs,
                **_memory_options(budgets.memory_mb),
            )
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

        def receive(event: Any) -> bool:
            if not event_valid(event, assignment) or event.sequence != len(events) - event_offset:
                return False
            events.append(dataclasses.replace(event, receipt_index=len(events)))
            persist()  # durable before worker acknowledgement
            return True

        function: Callable[..., VerificationReportV2] = verifier
        args: tuple[Any, ...] = (problem, assignment)
        if verifier is verify_assignment:
            function, args = _verify_with_exceptions, (problem, assignment, exceptions)
        allowance = remaining(budgets.verification_seconds)
        proof_deadline = time.monotonic() + allowance
        outcome = stage_call(
            "verification",
            function,
            *args,
            timeout=allowance,
            **({"event_handler": receive} if bounded_call is _NATIVE_BOUNDED_CALL else {}),
        )
        proof_deadline = float("inf")
        if outcome.status == "complete" and _valid_report(problem, assignment, outcome.value):
            report = cast(VerificationReportV2, outcome.value)
            if isinstance(report, VerificationReportV3) and bounded_call is _NATIVE_BOUNDED_CALL:
                report = dataclasses.replace(report, events=tuple(events[event_offset:]))
            if report.verdict in {"VERIFIED_FEASIBLE", "VERIFIED_INFEASIBLE"}:
                verification_cache[theory_key] = report
            return report
        completed = (
            [event for event in events[event_offset:] if event_valid(event, assignment)]
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
        if path is None or not path.exists():
            raise ValueError("resume requires an existing search ledger")
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
        pending = {p.assignment: p for p in saved.pending}
        deferred = {p.assignment: p for p in saved.deferred}
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
            feasible[assignment] = report
            if lower is None or objective.score(assignment) > lower:
                incumbent, incumbent_report, lower = assignment, report, objective.score(assignment)
        for assignment, report in cuts:
            if (
                not _valid_report(problem, assignment, report)
                or report.verdict != "VERIFIED_INFEASIBLE"
                or not any(q.complete and q.verdict == "fail" for q in report.obligations)
            ):
                raise ValueError("invalid ledger logical exclusion")
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

    if baseline_evidence is not None:
        exceptions = baseline_evidence.exception_checks
    elif (
        verifier is verify_assignment
        and problem.policy.exceptions
        and not diagnose
        and remaining(1) > 0
        and checks < budgets.max_checks
    ):
        checks += 1
        outcome = stage_call(
            "baseline", _verify_exceptions, problem, timeout=remaining(budgets.verification_seconds)
        )
        exceptions = (
            outcome.value
            if outcome.status == "complete"
            else (ObligationV2("source_exception", "unknown", False, outcome.detail),)
        )
    elif problem.policy.exceptions:
        exceptions = (
            ObligationV2("source_exception", "unknown", False, "exception budget exhausted"),
        )

    if baseline_evidence is not None:
        if baseline_evidence.identity != baseline_identity(problem):
            raise ValueError("shared baseline does not match this input/policy")
        baseline = list(baseline_evidence.reports)
    elif diagnose and verifier is verify_assignment and not resume and remaining(1) > 0:
        remaining_problem = dataclasses.replace(
            problem,
            budgets=dataclasses.replace(
                budgets,
                total_seconds=remaining(budgets.total_seconds),
                max_checks=max(0, budgets.max_checks - checks),
            ),
        )
        captured = collect_baselines(remaining_problem)
        baseline, checks = list(captured.reports), checks + captured.checks
        exceptions = captured.exception_checks
        timings["baseline"] += captured.elapsed_seconds
        failures.extend(captured.failures)

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
                existing = dataclasses.replace(
                    existing,
                    obligations=exceptions + existing.obligations,
                    expected_obligations=tuple(q.name for q in exceptions)
                    + existing.expected_obligations,
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

    while not bypass and remaining(1) > 0 and checks < budgets.max_checks:
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
    if path is not None:
        _atomic_record(path, final_ledger)
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
