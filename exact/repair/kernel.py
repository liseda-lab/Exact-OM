"""Small finite-pool kernel: complete replacements, verified incumbents and bounds."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, cast

from .maxsat import solve_master
from .records import (
    ObjectiveV2,
    ObligationV2,
    PendingAssignmentV2,
    PolicyV2,
    RepairInputV2,
    RepairResultV2,
    VerificationReportV2,
    canonical_hash,
)
from .workers import bounded_call


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


def verify_theory(
    axioms: tuple[Any, ...],
    active: tuple[Any, ...],
    policy: PolicyV2,
    assignment_hash: str,
    *,
    backend: str = "hermit",
) -> VerificationReportV2:
    """Adapt shared-snapshot verification to the stable kernel evidence record."""
    from .owl import OwlVerifier, snapshot_from_axioms

    snapshot = snapshot_from_axioms(axioms)
    report = OwlVerifier(reasoner=backend).check_theory(
        snapshot,
        policy.monitored_classes,
        required=policy.required,
        prohibited=policy.prohibited,
        activated=active,
        exceptions=policy.exceptions,
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
    return VerificationReportV2(
        assignment_hash,
        canonical_hash((axioms, active)),
        policy.content_hash,
        report.logical_status,
        report.verification_scope,
        obligations,
        backend=f"{report.support.reasoner}/{report.support.package_version}:{report.support.backend}",
        support=tuple(sorted(dataclasses.asdict(report.support).items())),
    )


def verify_assignment(problem: RepairInputV2, assignment: tuple[int, ...]) -> VerificationReportV2:
    """Fresh full-theory check before incumbent promotion."""
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
    return dataclasses.replace(report, obligations=exception_checks + report.obligations)


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
    return (
        isinstance(report, VerificationReportV2)
        and report.assignment_hash == canonical_hash(assignment)
        and report.theory_hash == canonical_hash(materialize(problem, assignment))
        and report.policy_hash == problem.policy.content_hash
    )


def repair(
    problem: RepairInputV2,
    objective: ObjectiveV2,
    *,
    verifier: Callable[[RepairInputV2, tuple[int, ...]], VerificationReportV2] = verify_assignment,
    diagnose: bool = True,
    preserve_verified_input: bool = True,
) -> RepairResultV2:
    """Search a frozen finite inventory under supervised time and call budgets.

    Unknown results receive scheduling blocks only. Rebuilding the exact master
    keeps all solver state inside killable workers while the parent owns evidence.
    No fallback assignment is assumed feasible, including the all-delete state.
    """
    if len(objective.unary) != len(problem.objects) or any(
        len(row) != len(obj.candidates) for row, obj in zip(objective.unary, problem.objects)
    ):
        raise ValueError("objective shape does not match the frozen inventory")
    budgets = problem.budgets
    started = time.monotonic()
    deadline = started + budgets.total_seconds
    timings = {"baseline": 0.0, "verification": 0.0, "master": 0.0}
    upper: int | None = objective.upper_cap
    lower: int | None = None
    incumbent: tuple[int, ...] | None = None
    incumbent_report: VerificationReportV2 | None = None
    cuts: list[tuple[tuple[int, ...], VerificationReportV2]] = []
    pending: dict[tuple[int, ...], PendingAssignmentV2] = {}
    checked_feasible: set[tuple[int, ...]] = set()
    failures: list[str] = []
    baseline: list[tuple[str, VerificationReportV2]] = []
    solves = checks = 0
    status = "UNRESOLVED"

    def remaining(stage: float) -> float:
        return min(stage, max(0.0, deadline - time.monotonic()))

    def stage_call(stage: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        before = time.monotonic()
        try:
            return bounded_call(function, *args, **kwargs)
        finally:
            timings[stage] += time.monotonic() - before

    def check(assignment: tuple[int, ...]) -> VerificationReportV2:
        nonlocal checks
        checks += 1
        outcome = stage_call(
            "verification",
            verifier,
            problem,
            assignment,
            timeout=remaining(budgets.verification_seconds),
        )
        if outcome.status == "complete" and _valid_report(problem, assignment, outcome.value):
            return cast(VerificationReportV2, outcome.value)
        detail = outcome.detail or "verifier returned a report for a different theory/policy"
        failures.append(f"verification: {outcome.status}: {detail}")
        return _unknown(problem, assignment, detail)

    def accept_report(
        assignment: tuple[int, ...], report: VerificationReportV2, attempts: int = 1
    ) -> None:
        nonlocal incumbent, incumbent_report, lower
        value = objective.score(assignment)
        if report.authorizes:
            pending.pop(assignment, None)
            checked_feasible.add(assignment)
            if lower is None or value > lower:
                incumbent, incumbent_report, lower = assignment, report, value
        elif report.verdict == "VERIFIED_INFEASIBLE" and any(
            q.complete and q.verdict == "fail" for q in report.obligations
        ):
            pending.pop(assignment, None)
            cuts.append((assignment, report))
        else:
            pending[assignment] = PendingAssignmentV2(assignment, value, report, attempts)

    # Baseline results are diagnostic. They never manufacture source exceptions.
    if diagnose and verifier is verify_assignment:
        diagnostic_policy = PolicyV2(monitored_classes=problem.policy.monitored_classes)
        for name, axioms in (
            ("source", problem.source_axioms),
            ("target", problem.target_axioms),
            ("union", tuple(set(problem.source_axioms + problem.target_axioms))),
        ):
            if checks >= budgets.max_checks or remaining(budgets.verification_seconds) <= 0:
                failures.append(f"baseline {name}: budget exhausted")
                continue
            checks += 1
            canonical_axioms = tuple(sorted(set(axioms), key=canonical_hash))
            outcome = stage_call(
                "baseline",
                verify_theory,
                canonical_axioms,
                (),
                diagnostic_policy,
                canonical_hash(name),
                timeout=remaining(budgets.verification_seconds),
            )
            if outcome.status == "complete":
                baseline.append((name, outcome.value))
            else:
                failures.append(f"baseline {name}: {outcome.status}: {outcome.detail}")
    # Exception proofs are rechecked against immutable source theories by the
    # verifier. Diagnosis never expands the caller's frozen exception set.

    original = []
    for obj in problem.objects:
        match = next(
            (
                a
                for a, c in enumerate(obj.candidates)
                if set(c.axioms) == set(obj.original_axioms) and not c.active_expressions
            ),
            None,
        )
        if match is None:
            break
        original.append(match)
    if len(original) == len(problem.objects) and checks < budgets.max_checks and remaining(1) > 0:
        assignment = tuple(original)
        report = check(assignment)
        baseline.append(("alignment", report))
        accept_report(assignment, report)
        if report.authorizes and preserve_verified_input:
            status = "OPTIMAL_IN_POOL" if lower == upper else "INCUMBENT_WITH_GAP"
            return _result(
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
                elapsed_seconds=time.monotonic() - started,
                stage_seconds=tuple(sorted(timings.items())),
            )

    while solves < budgets.max_solves and checks < budgets.max_checks and remaining(1) > 0:
        blocked = tuple(a for a, _ in cuts) + tuple(pending) + tuple(sorted(checked_feasible))
        solves += 1
        outcome = stage_call(
            "master", solve_master, objective, blocked, timeout=remaining(budgets.solver_seconds)
        )
        if outcome.status != "complete":
            failures.append(f"master: {outcome.status}: {outcome.detail}")
            break
        master = outcome.value
        values = ([lower] if lower is not None else []) + [p.value for p in pending.values()]
        if master.upper_bound is not None:
            values.append(master.upper_bound)
        upper = min(upper, max(values)) if values and upper is not None else None
        if lower is not None and upper == lower:
            status = "OPTIMAL_IN_POOL"
            break
        if master.assignment is None or (lower is not None and master.upper_bound <= lower):
            retry = next(
                (
                    p
                    for p in sorted(pending.values(), key=lambda p: -p.value)
                    if p.attempts <= budgets.retries and (lower is None or p.value > lower)
                ),
                None,
            )
            if retry is not None:
                accept_report(retry.assignment, check(retry.assignment), retry.attempts + 1)
                continue
            status = (
                "INCUMBENT_WITH_GAP"
                if lower is not None
                else "UNRESOLVED" if pending else "NO_FEASIBLE_IN_POOL"
            )
            break
        if remaining(budgets.verification_seconds) <= 0:
            break
        accept_report(master.assignment, check(master.assignment))
    if status == "UNRESOLVED" and lower is not None:
        status = "OPTIMAL_IN_POOL" if upper == lower else "INCUMBENT_WITH_GAP"
    return _result(
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
        elapsed_seconds=time.monotonic() - started,
        stage_seconds=tuple(sorted(timings.items())),
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
