"""Opt-in paired comparisons over captured, frozen repair cases.

This harness measures implementation outcomes. It neither runs a matcher nor
manufactures Conference/Bio-ML benchmark inputs, labels, or trained models.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import resource
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Sequence

from .api import write_artifact
from .checkpointing import CumulativeBudget
from .evaluation import outcome_metrics
from .kernel import (
    _compose_exceptions,
    _valid_report,
    _validated_baseline_exceptions,
    _verify_with_exceptions,
    baseline_identity,
    collect_baselines,
    materialize,
    repair,
    verify_assignment,
)
from .records import (
    SCHEMA_V3,
    BaselineReportV3,
    ObjectiveV2,
    ObjectiveV3,
    ObligationV2,
    PendingAssignmentV2,
    Record,
    RepairInputV2,
    RepairResultV2,
    RepairResultV3,
    VerificationReportV2,
    VerificationReportV3,
    canonical_hash,
    canonical_json,
    read_record,
)
from .workers import bounded_call

STUDY_SCHEMA = "exact-repair/study/v3"
COHORTS = frozenset(
    {"generated", "real_structure", "conference_2025", "bioml_2024_selected", "bioml_2026_whole"}
)
LANGUAGES = frozenset({"no_repair", "deletion", "directional", "rich_mapping", "ontology_edits"})
Verifier = Callable[[RepairInputV2, tuple[int, ...]], VerificationReportV2]


@dataclasses.dataclass(frozen=True)
class StudyCaseV2(Record):
    """One captured case, with evaluation-only metadata outside deployment input."""

    case_id: str
    cohort: str
    source_version: str
    group_id: str
    problem: RepairInputV2 | None = None
    objective: ObjectiveV2 | None = None
    capture_hash: str = ""
    artifact_hashes: tuple[tuple[str, str], ...] = ()
    objective_variants: tuple[tuple[str, ObjectiveV2], ...] = ()
    mapping_scores: tuple[tuple[str, float], ...] = ()
    external_objective: ObjectiveV2 | None = None
    complete_teacher_optimum: int | None = None
    teacher_cache_hash: str = ""
    availability: str = "available"
    detail: str = ""
    declared_split: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not self.case_id
            or not self.source_version
            or not self.group_id
            or self.cohort not in COHORTS
        ):
            raise ValueError(
                "case requires an ID, explicit cohort/version, and structural parent/pair group"
            )
        if self.declared_split not in {"", "train", "development", "test"}:
            raise ValueError("invalid declared study split")
        if self.availability not in {"available", "unavailable", "invalid"}:
            raise ValueError("invalid case availability")
        if self.availability == "available":
            if self.problem is None or self.objective is None or not self.capture_hash:
                raise ValueError(
                    "available cases require frozen problem, objective and capture identity"
                )
            for objective in [
                self.objective,
                self.external_objective,
                *dict(self.objective_variants).values(),
            ]:
                if objective is not None and (
                    len(objective.unary) != len(self.problem.objects)
                    or any(
                        len(row) != len(obj.candidates)
                        for row, obj in zip(objective.unary, self.problem.objects)
                    )
                ):
                    raise ValueError("study objective does not match the common frozen inventory")
        if len(dict(self.objective_variants)) != len(self.objective_variants):
            raise ValueError("objective variant names must be unique")
        if len(dict(self.mapping_scores)) != len(self.mapping_scores) or any(
            not math.isfinite(s) for _, s in self.mapping_scores
        ):
            raise ValueError("captured mapping scores must be unique and finite")
        if self.complete_teacher_optimum is not None and (
            type(self.complete_teacher_optimum) is not int
            or not self.teacher_cache_hash
            or self.external_objective is None
        ):
            raise ValueError(
                "exact regret requires a complete teacher cache and its external objective"
            )


@dataclasses.dataclass(frozen=True)
class StudyArmV2(Record):
    """An action/selection contrast or a named already-frozen model/profile control."""

    arm_id: str
    language: str = "ontology_edits"
    selector: str = "exact"
    pairwise: bool = False
    objective_key: str = "frozen"
    omitted_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not self.arm_id
            or self.language not in LANGUAGES
            or self.selector not in {"exact", "score_greedy"}
        ):
            raise ValueError("unknown study arm language or selector")
        if self.selector == "score_greedy" and self.language != "deletion":
            raise ValueError("score-greedy is the captured-score deletion control")
        if set(self.omitted_actions) & {"keep", "delete", "retain_subsumption"}:
            raise ValueError("family ablations must preserve elementary controls")


DEFAULT_ARMS = (
    StudyArmV2("no_repair", "no_repair"),
    StudyArmV2("score_greedy_deletion", "deletion", "score_greedy"),
    StudyArmV2("fixed_cost_deletion", "deletion", objective_key="fixed_cost"),
    StudyArmV2("directional", "directional"),
    StudyArmV2("rich_mapping", "rich_mapping"),
    StudyArmV2("ontology_edits", "ontology_edits"),
    StudyArmV2("exact_unary"),
    StudyArmV2("exact_pairwise", pairwise=True),
)


@dataclasses.dataclass(frozen=True)
class StudyOutcomeV2(Record):
    """One denominator-preserving scheduled outcome, including unavailable cases."""

    case_id: str
    arm_id: str
    case_hash: str
    arm_hash: str
    status: str
    split: str
    result: RepairResultV2 | None = None
    filtered_input: RepairInputV2 | None = None
    filtered_objective: ObjectiveV2 | None = None
    common_assignment: tuple[int, ...] | None = None
    external_value: int | None = None
    exact_external_regret: int | None = None
    resources: tuple[tuple[str, float], ...] = ()
    detail: str = ""
    metrics: tuple[tuple[str, Any], ...] = ()


@dataclasses.dataclass(frozen=True)
class StudyOutcomeV3(StudyOutcomeV2):
    """New outcomes explicitly identify common diagnosis and XR-2.1 accounting."""

    schema_version = SCHEMA_V3
    baseline_hash: str = ""


def grouped_splits(cases: Sequence[StudyCaseV2], *, seed: int = 0) -> dict[str, str]:
    """Assign whole declared clean-parent/pair groups before any sibling is evaluated.

    This is parent/pair holdout. Whole-ontology holdout requires separately formed
    groups containing every case sharing that ontology and is not implied here.
    """
    if type(seed) is not int:
        raise ValueError("split seed must be an integer")
    declared: dict[str, str] = {}
    for case in cases:
        if case.declared_split:
            if case.group_id in declared and declared[case.group_id] != case.declared_split:
                raise ValueError("declared splits divide a structural/pair group")
            declared[case.group_id] = case.declared_split
    result = {}
    for case in cases:
        group = case.group_id
        bucket = int(canonical_hash((seed, group))[:16], 16) % 100
        result[case.case_id] = declared.get(
            group, "train" if bucket < 70 else "development" if bucket < 85 else "test"
        )
    return result


def filter_inventory(
    case: StudyCaseV2, arm: StudyArmV2
) -> tuple[RepairInputV2, ObjectiveV2, tuple[tuple[int, ...], ...]]:
    """Filter a single common pool without regeneration, refilling or rescoring."""
    if case.problem is None or case.objective is None:
        raise ValueError("case has no frozen inventory")
    problem = case.problem
    if arm.objective_key == "fixed_cost":
        original_objective = ObjectiveV2(
            tuple(
                tuple(
                    -case.objective.scale if obj.original_axioms and not c.axioms else 0
                    for c in obj.candidates
                )
                for obj in problem.objects
            ),
            scale=case.objective.scale,
            profile=(("delete", 1.0),),
        )
    elif arm.objective_key == "frozen":
        original_objective = case.objective
    else:
        try:
            original_objective = dict(case.objective_variants)[arm.objective_key]
        except KeyError as exc:
            raise LookupError(f"frozen objective variant unavailable: {arm.objective_key}") from exc
    allowed = {"keep"}
    if arm.language != "no_repair":
        allowed.add("delete")
    if arm.language in {"directional", "rich_mapping", "ontology_edits"}:
        allowed.add("retain_subsumption")
    objects, indices = [], []
    for obj in problem.objects:
        chosen = []
        for i, candidate in enumerate(obj.candidates):
            tags = set(candidate.action_tags)
            keep = (
                "keep" in tags
                and set(candidate.axioms) == set(obj.original_axioms)
                and not candidate.active_expressions
            )
            editable = obj.kind == "mapping" or arm.language == "ontology_edits"
            rich = arm.language == "ontology_edits" or (
                obj.kind == "mapping" and arm.language == "rich_mapping"
            )
            elementary = bool(tags & allowed)
            omitted = bool(tags & set(arm.omitted_actions)) and not bool(
                tags & {"keep", "delete", "retain_subsumption"}
            )
            if keep or (editable and (rich or elementary) and not omitted):
                chosen.append(i)
        if not chosen:
            raise ValueError(
                f"filtered object lacks an applicable original control: {obj.object_id}"
            )
        indices.append(tuple(chosen))
        objects.append(
            dataclasses.replace(obj, candidates=tuple(obj.candidates[i] for i in chosen))
        )
    remap = [{old: new for new, old in enumerate(row)} for row in indices]
    pairs = tuple(
        (i, remap[i][a], j, remap[j][b], weight)
        for i, a, j, b, weight in original_objective.pairs
        if arm.pairwise and a in remap[i] and b in remap[j]
    )

    def rows(values: tuple[tuple[Any, ...], ...]) -> tuple[tuple[Any, ...], ...]:
        return (
            tuple(tuple(row[i] for i in selected) for row, selected in zip(values, indices))
            if values
            else ()
        )

    objective = dataclasses.replace(
        original_objective,
        unary=rows(original_objective.unary),
        pairs=pairs,
        benefit=rows(original_objective.benefit),
        costs=rows(original_objective.costs),
    )
    if isinstance(objective, ObjectiveV3):
        objective = dataclasses.replace(
            objective,
            pool_hash=canonical_hash(tuple(objects)),
            raw_pairs=tuple(
                (i, remap[i][a], j, remap[j][b], w)
                for i, a, j, b, w in objective.raw_pairs
                if arm.pairwise and a in remap[i] and b in remap[j]
            ),
        )
    return dataclasses.replace(problem, objects=tuple(objects)), objective, tuple(indices)


def _captured_scores(case: StudyCaseV2) -> dict[str, float]:
    scores = dict(case.mapping_scores)
    assert case.problem is not None
    for obj in case.problem.objects:
        record = dict(case.problem.evidence).get(obj.object_id, {})
        if isinstance(record, Mapping):
            mapping = record.get("mapping", {})
            score = mapping.get("Score") if isinstance(mapping, Mapping) else None
            if score is not None and math.isfinite(float(score)):
                scores.setdefault(obj.object_id, float(score))
    return scores


def _baseline_with_exceptions(problem, current, report, baseline_evidence):
    exceptions = _validated_baseline_exceptions(problem, baseline_evidence)
    return _compose_exceptions(problem, current, report, exceptions, baseline_evidence)


def _greedy(
    problem: RepairInputV2,
    objective: ObjectiveV2,
    scores: dict[str, float],
    verifier: Verifier,
    baseline_evidence: BaselineReportV3 | None = None,
) -> RepairResultV2:
    """Delete in captured-score order, requiring full authorization at every stop."""
    started = time.monotonic()
    assignment, edits = [], []
    for i, obj in enumerate(problem.objects):
        keep = next(
            (
                a
                for a, c in enumerate(obj.candidates)
                if "keep" in c.action_tags
                and set(c.axioms) == set(obj.original_axioms)
                and not c.active_expressions
            ),
            None,
        )
        if keep is None:
            raise ValueError(f"greedy control requires the original state: {obj.object_id}")
        assignment.append(keep)
        deletion = next(
            (a for a, c in enumerate(obj.candidates) if "delete" in c.action_tags), None
        )
        if obj.kind == "mapping" and deletion is not None:
            if obj.object_id not in scores:
                raise LookupError(f"captured mapping score unavailable: {obj.object_id}")
            edits.append((scores[obj.object_id], obj.object_id, i, deletion))
    pending, exclusions, failures = [], [], []
    baselines = list(baseline_evidence.reports) if baseline_evidence is not None else []
    selected = None
    report = None
    checks = 0
    for step in [None, *sorted(edits)]:
        if step is not None:
            assignment[step[2]] = step[3]
        remaining = problem.budgets.total_seconds - (time.monotonic() - started)
        if remaining <= 0 or checks >= problem.budgets.max_checks:
            failures.append("greedy verification budget exhausted")
            break
        current = tuple(assignment)
        existing = dict(baselines).get("alignment") if step is None else None
        if existing is not None and existing.theory_hash == canonical_hash(
            materialize(problem, current)
        ):
            from .workers import CallResult

            candidate_report = dataclasses.replace(
                existing, assignment_hash=canonical_hash(current)
            )
            assert baseline_evidence is not None
            if problem.policy.exceptions and candidate_report.authorizes:
                outcome = bounded_call(
                    _baseline_with_exceptions,
                    problem,
                    current,
                    candidate_report,
                    baseline_evidence,
                    timeout=min(remaining, problem.budgets.verification_seconds),
                    **(
                        {"memory_mb": problem.budgets.memory_mb}
                        if problem.budgets.memory_mb is not None
                        else {}
                    ),
                )
            else:
                outcome = CallResult("complete", candidate_report)
        else:
            checks += 1
            function: Callable[..., VerificationReportV2] = verifier
            args: tuple[Any, ...] = (problem, current)
            if verifier is verify_assignment and baseline_evidence is not None:
                function, args = _verify_with_exceptions, (
                    problem,
                    current,
                    baseline_evidence.exception_checks,
                )
                if problem.policy.exceptions:
                    args += (baseline_evidence,)
            worker_limits: dict[str, Any] = (
                {"memory_mb": problem.budgets.memory_mb}
                if problem.budgets.memory_mb is not None
                else {}
            )
            outcome = bounded_call(
                function,
                *args,
                timeout=min(remaining, problem.budgets.verification_seconds),
                **worker_limits,
            )
        candidate_report = outcome.value
        if outcome.status != "complete" or not _valid_report(problem, current, candidate_report):
            detail = outcome.detail or "verification returned an unrelated or malformed report"
            failures.append(detail)
            candidate_report = VerificationReportV3(
                canonical_hash(current),
                canonical_hash(materialize(problem, current)),
                problem.policy.content_hash,
                "UNKNOWN",
                "partial_detection",
                (ObligationV2("verification", "unknown", False, detail),),
                detail=detail,
            )
        if step is None and not any(name == "alignment" for name, _ in baselines):
            baselines.append(("alignment", candidate_report))
        if candidate_report.authorizes:
            selected, report = current, candidate_report
            break
        if candidate_report.verdict == "VERIFIED_INFEASIBLE" and any(
            q.complete and q.verdict == "fail" for q in candidate_report.obligations
        ):
            exclusions.append((current, candidate_report))
        else:
            pending.append(PendingAssignmentV2(current, objective.score(current), candidate_report))
    candidates = (
        ()
        if selected is None
        else tuple(obj.candidates[a] for obj, a in zip(problem.objects, selected))
    )
    value = None if selected is None else objective.score(selected)
    search = (
        "UNRESOLVED"
        if selected is None
        else "OPTIMAL_IN_POOL" if value == objective.upper_cap else "INCUMBENT_WITH_GAP"
    )
    return RepairResultV3(
        input_hash=problem.content_hash,
        objective_hash=objective.content_hash,
        logical_status="UNKNOWN" if selected is None else "VERIFIED_FEASIBLE",
        search_status=search,
        verification_scope=report.scope if report else "partial_detection",
        candidate_coverage=problem.candidate_coverage,
        assignment=selected,
        selected=candidates,
        alignment=(
            None
            if selected is None
            else tuple(
                ax
                for obj, c in zip(problem.objects, candidates)
                if obj.kind == "mapping"
                for ax in c.axioms
            )
        ),
        ontology_patch=(),
        lower_bound=value,
        upper_bound=objective.upper_cap,
        verification=report,
        pending=tuple(pending),
        exclusions=tuple(exclusions),
        failures=tuple(failures),
        checks=checks,
        baseline=tuple(baselines),
        model_status=problem.model_status,
        elapsed_seconds=time.monotonic() - started,
        generation_status={
            "bounded_enumerated": "COMPLETE_DECLARED_ENUMERATION",
            "sampled": "SAMPLED",
        }.get(problem.candidate_coverage, "PARTIAL_RESOURCE_LIMIT"),
    )


def _usage() -> tuple[float, float]:
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(
        resource.RUSAGE_CHILDREN
    )
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime, float(own.ru_maxrss)


def evaluate_case(
    case: StudyCaseV2,
    arm: StudyArmV2,
    *,
    split: str,
    verifier: Verifier = verify_assignment,
    seconds: float | None = None,
    baseline_evidence: BaselineReportV3 | None = None,
) -> StudyOutcomeV3:
    """Produce a status for every scheduled arm, preserving unknowns and failures."""
    started = time.monotonic()
    cpu_before, _ = _usage()
    result = problem = objective = None
    assignment = None
    external_value = regret = None
    status, detail = "completed", ""
    preparation_seconds = selection_seconds = 0.0
    try:
        if case.availability != "available":
            status, detail = case.availability, case.detail
        else:
            stage_started = time.monotonic()
            if baseline_evidence is None and verifier is verify_assignment:
                assert case.problem is not None
                baseline_evidence = collect_baselines(case.problem)
            problem, objective, indices = filter_inventory(case, arm)
            if baseline_evidence is not None and baseline_evidence.identity != baseline_identity(
                problem
            ):
                raise ValueError("shared baseline input/policy mismatch")
            preparation_seconds = time.monotonic() - stage_started
            remaining = (
                problem.budgets.total_seconds
                if seconds is None
                else min(seconds, problem.budgets.total_seconds)
            ) - preparation_seconds
            if remaining <= 0:
                raise TimeoutError("case budget exhausted during preparation")
            problem = dataclasses.replace(
                problem, budgets=dataclasses.replace(problem.budgets, total_seconds=remaining)
            )
            stage_started = time.monotonic()
            result = (
                _greedy(problem, objective, _captured_scores(case), verifier, baseline_evidence)
                if arm.selector == "score_greedy"
                else repair(
                    problem,
                    objective,
                    verifier=verifier,
                    preserve_verified_input=False,
                    diagnose=False,
                    baseline_evidence=baseline_evidence,
                )
            )
            selection_seconds = time.monotonic() - stage_started
            if result.assignment is not None:
                assignment = tuple(row[a] for row, a in zip(indices, result.assignment))
                if case.external_objective is not None:
                    external_value = case.external_objective.score(assignment)
                    if case.complete_teacher_optimum is not None:
                        regret = case.complete_teacher_optimum - external_value
                        if regret < 0:
                            raise ValueError(
                                "selected external value exceeds the claimed complete teacher optimum"
                            )
    except LookupError as exc:
        status, detail = "unavailable", str(exc)
    except Exception as exc:
        status, detail = "failed", f"{type(exc).__name__}: {exc}"
    cpu_after, peak_rss = _usage()
    return StudyOutcomeV3(
        case.case_id,
        arm.arm_id,
        case.content_hash,
        arm.content_hash,
        status,
        split,
        result,
        problem,
        objective,
        assignment,
        external_value,
        regret,
        (
            ("elapsed_seconds", time.monotonic() - started),
            ("cpu_seconds", cpu_after - cpu_before),
            ("preparation_seconds", preparation_seconds),
            (
                "shared_baseline_seconds",
                baseline_evidence.elapsed_seconds if baseline_evidence else 0.0,
            ),
            (
                "shared_baseline_checks",
                float(baseline_evidence.checks) if baseline_evidence else 0.0,
            ),
            ("selection_seconds", selection_seconds),
            ("parent_process_high_water_rss_kib", peak_rss),
        ),
        detail,
        (
            tuple(sorted(outcome_metrics(problem, result).items()))
            if problem is not None and result is not None
            else ()
        ),
        baseline_hash=baseline_evidence.content_hash if baseline_evidence else "",
    )


def compiler_accounting(
    problem: RepairInputV2 | None, *, policy: str = "same_frozen_cache_state_per_arm"
) -> dict[str, Any]:
    """Export exact compiler receipts and charge their observed preparation once.

    These pools were prepared before matched selection. Their costs are a shared
    preparation ledger, not repeated execution inside each arm's selection budget.
    Cold-required/warm-required comparisons reject incompatible measured states;
    historical rows without receipts remain explicitly unavailable.
    """
    if policy not in {"same_frozen_cache_state_per_arm", "cold_required", "warm_required"}:
        raise ValueError("unknown compiler comparison cache policy")
    receipts, unavailable = [], []
    if problem is None:
        unavailable.append(dict(object_id=None, family=None, status="input_unavailable"))
    for report in problem.proposal_provenance if problem is not None else ():
        if not hasattr(report, "get"):
            continue
        telemetry = report.get("compiler_telemetry", ())
        if not telemetry and report.get("compilation_seconds", 0) > 0:
            unavailable.append(
                dict(
                    object_id=report.get("object_id"),
                    family="unknown",
                    status="historical_measurements_unavailable",
                )
            )
        families = (
            telemetry
            if telemetry and len(telemetry[0]) == 4
            else (("monolithic", "resolved", "", telemetry),)
        )
        for family, status, detail, rows in families:
            receipt = dict(rows).get("measurement_receipt") if rows else None
            if receipt is not None:
                receipt = json.loads(canonical_json(receipt))
                receipts.append(
                    dict(
                        object_id=report.get("object_id"),
                        family=family,
                        receipt_hash=canonical_hash(receipt),
                        receipt=receipt,
                    )
                )
            elif rows or status not in {"resolved", "empty_language"}:
                unavailable.append(
                    dict(
                        object_id=report.get("object_id"),
                        family=family,
                        status=status,
                        detail=detail,
                    )
                )
    # A frozen preparation can be referenced by several arms without duplicating cost.
    unique = {row["receipt_hash"]: row for row in receipts}
    expected = {"cold_compile"} if policy == "cold_required" else {"warm_load", "memory_reuse"}
    if (
        problem is not None
        and policy != "same_frozen_cache_state_per_arm"
        and (
            unavailable
            or not unique
            or any(row["receipt"]["cache_mode"] not in expected for row in unique.values())
        )
    ):
        raise ValueError(
            "compiler receipts do not establish the declared cold/warm comparison policy"
        )
    wall, cpu = [], []
    for row in unique.values():
        receipt = row["receipt"]
        supervised = receipt.get("supervised_call")
        if supervised is not None:
            wall.append(supervised["wall_seconds"])
            cpu.append(dict(supervised["resources"]).get("cpu_seconds"))
        else:
            total = receipt.get("total") or {}
            wall.append(total.get("wall_seconds"))
            cpu.append(total.get("cpu_seconds"))
    complete = not unavailable and all(value is not None for value in wall)
    populations: dict[str, dict[str, Any]] = {}
    for row in unique.values():
        receipt = row["receipt"]
        cold = receipt.get("cold_receipt")
        if cold is not None:
            identity = canonical_hash(cold)
            population = populations.setdefault(
                identity, dict(cold_receipt=cold, already_charged=False)
            )
            population["already_charged"] |= receipt["cache_mode"] == "cold_compile"
    # An originating cold body receipt is deliberately not called a measured
    # end-to-end population duration: startup/publication overhead is unavailable
    # unless its original per-call record was also supplied to the comparison.
    population_accounting = tuple(
        dict(
            identity=identity,
            **value,
            measured_phase_wall_seconds=sum(
                phase["wall_seconds"] for phase in value["cold_receipt"]["phases"]
            ),
            measured_phase_cpu_seconds=sum(
                phase["cpu_seconds"] for phase in value["cold_receipt"]["phases"]
            ),
            additional_total_seconds=None if not value["already_charged"] else 0.0,
            policy="originating population disclosed separately; never substitute warm time for cold work",
        )
        for identity, value in sorted(populations.items())
    )
    return dict(
        schema="exact-repair/compiler-accounting/review-1",
        policy=policy,
        charge_scope="shared frozen preparation, once per receipt; separate from selection execution",
        shared_charge_key=canonical_hash(tuple(sorted(unique))),
        cache_population=population_accounting,
        receipts=tuple(unique[key] for key in sorted(unique)),
        unavailable=tuple(unavailable),
        charged_wall_seconds=sum(wall) if complete else None,
        charged_cpu_seconds=(
            sum(cpu) if complete and all(value is not None for value in cpu) else None
        ),
    )


def runtime_manifest() -> dict[str, Any]:
    """Identify code, dependencies and resource measurement scope without importing ML."""
    versions = {}
    for package in (
        "pyowl-core",
        "python-sat",
        "pysdd",
        "torch",
        "torch-geometric",
        "pyhermit",
        "pyelk-reasoner",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    implementations = {}
    for module in ("pyowl_core", "pyowl2vec_star_projector", "pyhermit", "pyelk"):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            implementations[module] = {"status": "unavailable"}
            continue
        root = Path(spec.origin).parent
        digest = hashlib.sha256()
        for path in sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".so"}
        ):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        implementations[module] = {
            "path": str(root),
            "source_and_native_sha256": digest.hexdigest(),
        }
    code = {
        path.name: canonical_hash(path.read_bytes().hex())
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "dependencies": versions,
        "ontology_implementations": implementations,
        "code_hashes": code,
        "cache_policy": "fresh verifier/master workers; frozen artifacts reused",
        "memory_scope": "parent process lifetime high-water RSS; not isolated per-arm peak",
    }


def run_study(
    cases: Sequence[StudyCaseV2],
    directory: str | Path,
    *,
    arms: Sequence[StudyArmV2] = DEFAULT_ARMS,
    seed: int = 0,
    verifier: Verifier = verify_assignment,
    max_attempts: int = 2,
    campaign_seconds: float = 3600.0,
    compiler_cache_policy: str = "same_frozen_cache_state_per_arm",
) -> dict[str, Any]:
    """Persist a fixed schedule, resume completed outcomes, and cap interrupted retries."""
    if len({case.case_id for case in cases}) != len(cases) or len(
        {arm.arm_id for arm in arms}
    ) != len(arms):
        raise ValueError("scheduled case and arm IDs must be unique")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not math.isfinite(campaign_seconds) or campaign_seconds <= 0:
        raise ValueError("campaign_seconds must be positive and finite")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    splits = grouped_splits(cases, seed=seed)
    compiler_costs = {
        case.case_id: compiler_accounting(case.problem, policy=compiler_cache_policy)
        for case in cases
    }
    plan = {
        "schema": STUDY_SCHEMA,
        "compiler_cache_policy": compiler_cache_policy,
        "shared_compiler_preparation": compiler_costs,
        "seed": seed,
        "max_attempts": max_attempts,
        "campaign_seconds": campaign_seconds,
        "cases": [case.to_dict() for case in cases],
        "arms": [arm.to_dict() for arm in arms],
        "splits": splits,
        "split_scope": "declared clean-parent/pair group holdout",
        "runtime": runtime_manifest(),
        "verifier": f"{verifier.__module__}.{verifier.__qualname__}",
    }
    plan_hash = canonical_hash(plan)
    plan_path = root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing.get("hash") != plan_hash or canonical_hash(existing.get("plan")) != plan_hash:
            raise ValueError(
                "resume plan differs from frozen cases, arms, code, runtime or split manifest"
            )
    else:
        write_artifact(plan_path, {"hash": plan_hash, "plan": plan})
    with CumulativeBudget(root / "budget.json", plan_hash, campaign_seconds) as budget:
        rows = []
        for case in cases:
            baseline_evidence: BaselineReportV3 | None = None
            if case.problem is not None and verifier is verify_assignment:
                baseline_path = root / "baselines" / f"{baseline_identity(case.problem)}.json"
                if baseline_path.exists():
                    baseline_record = read_record(json.loads(baseline_path.read_text()))
                    if not isinstance(
                        baseline_record, BaselineReportV3
                    ) or baseline_record.identity != baseline_identity(case.problem):
                        raise ValueError("invalid shared baseline artifact")
                    baseline_evidence = baseline_record
                elif budget.remaining > 0:
                    reserved = budget.begin(
                        min(budget.remaining, case.problem.budgets.total_seconds)
                    )
                    try:
                        bounded_problem = dataclasses.replace(
                            case.problem,
                            budgets=dataclasses.replace(
                                case.problem.budgets, total_seconds=reserved
                            ),
                        )
                        baseline_evidence = collect_baselines(bounded_problem)
                        write_artifact(baseline_path, baseline_evidence.to_dict())
                    finally:
                        budget.finish()
            for arm in arms:
                job = canonical_hash((case.case_id, arm.arm_id))
                destination, state_path = (
                    root / "outcomes" / f"{job}.json",
                    root / "state" / f"{job}.json",
                )
                if destination.exists():
                    value = read_record(json.loads(destination.read_text()))
                    if (
                        not isinstance(value, StudyOutcomeV3)
                        or value.case_hash != case.content_hash
                        or value.arm_hash != arm.content_hash
                    ):
                        raise ValueError("resumed outcome does not belong to the frozen schedule")
                    outcome = value
                else:
                    state = (
                        json.loads(state_path.read_text())
                        if state_path.exists()
                        else {"attempts": 0}
                    )
                    if state_path.exists() and state.get("plan_hash") != plan_hash:
                        raise ValueError("interrupted state belongs to a different frozen plan")
                    attempts = state["attempts"]
                    if type(attempts) is not int or attempts < 0:
                        raise ValueError("invalid interrupted-attempt counter")
                    if attempts >= max_attempts:
                        outcome = StudyOutcomeV3(
                            case.case_id,
                            arm.arm_id,
                            case.content_hash,
                            arm.content_hash,
                            "failed",
                            splits[case.case_id],
                            detail="interrupted attempt limit exhausted",
                        )
                    else:
                        write_artifact(
                            state_path,
                            {"plan_hash": plan_hash, "attempts": attempts + 1, "status": "running"},
                        )
                        remaining = budget.remaining
                        if remaining <= 0:
                            outcome = StudyOutcomeV3(
                                case.case_id,
                                arm.arm_id,
                                case.content_hash,
                                arm.content_hash,
                                "timeout",
                                splits[case.case_id],
                                detail="campaign deadline exhausted",
                            )
                        else:
                            reserved = budget.begin(
                                min(remaining, case.problem.budgets.total_seconds)
                                if case.problem is not None
                                else remaining
                            )
                            try:
                                outcome = evaluate_case(
                                    case,
                                    arm,
                                    split=splits[case.case_id],
                                    verifier=verifier,
                                    seconds=reserved,
                                    baseline_evidence=baseline_evidence,
                                )
                            finally:
                                budget.finish()
                    outcome = dataclasses.replace(
                        outcome,
                        resources=outcome.resources
                        + (
                            ("attempt", float(min(attempts + 1, max_attempts))),
                            ("interrupted_attempts", float(attempts)),
                        ),
                    )
                    write_artifact(destination, outcome.to_dict())
                    write_artifact(
                        state_path,
                        {
                            "plan_hash": plan_hash,
                            "attempts": min(attempts + 1, max_attempts),
                            "status": outcome.status,
                        },
                    )
                rows.append(
                    {
                        "case_id": case.case_id,
                        "arm_id": arm.arm_id,
                        "cohort": case.cohort,
                        "source_version": case.source_version,
                        "group_id": case.group_id,
                        "status": outcome.status,
                        "logical_status": (
                            outcome.result.logical_status if outcome.result else "UNKNOWN"
                        ),
                        "search_status": (
                            outcome.result.search_status if outcome.result else "UNRESOLVED"
                        ),
                        "verification_scope": (
                            outcome.result.verification_scope if outcome.result else "unavailable"
                        ),
                        "artifact": str(destination.relative_to(root)),
                        "artifact_hash": outcome.content_hash,
                        "external_value": outcome.external_value,
                        "exact_external_regret": outcome.exact_external_regret,
                        "metrics": json.loads(canonical_json(dict(outcome.metrics))),
                        "resources": dict(outcome.resources),
                        "compiler_preparation": compiler_costs[case.case_id],
                    }
                )
                write_artifact(
                    root / "results.json",
                    {
                        "schema": STUDY_SCHEMA,
                        "plan_hash": plan_hash,
                        "scheduled": len(cases) * len(arms),
                        "recorded": len(rows),
                        "rows": rows,
                        "evidence_scope": "frozen-case measurements; not an unperformed benchmark",
                    },
                )
        return {
            "plan_hash": plan_hash,
            "scheduled": len(cases) * len(arms),
            "recorded": len(rows),
            "rows": rows,
        }


def load_schedule(path: str | Path) -> tuple[tuple[StudyCaseV2, ...], tuple[StudyArmV2, ...], int]:
    """Load scheduled case artifacts once; retain missing/corrupt entries as outcomes."""
    schedule_path = Path(path)
    payload = json.loads(schedule_path.read_text())
    if payload.get("schema") != "exact-repair/study-schedule/v2":
        raise ValueError("unsupported study schedule schema")
    cases = []
    for entry in payload["cases"]:
        metadata = {key: entry[key] for key in ("case_id", "cohort", "source_version", "group_id")}
        if "declared_split" in entry:
            metadata["declared_split"] = entry["declared_split"]
        try:
            record = read_record(json.loads((schedule_path.parent / entry["artifact"]).read_text()))
            if not isinstance(record, StudyCaseV2) or any(
                getattr(record, key) != value for key, value in metadata.items()
            ):
                raise ValueError("case artifact differs from its scheduled identity")
            cases.append(record)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            cases.append(
                StudyCaseV2(
                    **metadata,
                    availability="unavailable" if isinstance(exc, OSError) else "invalid",
                    detail=str(exc),
                )
            )
    arms = (
        tuple(StudyArmV2(**entry) for entry in payload["arms"])
        if "arms" in payload
        else DEFAULT_ARMS
    )
    return tuple(cases), arms, payload.get("seed", 0)
