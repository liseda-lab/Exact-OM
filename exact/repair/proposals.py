"""Matched finite-language proposal controls and evaluation-only coverage reports."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

from .candidates import deduplicate_candidates, finite_expression_menu
from .records import (
    RepairInputV2,
    ReplacementCandidateV2,
    canonical_hash,
    canonical_json,
)
from .workers import bounded_call

if TYPE_CHECKING:
    from .learning import TeacherCache


PROPOSAL_ARMS = (
    "bounded_enumeration",
    "grammar_uniform",
    "rejection",
    "grammar_product",
    "grammar_mixture",
)


def enumerate_grammar_candidates(
    encoding: Any,
    *,
    max_expressions: int = 10000,
    deadline: float | None = None,
) -> tuple[ReplacementCandidateV2, ...]:
    """Enumerate exactly the same typed templates/menus/bounds as the direct grammar.

    ``deadline`` is an absolute monotonic deadline. An exceeded expression or
    wall budget raises; callers must record an unresolved enumeration arm, never
    call a truncated prefix exhaustive. Fixed elementary states remain in the
    optimizer's independently captured input even if this comparison fails.
    """
    options = {
        "max_depth": encoding.max_depth,
        "max_constructors": encoding.max_constructors,
        "max_expressions": max_expressions,
    }
    if deadline is None:
        expressions = finite_expression_menu(encoding.classes, encoding.properties, **options)
    else:
        result = bounded_call(
            finite_expression_menu,
            encoding.classes,
            encoding.properties,
            timeout=deadline - monotonic(),
            **options,
        )
        if result.status == "timeout":
            raise TimeoutError("grammar enumeration deadline exhausted")
        if result.status != "complete":
            raise ValueError(f"grammar enumeration failed: {result.detail}")
        expressions = result.value
    pool = list(encoding.elementary_candidates)
    for expression in expressions:
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError("grammar enumeration deadline exhausted")
        for template in encoding.templates:
            if template.fixed is not None:
                continue
            assignment = encoding.assignment(template, expression)
            if encoding.accepts(assignment):
                pool.append(encoding.emit(template, expression))
    return deduplicate_candidates(pool)


class UnconditionedMixture:
    """Raw Bernoulli-mixture mass for rejection; no compiler or global normalizer.

    Accepted assignment/bundle masses are not normalized over valid proposals.
    They are useful for a rejection baseline and must never be substituted for
    the conditioned likelihood used to train the main proposal model.
    """

    def __init__(self, encoding: Any, literal_logits: Any, component_logits: Any = None):
        import torch
        import torch.nn.functional as functional

        if not isinstance(literal_logits, torch.Tensor) or not literal_logits.is_floating_point():
            raise TypeError("literal_logits must be a floating-point tensor")
        if literal_logits.ndim != 2 or literal_logits.shape[1] != encoding.variable_count:
            raise ValueError("literal logits must have shape (components, Boolean variables)")
        count = literal_logits.shape[0]
        if count < 1 or torch.isnan(literal_logits).any():
            raise ValueError("mixture requires non-NaN logits and at least one component")
        if component_logits is None:
            component_logits = literal_logits.new_zeros(count)
        if (
            not isinstance(component_logits, torch.Tensor)
            or component_logits.shape != (count,)
            or component_logits.device != literal_logits.device
            or not torch.isfinite(component_logits).all()
        ):
            raise ValueError("component logits must be a finite vector on the literal-logit device")
        self.encoding = encoding
        self.literal_logits = literal_logits
        self.log_mixture = torch.log_softmax(component_logits, dim=0)
        self.log_positive = functional.logsigmoid(literal_logits)
        self.log_negative = functional.logsigmoid(-literal_logits)

    def candidate(self, assignment: Sequence[bool]) -> ReplacementCandidateV2:
        if not self.encoding.accepts(assignment):
            raise ValueError("assignment is outside the constrained grammar")
        return cast(ReplacementCandidateV2, self.encoding.decode(assignment))

    def log_probability(self, assignment: Sequence[bool]) -> Any:
        """Return unconditioned valid-assignment log mass, with invalid mass masked."""
        import torch

        if not self.encoding.accepts(assignment):
            return self.literal_logits.new_tensor(-float("inf"))
        bits = torch.tensor(tuple(assignment), device=self.literal_logits.device, dtype=torch.bool)
        components = torch.where(bits[None, :], self.log_positive, self.log_negative).sum(dim=1)
        return torch.logsumexp(self.log_mixture + components, dim=0)

    def candidate_log_probability(self, candidate: ReplacementCandidateV2) -> Any:
        """Sum all valid inverse encodings; this is unconditioned bundle mass."""
        import torch

        terms = [
            self.log_probability(bits) for bits in self.encoding.candidate_assignments(candidate)
        ]
        if not terms:
            return self.literal_logits.new_tensor(-float("inf"))
        return torch.logsumexp(torch.stack(terms), dim=0)


def _coverage(
    requested: Mapping[str, Iterable[Any]], available: Mapping[str, Iterable[Any]]
) -> dict[str, Any]:
    rows = []
    requested_total = found_total = 0
    for object_id, values in sorted(requested.items()):
        expected, offered = set(values), set(available.get(object_id, ()))
        found = len(expected & offered)
        requested_total += len(expected)
        found_total += found
        rows.append({"object_id": object_id, "requested": len(expected), "found": found})
    return {
        "requested": requested_total,
        "found": found_total,
        "recall": found_total / requested_total if requested_total else None,
        "objects": rows,
    }


def compare_proposals(
    problem: RepairInputV2,
    model: Any,
    *,
    arms: Sequence[str] = PROPOSAL_ARMS,
    seconds_per_arm: float = 30.0,
    retrieval_seconds: float = 10.0,
    options: Mapping[str, Any] | None = None,
    useful_candidate_ids: Mapping[str, Iterable[str]] | None = None,
    required_symbols: Mapping[str, Iterable[Any]] | None = None,
    teacher_cache: TeacherCache | None = None,
    solve: bool = False,
    verification_seconds: float = 10.0,
) -> dict[str, Any]:
    """Compare one explicitly supplied model/input, retaining every scheduled arm.

    All arms share captured retrieval, model weights, draws, pool limits, seed,
    grammar bounds and verification budgets. Useful candidates/required symbols
    are evaluator-only labels and are never passed to proposal or selection.
    Neural objective values are reported as model values, not teacher utility.
    """
    from .evaluation import outcome_metrics
    from .kernel import repair
    from .pipeline import bounded_freeze_neural_round, model_digest
    from .retrieval import retrieve_vocabulary

    arms = tuple(arms)
    if len(set(arms)) != len(arms) or not set(arms) <= set(PROPOSAL_ARMS):
        raise ValueError("proposal arms must be unique known controls")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (seconds_per_arm, retrieval_seconds, verification_seconds)
    ):
        raise ValueError("proposal comparison budgets must be positive")
    options = dict(options or {})
    useful_candidate_ids = {
        key: tuple(value) for key, value in (useful_candidate_ids or {}).items()
    }
    required_symbols = {key: tuple(value) for key, value in (required_symbols or {}).items()}
    if {"proposal_arm", "seconds"} & options.keys():
        raise ValueError("comparison owns proposal arm and wall-budget selection")
    options.setdefault("seed", 13)
    options.setdefault("candidate_cap", 64)
    options.setdefault("draws_per_object", 32)
    options.setdefault("max_depth", 2)
    options.setdefault("max_constructors", 2)
    options.setdefault("compile_seconds", min(seconds_per_arm, 20.0))
    original_problem = problem
    if teacher_cache is not None:
        expected = {
            "input": problem.content_hash,
            "policy": problem.policy.content_hash,
            "inventory": canonical_hash(tuple(obj.candidates for obj in problem.objects)),
            "profile": canonical_hash(tuple(options.get("profile", ()))),
        }
        if any(dict(teacher_cache.hashes).get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Evaluation teacher cache does not match the captured input/inventory/profile"
            )
        if teacher_cache.candidate_counts != tuple(len(obj.candidates) for obj in problem.objects):
            raise ValueError("Evaluation teacher cache candidate counts do not match")
    started = monotonic()
    retrieved = bounded_call(
        retrieve_vocabulary,
        problem,
        config=options.get("retrieval_config"),
        timeout=retrieval_seconds,
    )
    retrieval_elapsed = monotonic() - started
    model_hash = model_digest(model)
    rows = []
    if retrieved.status != "complete":
        rows = [
            {
                "arm": arm,
                "status": retrieved.status,
                "stage": "retrieval",
                "detail": retrieved.detail,
            }
            for arm in arms
        ]
        vocabulary = None
        retrieval_hash = None
    else:
        retrieval = retrieved.value
        problem = replace(
            problem,
            evidence=tuple(
                sorted({**dict(problem.evidence), "retrieval": retrieval.capture()}.items())
            ),
        )
        vocabulary = _coverage(
            required_symbols or {},
            {menu.object_id: (*menu.classes, *menu.properties) for menu in retrieval.menus},
        )
        retrieval_hash = canonical_hash(retrieval)
        for arm in arms:
            arm_started = monotonic()
            outcome = bounded_freeze_neural_round(
                problem, model, seconds=seconds_per_arm, proposal_arm=arm, **options
            )
            row: dict[str, Any] = {
                "arm": arm,
                "status": outcome.status,
                "elapsed_seconds": monotonic() - arm_started,
                "detail": outcome.detail,
                "logical_status": "UNKNOWN",
                "search_status": "UNRESOLVED",
                "verification_scope": "not_run",
            }
            if outcome.status == "complete":
                frozen = outcome.value
                offered = {
                    obj.object_id: tuple(candidate.candidate_id for candidate in obj.candidates)
                    for obj in frozen.problem.objects
                }
                row.update(
                    input=frozen.problem.to_dict(),
                    objective=frozen.objective.to_dict(),
                    graph_hash=frozen.graph_hash,
                    model_hash=frozen.model_hash,
                    candidate_coverage=frozen.problem.candidate_coverage,
                    proposal_reports=tuple(dict(report) for report in frozen.proposal_reports),
                    useful_candidate_coverage=_coverage(useful_candidate_ids or {}, offered),
                    retained_candidates=sum(map(len, offered.values())),
                )
                if solve:
                    # The same separate verification/selection budget applies to
                    # every arm, including enumeration and rejection controls.
                    bounded_problem = replace(
                        frozen.problem,
                        budgets=replace(
                            frozen.problem.budgets,
                            total_seconds=verification_seconds,
                            solver_seconds=min(
                                frozen.problem.budgets.solver_seconds, verification_seconds
                            ),
                            verification_seconds=min(
                                frozen.problem.budgets.verification_seconds, verification_seconds
                            ),
                        ),
                    )
                    result = repair(
                        bounded_problem, frozen.objective, preserve_verified_input=False
                    )
                    row.update(
                        input=bounded_problem.to_dict(),
                        result=result.to_dict(),
                        metrics=outcome_metrics(bounded_problem, result),
                        logical_status=result.logical_status,
                        search_status=result.search_status,
                        verification_scope=result.verification_scope,
                        selected_model_objective=result.lower_bound,
                    )
                    if teacher_cache is not None:
                        row["teacher_evaluation"] = _teacher_evaluation(
                            original_problem, result, teacher_cache
                        )
            rows.append(row)
    report = {
        "schema": "exact-repair/proposal-comparison/v2",
        "input_hash": problem.content_hash,
        "model_hash": model_hash,
        "retrieval_hash": retrieval_hash,
        "retrieval_seconds": retrieval_elapsed,
        "vocabulary_coverage": vocabulary,
        "scheduled": len(arms),
        "recorded": len(rows),
        "options": options,
        "seconds_per_arm": seconds_per_arm,
        "verification_seconds": verification_seconds if solve else None,
        "rows": rows,
        "evidence_scope": "one frozen case/model; evaluator-only coverage; no inferred teacher utility",
    }

    return cast(dict[str, Any], json.loads(canonical_json(report)))


def _teacher_evaluation(problem: RepairInputV2, result: Any, cache: TeacherCache) -> dict[str, Any]:
    """Score selected IDs only where the common evaluator cache actually has labels."""
    choices = []
    if result.assignment is not None:
        for obj, candidate in zip(problem.objects, result.selected):
            by_id = {item.candidate_id: index for index, item in enumerate(obj.candidates)}
            if candidate.candidate_id not in by_id:
                break
            choices.append(by_id[candidate.candidate_id])
    label = (
        next((row for row in cache.labels if row.assignment == tuple(choices)), None)
        if result.assignment is not None and len(choices) == len(problem.objects)
        else None
    )
    utility = (
        float(cast(float, label.benefit)) - label.cost
        if label is not None and label.usable
        else None
    )
    feasible = [float(cast(float, row.benefit)) - row.cost for row in cache.labels if row.usable]
    optimum = max(feasible) if cache.complete and feasible else None
    return {
        "teacher_cache_hash": canonical_hash(cache),
        "complete": cache.complete,
        "scope": "common finite evaluator inventory",
        "labelled_selected_utility": utility,
        "exact_teacher_optimum": optimum,
        "exact_regret": optimum - utility if optimum is not None and utility is not None else None,
        "status": "labelled" if utility is not None else "outside_or_unknown_in_evaluator_cache",
    }
