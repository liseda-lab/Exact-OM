"""Evaluation-only repair metrics and grouped paired effects.

These functions never feed reference judgements or teacher labels into retrieval,
proposal, scoring, or selection. Unknown scheduled outcomes keep their denominator.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence, cast

from .candidates import expression_size
from .records import RepairInputV2, RepairResultV2


def outcome_metrics(problem: RepairInputV2, result: RepairResultV2) -> dict[str, Any]:
    """Measure observed status, edits and bounded-stage costs without imputing labels."""
    obligations = result.verification.obligations if result.verification else ()
    edits: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    depths: list[int] = []
    constructors: list[int] = []
    if result.assignment is not None:
        for obj, selected in zip(problem.objects, result.selected):
            changed = set(selected.axioms) != set(obj.original_axioms)
            if changed:
                edits.update(selected.action_tags)
                provenance.update((f'{obj.kind}:{obj.source or "unknown"}:{obj.authorship}',))
            for expression in selected.active_expressions:
                depth, count = expression_size(expression)
                depths.append(depth)
                constructors.append(count)
    return {
        "logical_status": result.logical_status,
        "search_status": result.search_status,
        "verification_scope": result.verification_scope,
        "candidate_coverage": result.candidate_coverage,
        "verified": result.logical_status == "VERIFIED_FEASIBLE",
        "witnessed_violations": sum(q.verdict == "fail" for q in obligations),
        "unknown_obligations": sum(q.verdict == "unknown" or not q.complete for q in obligations),
        "edits_by_family": dict(sorted(edits.items())),
        "edits_by_provenance": dict(sorted(provenance.items())),
        "maximum_expression_depth": max(depths, default=0),
        "activated_constructor_count": sum(constructors),
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "absolute_integer_gap": result.gap,
        "pending_assignments": len(result.pending),
        "logical_cuts": len(result.exclusions),
        "checks": result.checks,
        "failures": result.failures,
        "stage_seconds": dict(result.stage_seconds),
        "first_verified_seconds": result.first_verified_seconds,
    }


def partial_reference_metrics(
    predicted: Iterable[tuple[str, str, str, str]],
    positive: Iterable[tuple[str, str, str, str]],
    negative: Iterable[tuple[str, str, str, str]] = (),
    *,
    entity_kinds: Iterable[str] = ("class",),
) -> dict[str, Any]:
    """Evaluate only explicitly labelled correspondences; absence stays unspecified.

    Rows are (entity_kind, source, target, relation). No complex OWL bundle is
    silently projected to an equivalence, and unlabelled predictions are counted.
    """
    kinds = frozenset(entity_kinds)
    sets = [{row for row in rows if row[0] in kinds} for rows in (predicted, positive, negative)]
    predictions, positives, negatives = sets
    if positives & negatives:
        raise ValueError("reference positive and negative labels overlap")
    tp, fp, fn = (
        len(predictions & positives),
        len(predictions & negatives),
        len(positives - predictions),
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision is not None and recall is not None else None
    )
    return {
        "entity_kinds": sorted(kinds),
        "label_scope": "explicit_partial_judgements",
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "unlabelled_predictions": len(predictions - positives - negatives),
        "labelled_precision": precision,
        "positive_recall": recall,
        "labelled_f1": f1,
    }


def paired_group_effects(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    metric: str,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 13,
    confidence_level: float = 0.95,
    sign_flip_replicates: int = 10000,
) -> dict[str, Any]:
    """Average paired cases within groups, then bootstrap independent declared groups.

    Missing, failed and unknown outcomes are excluded only from the quality
    contrast; their counts and all scheduled-case verification coverage are shown.
    This estimates exploratory uncertainty, not a powered confirmatory test.
    """
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if left == right:
        raise ValueError("paired contrasts require two distinct arms")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and strictly between zero and one")
    if type(sign_flip_replicates) is not int or not 1 <= sign_flip_replicates <= 1000000:
        raise ValueError("sign_flip_replicates must be an integer between one and one million")
    selected = [r for r in rows if r["arm_id"] in {left, right}]
    cases: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected:
        if row["arm_id"] in cases[row["case_id"]]:
            raise ValueError("paired rows require unique case and arm identities")
        cases[row["case_id"]][row["arm_id"]] = row
    groups: dict[str, list[float]] = defaultdict(list)
    status: dict[str, Counter[str]] = {left: Counter(), right: Counter()}
    for row in selected:
        status[row["arm_id"]].update((str(row.get("logical_status", "UNKNOWN")),))
    paired = 0
    for pair in cases.values():
        if left not in pair or right not in pair:
            continue
        a, b = pair[left], pair[right]
        if a["group_id"] != b["group_id"]:
            raise ValueError("paired arms must share structural/pair ancestry")
        values = a.get(metric), b.get(metric)
        if any(type(v) not in (int, float) or not math.isfinite(cast(float, v)) for v in values):
            continue
        if any(r.get("logical_status") != "VERIFIED_FEASIBLE" for r in (a, b)):
            continue
        groups[str(a["group_id"])].append(
            float(cast(float, values[1])) - float(cast(float, values[0]))
        )
        paired += 1
    effects = {g: sum(v) / len(v) for g, v in sorted(groups.items())}
    samples = list(effects.values())
    mean = sum(samples) / len(samples) if samples else None
    interval = None
    if len(samples) >= 2:
        rng = random.Random(seed)
        draws = sorted(
            sum(rng.choices(samples, k=len(samples))) / len(samples)
            for _ in range(bootstrap_replicates)
        )
        tail = (1.0 - confidence_level) / 2.0
        interval = (
            draws[int((len(draws) - 1) * tail)],
            draws[int((len(draws) - 1) * (1.0 - tail))],
        )
    for arm in (left, right):
        missing = len(cases) - sum(status[arm].values())
        if missing:
            status[arm]["NOT_RECORDED"] = missing
    pvalue = None
    draws_count = 0
    method = "unavailable"
    if samples:
        observed = abs(sum(samples))
        tolerance = 1e-12 * max(1.0, sum(abs(value) for value in samples))
        if len(samples) <= 16:
            totals = (
                sum(sign * value for sign, value in zip(signs, samples))
                for signs in itertools.product((-1, 1), repeat=len(samples))
            )
            draws_count = 2 ** len(samples)
            pvalue = sum(abs(value) >= observed - tolerance for value in totals) / draws_count
            method = "exact_group_sign_flip"
        else:
            rng = random.Random(seed + 1)
            extremes = sum(
                abs(sum(value if rng.getrandbits(1) else -value for value in samples))
                >= observed - tolerance
                for _ in range(sign_flip_replicates)
            )
            draws_count = sign_flip_replicates
            pvalue = (extremes + 1) / (draws_count + 1)
            method = "monte_carlo_group_sign_flip_plus_one"
    report = {
        "left": left,
        "right": right,
        "metric": metric,
        "effect": "right_minus_left",
        "scheduled_cases": len(cases),
        "paired_verified_cases": paired,
        "groups": len(groups),
        "per_group_effect": effects,
        "mean_group_effect": mean,
        "exploratory_interval": interval,
        "confidence_level": confidence_level,
        "paired_pvalue": pvalue,
        "pvalue_method": method,
        "sign_flip_draws": draws_count,
        "pvalue_assumption": "independent groups with sign-symmetric paired null effects",
        "all_scheduled_status": {arm: dict(counts) for arm, counts in status.items()},
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }

    if confidence_level == 0.95:
        report["exploratory_95_percent_interval"] = interval
    return report


def holm_adjust(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm step-down familywise adjustment; unavailable contrasts remain explicit."""
    present = []
    for key, value in pvalues.items():
        if value is None:
            continue
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("p-values must be finite and between zero and one")
        present.append((value, key))
    result: dict[str, float | None] = {key: None for key in pvalues}
    running = 0.0
    for index, (value, key) in enumerate(sorted(present)):
        running = max(running, min(1.0, (len(present) - index) * value))
        result[key] = running
    return result
