"""Deterministic retrieval diagnostics and E05 development selection.

The functions in this module are deliberately independent of the experiment
runner.  They consume already-produced, reference-aware diagnostic summaries;
candidate generation itself remains gold-free.  Keeping the selection rule
here makes the scientific decision executable without coupling production
retrieval to paper orchestration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_SHA256_HEX_LENGTH = 64


def retrieval_diagnostic_record(
    *,
    arm_id: str,
    task_id: str,
    analysis: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    runtime_seconds: float,
    split_role: str = "development",
    seed: int | None = None,
) -> dict[str, Any]:
    """Build one canonical selection row from recall and pool summaries.

    ``analysis`` follows :func:`exact.analysis.candidate_recall.analyze_candidate_recall`.
    The pool manifest remains gold-free; this helper joins its size/fingerprint
    metadata to the separately computed development-reference diagnostics.
    """

    metrics = _mapping(analysis.get("metrics"), "analysis.metrics")
    ranks = _mapping(analysis.get("gold_rank"), "analysis.gold_rank")
    pool = _mapping(pool_manifest.get("gold_free_summary"), "pool_manifest.gold_free_summary")
    fingerprint = _sha256(pool_manifest.get("fingerprint"), "pool_manifest.fingerprint")

    recall = _finite_float(
        metrics.get("generated_candidate_recall"),
        "analysis.metrics.generated_candidate_recall",
    )
    if not 0.0 <= recall <= 1.0:
        raise ValueError("generated candidate recall must be between zero and one")
    mean_pool_size = _finite_float(
        pool.get("mean_pool_size"),
        "pool_manifest.gold_free_summary.mean_pool_size",
    )
    if mean_pool_size < 0.0:
        raise ValueError("mean pool size cannot be negative")
    runtime = _finite_float(runtime_seconds, "runtime_seconds")
    if runtime < 0.0:
        raise ValueError("runtime_seconds cannot be negative")

    return {
        "arm_id": _nonempty(arm_id, "arm_id"),
        "task_id": _nonempty(task_id, "task_id"),
        "split_role": _nonempty(split_role, "split_role"),
        "seed": int(seed) if seed is not None else None,
        "candidate_recall": recall,
        "mean_pool_size": mean_pool_size,
        "gold_rank_p90": _optional_finite_float(ranks.get("rank_p90"), "gold rank p90"),
        "gold_rank_median": _optional_finite_float(ranks.get("rank_median"), "gold rank median"),
        "runtime_seconds": runtime,
        "candidate_pool_fingerprint": fingerprint,
    }


def matched_mean_pool_size(
    baseline_mean: float,
    candidate_mean: float,
    *,
    relative_tolerance: float = 0.02,
    absolute_tolerance: float = 1.0,
) -> dict[str, Any]:
    """Evaluate the pre-registered matched-mean candidate-pool constraint."""

    baseline = _finite_float(baseline_mean, "baseline_mean")
    candidate = _finite_float(candidate_mean, "candidate_mean")
    relative = _finite_float(relative_tolerance, "relative_tolerance")
    absolute = _finite_float(absolute_tolerance, "absolute_tolerance")
    if baseline < 0.0 or candidate < 0.0:
        raise ValueError("mean pool sizes cannot be negative")
    if relative < 0.0 or absolute < 0.0:
        raise ValueError("pool-size tolerances cannot be negative")
    allowed = max(absolute, relative * baseline)
    difference = abs(candidate - baseline)
    return {
        "matched": difference <= allowed + 1e-12,
        "baseline_mean_pool_size": baseline,
        "candidate_mean_pool_size": candidate,
        "absolute_difference": difference,
        "allowed_absolute_difference": allowed,
    }


def select_e05_candidate(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str,
    candidate_arms: Sequence[str],
    required_tasks: Sequence[str] | None = None,
    decision_id: str = "retrieval_configuration",
    minimum_macro_recall_gain: float = 0.005,
    maximum_task_recall_loss: float = 0.005,
    pool_relative_tolerance: float = 0.02,
    pool_absolute_tolerance: float = 1.0,
    complexity_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply the binding E05 development-selection rule.

    Candidate pools are matched to the baseline independently for every task.
    Eligible candidates must improve task-macro recall by at least 0.5 points
    and may not lose more than 0.5 points on any task.  Ranking then follows
    macro recall, gold-rank p90, gold-rank median, runtime, and declared
    simplicity.  Reporting/test records are rejected before any decision is
    computed.
    """

    baseline_id = _nonempty(baseline_arm, "baseline_arm")
    candidates = _unique_nonempty(candidate_arms, "candidate_arms")
    if baseline_id in candidates:
        raise ValueError("baseline_arm cannot also be a candidate arm")
    minimum_gain = _probability_delta(minimum_macro_recall_gain, "minimum_macro_recall_gain")
    maximum_loss = _probability_delta(maximum_task_recall_loss, "maximum_task_recall_loss")

    by_arm_task: dict[tuple[str, str], dict[str, Any]] = {}
    admitted_arms = {baseline_id, *candidates}
    for raw in records:
        record = _validated_selection_record(raw)
        if record["arm_id"] not in admitted_arms:
            continue
        split_role = record["split_role"].lower()
        if split_role in {"reporting", "test", "confirm"}:
            raise ValueError(
                "E05 development selection cannot consume reporting/test diagnostics "
                f"({record['arm_id']}/{record['task_id']})"
            )
        key = (record["arm_id"], record["task_id"])
        if key in by_arm_task:
            raise ValueError(
                "E05 selection requires exactly one deterministic diagnostic per arm/task; "
                f"duplicate {record['arm_id']}/{record['task_id']}"
            )
        by_arm_task[key] = record

    baseline_tasks = sorted(task for arm, task in by_arm_task if arm == baseline_id)
    tasks = (
        _unique_nonempty(required_tasks, "required_tasks")
        if required_tasks is not None
        else baseline_tasks
    )
    if not tasks:
        raise ValueError("E05 selection has no required development tasks")
    missing_baseline = [task for task in tasks if (baseline_id, task) not in by_arm_task]
    if missing_baseline:
        raise ValueError(
            "E05 baseline is missing required task diagnostics: " + ", ".join(missing_baseline)
        )

    declared_complexity = list(complexity_order or candidates)
    complexity = {arm: index for index, arm in enumerate(declared_complexity)}
    unknown_complexity = [arm for arm in candidates if arm not in complexity]
    if unknown_complexity:
        raise ValueError(
            "complexity_order is missing candidate arms: " + ", ".join(unknown_complexity)
        )

    baseline_rows = [by_arm_task[(baseline_id, task)] for task in tasks]
    baseline_summary = _aggregate_rows(baseline_rows)
    arm_reports: list[dict[str, Any]] = []
    eligible_ranking: list[tuple[tuple[float, float, float, float, int, str], str]] = []

    for arm in candidates:
        reasons: list[dict[str, Any]] = []
        missing = [task for task in tasks if (arm, task) not in by_arm_task]
        if missing:
            reasons.append({"code": "missing_task_diagnostics", "tasks": missing})
            rows: list[dict[str, Any]] = []
        else:
            rows = [by_arm_task[(arm, task)] for task in tasks]

        pool_checks: list[dict[str, Any]] = []
        task_deltas: dict[str, float] = {}
        if rows:
            for task, baseline_row, candidate_row in zip(tasks, baseline_rows, rows):
                check = matched_mean_pool_size(
                    baseline_row["mean_pool_size"],
                    candidate_row["mean_pool_size"],
                    relative_tolerance=pool_relative_tolerance,
                    absolute_tolerance=pool_absolute_tolerance,
                )
                check["task_id"] = task
                pool_checks.append(check)
                if not check["matched"]:
                    reasons.append({"code": "mean_pool_size_not_matched", **check})
                delta = candidate_row["candidate_recall"] - baseline_row["candidate_recall"]
                task_deltas[task] = delta
                if delta < -maximum_loss - 1e-12:
                    reasons.append(
                        {
                            "code": "task_recall_regression",
                            "task_id": task,
                            "delta": delta,
                            "maximum_loss": maximum_loss,
                        }
                    )

        summary = _aggregate_rows(rows) if rows else None
        macro_delta = (
            summary["macro_candidate_recall"] - baseline_summary["macro_candidate_recall"]
            if summary is not None
            else None
        )
        if macro_delta is not None and macro_delta < minimum_gain - 1e-12:
            reasons.append(
                {
                    "code": "insufficient_macro_recall_gain",
                    "delta": macro_delta,
                    "minimum_gain": minimum_gain,
                }
            )

        eligible = not reasons
        report = {
            "arm_id": arm,
            "eligible": eligible,
            "exclusion_reasons": reasons,
            "task_recall_deltas": task_deltas,
            "pool_checks": pool_checks,
            "aggregate": summary,
            "macro_candidate_recall_delta": macro_delta,
            "complexity_rank": complexity[arm],
        }
        arm_reports.append(report)
        if eligible and summary is not None:
            rank_key = (
                -summary["macro_candidate_recall"],
                _rank_value(summary["macro_gold_rank_p90"]),
                _rank_value(summary["macro_gold_rank_median"]),
                summary["total_runtime_seconds"],
                complexity[arm],
                arm,
            )
            eligible_ranking.append((rank_key, arm))

    eligible_ranking.sort(key=lambda item: item[0])
    selected = eligible_ranking[0][1] if eligible_ranking else None
    return {
        "schema_version": 1,
        "decision_id": _nonempty(decision_id, "decision_id"),
        "status": "selected" if selected is not None else "screened_out",
        "selected_arm": selected,
        "baseline_arm": baseline_id,
        "required_tasks": tasks,
        "selection_rule": {
            "primary_endpoint": "task_macro_candidate_recall",
            "minimum_macro_recall_gain": minimum_gain,
            "maximum_task_recall_loss": maximum_loss,
            "matched_mean_pool_size": {
                "absolute_tolerance": float(pool_absolute_tolerance),
                "relative_tolerance": float(pool_relative_tolerance),
                "rule": "max(absolute_tolerance, relative_tolerance * baseline_mean)",
            },
            "rank_order": [
                "higher_macro_candidate_recall",
                "lower_macro_gold_rank_p90",
                "lower_macro_gold_rank_median",
                "lower_total_runtime",
                "simpler_declared_configuration",
                "arm_id",
            ],
        },
        "baseline": baseline_summary,
        "candidates": arm_reports,
    }


def _validated_selection_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(raw)
    recall = _finite_float(record.get("candidate_recall"), "candidate_recall")
    if not 0.0 <= recall <= 1.0:
        raise ValueError("candidate_recall must be between zero and one")
    mean_pool = _finite_float(record.get("mean_pool_size"), "mean_pool_size")
    runtime = _finite_float(record.get("runtime_seconds"), "runtime_seconds")
    if mean_pool < 0.0 or runtime < 0.0:
        raise ValueError("mean_pool_size and runtime_seconds cannot be negative")
    return {
        "arm_id": _nonempty(record.get("arm_id"), "arm_id"),
        "task_id": _nonempty(record.get("task_id"), "task_id"),
        "split_role": _nonempty(record.get("split_role", "development"), "split_role"),
        "candidate_recall": recall,
        "mean_pool_size": mean_pool,
        "gold_rank_p90": _optional_finite_float(record.get("gold_rank_p90"), "gold_rank_p90"),
        "gold_rank_median": _optional_finite_float(
            record.get("gold_rank_median"), "gold_rank_median"
        ),
        "runtime_seconds": runtime,
        "candidate_pool_fingerprint": _sha256(
            record.get("candidate_pool_fingerprint"), "candidate_pool_fingerprint"
        ),
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(rows),
        "macro_candidate_recall": _mean([float(row["candidate_recall"]) for row in rows]),
        "macro_gold_rank_p90": _optional_mean([row["gold_rank_p90"] for row in rows]),
        "macro_gold_rank_median": _optional_mean([row["gold_rank_median"] for row in rows]),
        "total_runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows),
        "candidate_pool_fingerprints": {
            str(row["task_id"]): str(row["candidate_pool_fingerprint"]) for row in rows
        },
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label).lower()
    if len(text) != _SHA256_HEX_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return text


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _optional_finite_float(value: Any, label: str) -> float | None:
    return None if value is None else _finite_float(value, label)


def _probability_delta(value: Any, label: str) -> float:
    delta = _finite_float(value, label)
    if not 0.0 <= delta <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return delta


def _unique_nonempty(values: Sequence[Any], label: str) -> list[str]:
    result = [_nonempty(value, label) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _optional_mean(values: Sequence[float | None]) -> float | None:
    return None if not values or any(value is None for value in values) else _mean(values)  # type: ignore[arg-type]


def _rank_value(value: float | None) -> float:
    return math.inf if value is None else float(value)


__all__ = [
    "matched_mean_pool_size",
    "retrieval_diagnostic_record",
    "select_e05_candidate",
]
