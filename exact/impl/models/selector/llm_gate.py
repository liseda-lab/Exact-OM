"""Deterministic, data-only gate helpers for E25."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

Pair = Tuple[str, str]


def _gate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[float, str, str]]:
    clean: List[Tuple[float, str, str]] = []
    seen: set[Pair] = set()
    for row in rows:
        try:
            uncertainty = float(row["U"])
            source = str(row["source_iri"])
            target = str(row["target_iri"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("gate rows require U, source_iri, and target_iri") from exc
        if not math.isfinite(uncertainty):
            raise ValueError("gate uncertainty values must be finite")
        pair = (source, target)
        if pair in seen:
            raise ValueError(f"duplicate gate pair: {pair!r}")
        seen.add(pair)
        clean.append((uncertainty, source, target))
    if not clean:
        raise ValueError("gate fitting requires at least one development row")
    return clean


def fit_quantile_gate_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    task_id: str,
    entity_kind: str,
) -> Dict[str, Any]:
    """Select exactly ``ceil(fraction*N)`` rows under the clarified tie rule."""

    p = float(fraction)
    if not math.isfinite(p) or p <= 0.0 or p > 1.0:
        raise ValueError("quantile gate fraction must be in (0, 1]")
    ordered = sorted(_gate_rows(rows), key=lambda row: (-row[0], row[1], row[2]))
    selected_count = int(math.ceil(p * len(ordered)))
    selected = ordered[:selected_count]
    excluded = ordered[selected_count:]
    selected_pairs = [[source, target] for _, source, target in selected]
    return {
        "schema_version": 1,
        "kind": "llm_gate",
        "mode": "quantile",
        "task_id": str(task_id),
        "entity_kind": str(entity_kind),
        "fraction": p,
        "row_count": len(ordered),
        "selected_count": selected_count,
        "threshold": float(selected[-1][0]),
        "pairs": selected_pairs,
        "boundary_ids": {
            "selected_last": [selected[-1][1], selected[-1][2]],
            "excluded_first": ([excluded[0][1], excluded[0][2]] if excluded else None),
        },
        "tie_rule": "(-U, source_iri, target_iri)",
    }


def fit_forced_sample_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    task_id: str,
    entity_kind: str,
) -> Dict[str, Any]:
    """Freeze a bounded deterministic hash sample without consulting labels."""

    clean = _gate_rows(rows)
    count = int(sample_size)
    if count < 1 or count > len(clean):
        raise ValueError("forced sample_size must be between one and the row count")

    def _key(row: Tuple[float, str, str]) -> Tuple[bytes, str, str]:
        _, source, target = row
        digest = hashlib.sha256(f"{int(seed)}\x00{source}\x00{target}".encode("utf-8")).digest()
        return digest, source, target

    selected = sorted(clean, key=_key)[:count]
    pairs = sorted((source, target) for _, source, target in selected)
    return {
        "schema_version": 1,
        "kind": "llm_gate",
        "mode": "forced_sample",
        "task_id": str(task_id),
        "entity_kind": str(entity_kind),
        "seed": int(seed),
        "row_count": len(clean),
        "sample_size": count,
        "pairs": [[source, target] for source, target in pairs],
        "selection_rule": "sha256(seed, source_iri, target_iri)",
    }


@dataclass(frozen=True)
class AnalyticalOracleResult:
    corrected_scores: Tuple[float, ...]
    routed: Tuple[bool, ...]
    baseline_predictions: Tuple[bool, ...]
    labels: Tuple[bool, ...]
    llm_invocations: int = 0
    deployable: bool = False


def analytical_oracle_ceiling(
    baseline_scores: Sequence[float],
    labels: Sequence[float],
    *,
    threshold: float,
) -> AnalyticalOracleResult:
    """Return an always-correct reference ceiling without invoking an LLM.

    Only baseline errors are routed. Their score is replaced directly with the
    correct binary outcome; unchanged rows retain the baseline score. This is
    an oracle diagnostic and must never be used as a selectable product arm.
    """

    if len(baseline_scores) != len(labels) or not baseline_scores:
        raise ValueError("oracle scores and labels must be equally sized and non-empty")
    cutoff = float(threshold)
    if not math.isfinite(cutoff) or cutoff <= 0.0 or cutoff > 1.0:
        raise ValueError("oracle threshold must be in (0, 1]")
    scores = tuple(float(value) for value in baseline_scores)
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("oracle baseline scores must be finite")
    binary_labels: List[bool] = []
    for label in labels:
        value = float(label)
        if value not in {0.0, 1.0}:
            raise ValueError("oracle labels must be binary")
        binary_labels.append(bool(value))
    predictions = tuple(score >= cutoff for score in scores)
    routed = tuple(prediction != label for prediction, label in zip(predictions, binary_labels))
    corrected = tuple(
        (1.0 if label else 0.0) if should_route else score
        for score, label, should_route in zip(scores, binary_labels, routed)
    )
    return AnalyticalOracleResult(
        corrected_scores=corrected,
        routed=routed,
        baseline_predictions=predictions,
        labels=tuple(binary_labels),
    )


__all__ = [
    "AnalyticalOracleResult",
    "analytical_oracle_ceiling",
    "fit_forced_sample_artifact",
    "fit_quantile_gate_artifact",
]
