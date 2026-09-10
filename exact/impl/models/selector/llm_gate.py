"""Deterministic, data-only gate helpers for E25."""

from __future__ import annotations

import hashlib
import json
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


def select_inference_gate_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    fraction: float,
    dataset_signature: str,
    source_universe: Sequence[str] = (),
) -> Dict[str, Any]:
    """Freeze exact-budget routing over the current unlabeled inference pool.

    Empty sources stay in the audit universe but cannot consume a judgment.
    Pair rows use U; source routing uses the top-two pre-LLM score margin.
    """
    if mode not in {"source_top_fraction", "pair_top_fraction"}:
        raise ValueError("inference gate mode must be source_top_fraction or pair_top_fraction")
    p = float(fraction)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("inference gate fraction must be in [0, 1]")
    if not dataset_signature:
        raise ValueError("inference routing requires a dataset signature")
    grouped: Dict[str, List[Tuple[float, str]]] = {}
    population: List[List[Any]] = []
    for row in rows:
        source, target = str(row["source_iri"]), str(row["target_iri"])
        value = float(row["score"] if mode == "source_top_fraction" else row["U"])
        if not source or not target or not math.isfinite(value):
            raise ValueError("gate rows require nonempty identities and finite scores")
        grouped.setdefault(source, []).append((value, target))
        population.append([source, target, value])
    if len({(r[0], r[1]) for r in population}) != len(population):
        raise ValueError("inference gate population contains duplicate pairs")
    population.sort()
    if mode == "source_top_fraction":
        ranked = []
        for source, candidates in grouped.items():
            scores = sorted((value for value, _ in candidates), reverse=True)
            margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
            ranked.append((1.0 - min(1.0, max(0.0, margin)), source, ""))
        statistic = "one_minus_top_two_margin_singleton_one_minus_top_score"
    else:
        ranked = [(value, source, target) for source, target, value in population]
        statistic = "U"
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    count = math.ceil(p * len(ranked))
    selected = ranked[:count]
    selected_sources = sorted({source for _, source, _ in selected})
    pairs = (
        [[source, target] for source, target, _ in population if source in selected_sources]
        if mode == "source_top_fraction"
        else sorted([[source, target] for _, source, target in selected])
    )
    universe = sorted(set(str(source) for source in source_universe) | set(grouped))
    return {
        "schema_version": 1,
        "kind": "llm_gate",
        "mode": mode,
        "dataset_signature": str(dataset_signature),
        "fraction": p,
        "population_fingerprint": hashlib.sha256(
            json.dumps(
                {"rows": population, "sources": universe}, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "population_pairs": [[source, target] for source, target, _ in population],
        "population_sources": universe,
        "row_count": len(population),
        "eligible_count": len(ranked),
        "selected_count": count,
        "selected_sources": selected_sources,
        "pairs": pairs,
        "no_candidate_sources": sorted(set(universe) - set(grouped)),
        "no_candidate_fallback": "base_abstain",
        "statistic": statistic,
        "tie_rule": "(-statistic, source_iri, target_iri)",
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
    "select_inference_gate_artifact",
    "AnalyticalOracleResult",
    "analytical_oracle_ceiling",
    "fit_forced_sample_artifact",
    "fit_quantile_gate_artifact",
]
