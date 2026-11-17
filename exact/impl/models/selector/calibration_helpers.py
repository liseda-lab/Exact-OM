"""Small dependency-free score calibration and threshold primitives.

These functions operate on already-frozen score frames.  They deliberately do
not read references or choose data splits; the experiment harness remains
responsible for supplying development/OOF labels and enforcing leakage rules.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _validated_samples(
    scores: Sequence[float],
    labels: Sequence[float],
    sample_weights: Optional[Sequence[float]] = None,
) -> Tuple[List[float], List[float], List[float]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("at least one calibration sample is required")
    if sample_weights is not None and len(sample_weights) != len(scores):
        raise ValueError("sample_weights must have the same length as scores")
    clean_scores: List[float] = []
    clean_labels: List[float] = []
    clean_weights: List[float] = []
    weights = sample_weights if sample_weights is not None else [1.0] * len(scores)
    for score, label, weight in zip(scores, labels, weights):
        score_value = float(score)
        label_value = float(label)
        weight_value = float(weight)
        if not math.isfinite(score_value):
            raise ValueError("calibration scores must be finite")
        if label_value not in {0.0, 1.0}:
            raise ValueError("calibration labels must be binary")
        if not math.isfinite(weight_value) or weight_value < 0.0:
            raise ValueError("sample weights must be finite and non-negative")
        if weight_value == 0.0:
            continue
        clean_scores.append(score_value)
        clean_labels.append(label_value)
        clean_weights.append(weight_value)
    if not clean_scores:
        raise ValueError("at least one positive-weight calibration sample is required")
    return clean_scores, clean_labels, clean_weights


@dataclass(frozen=True)
class PlattCalibrator:
    """Logistic calibration over one scalar score."""

    slope: float
    intercept: float

    def predict_one(self, score: float) -> float:
        return _sigmoid(self.slope * float(score) + self.intercept)

    def predict(self, scores: Iterable[float]) -> List[float]:
        return [self.predict_one(score) for score in scores]

    def to_dict(self) -> Mapping[str, object]:
        return {"mode": "platt", "slope": self.slope, "intercept": self.intercept}


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Piecewise-constant monotone calibration fitted by PAVA."""

    upper_bounds: Tuple[float, ...]
    probabilities: Tuple[float, ...]

    def predict_one(self, score: float) -> float:
        if not self.upper_bounds:
            raise ValueError("isotonic calibrator has no fitted blocks")
        index = bisect.bisect_left(self.upper_bounds, float(score))
        index = min(index, len(self.probabilities) - 1)
        return float(self.probabilities[index])

    def predict(self, scores: Iterable[float]) -> List[float]:
        return [self.predict_one(score) for score in scores]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "mode": "isotonic",
            "upper_bounds": list(self.upper_bounds),
            "probabilities": list(self.probabilities),
        }


ScoreCalibrator = Union[PlattCalibrator, IsotonicCalibrator]


def score_calibrator_from_dict(
    payload: Mapping[str, Any],
    *,
    expected_mode: Optional[str] = None,
) -> ScoreCalibrator:
    """Reconstruct and validate an immutable fitted score calibrator.

    Fitting and split enforcement belong to the experiment harness. This
    runtime helper is deliberately data-only: it refuses malformed or
    mode-mismatched artifacts instead of accepting a configured calibration
    arm that behaves like ``none``.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("score calibrator payload must be an object")
    mode = str(payload.get("mode") or "").strip().lower()
    if expected_mode is not None and mode != str(expected_mode).strip().lower():
        raise ValueError(
            f"score calibrator mode mismatch: expected {expected_mode!r}, found {mode!r}"
        )
    if mode == "platt":
        try:
            slope = float(payload["slope"])
            intercept = float(payload["intercept"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Platt calibrator requires numeric slope and intercept") from exc
        if not math.isfinite(slope) or not math.isfinite(intercept):
            raise ValueError("Platt calibrator parameters must be finite")
        return PlattCalibrator(slope=slope, intercept=intercept)
    if mode == "isotonic":
        raw_bounds = payload.get("upper_bounds")
        raw_probabilities = payload.get("probabilities")
        if not isinstance(raw_bounds, (list, tuple)) or not isinstance(
            raw_probabilities, (list, tuple)
        ):
            raise ValueError("isotonic calibrator requires upper_bounds and probabilities arrays")
        try:
            bounds = tuple(float(value) for value in raw_bounds)
            probabilities = tuple(float(value) for value in raw_probabilities)
        except (TypeError, ValueError) as exc:
            raise ValueError("isotonic calibrator arrays must be numeric") from exc
        if not bounds or len(bounds) != len(probabilities):
            raise ValueError("isotonic calibrator arrays must be equally sized and non-empty")
        if any(not math.isfinite(value) for value in (*bounds, *probabilities)):
            raise ValueError("isotonic calibrator values must be finite")
        if any(left >= right for left, right in zip(bounds, bounds[1:])):
            raise ValueError("isotonic upper_bounds must be strictly increasing")
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("isotonic probabilities must be between zero and one")
        if any(left > right for left, right in zip(probabilities, probabilities[1:])):
            raise ValueError("isotonic probabilities must be non-decreasing")
        return IsotonicCalibrator(upper_bounds=bounds, probabilities=probabilities)
    raise ValueError("score calibrator mode must be 'platt' or 'isotonic'")


def _logistic_loss(
    scores: Sequence[float],
    labels: Sequence[float],
    weights: Sequence[float],
    slope: float,
    intercept: float,
    l2: float,
) -> float:
    loss = 0.0
    total_weight = max(sum(weights), 1.0e-12)
    for score, label, weight in zip(scores, labels, weights):
        logit = slope * score + intercept
        # softplus(logit) - y*logit is stable binary cross entropy.
        softplus = max(logit, 0.0) + math.log1p(math.exp(-abs(logit)))
        loss += weight * (softplus - label * logit)
    return loss / total_weight + 0.5 * float(l2) * slope * slope


def fit_platt_calibrator(
    scores: Sequence[float],
    labels: Sequence[float],
    sample_weights: Optional[Sequence[float]] = None,
    *,
    l2: float = 1.0e-6,
    max_iterations: int = 100,
    tolerance: float = 1.0e-8,
) -> PlattCalibrator:
    """Fit deterministic one-dimensional Platt scaling with Newton steps."""

    x, y, weights = _validated_samples(scores, labels, sample_weights)
    positive_weight = sum(weight * label for label, weight in zip(y, weights))
    total_weight = sum(weights)
    negative_weight = total_weight - positive_weight
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        raise ValueError("Platt calibration requires both positive and negative labels")

    slope = 1.0
    prior = (positive_weight + 1.0) / (total_weight + 2.0)
    intercept = math.log(prior / (1.0 - prior))
    regularization = max(0.0, float(l2))
    current_loss = _logistic_loss(x, y, weights, slope, intercept, regularization)
    denom = max(total_weight, 1.0e-12)

    for _ in range(max(1, int(max_iterations))):
        grad_slope = regularization * slope
        grad_intercept = 0.0
        h_ss = regularization
        h_si = 0.0
        h_ii = 0.0
        for score, label, weight in zip(x, y, weights):
            probability = _sigmoid(slope * score + intercept)
            error = probability - label
            curvature = max(probability * (1.0 - probability), 1.0e-12)
            normalized_weight = weight / denom
            grad_slope += normalized_weight * error * score
            grad_intercept += normalized_weight * error
            h_ss += normalized_weight * curvature * score * score
            h_si += normalized_weight * curvature * score
            h_ii += normalized_weight * curvature

        determinant = h_ss * h_ii - h_si * h_si
        if determinant <= 1.0e-18:
            break
        step_slope = (h_ii * grad_slope - h_si * grad_intercept) / determinant
        step_intercept = (-h_si * grad_slope + h_ss * grad_intercept) / determinant
        if max(abs(step_slope), abs(step_intercept)) <= float(tolerance):
            break

        step_scale = 1.0
        accepted = False
        while step_scale >= 1.0e-6:
            candidate_slope = slope - step_scale * step_slope
            candidate_intercept = intercept - step_scale * step_intercept
            candidate_loss = _logistic_loss(
                x,
                y,
                weights,
                candidate_slope,
                candidate_intercept,
                regularization,
            )
            if candidate_loss <= current_loss:
                slope = candidate_slope
                intercept = candidate_intercept
                current_loss = candidate_loss
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break

    return PlattCalibrator(slope=float(slope), intercept=float(intercept))


def fit_isotonic_calibrator(
    scores: Sequence[float],
    labels: Sequence[float],
    sample_weights: Optional[Sequence[float]] = None,
) -> IsotonicCalibrator:
    """Fit a non-decreasing weighted isotonic mapping with PAVA."""

    x, y, weights = _validated_samples(scores, labels, sample_weights)
    ordered = sorted(zip(x, y, weights), key=lambda item: item[0])
    grouped: List[dict[str, float]] = []
    for score, label, weight in ordered:
        if grouped and score == grouped[-1]["upper"]:
            grouped[-1]["weight"] += weight
            grouped[-1]["positive"] += weight * label
            continue
        grouped.append(
            {
                "lower": score,
                "upper": score,
                "weight": weight,
                "positive": weight * label,
            }
        )

    blocks: List[dict[str, float]] = []
    for group in grouped:
        blocks.append(dict(group))
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["positive"] / left["weight"]
            right_mean = right["positive"] / right["weight"]
            if left_mean <= right_mean:
                break
            blocks[-2:] = [
                {
                    "lower": left["lower"],
                    "upper": right["upper"],
                    "weight": left["weight"] + right["weight"],
                    "positive": left["positive"] + right["positive"],
                }
            ]

    return IsotonicCalibrator(
        upper_bounds=tuple(float(block["upper"]) for block in blocks),
        probabilities=tuple(float(block["positive"] / block["weight"]) for block in blocks),
    )


def fit_score_calibrator(
    mode: str,
    scores: Sequence[float],
    labels: Sequence[float],
    sample_weights: Optional[Sequence[float]] = None,
) -> ScoreCalibrator:
    normalized = str(mode or "none").strip().lower()
    if normalized == "platt":
        return fit_platt_calibrator(scores, labels, sample_weights)
    if normalized == "isotonic":
        return fit_isotonic_calibrator(scores, labels, sample_weights)
    raise ValueError("score calibration mode must be 'platt' or 'isotonic'")


def otsu_threshold(scores: Sequence[float]) -> float:
    """Return the continuous Otsu split over finite scalar scores."""

    values = sorted(float(score) for score in scores)
    if not values:
        raise ValueError("Otsu threshold requires at least one score")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("threshold scores must be finite")
    if values[0] == values[-1]:
        return values[0]

    total = len(values)
    total_sum = sum(values)
    lower_sum = 0.0
    best_key: Optional[Tuple[float, float]] = None
    best_threshold = values[0]
    for index in range(total - 1):
        lower_sum += values[index]
        if values[index] == values[index + 1]:
            continue
        lower_count = index + 1
        upper_count = total - lower_count
        lower_mean = lower_sum / lower_count
        upper_mean = (total_sum - lower_sum) / upper_count
        between_variance = lower_count * upper_count * (lower_mean - upper_mean) ** 2
        threshold = (values[index] + values[index + 1]) / 2.0
        key = (between_variance, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def knee_threshold(scores: Sequence[float]) -> float:
    """Return the midpoint of the largest adjacent score-distribution gap."""

    values = sorted(set(float(score) for score in scores))
    if not values:
        raise ValueError("knee threshold requires at least one score")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("threshold scores must be finite")
    if len(values) == 1:
        return values[0]
    gaps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    # Prefer the higher threshold on exact gap ties; this is deterministic and
    # conservative for acceptance.
    best_index = max(range(len(gaps)), key=lambda index: (gaps[index], values[index + 1]))
    return float((values[best_index] + values[best_index + 1]) / 2.0)


def select_distribution_threshold(
    scores: Sequence[float],
    mode: str,
    *,
    fixed_threshold: float = 0.7,
) -> float:
    normalized = str(mode or "fixed").strip().lower()
    if normalized == "fixed":
        return float(fixed_threshold)
    if normalized == "otsu":
        return otsu_threshold(scores)
    if normalized == "knee":
        return knee_threshold(scores)
    raise ValueError("threshold mode must be fixed, otsu, or knee")


def brier_score(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Brier score requires equally sized non-empty inputs")
    return float(
        sum(
            (float(probability) - float(label)) ** 2
            for probability, label in zip(probabilities, labels)
        )
        / len(probabilities)
    )


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("ECE requires equally sized non-empty inputs")
    if int(bins) < 1:
        raise ValueError("bins must be at least one")
    bucket_probabilities: List[List[float]] = [[] for _ in range(int(bins))]
    bucket_labels: List[List[float]] = [[] for _ in range(int(bins))]
    for probability, label in zip(probabilities, labels):
        p = min(1.0, max(0.0, float(probability)))
        y = float(label)
        if y not in {0.0, 1.0}:
            raise ValueError("ECE labels must be binary")
        bucket = min(int(bins) - 1, int(p * int(bins)))
        bucket_probabilities[bucket].append(p)
        bucket_labels[bucket].append(y)
    total = len(probabilities)
    error = 0.0
    for probs, ys in zip(bucket_probabilities, bucket_labels):
        if not probs:
            continue
        error += (len(probs) / total) * abs((sum(probs) / len(probs)) - (sum(ys) / len(ys)))
    return float(error)


__all__ = [
    "IsotonicCalibrator",
    "PlattCalibrator",
    "ScoreCalibrator",
    "brier_score",
    "expected_calibration_error",
    "fit_isotonic_calibrator",
    "fit_platt_calibrator",
    "fit_score_calibrator",
    "knee_threshold",
    "otsu_threshold",
    "score_calibrator_from_dict",
    "select_distribution_threshold",
]
