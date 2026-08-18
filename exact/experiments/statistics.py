"""Small deterministic paired-bootstrap utility used by E00."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BootstrapResult:
    n_units: int
    resamples: int
    seed: int
    delta: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _finite_values(values: Sequence[float], label: str) -> list[float]:
    converted = [float(value) for value in values]
    bad = [index for index, value in enumerate(converted) if not math.isfinite(value)]
    if bad:
        raise ValueError(f"{label} contains non-finite values at positions {bad}")
    return converted


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    baseline: Mapping[str, float] | Sequence[float],
    candidate: Mapping[str, float] | Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Bootstrap the mean paired candidate-minus-baseline delta.

    Mappings are aligned by key and must have identical key sets. Sequences are
    paired positionally. Resampling uses independent units, never candidate rows.
    """

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if isinstance(baseline, Mapping) != isinstance(candidate, Mapping):
        raise TypeError("baseline and candidate must both be mappings or both be sequences")
    if isinstance(baseline, Mapping):
        assert isinstance(candidate, Mapping)
        if set(baseline) != set(candidate):
            missing_left = sorted(set(candidate).difference(baseline))
            missing_right = sorted(set(baseline).difference(candidate))
            raise ValueError(
                "paired mappings must have identical units "
                f"(baseline_missing={missing_left}, candidate_missing={missing_right})"
            )
        keys = sorted(baseline)
        left = _finite_values([baseline[key] for key in keys], "baseline")
        right = _finite_values([candidate[key] for key in keys], "candidate")
        deltas = [new - old for old, new in zip(left, right)]
    else:
        assert not isinstance(candidate, Mapping)
        left = _finite_values(baseline, "baseline")
        right = _finite_values(candidate, "candidate")
        if len(left) != len(right):
            raise ValueError("paired sequences must have equal length")
        deltas = [new - old for old, new in zip(left, right)]
    if not deltas:
        raise ValueError("paired bootstrap requires at least one independent unit")
    _finite_values(deltas, "paired deltas")

    point = fmean(deltas)
    generator = random.Random(seed)
    n_units = len(deltas)
    samples = [
        fmean(deltas[generator.randrange(n_units)] for _ in range(n_units))
        for _ in range(resamples)
    ]
    _finite_values([point, *samples], "paired-bootstrap results")
    alpha = (1.0 - confidence) / 2.0
    ci_low = _quantile(samples, alpha)
    ci_high = _quantile(samples, 1.0 - alpha)
    _finite_values([ci_low, ci_high], "paired-bootstrap interval")
    return BootstrapResult(
        n_units=n_units,
        resamples=resamples,
        seed=seed,
        delta=point,
        ci_low=ci_low,
        ci_high=ci_high,
    )


__all__ = ["BootstrapResult", "paired_bootstrap"]
