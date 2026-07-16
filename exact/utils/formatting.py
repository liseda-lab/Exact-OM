"""Small formatting and numeric helpers shared across Exact-OM subsystems."""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence


def format_duration(total_seconds: float) -> str:
    """Render seconds as ``days:hours:minutes:seconds``."""
    seconds = max(0, int(round(total_seconds)))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d:{hours:02d}:{minutes:02d}:{seconds:02d}"


def strip_code_fences(text: str) -> str:
    """Remove one optional Markdown code fence from a model response."""
    blob = (text or "").strip()
    if not blob:
        return ""
    blob = re.sub(r"^\s*```(?:json)?\s*", "", blob, flags=re.IGNORECASE)
    blob = re.sub(r"\s*```\s*$", "", blob)
    return blob.strip()


def clip01(value: float) -> float:
    """Clamp a numeric value to the closed unit interval."""
    return max(0.0, min(1.0, float(value)))


def safe_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or zero for an empty sequence."""
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def safe_div(numerator: float, denominator: float) -> float:
    """Divide two values, returning zero when the denominator is zero."""
    return float(numerator) / float(denominator) if denominator else 0.0


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    """Return a linearly interpolated quantile, or ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])
