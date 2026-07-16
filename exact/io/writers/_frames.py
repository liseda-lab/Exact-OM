"""Mapping-table normalization shared by built-in writers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

import pandas as pd

from exact.io.writers.base import WriterOptionsError

_ALIASES = {
    "SrcEntity": ("Src", "source", "src", "head"),
    "TgtEntity": ("Tgt", "target", "tgt", "tail"),
    "Score": ("score", "confidence", "measure"),
    "Relation": ("relation",),
    "Kind": ("kind", "SrcKind"),
    "TgtCandidates": ("Candidates", "candidates"),
}


def _records(values: Iterable[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            records.append(dict(value))
            continue
        if hasattr(value, "head") and hasattr(value, "tail"):
            records.append(
                {
                    "SrcEntity": str(value.head),
                    "TgtEntity": str(value.tail),
                    "Score": getattr(value, "score", 0.0),
                    "Relation": getattr(value, "relation", "="),
                    "Kind": getattr(value, "kind", "class"),
                }
            )
            continue
        raise WriterOptionsError(
            "Mappings must be a pandas DataFrame, mappings, or EntityMapping-like objects"
        )
    return records


def as_frame(mappings: Any) -> pd.DataFrame:
    """Return a defensive DataFrame copy from supported mapping containers."""

    if isinstance(mappings, pd.DataFrame):
        return mappings.copy()
    if isinstance(mappings, Mapping):
        return pd.DataFrame([dict(mappings)])
    if isinstance(mappings, (str, bytes)) or not isinstance(mappings, Iterable):
        raise WriterOptionsError("Mappings must be tabular or iterable")
    return pd.DataFrame(_records(mappings))


def canonical_frame(
    mappings: Any,
    *,
    require_score: bool = True,
    require_candidates: bool = False,
) -> pd.DataFrame:
    """Normalize mapping column aliases without changing row order."""

    frame = as_frame(mappings)
    rename: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        if canonical in frame.columns:
            continue
        alias = next((item for item in aliases if item in frame.columns), None)
        if alias is not None:
            rename[alias] = canonical
    frame = frame.rename(columns=rename)
    required = ["SrcEntity", "TgtEntity"]
    if require_score:
        required.append("Score")
    if require_candidates:
        required.append("TgtCandidates")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise WriterOptionsError(
            f"Mapping table is missing required column(s): {', '.join(missing)}"
        )
    if "Relation" not in frame.columns:
        frame["Relation"] = "="
    else:
        frame["Relation"] = frame["Relation"].replace({"<?rel>": "="}).fillna("=")
    if "Kind" not in frame.columns:
        frame["Kind"] = "class"
    return frame


def validated_mapping_frame(mappings: Any) -> pd.DataFrame:
    """Normalize a scored table and validate identifiers and finite scores."""

    frame = canonical_frame(mappings)
    for column in ("SrcEntity", "TgtEntity"):
        if frame[column].isna().any() or any(not str(value).strip() for value in frame[column]):
            raise WriterOptionsError(f"{column} values must be non-empty")
        frame[column] = frame[column].map(str)
    try:
        frame["Score"] = pd.to_numeric(frame["Score"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise WriterOptionsError("Score values must be numeric") from exc
    if any(not isfinite(float(value)) for value in frame["Score"]):
        raise WriterOptionsError("Score values must be finite")
    return frame


__all__ = ["as_frame", "canonical_frame", "validated_mapping_frame"]
