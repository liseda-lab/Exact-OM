"""Kind-aware selector grouping with class-only compatibility."""

from __future__ import annotations

from typing import Any, Iterator

import pandas as pd


def _group_columns(
    frame: pd.DataFrame,
    entity_column: str,
    kind_column: str,
) -> str | list[str]:
    if (
        kind_column in frame.columns
        and frame[kind_column].nunique(dropna=False) > 1
    ):
        return [entity_column, kind_column]
    return entity_column


def groupby_source(frame: pd.DataFrame):
    """Group candidate rows by source and, for mixed runs, source kind."""

    return frame.groupby(
        _group_columns(frame, "Src", "SrcKind"), sort=False
    )


def groupby_target(frame: pd.DataFrame):
    """Group candidate rows by target and, for mixed runs, target kind."""

    return frame.groupby(
        _group_columns(frame, "Tgt", "TgtKind"), sort=False
    )


def source_from_group_key(key: Any) -> str:
    """Return the source IRI from a scalar or composite pandas group key."""

    return str(key[0]) if isinstance(key, tuple) else str(key)


def kind_from_group_key(key: Any) -> str:
    """Return a group kind, defaulting legacy/single-kind keys to class."""

    return str(key[1]) if isinstance(key, tuple) and len(key) > 1 else "class"


def source_group_id(key: Any) -> str:
    """Return a collision-safe decision key for a selector source group."""

    if isinstance(key, tuple):
        return f"{source_from_group_key(key)}\x1f{kind_from_group_key(key)}"
    return str(key)


def iter_source_groups(
    frame: pd.DataFrame,
) -> Iterator[tuple[str, str, str, pd.DataFrame]]:
    """Yield decision id, source IRI, kind, and group in input order."""

    for key, group in groupby_source(frame):
        yield (
            source_group_id(key),
            source_from_group_key(key),
            kind_from_group_key(key),
            group,
        )


def count_source_groups(frame: pd.DataFrame) -> int:
    """Count source pools without merging kinds in mixed runs."""

    if frame.empty or "Src" not in frame.columns:
        return 0
    return int(groupby_source(frame).ngroups)


__all__ = [
    "count_source_groups",
    "groupby_source",
    "groupby_target",
    "iter_source_groups",
    "kind_from_group_key",
    "source_from_group_key",
    "source_group_id",
]
