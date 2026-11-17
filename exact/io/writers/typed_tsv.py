"""BioKG-compatible typed TSV writer and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from exact.io.writers._frames import validated_mapping_frame
from exact.io.writers.base import WriterOptionsError

RELATION_TO_TYPED = {
    "=": "equivalent",
    "<": "source_subsumed_by_target",
    ">": "source_subsumes_target",
    "equivalent": "equivalent",
    "source_subsumed_by_target": "source_subsumed_by_target",
    "source_subsumes_target": "source_subsumes_target",
}
TYPED_RELATIONS = frozenset(RELATION_TO_TYPED.values())
TYPED_COLUMNS = ("SrcEntity", "TgtEntity", "Relation", "Score")


def validate_typed_frame(
    mappings: Any,
    *,
    sort: bool = True,
    reject_duplicates: bool = True,
) -> pd.DataFrame:
    """Return a validated BioKG submission table.

    Internal relation symbols are converted to the competition vocabulary and
    rows are ranked by descending finite score.
    """

    frame = validated_mapping_frame(mappings)
    normalized_relations: list[str] = []
    for relation in frame["Relation"]:
        try:
            normalized_relations.append(RELATION_TO_TYPED[str(relation).strip()])
        except KeyError as exc:
            allowed = ", ".join(sorted(RELATION_TO_TYPED))
            raise WriterOptionsError(
                f"Unsupported relation {relation!r}; expected one of: {allowed}"
            ) from exc
    frame["Relation"] = normalized_relations
    if reject_duplicates and frame.duplicated(["SrcEntity", "TgtEntity"]).any():
        duplicate = frame.loc[
            frame.duplicated(["SrcEntity", "TgtEntity"], keep=False),
            ["SrcEntity", "TgtEntity"],
        ].iloc[0]
        raise WriterOptionsError(
            "Duplicate mapping in typed submission: "
            f"{duplicate['SrcEntity']} -> {duplicate['TgtEntity']}"
        )
    if sort:
        frame = frame.sort_values(
            ["Score", "SrcEntity", "TgtEntity"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return frame[list(TYPED_COLUMNS)]


class TypedTsvWriter:
    """Write ranked mappings using the BioKG submission vocabulary."""

    name = "typed-tsv"
    default_filename = "alignment.typed.tsv"

    def write(
        self,
        mappings: Any,
        path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        normalized = dict(options or {})
        unknown = sorted(set(normalized) - {"reject_duplicates", "sort"})
        if unknown:
            raise WriterOptionsError(f"Unknown typed TSV option(s): {', '.join(unknown)}")
        frame = validate_typed_frame(
            mappings,
            sort=bool(normalized.get("sort", True)),
            reject_duplicates=bool(normalized.get("reject_duplicates", True)),
        )
        frame.to_csv(path, sep="\t", index=False)
        return Path(path)


WRITER = TypedTsvWriter()

__all__ = [
    "RELATION_TO_TYPED",
    "TYPED_COLUMNS",
    "TYPED_RELATIONS",
    "TypedTsvWriter",
    "WRITER",
    "validate_typed_frame",
]
