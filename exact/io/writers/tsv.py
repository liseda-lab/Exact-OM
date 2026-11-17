"""Legacy byte-compatible global and local TSV writers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from exact.io.writers._frames import canonical_frame
from exact.io.writers.base import WriterOptionsError


def _reject_options(options: Mapping[str, Any] | None) -> None:
    if options:
        raise WriterOptionsError(
            f"Legacy TSV writers do not accept options: {', '.join(sorted(options))}"
        )


class GlobalTsvWriter:
    """Write the historical three-column global alignment table."""

    name = "tsv-global"
    default_filename = "src2tgt.maps_global.tsv"

    def write(
        self,
        mappings: Any,
        path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        _reject_options(options)
        frame = canonical_frame(mappings)
        frame[["SrcEntity", "TgtEntity", "Score"]].to_csv(path, sep="\t", index=False)
        return Path(path)


class LocalTsvWriter:
    """Write the historical three-column local/ranking alignment table."""

    name = "tsv-local"
    default_filename = "src2tgt.maps_local.tsv"

    def write(
        self,
        mappings: Any,
        path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        _reject_options(options)
        frame = canonical_frame(mappings, require_score=False, require_candidates=True)
        frame[["SrcEntity", "TgtEntity", "TgtCandidates"]].to_csv(path, sep="\t", index=False)
        return Path(path)


GLOBAL_WRITER = GlobalTsvWriter()
LOCAL_WRITER = LocalTsvWriter()

__all__ = [
    "GLOBAL_WRITER",
    "LOCAL_WRITER",
    "GlobalTsvWriter",
    "LocalTsvWriter",
]
