"""Small, deterministic provenance helpers for user-supplied artifacts."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tabular_row_count(path: Path) -> Optional[int]:
    """Count data rows for CSV/TSV inputs, excluding an optional header row."""

    path = Path(path)
    if path.suffix.lower() not in {".csv", ".tsv", ".txt", ".cands"}:
        return None
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".cands"} else ","
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream, delimiter=delimiter) if any(cell for cell in row)]
    if not rows:
        return 0
    header_tokens = {cell.strip().lower() for cell in rows[0]}
    has_header = bool(
        header_tokens
        & {
            "src",
            "source",
            "srcentity",
            "tgt",
            "target",
            "tgtentity",
            "tgtcandidates",
            "score",
        }
    )
    return max(0, len(rows) - int(has_header))


def file_provenance(path: Path) -> dict[str, Any]:
    """Describe an input file using its resolved path, digest, size, and row count."""

    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "rows": tabular_row_count(resolved),
    }
