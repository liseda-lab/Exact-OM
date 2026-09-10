"""Small, deterministic provenance helpers for user-supplied artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash file bytes or a directory's relative filenames and file-byte identities."""
    path = Path(path)
    if not path.is_dir():
        return sha256_file(path)
    files = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if not files:
        raise ValueError(f"Input directory contains no files: {path}")
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
    if resolved.is_dir():
        files = {
            item.relative_to(resolved).as_posix(): sha256_file(item)
            for item in sorted(resolved.rglob("*"))
            if item.is_file()
        }
        return {
            "path": str(resolved),
            "sha256": sha256_path(resolved),
            "files": files,
            "bytes": sum((resolved / name).stat().st_size for name in files),
            "rows": None,
        }
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "rows": tabular_row_count(resolved),
    }
