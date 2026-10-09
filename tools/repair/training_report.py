"""Lossless immutable report transport outside the bounded worker message frame."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def publish_report(report: dict, directory: Path) -> dict:
    """Publish full acquisition/selection evidence before returning a small receipt."""
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".report-", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = os.fstat(stream.fileno()).st_size
        path = directory / (digest + ".json")
        os.replace(temporary, path)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return dict(
            schema="exact-repair/training-report-artifact/v1",
            path=str(path.resolve()),
            sha256=digest,
            size_bytes=size,
        )
    finally:
        temporary.unlink(missing_ok=True)


def read_report(artifact: dict, directory: Path) -> dict:
    """Check location, size and digest before admitting inert JSON evidence."""
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"schema", "path", "sha256", "size_bytes"}
        or artifact["schema"] != "exact-repair/training-report-artifact/v1"
    ):
        raise ValueError("Expected a training report artifact descriptor")
    digest, size = artifact["sha256"], artifact["size_bytes"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(size) is not int
        or size < 1
        or not isinstance(artifact["path"], str)
    ):
        raise ValueError("Invalid report artifact digest, size or path")
    path = Path(artifact["path"])
    if (
        not path.is_absolute()
        or path != directory.resolve() / (digest + ".json")
        or path.is_symlink()
    ):
        raise ValueError("Report artifact is outside its declared output directory")
    try:
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != size:
                raise ValueError("Report artifact size mismatch")
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Report artifact integrity mismatch")
            stream.seek(0)
            report = json.load(stream)
    except FileNotFoundError as error:
        raise ValueError("Report artifact is missing") from error
    if not isinstance(report, dict) or report.get("schema") != "exact-repair/training/v3":
        raise ValueError("Report artifact requires training schema v3")
    return report
