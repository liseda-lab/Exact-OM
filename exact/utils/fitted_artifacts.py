"""Canonical fingerprints and durable immutable research artifacts."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _json_chunks(payload):
    """Bound encoded memory while retaining the original canonical JSON bytes."""
    chunks, size = [], 0
    for chunk in json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False).iterencode(payload):
        encoded = chunk.encode("utf-8")
        chunks.append(encoded)
        size += len(encoded)
        if size >= 1024**2:
            yield b"".join(chunks)
            chunks, size = [], 0
    chunks.append(b"\n")
    yield b"".join(chunks)


def freeze_json(path, payload, *, max_bytes=None):
    """Stream immutable JSON with bounded size and reserved filesystem headroom."""
    path = Path(path)
    limit = int(os.environ.get("EXACT_JSON_MAX_BYTES", 8 * 1024**3))
    if max_bytes is not None:
        limit = min(limit, max_bytes)
    reserve = int(os.environ.get("EXACT_STORAGE_MIN_FREE_BYTES", 1024**3))
    if limit <= 0 or reserve < 0:
        raise ValueError("JSON size limit must be positive and storage reserve nonnegative")
    if path.exists():
        with path.open("rb") as stream:
            for encoded in _json_chunks(payload):
                if stream.read(len(encoded)) != encoded:
                    raise ValueError(f"Fitted artifact identity conflict: {path}")
            if stream.read(1):
                raise ValueError(f"Fitted artifact identity conflict: {path}")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    written = 0
    try:
        with temporary.open("wb") as stream:
            for encoded in _json_chunks(payload):
                if written + len(encoded) > limit:
                    raise ValueError(f"JSON artifact exceeds {limit} byte safety limit: {path}")
                if shutil.disk_usage(path.parent).free < reserve + len(encoded):
                    raise OSError(f"Storage reserve of {reserve} bytes would be consumed: {path}")
                stream.write(encoded)
                written += len(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload
