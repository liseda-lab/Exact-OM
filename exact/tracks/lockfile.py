"""Atomic dataset lockfile and local integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .provider import IntegrityError

LOCKFILE_NAME = "datasets.lock.json"
LOCKFILE_VERSION = 1


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_within(root: Path, relative: str) -> Path:
    """Resolve a lockfile path and reject traversal outside ``root``."""

    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise IntegrityError(f"Unsafe path {relative!r} in dataset lock") from exc
    return candidate


def relative_to_root(path: Path, root: Path) -> str:
    """Return a normalized POSIX path relative to ``root``."""

    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as exc:
        raise IntegrityError(f"Path {path} is outside data root {root}") from exc


def empty_lock() -> dict[str, Any]:
    """Create a new lockfile document."""

    return {"lock_version": LOCKFILE_VERSION, "tasks": {}}


def read_lock(data_root: Path) -> dict[str, Any]:
    """Read and minimally validate ``datasets.lock.json``."""

    path = Path(data_root) / LOCKFILE_NAME
    if not path.exists():
        return empty_lock()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Could not read dataset lock {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("lock_version") != LOCKFILE_VERSION:
        raise IntegrityError(
            f"Unsupported or malformed dataset lock {path}; expected lock_version {LOCKFILE_VERSION}"
        )
    if not isinstance(value.get("tasks"), dict):
        raise IntegrityError(f"Malformed dataset lock {path}: 'tasks' must be an object")
    return value


def write_lock(data_root: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace the dataset lock and fsync its contents."""

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCKFILE_NAME
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(root, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock_operation(data_root: Path, *, timeout: float = 30.0) -> Iterator[None]:
    """Serialize lock/materialization updates using an exclusive sentinel file."""

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    sentinel = root / ".datasets.lock.write"
    deadline = time.monotonic() + timeout
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for dataset lock at {sentinel}")
            time.sleep(0.05)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        sentinel.unlink(missing_ok=True)


def hash_tree(path: Path, data_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every regular, non-symlink file below ``path``."""

    records: dict[str, dict[str, Any]] = {}
    for file_path in sorted(Path(path).rglob("*")):
        if file_path.is_symlink():
            raise IntegrityError(f"Materialized tracks may not contain symlinks: {file_path}")
        if file_path.is_file():
            records[relative_to_root(file_path, data_root)] = {
                "sha256": sha256_file(file_path),
                "size": file_path.stat().st_size,
            }
    return records


def verify_hashes(files: Mapping[str, Any], data_root: Path) -> tuple[list[str], int]:
    """Compare local files to lock records, returning issues and checked count."""

    issues: list[str] = []
    checked = 0
    for relative, record in sorted(files.items()):
        if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
            issues.append(f"invalid lock record for {relative}")
            continue
        path = path_within(data_root, relative)
        if not path.is_file() or path.is_symlink():
            issues.append(f"missing file: {relative}")
            continue
        checked += 1
        actual = sha256_file(path)
        if actual != record["sha256"]:
            issues.append(
                f"checksum mismatch for {relative}: locked {record['sha256']}, found {actual}"
            )
    return issues, checked
