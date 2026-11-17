"""Traversal-safe archive extraction used by declarative HTTP tracks."""

from __future__ import annotations

import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from .provider import IntegrityError


def _safe_member(destination: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise IntegrityError(f"Archive contains unsafe member path: {member_name!r}")
    candidate = destination.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(destination.resolve())
    except ValueError as exc:
        raise IntegrityError(f"Archive member escapes destination: {member_name!r}") from exc
    return candidate


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP without following links or accepting traversal paths."""

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise IntegrityError(f"ZIP archive contains a symbolic link: {info.filename}")
            target = _safe_member(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a TAR archive while rejecting links, devices, and traversal paths."""

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:*") as bundle:
        for info in bundle:
            if info.issym() or info.islnk() or info.isdev() or info.isfifo():
                raise IntegrityError(f"TAR archive contains unsupported member: {info.name}")
            target = _safe_member(destination, info.name)
            if info.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not info.isfile():
                raise IntegrityError(f"TAR archive contains unsupported member: {info.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(info)
            if source is None:
                raise IntegrityError(f"Could not read TAR member: {info.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def safe_extract_archive(archive: Path, destination: Path) -> None:
    """Extract a supported ZIP or TAR archive safely."""

    if zipfile.is_zipfile(archive):
        safe_extract_zip(archive, destination)
    elif tarfile.is_tarfile(archive):
        safe_extract_tar(archive, destination)
    else:
        raise IntegrityError(f"Unsupported archive format: {archive}")
