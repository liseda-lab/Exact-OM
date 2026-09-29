"""Share verified prepared datasets across scorer variants, never predictions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

from exact.utils.provenance import sha256_file

_REQUIRED = {
    "dataset.csv",
    "dataset.meta.json",
    "candidate_pool_manifest.json",
    "candidate_recall.tsv",
}
_TEMPLATES = {"verbalization_templates.json", "verbalization_templates.meta.json"}
_VERIFIED = OrderedDict()


def _signature(directory: Path) -> list:
    """An immutable-file receipt is invalidated by content writes or replacement."""
    result = []
    for name in sorted(_REQUIRED | _TEMPLATES | {"manifest.json"}):
        path = directory / name
        if path.exists():
            stat = path.stat()
            result.append(
                [name, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            )
    return result


def _location(fingerprint: str) -> tuple[Path, str] | None:
    root, role = os.getenv("EXACT_DATASET_CACHE_DIR"), os.getenv("EXACT_EXPERIMENT_ROLE")
    if not root:
        return None
    if not role or role in {".", ".."} or Path(role).name != role:
        raise ValueError("Shared dataset cache requires an explicit experiment role")
    return Path(root) / role / fingerprint, role


def _verify(directory: Path, fingerprint: str, role: str) -> dict:
    signature = _signature(directory)
    key = (str(directory.resolve()), fingerprint, role)
    cached = _VERIFIED.get(key)
    if cached is not None and cached[0] == signature:
        _VERIFIED.move_to_end(key)
        return cached[1]
    record: dict = json.loads((directory / "manifest.json").read_text())
    if not isinstance(record, dict):
        raise ValueError("Shared dataset cache has an incompatible manifest")
    files = record.get("files", {})
    if (
        record.get("schema_version") != 1
        or record.get("fingerprint") != fingerprint
        or record.get("role") != role
        or not _REQUIRED.issubset(files)
        or set(files) - (_REQUIRED | _TEMPLATES)
        or bool(set(files) & _TEMPLATES) != _TEMPLATES.issubset(files)
    ):
        raise ValueError("Shared dataset cache has an incompatible manifest")
    for name, expected in files.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Shared dataset cache has changed: {name}")
    metadata = json.loads((directory / "dataset.meta.json").read_text())
    pool = json.loads((directory / "candidate_pool_manifest.json").read_text())
    if (
        metadata.get("fingerprint") != fingerprint
        or metadata.get("candidate_pool_fingerprint") != pool.get("fingerprint")
        or metadata.get("candidate_recall_sha256") != files["candidate_recall.tsv"]
    ):
        raise ValueError("Shared dataset cache metadata does not match its prepared inputs")
    if _signature(directory) != signature:
        raise ValueError("Shared dataset cache changed during verification")
    _VERIFIED[key] = (signature, record)
    _VERIFIED.move_to_end(key)
    while len(_VERIFIED) > 128:
        _VERIFIED.popitem(last=False)
    return record


def _local_source(source: Path, fingerprint: str, role: str) -> tuple[Path, dict]:
    root = os.getenv("EXACT_DATASET_CACHE_LOCAL_DIR")
    if not root:
        return source, _verify(source, fingerprint, role)
    local_root = Path(root) / "prepared-datasets"
    key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    staged = local_root / key
    receipt_path = staged / "source-receipt.json"
    signature = _signature(source)
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text())
            if receipt["source"] == signature:
                record = _verify(staged, fingerprint, role)
                # The local copy was fully hashed when published. Do not re-read
                # unchanged multi-gigabyte NAS files for every new scorer arm.
                return staged, record
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    record = _verify(source, fingerprint, role)
    # Never use an existing, altered local payload, even when the NAS copy is fine.
    if staged.exists():
        if _verify(staged, fingerprint, role) != record:
            raise ValueError("Shared dataset cache publication changed after local staging")
        receipt_path.write_text(json.dumps({"source": signature}) + "\n")
        return staged, record
    temporary = None
    try:
        local_root.mkdir(parents=True, exist_ok=True)
        with (local_root / ".stage.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if staged.exists():
                if _verify(staged, fingerprint, role) != record:
                    raise ValueError("Shared dataset cache publication changed after local staging")
                return staged, record
            needed = sum((source / name).stat().st_size for name in record["files"])
            limit = max(0, int(os.getenv("EXACT_DATASET_CACHE_LOCAL_MAX_BYTES", str(64 * 1024**3))))
            used = sum(
                path.stat().st_size
                for directory in local_root.iterdir()
                if directory.is_dir()
                for path in directory.iterdir()
                if path.is_file()
            )
            if used + needed > limit or shutil.disk_usage(local_root).free < needed + 1024**3:
                return source, record
            temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=local_root))
            for name in record["files"]:
                shutil.copyfile(source / name, temporary / name)
            shutil.copyfile(source / "manifest.json", temporary / "manifest.json")
            _verify(temporary, fingerprint, role)
            if _signature(source) != signature:
                raise ValueError("Shared dataset cache changed during local staging")
            (temporary / "source-receipt.json").write_text(json.dumps({"source": signature}) + "\n")
            try:
                temporary.rename(staged)
            except OSError:
                if not staged.is_dir():
                    raise
                if _verify(staged, fingerprint, role) != record:
                    raise ValueError("Shared dataset cache publication changed after local staging")
            return staged, record
    except OSError:
        # Full/unavailable scratch is a cache miss, never a failed experiment.
        return source, record
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def restore(directory: Path, fingerprint: str) -> bool:
    location = _location(fingerprint)
    if location is None or not location[0].exists():
        return False
    source, role = location
    source, record = _local_source(source, fingerprint, role)
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(record["files"], key=lambda name: name == "dataset.meta.json"):
        temporary = directory / f".{name}.shared.tmp"
        shutil.copyfile(source / name, temporary)
        os.replace(temporary, directory / name)
    return True


def publish(directory: Path, fingerprint: str) -> Path | None:
    location = _location(fingerprint)
    if location is None or not all((directory / name).is_file() for name in _REQUIRED):
        return None
    destination, role = location
    if destination.exists():
        _verify(destination, fingerprint, role)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepared-", dir=destination.parent))
    try:
        names = _REQUIRED | (
            _TEMPLATES if all((directory / name).is_file() for name in _TEMPLATES) else set()
        )
        for name in names:
            shutil.copyfile(directory / name, temporary / name)
        record = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "role": role,
            "files": {name: sha256_file(temporary / name) for name in sorted(names)},
        }
        (temporary / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        _verify(temporary, fingerprint, role)
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            _verify(destination, fingerprint, role)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
