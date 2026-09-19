"""Share verified prepared datasets across scorer variants, never predictions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from exact.utils.provenance import sha256_file

_REQUIRED = {
    "dataset.csv",
    "dataset.meta.json",
    "candidate_pool_manifest.json",
    "candidate_recall.tsv",
}
_TEMPLATES = {"verbalization_templates.json", "verbalization_templates.meta.json"}


def _location(fingerprint: str) -> tuple[Path, str] | None:
    root, role = os.getenv("EXACT_DATASET_CACHE_DIR"), os.getenv("EXACT_EXPERIMENT_ROLE")
    if not root:
        return None
    if not role or role in {".", ".."} or Path(role).name != role:
        raise ValueError("Shared dataset cache requires an explicit experiment role")
    return Path(root) / role / fingerprint, role


def _verify(directory: Path, fingerprint: str, role: str) -> dict:
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
    return record


def restore(directory: Path, fingerprint: str) -> bool:
    location = _location(fingerprint)
    if location is None or not location[0].exists():
        return False
    source, role = location
    record = _verify(source, fingerprint, role)
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
