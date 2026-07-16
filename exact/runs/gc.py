"""Foreign-file-safe retention and cleanup for run directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .layout import RunLayout
from .manifest import RunManifest

CheckpointRetention = Literal["latest", "all", "none"]
_SIDECAR_SUFFIXES = ("_audit", "_candidates", "_overlay")
_CHECKPOINT_PREFIXES = (
    "inference_",
    "train_",
    "validation_",
    "test_",
    "prefiltered_",
)


@dataclass(frozen=True)
class CleanResult:
    """Files selected by a cleanup operation."""

    paths: tuple[Path, ...]
    bytes: int
    dry_run: bool

    @property
    def count(self) -> int:
        return len(self.paths)


def _known_checkpoint_file(path: Path, checkpoint_dir: Path) -> bool:
    try:
        relative = path.relative_to(checkpoint_dir)
    except ValueError:
        return False
    if len(relative.parts) == 1:
        name = relative.name
        return name.endswith(".json") and (
            name.startswith(_CHECKPOINT_PREFIXES)
            or any(token in name for token in ("_additional_models_", "_rationales_"))
        )
    parent = relative.parts[0]
    if not parent.endswith(_SIDECAR_SUFFIXES):
        return False
    name = relative.name
    return name == "manifest.json" or (
        name.startswith("shard-") and (".jsonl" in name or name.endswith(".json"))
    )


def _manifest_paths(layout: RunLayout, kinds: Iterable[str]) -> set[Path]:
    if not layout.manifest_path.is_file():
        return set()
    manifest = RunManifest.open(layout)
    return set(manifest.paths(kinds))


def cleanup_plan(
    run_dir: Path,
    *,
    keep_resume: bool = False,
    include_dataset_cache: bool = False,
) -> tuple[Path, ...]:
    """Return only manifest-owned or recognized disposable files."""

    layout = RunLayout.open(run_dir)
    selected: set[Path] = set()
    if not keep_resume:
        selected.update(_manifest_paths(layout, {"checkpoint"}))
        if layout.checkpoints_dir.is_dir():
            selected.update(
                path
                for path in layout.checkpoints_dir.rglob("*")
                if path.is_file() and _known_checkpoint_file(path, layout.checkpoints_dir)
            )
    if include_dataset_cache:
        selected.update(_manifest_paths(layout, {"dataset_cache", "cache"}))
    return tuple(sorted(path for path in selected if path.is_file()))


def clean_run(
    run_dir: Path,
    *,
    keep_resume: bool = False,
    include_dataset_cache: bool = False,
    dry_run: bool = False,
) -> CleanResult:
    """Remove disposable artifacts while preserving all foreign files."""

    layout = RunLayout.open(run_dir)
    paths = cleanup_plan(
        run_dir,
        keep_resume=keep_resume,
        include_dataset_cache=include_dataset_cache,
    )
    size = sum(path.stat().st_size for path in paths)
    if not dry_run:
        for path in paths:
            path.unlink(missing_ok=True)
        if layout.checkpoints_dir.is_dir():
            for directory in sorted(
                (path for path in layout.checkpoints_dir.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        if layout.manifest_path.is_file():
            manifest = RunManifest.open(layout)
            manifest.remove_missing()
            manifest.write()
    return CleanResult(paths=paths, bytes=size, dry_run=dry_run)


def _checkpoint_references(path: Path) -> set[Path]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    references: set[Path] = set()
    for key in (
        "audit_manifest_path",
        "candidate_records_manifest_path",
        "final_overlay_manifest_path",
        "candidate_records_path",
        "results_path",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            continue
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = (path.parent / resolved).resolve()
        references.add(resolved)
        if resolved.name == "manifest.json" and resolved.parent.is_dir():
            references.update(item for item in resolved.parent.rglob("*") if item.is_file())
    return references


def _valid_checkpoint(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("processed_examples"), int)
        and bool(payload.get("checkpoint_fingerprint"))
    )


def prune_checkpoints(
    run_dir: Path,
    policy: CheckpointRetention = "latest",
) -> CleanResult:
    """Apply the finalization checkpoint-retention policy."""

    if policy not in {"latest", "all", "none"}:
        raise ValueError(f"Unknown checkpoint retention policy: {policy!r}")
    layout = RunLayout.open(run_dir)
    if policy == "all" or not layout.checkpoints_dir.is_dir():
        return CleanResult(paths=(), bytes=0, dry_run=False)
    known = set(cleanup_plan(run_dir))
    if policy == "latest":
        manifests = [
            path
            for path in known
            if path.parent == layout.checkpoints_dir
            and path.suffix == ".json"
            and _valid_checkpoint(path)
        ]
        if manifests:
            latest = max(manifests, key=lambda path: (path.stat().st_mtime_ns, path.name))
            keep = {latest, *_checkpoint_references(latest)}
            known.difference_update(keep)
    paths = tuple(sorted(path for path in known if path.is_file()))
    size = sum(path.stat().st_size for path in paths)
    for path in paths:
        path.unlink(missing_ok=True)
    if layout.manifest_path.is_file():
        manifest = RunManifest.open(layout)
        manifest.remove_missing()
        manifest.write()
    return CleanResult(paths=paths, bytes=size, dry_run=False)


def format_bytes(value: int) -> str:
    """Render byte counts for CLI reports."""

    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


__all__ = [
    "CheckpointRetention",
    "CleanResult",
    "clean_run",
    "cleanup_plan",
    "format_bytes",
    "prune_checkpoints",
]
