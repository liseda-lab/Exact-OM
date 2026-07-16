"""Atomic run-manifest creation and artifact registration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Optional

from .layout import LAYOUT_VERSION, RunLayout


MANIFEST_SCHEMA_VERSION = 1
DELIVERABLE_KINDS = {"alignment", "evaluation"}


def _exact_version() -> str:
    try:
        return version("exact-om")
    except PackageNotFoundError:
        return "unknown"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash ``path`` without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class RunManifest:
    """Mutable manifest facade with atomic persistence."""

    layout: RunLayout
    payload: dict[str, Any]

    @classmethod
    def create(cls, layout: RunLayout, run_id: Optional[str] = None) -> "RunManifest":
        if layout.version != LAYOUT_VERSION:
            raise ValueError("Run manifests can only be created for v2 layouts")
        sessions = [run_id] if run_id else []
        return cls(
            layout,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "layout_version": LAYOUT_VERSION,
                "exact_version": _exact_version(),
                "sessions": sessions,
                "artifacts": [],
            },
        )

    @classmethod
    def open(cls, layout_or_root: RunLayout | Path) -> "RunManifest":
        layout = (
            layout_or_root
            if isinstance(layout_or_root, RunLayout)
            else RunLayout.open(layout_or_root)
        )
        try:
            payload = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Run manifest not found: {layout.manifest_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid run manifest at {layout.manifest_path}: {exc}") from exc
        if int(payload.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported run manifest schema version: "
                f"{payload.get('schema_version')!r}"
            )
        if int(payload.get("layout_version", -1)) != layout.version:
            raise ValueError("Manifest layout version does not match the detected run layout")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("Run manifest 'artifacts' must be a list")
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                raise ValueError("Every manifest artifact must have a string path")
            layout.resolve_relative(artifact["path"])
        return cls(layout, payload)

    def add_session(self, run_id: str) -> None:
        sessions = self.payload.setdefault("sessions", [])
        if run_id not in sessions:
            sessions.append(run_id)

    def register(
        self,
        path: Path,
        *,
        kind: str,
        schema_version: Optional[int] = None,
        run_id: Optional[str] = None,
        checksum: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Register an existing artifact, replacing an entry for the same path."""

        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cannot register missing artifact: {path}")
        relative = self.layout.relative(path)
        artifact: dict[str, Any] = {
            "path": relative,
            "kind": str(kind),
            "bytes": path.stat().st_size,
        }
        if schema_version is not None:
            artifact["schema_version"] = int(schema_version)
        if run_id is not None:
            artifact["run_id"] = str(run_id)
            self.add_session(str(run_id))
        should_hash = kind in DELIVERABLE_KINDS if checksum is None else checksum
        if should_hash:
            artifact["sha256"] = sha256_file(path)
        artifacts = list(self.payload.get("artifacts") or [])
        artifacts = [entry for entry in artifacts if entry.get("path") != relative]
        artifacts.append(artifact)
        artifacts.sort(key=lambda entry: str(entry.get("path", "")))
        self.payload["artifacts"] = artifacts
        return artifact

    def remove_missing(self) -> None:
        """Drop entries whose files no longer exist."""

        self.payload["artifacts"] = [
            artifact
            for artifact in self.payload.get("artifacts") or []
            if self.layout.resolve_relative(str(artifact["path"])).is_file()
        ]

    def write(self) -> Path:
        self.payload["schema_version"] = MANIFEST_SCHEMA_VERSION
        self.payload["layout_version"] = self.layout.version
        self.payload["exact_version"] = _exact_version()
        _atomic_json(self.layout.manifest_path, self.payload)
        return self.layout.manifest_path

    def paths(self, kinds: Optional[Iterable[str]] = None) -> list[Path]:
        selected = set(kinds or [])
        return [
            self.layout.resolve_relative(str(artifact["path"]))
            for artifact in self.payload.get("artifacts") or []
            if not selected or artifact.get("kind") in selected
        ]


def refresh_manifest(
    layout: RunLayout,
    *,
    run_id: Optional[str] = None,
    manifest: Optional[RunManifest] = None,
) -> RunManifest:
    """Register every known v2 artifact currently present in ``layout``."""

    current = manifest or (
        RunManifest.open(layout) if layout.manifest_path.exists() else RunManifest.create(layout)
    )
    if run_id:
        current.add_session(run_id)

    candidates: list[tuple[Path, str, Optional[int], Optional[bool]]] = [
        (layout.timings_path, "timing", 1, False),
        (layout.log_path, "log", None, False),
        (layout.mapping_path("global"), "alignment", None, True),
        (layout.mapping_path("local"), "alignment", None, True),
        (layout.evaluation_path("json"), "evaluation", 1, True),
        (layout.evaluation_path("csv"), "evaluation", None, True),
        (layout.run_stats_path, "stats", 1, False),
        (layout.stats_dir / "run_stats.csv", "stats", None, False),
        (layout.summary_metrics_path, "stats", None, False),
        (layout.stats_dir / "llm_calibration.json", "stats", 1, False),
        (layout.explanation_index_path, "explanations", 1, False),
        (layout.full_explanations_path, "explanations_export", 1, False),
    ]
    for path, kind, schema_version, checksum in candidates:
        if path.is_file():
            current.register(
                path,
                kind=kind,
                schema_version=schema_version,
                run_id=run_id,
                checksum=checksum,
            )
    for directory, kind in (
        (layout.explanation_shards_dir, "explanations"),
        (layout.plots_dir, "plot"),
        (layout.checkpoints_dir, "checkpoint"),
    ):
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            current.register(path, kind=kind, run_id=run_id, checksum=False)
    current.remove_missing()
    current.write()
    return current


def finalize_artifacts(
    run_dir: Path,
    *,
    run_id: Optional[str] = None,
    save_full_explanations: bool = False,
    checkpoint_retention: str = "latest",
) -> dict[str, Any]:
    """Finalize a successful run and return a compact size report."""

    from .gc import prune_checkpoints
    from .store import ExplanationStore

    layout = RunLayout.open(run_dir)
    if layout.version != LAYOUT_VERSION:
        raise ValueError("Only layout-v2 runs can be finalized")
    before = {
        name: sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        for name, directory in (
            ("alignment", layout.alignment_dir),
            ("evaluation", layout.evaluation_dir),
            ("explanations", layout.explanations_dir),
            ("stats", layout.stats_dir),
            ("plots", layout.plots_dir),
            ("checkpoints", layout.checkpoints_dir),
        )
        if directory.is_dir()
    }
    compaction: dict[str, int] = {}
    if layout.explanation_index_path.is_file():
        store = ExplanationStore(layout.explanations_dir, run_id=run_id)
        compaction = store.compact()
        if save_full_explanations:
            store.export(layout.full_explanations_path, format="json")
    retention = prune_checkpoints(run_dir, policy=checkpoint_retention)  # type: ignore[arg-type]
    manifest = refresh_manifest(layout, run_id=run_id)
    after = {
        name: sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        for name, directory in (
            ("alignment", layout.alignment_dir),
            ("evaluation", layout.evaluation_dir),
            ("explanations", layout.explanations_dir),
            ("stats", layout.stats_dir),
            ("plots", layout.plots_dir),
            ("checkpoints", layout.checkpoints_dir),
        )
        if directory.is_dir()
    }
    return {
        "before_bytes": before,
        "after_bytes": after,
        "compaction": compaction,
        "checkpoints_removed": retention.count,
        "checkpoint_bytes_removed": retention.bytes,
        "manifest": manifest.layout.manifest_path,
    }


__all__ = [
    "DELIVERABLE_KINDS",
    "MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "finalize_artifacts",
    "refresh_manifest",
    "sha256_file",
]
