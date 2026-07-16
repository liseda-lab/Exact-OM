"""Version-aware path resolution for Exact run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


LAYOUT_VERSION = 2
MANIFEST_FILENAME = "run_manifest.json"


@dataclass(frozen=True)
class RunLayout:
    """Resolve artifact paths without leaking layout details to consumers.

    ``open`` is side-effect free and auto-detects historical (v1) runs. New
    producers should call ``create`` and write only through the v2 properties.
    """

    root: Path
    version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if self.version not in {1, LAYOUT_VERSION}:
            raise ValueError(f"Unsupported run layout version: {self.version}")

    @classmethod
    def open(cls, root: Path) -> "RunLayout":
        """Open an existing run, defaulting to v1 when no manifest exists."""

        resolved = Path(root).expanduser().resolve()
        manifest_path = resolved / MANIFEST_FILENAME
        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid run manifest at {manifest_path}: {exc}") from exc
            version = int(payload.get("layout_version", LAYOUT_VERSION))
            return cls(resolved, version)
        if (resolved / "explanations" / "index.json").is_file():
            return cls(resolved, LAYOUT_VERSION)
        v2_directories = {"alignment", "evaluation", "explanations", "stats", "checkpoints"}
        if all((resolved / name).is_dir() for name in v2_directories) and not (
            resolved / "model"
        ).exists():
            return cls(resolved, LAYOUT_VERSION)
        return cls(resolved, 1)

    @classmethod
    def create(cls, root: Path) -> "RunLayout":
        """Create the directory skeleton for a new v2 run."""

        layout = cls(Path(root), LAYOUT_VERSION)
        layout.ensure_directories()
        return layout

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def timings_path(self) -> Path:
        return self.root / "timings.json"

    @property
    def log_path(self) -> Path:
        return self.root / "exact.log"

    @property
    def config_path(self) -> Path:
        candidates = [self.root / "config.yaml", self.root / "config.yml"]
        return self._first_existing(candidates)

    @property
    def alignment_dir(self) -> Path:
        if self.version == LAYOUT_VERSION:
            return self.root / "alignment"
        return self.root / "model" / "alignment"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation" if self.version == LAYOUT_VERSION else self.root

    @property
    def explanations_dir(self) -> Path:
        if self.version == LAYOUT_VERSION:
            return self.root / "explanations"
        return self.alignment_dir / "default"

    @property
    def explanation_shards_dir(self) -> Path:
        return self.explanations_dir / "shards"

    @property
    def explanation_index_path(self) -> Path:
        return self.explanations_dir / "index.json"

    @property
    def full_explanations_path(self) -> Path:
        if self.version == LAYOUT_VERSION:
            return self.explanations_dir / "full_explanations.json"
        return self._first_existing(
            [
                self.alignment_dir / "default" / "full_explanations.json",
                self.alignment_dir / "full_explanations.json",
                self.root / "full_explanations.json",
            ]
        )

    @property
    def stats_dir(self) -> Path:
        if self.version == LAYOUT_VERSION:
            return self.root / "stats"
        return self.alignment_dir / "default"

    @property
    def run_stats_path(self) -> Path:
        return self._first_existing(
            [
                self.stats_dir / "run_stats.json",
                self.root / "run_stats.json",
            ]
        )

    @property
    def summary_metrics_path(self) -> Path:
        return self._first_existing(
            [
                self.stats_dir / "summary_metrics.csv",
                self.alignment_dir / "default" / "summary_metrics.csv",
            ]
        )

    @property
    def plots_dir(self) -> Path:
        if self.version == LAYOUT_VERSION:
            return self.root / "plots"
        return self.root / "model" / "plots"

    @property
    def checkpoints_dir(self) -> Path:
        if self.version == LAYOUT_VERSION:
            return self.root / "checkpoints"
        return self.root / "model" / "checkpoints"

    def mapping_path(self, kind: Literal["global", "local"]) -> Path:
        """Return the canonical or existing mapping path for ``kind``."""

        if kind not in {"global", "local"}:
            raise ValueError(f"Unknown mapping kind: {kind!r}")
        if self.version == LAYOUT_VERSION:
            return self._first_existing(
                [
                    self.alignment_dir / f"maps_{kind}.tsv",
                    self.alignment_dir / f"src2tgt.maps_{kind}.tsv",
                ]
            )
        return self._first_existing(
            [
                self.alignment_dir / f"src2tgt.maps_{kind}.tsv",
                self.alignment_dir / f"maps_{kind}.tsv",
                self.root / "alignment" / f"src2tgt.maps_{kind}.tsv",
            ]
        )

    def evaluation_path(self, suffix: Literal["json", "csv"] = "json") -> Path:
        """Return the canonical or existing evaluation report path."""

        name = f"evaluation_results.{suffix}"
        if self.version == LAYOUT_VERSION:
            return self.evaluation_dir / name
        return self._first_existing(
            [
                self.root / name,
                self.alignment_dir / "default" / name,
            ]
        )

    def ensure_directories(self) -> None:
        """Create every producer-owned v2 directory."""

        if self.version != LAYOUT_VERSION:
            raise ValueError("Cannot create artifacts in a historical v1 layout")
        for path in (
            self.alignment_dir,
            self.evaluation_dir,
            self.explanation_shards_dir,
            self.stats_dir,
            self.plots_dir,
            self.checkpoints_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        """Return a manifest-safe path relative to the run root."""

        resolved = Path(path).expanduser().resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes run directory: {path}") from exc

    def resolve_relative(self, value: str) -> Path:
        """Resolve and validate a manifest path."""

        path = (self.root / value).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Manifest artifact path escapes run directory: {value!r}") from exc
        return path

    @staticmethod
    def _first_existing(candidates: list[Path]) -> Path:
        return next((path for path in candidates if path.exists()), candidates[0])


__all__ = ["LAYOUT_VERSION", "MANIFEST_FILENAME", "RunLayout"]
