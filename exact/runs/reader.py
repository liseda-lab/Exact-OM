"""Stable read seam for historical and layout-v2 run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import pandas as pd

from exact.utils.data import read_table

from .layout import LAYOUT_VERSION, RunLayout
from .manifest import RunManifest
from .store import ExplanationStore


class RunReader:
    """Read mappings, explanations, stats, and provenance from any run layout."""

    def __init__(self, layout: RunLayout):
        self.layout = layout
        self._store: Optional[ExplanationStore] = None
        self._legacy_explanations: Optional[list[dict[str, Any]]] = None

    @classmethod
    def open(cls, run_dir: Path) -> "RunReader":
        layout = RunLayout.open(run_dir)
        if not layout.root.exists():
            raise FileNotFoundError(f"Run directory not found: {layout.root}")
        return cls(layout)

    def mappings(self, kind: Literal["global", "local"]) -> pd.DataFrame:
        path = self.layout.mapping_path(kind)
        if not path.is_file():
            raise FileNotFoundError(f"{kind.title()} mapping artifact not found: {path}")
        return read_table(path)

    def explanations_for(self, src_iri: str) -> list[dict[str, Any]]:
        if self._has_store:
            return self._explanation_store.get(src_iri)
        return [
            record
            for record in self._load_legacy_explanations()
            if self._record_source(record) == str(src_iri)
        ]

    def iter_explanations(self) -> Iterator[dict[str, Any]]:
        if self._has_store:
            yield from self._explanation_store.iter_all()
            return
        yield from self._load_legacy_explanations()

    def manifest(self) -> dict[str, Any]:
        if self.layout.manifest_path.is_file():
            return dict(RunManifest.open(self.layout).payload)
        artifacts: list[dict[str, Any]] = []
        candidates = [
            (self.layout.mapping_path("global"), "alignment"),
            (self.layout.mapping_path("local"), "alignment"),
            (self.layout.full_explanations_path, "explanations_export"),
            (self.layout.run_stats_path, "stats"),
            (self.layout.summary_metrics_path, "stats"),
            (self.layout.timings_path, "timing"),
        ]
        for path, kind in candidates:
            if path.is_file():
                artifacts.append(
                    {
                        "path": self.layout.relative(path),
                        "kind": kind,
                        "bytes": path.stat().st_size,
                    }
                )
        return {
            "schema_version": 1,
            "layout_version": 1,
            "exact_version": "unknown",
            "sessions": [],
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
            "synthesized": True,
        }

    def stats(self) -> dict[str, Any]:
        path = self.layout.run_stats_path
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid run statistics at {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Run statistics must be a JSON object: {path}")
        return payload

    @property
    def explanation_shard_reads(self) -> int:
        return self._store.shard_reads if self._store is not None else 0

    @property
    def _has_store(self) -> bool:
        return (
            self.layout.version == LAYOUT_VERSION and self.layout.explanation_index_path.is_file()
        )

    @property
    def _explanation_store(self) -> ExplanationStore:
        if self._store is None:
            self._store = ExplanationStore(self.layout.explanations_dir)
        return self._store

    def _load_legacy_explanations(self) -> list[dict[str, Any]]:
        if self._legacy_explanations is None:
            path = self.layout.full_explanations_path
            if not path.is_file():
                raise FileNotFoundError(f"Explanation artifact not found: {path}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid explanation artifact at {path}: {exc}") from exc
            if not isinstance(payload, list) or not all(
                isinstance(record, dict) for record in payload
            ):
                raise ValueError(f"Explanation artifact must be a JSON array of objects: {path}")
            self._legacy_explanations = payload
        return self._legacy_explanations

    @staticmethod
    def _record_source(record: dict[str, Any]) -> str:
        for key in ("src_iri", "Src", "source_iri", "source"):
            if record.get(key) is not None:
                return str(record[key])
        return ""


__all__ = ["RunReader"]
