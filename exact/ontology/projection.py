"""Narrow Exact adapter over :mod:`pyowl2vec_star_projector`.

Projection semantics live upstream.  This module owns only normalized Exact
configuration, semantic cache identity, and bounded conversion of returned rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Literal, Mapping, TypeAlias, cast

import pyowl2vec_star_projector as shared_projector
import pyowl_core
from pyowl2vec_star_projector import REFERENCE_PROFILE, ProjectionOptions, Projector
from pyowl_core import OntologySnapshot

from exact.core.entities.graph import Edge

ProjectorBackend: TypeAlias = Literal["auto", "native", "python"]
ProjectionMethod: TypeAlias = Literal["owl2vecstar", "taxonomy"]


@dataclass(frozen=True, slots=True)
class ProjectorSettings:
    """Exact's intentionally small shared-projector configuration surface."""

    backend: ProjectorBackend = "auto"
    profile: str = REFERENCE_PROFILE

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "native", "python"}:
            raise ValueError("projector backend must be one of: auto, native, python")
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("projector profile must be a nonempty string")
        # Upstream is authoritative for supported profiles and validation text.
        ProjectionOptions(profile=self.profile, backend=self.backend)

    @classmethod
    def from_value(cls, value: object = None) -> "ProjectorSettings":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="python")
        if not isinstance(value, Mapping):
            raise TypeError("projector configuration must be a mapping")
        unknown = sorted(set(map(str, value)) - {"backend", "profile"})
        if unknown:
            raise ValueError(f"unknown projector option(s): {', '.join(unknown)}")
        backend = value.get("backend", "auto")
        profile = value.get("profile", REFERENCE_PROFILE)
        return cls(
            backend=cast(ProjectorBackend, str(backend)),
            profile=str(profile),
        )


@dataclass(frozen=True, slots=True)
class ProjectionCacheKey:
    """Complete semantic identity for one in-process projection result."""

    structural_fingerprint: str
    core_package_version: str
    core_api_version: tuple[int, int]
    core_model_schema_version: int
    core_wire_format_version: tuple[int, int]
    projector_package_version: str
    projector_api_version: int
    projector_compiler_cache_schema: str
    method: ProjectionMethod
    profile: str
    backend: ProjectorBackend
    include_literals: bool
    duplicates: Literal["unique"] = "unique"
    order: Literal["canonical"] = "canonical"
    compatibility_state: Literal["isolated"] = "isolated"


def normalize_method(method: str) -> ProjectionMethod:
    normalized = str(method).lower().replace("_", "").replace("-", "")
    if normalized not in {"taxonomy", "owl2vecstar"}:
        raise ValueError("method must be one of {'taxonomy', 'owl2vecstar'}")
    return cast(ProjectionMethod, normalized)


def cache_key(
    snapshot: OntologySnapshot,
    settings: ProjectorSettings,
    *,
    method: str,
    include_literals: bool,
) -> ProjectionCacheKey:
    """Build the versioned semantic key required by the WP-M cache contract."""

    if not isinstance(snapshot, OntologySnapshot):
        raise TypeError("snapshot must be a concrete pyowl_core.OntologySnapshot")
    fingerprint = snapshot.structural_fingerprint
    return ProjectionCacheKey(
        structural_fingerprint=(f"{fingerprint.algorithm}:{fingerprint.schema}:{fingerprint.hex}"),
        core_package_version=str(pyowl_core.__version__),
        core_api_version=(pyowl_core.API_VERSION[0], pyowl_core.API_VERSION[1]),
        core_model_schema_version=int(pyowl_core.MODEL_SCHEMA_VERSION),
        core_wire_format_version=(
            pyowl_core.WIRE_FORMAT_VERSION[0],
            pyowl_core.WIRE_FORMAT_VERSION[1],
        ),
        projector_package_version=str(shared_projector.__version__),
        projector_api_version=int(shared_projector.PROJECTOR_API_VERSION),
        projector_compiler_cache_schema=str(shared_projector.COMPILER_CACHE_SCHEMA),
        method=normalize_method(method),
        profile=settings.profile,
        backend=settings.backend,
        include_literals=bool(include_literals),
    )


def projector_cache_identity(settings: ProjectorSettings) -> dict[str, object]:
    """Return path-free semantic fields for Exact dataset cache fingerprints."""

    return {
        "package_version": str(shared_projector.__version__),
        "api_version": int(shared_projector.PROJECTOR_API_VERSION),
        "compiler_cache_schema": str(shared_projector.COMPILER_CACHE_SCHEMA),
        "profile": settings.profile,
        "backend": settings.backend,
        "duplicates": "unique",
        "order": "canonical",
        "compatibility_state": "isolated",
        "core_package_version": str(pyowl_core.__version__),
        "core_api_version": list(pyowl_core.API_VERSION),
        "core_model_schema_version": int(pyowl_core.MODEL_SCHEMA_VERSION),
        "core_wire_format_version": list(pyowl_core.WIRE_FORMAT_VERSION),
    }


class SharedProjectionAdapter:
    """Thread-safe projection/cache facade retaining the exact shared snapshot."""

    def __init__(
        self,
        snapshot: OntologySnapshot,
        settings: ProjectorSettings | None = None,
        *,
        projector: Projector | None = None,
    ) -> None:
        if not isinstance(snapshot, OntologySnapshot):
            raise TypeError("snapshot must be a concrete pyowl_core.OntologySnapshot")
        self.snapshot = snapshot
        self.settings = settings or ProjectorSettings()
        self.projector = projector or Projector()
        self._cache: dict[ProjectionCacheKey, tuple[Edge, ...]] = {}
        self._lock = RLock()

    @property
    def cache_keys(self) -> tuple[ProjectionCacheKey, ...]:
        with self._lock:
            return tuple(self._cache)

    def edges(self, *, method: str = "owl2vecstar", include_literals: bool = False) -> list[Edge]:
        key = cache_key(
            self.snapshot,
            self.settings,
            method=method,
            include_literals=include_literals,
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                if key.method == "taxonomy":
                    rows = self.projector.project_taxonomy(
                        self.snapshot,
                        duplicates="unique",
                        order="canonical",
                        backend=self.settings.backend,
                    )
                else:
                    options = ProjectionOptions(
                        profile=self.settings.profile,
                        include_literals=key.include_literals,
                        duplicates="unique",
                        order="canonical",
                        compatibility_state="isolated",
                        backend=self.settings.backend,
                    )
                    rows = self.projector.project(self.snapshot, options=options)
                cached = tuple(Edge(row.source, row.relation, row.destination) for row in rows)
                self._cache[key] = cached
        return list(cached)


def project(
    source: OntologySnapshot | pyowl_core.SnapshotProvider,
    method: str = "owl2vecstar",
    include_literals: bool = False,
    *,
    settings: ProjectorSettings | None = None,
) -> list[Edge]:
    """One-minor compatibility function, without parser/path fallback."""

    snapshot = pyowl_core.coerce_snapshot(source)
    if not isinstance(snapshot, OntologySnapshot):
        raise TypeError("projection requires a concrete OntologySnapshot")
    return SharedProjectionAdapter(snapshot, settings).edges(
        method=method,
        include_literals=include_literals,
    )


__all__ = [
    "ProjectionCacheKey",
    "ProjectorBackend",
    "ProjectorSettings",
    "SharedProjectionAdapter",
    "cache_key",
    "normalize_method",
    "project",
    "projector_cache_identity",
]
