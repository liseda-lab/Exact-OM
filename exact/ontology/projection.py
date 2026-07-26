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
from pyowl_core import OntologyView

from exact.core.entities.graph import Edge
from exact.ontology.view_contract import retain_ontology_view

ProjectorBackend: TypeAlias = Literal["auto", "native", "python"]
ProjectionMethod: TypeAlias = Literal["owl2vecstar", "taxonomy"]

_ENCODED_BUFFER_WIDTHS: Mapping[str, int] = {
    "field_kinds": 1,
    "field_lengths": 8,
    "field_values": 8,
    "item_kinds": 1,
    "item_lengths": 8,
    "item_values": 8,
    "node_field_offsets": 8,
    "node_tags": 2,
    "root_ids": 4,
    "root_kinds": 1,
    "scalar_bytes": 1,
}
_ENCODED_COMPILER_HANDOFF_FIELDS = frozenset(
    {
        "buffer_widths",
        "descriptor_sha256",
        "model_schema",
        "schema_name",
        "schema_version",
    }
)


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
class EncodedContractIdentity:
    """Import-time public schema identity without requesting an encoded view."""

    core_schema_name: str | None
    core_schema_version: int | None
    core_descriptor_sha256: str | None
    projector_schema_name: str | None
    projector_schema_version: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "core": {
                "schema_name": self.core_schema_name,
                "schema_version": self.core_schema_version,
                "descriptor_sha256": self.core_descriptor_sha256,
            },
            "projector": {
                "schema_name": self.projector_schema_name,
                "schema_version": self.projector_schema_version,
            },
        }


def _optional_schema(
    name: object,
    schema: object,
    label: str,
) -> tuple[str | None, int | None]:
    if name is None and schema is None:
        return None, None
    if not isinstance(name, str) or not name:
        raise TypeError(f"{label} encoded schema name must be nonempty text")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < 1:
        raise TypeError(f"{label} encoded schema version must be a positive integer")
    return name, schema


def encoded_contract_identity() -> EncodedContractIdentity:
    """Return stable core/projector compiler inputs without capability negotiation."""

    core_view = getattr(pyowl_core, "EncodedStructuralView", object())
    core_name, core_schema = _optional_schema(
        getattr(core_view, "SCHEMA_NAME", None),
        getattr(core_view, "SCHEMA_VERSION", None),
        "core",
    )
    projector_name, projector_schema = _optional_schema(
        getattr(shared_projector, "ENCODED_SCHEMA_NAME", None),
        getattr(shared_projector, "ENCODED_SCHEMA_VERSION", None),
        "projector",
    )
    digest = getattr(pyowl_core, "ENCODED_STRUCTURAL_DESCRIPTOR_SHA256_V1", None)
    if digest is not None and (type(digest) is not bytes or len(digest) != 32):
        raise TypeError("core encoded descriptor digest must be exact bytes32")
    return EncodedContractIdentity(
        core_schema_name=core_name,
        core_schema_version=core_schema,
        core_descriptor_sha256=None if digest is None else digest.hex(),
        projector_schema_name=projector_name,
        projector_schema_version=projector_schema,
    )


def _validate_encoded_compiler_handoff(value: object) -> dict[str, object]:
    """Validate a consumer's exact public pyowl-core structural schema attestation."""

    if not isinstance(value, Mapping):
        raise TypeError("consumer compiler_handoff must be a mapping")
    fields = set(value)
    if not all(isinstance(name, str) for name in fields):
        raise TypeError("consumer compiler_handoff keys must be strings")
    if fields != _ENCODED_COMPILER_HANDOFF_FIELDS:
        raise ValueError("consumer compiler_handoff fields are incompatible")

    contract = encoded_contract_identity()
    expected_scalars: tuple[tuple[str, object], ...] = (
        ("schema_name", contract.core_schema_name),
        ("schema_version", contract.core_schema_version),
        ("model_schema", int(pyowl_core.MODEL_SCHEMA_VERSION)),
        ("descriptor_sha256", contract.core_descriptor_sha256),
    )
    if any(expected is None for _name, expected in expected_scalars):
        raise RuntimeError("core encoded structural contract is unavailable")
    for name, expected in expected_scalars:
        actual = value[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"consumer compiler_handoff {name} is incompatible; "
                f"expected {expected!r}, received {actual!r}"
            )

    widths = value["buffer_widths"]
    if not isinstance(widths, Mapping):
        raise TypeError("consumer compiler_handoff buffer_widths must be a mapping")
    width_names = set(widths)
    if not all(isinstance(name, str) for name in width_names):
        raise TypeError("consumer compiler_handoff buffer names must be strings")
    if width_names != set(_ENCODED_BUFFER_WIDTHS):
        raise ValueError("consumer compiler_handoff buffer widths are incompatible")
    for name, expected in _ENCODED_BUFFER_WIDTHS.items():
        actual = widths[name]
        if type(actual) is not int or actual != expected:
            raise ValueError(
                f"consumer compiler_handoff width for {name!r} is incompatible; "
                f"expected {expected}, received {actual!r}"
            )

    return {
        "buffer_widths": dict(sorted(_ENCODED_BUFFER_WIDTHS.items())),
        "descriptor_sha256": contract.core_descriptor_sha256,
        "model_schema": int(pyowl_core.MODEL_SCHEMA_VERSION),
        "schema_name": contract.core_schema_name,
        "schema_version": contract.core_schema_version,
    }


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
    encoded_contract: EncodedContractIdentity
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
    snapshot: OntologyView,
    settings: ProjectorSettings,
    *,
    method: str,
    include_literals: bool,
) -> ProjectionCacheKey:
    """Build the versioned semantic key required by the WP-M cache contract."""

    snapshot = retain_ontology_view(snapshot)
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
        encoded_contract=encoded_contract_identity(),
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
        "encoded_contract": encoded_contract_identity().as_dict(),
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
        snapshot: OntologyView,
        settings: ProjectorSettings | None = None,
        *,
        projector: Projector | None = None,
    ) -> None:
        self.snapshot = retain_ontology_view(snapshot)
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
    source: OntologyView | pyowl_core.SnapshotProvider,
    method: str = "owl2vecstar",
    include_literals: bool = False,
    *,
    settings: ProjectorSettings | None = None,
) -> list[Edge]:
    """One-minor compatibility function, without parser/path fallback."""

    snapshot = pyowl_core.coerce_snapshot(source)
    return SharedProjectionAdapter(snapshot, settings).edges(
        method=method,
        include_literals=include_literals,
    )


__all__ = [
    "EncodedContractIdentity",
    "ProjectionCacheKey",
    "ProjectorBackend",
    "ProjectorSettings",
    "SharedProjectionAdapter",
    "cache_key",
    "encoded_contract_identity",
    "normalize_method",
    "project",
    "projector_cache_identity",
]
