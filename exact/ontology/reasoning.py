"""Shared-snapshot hierarchy reasoner adapters.

The built-in asserted mode remains dependency free.  ELK and HermiT are imported
only when explicitly selected and receive the exact snapshot owned by the source.
Process isolation uses pyowl-core's verified wire format, never an OWL source path
or a pickled ontology graph.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from importlib import import_module
from importlib.metadata import PackageNotFoundError, entry_points, version
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Mapping, Protocol, cast, runtime_checkable

import pyowl_core
from pyowl_core import IRI, Class, OntologyView

from exact.ontology.projection import encoded_contract_identity
from exact.ontology.store import OWL_NOTHING, OWL_THING, OwlOntologySource

ReasonerFallback = Literal["error", "asserted"]
_OWL_BOUNDS = frozenset({OWL_THING, OWL_NOTHING})
_WORKER_SCHEMA_VERSION = 2
_PATH_FRAGMENT = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/)[^\s\"']+")
_INGESTION_PATHS = frozenset({"scalar-python", "scalar-native", "scalar-wire", "encoded-native"})
_HANDOFF_COUNTERS = frozenset(
    {
        "base_flattening_bytes",
        "encoded_buffer_count",
        "encoded_buffer_bytes",
        "encoded_zero_copy_buffers",
        "encoded_detached_buffer_count",
        "encoded_indexed_buffer_count",
        "encoded_staging_copy_bytes",
        "encoded_private_ir_bytes",
        "encoded_segment_count",
        "encoded_referenced_view_count",
        "encoded_posting_bytes",
        "encoded_compiler_gil_released",
        "materialized_scalar_rows",
        "parser_calls",
        "per_row_ffi_calls",
        "resolver_calls",
        "scalar_axiom_materializations",
        "scalar_term_materializations",
        "structural_copy_bytes",
        "wire_decoder_calls",
        "wire_encoder_calls",
    }
)
_HANDOFF_OPTIONAL_FIELDS = frozenset(
    {
        "compiler_cache_schema_version",
        "consumer_compile_seconds",
        "encoded_view_publication_seconds",
        "implementation_version",
        "ir_schema_version",
        "native_abi_version",
    }
)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _failure_reason(error: Exception) -> str:
    """Return bounded diagnostics without persisting a machine or wire path."""

    message = _PATH_FRAGMENT.sub("<path>", str(error))
    return f"{type(error).__name__}: {message[:1000]}"


class ReasonerUnavailableError(ImportError):
    """An explicitly selected optional reasoner is not installed."""


class ReasonerWorkerError(RuntimeError):
    """A verified-wire reasoner worker failed or returned an invalid result."""


class ReasonerWorkerTimeoutError(ReasonerWorkerError, TimeoutError):
    """A verified-wire reasoner worker exceeded its configured deadline."""


@dataclass(frozen=True, slots=True)
class ConsumerHandoffProvenance:
    """Bounded public compiler diagnostics without consumer-private state."""

    ingestion_path: str
    compiler_digest: str | None = None
    counters: tuple[tuple[str, int | bool], ...] = ()
    compiler_cache_schema_version: int | None = None
    ir_schema_version: int | None = None
    native_abi_version: int | str | None = None
    implementation_version: str | None = None
    consumer_compile_seconds: float | None = None
    encoded_view_publication_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.ingestion_path not in _INGESTION_PATHS:
            raise ValueError("reasoner ingestion path is not recognized")
        if self.compiler_digest is not None and (
            not isinstance(self.compiler_digest, str)
            or _SHA256_HEX.fullmatch(self.compiler_digest) is None
        ):
            raise ValueError("reasoner compiler digest must be lowercase SHA-256 or None")
        for name in ("compiler_cache_schema_version", "ir_schema_version"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise TypeError(f"reasoner {name} must be a positive integer or None")
        native_abi = self.native_abi_version
        if native_abi is not None and (
            isinstance(native_abi, bool)
            or not isinstance(native_abi, (int, str))
            or (isinstance(native_abi, int) and native_abi < 1)
            or (isinstance(native_abi, str) and not native_abi)
        ):
            raise TypeError("reasoner native ABI version must be nonempty or None")
        if self.implementation_version is not None and (
            not isinstance(self.implementation_version, str) or not self.implementation_version
        ):
            raise TypeError("reasoner implementation version must be nonempty text or None")
        for name in ("consumer_compile_seconds", "encoded_view_publication_seconds"):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not float or not math.isfinite(value) or value < 0.0
            ):
                raise TypeError(f"reasoner {name} must be a finite nonnegative float or None")
        if (
            self.ingestion_path != "encoded-native"
            and self.encoded_view_publication_seconds is not None
        ):
            raise ValueError("scalar reasoner ingestion claimed encoded-view publication")
        names = tuple(name for name, _value in self.counters)
        if names != tuple(sorted(set(names))) or any(
            name not in _HANDOFF_COUNTERS for name in names
        ):
            raise ValueError("reasoner handoff counters are not canonical")
        for name, value in self.counters:
            if name == "encoded_compiler_gil_released":
                if not isinstance(value, bool):
                    raise TypeError("reasoner GIL diagnostic must be boolean")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError("reasoner handoff counters must be nonnegative integers")

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "ingestion_path": self.ingestion_path,
            "compiler_digest": self.compiler_digest,
            "counters": dict(self.counters),
        }
        for name in (
            "compiler_cache_schema_version",
            "ir_schema_version",
            "native_abi_version",
            "implementation_version",
            "consumer_compile_seconds",
            "encoded_view_publication_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class ReasonerSettings:
    """Small, reasoner-neutral runtime surface owned by Exact."""

    backend: str = "auto"
    workers: int = 0
    timeout_seconds: float | None = None
    fallback: ReasonerFallback = "error"
    worker_wire: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("reasoner backend must be a nonempty string")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise TypeError("reasoner workers must be a nonnegative integer")
        if self.workers < 0:
            raise ValueError("reasoner workers must be a nonnegative integer")
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                raise TypeError("reasoner timeout must be a finite positive number or None")
            timeout = float(self.timeout_seconds)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("reasoner timeout must be a finite positive number or None")
            object.__setattr__(self, "timeout_seconds", timeout)
        if self.fallback not in {"error", "asserted"}:
            raise ValueError("reasoner fallback must be 'error' or 'asserted'")
        if not isinstance(self.worker_wire, bool):
            raise TypeError("worker_wire must be a boolean")

    @classmethod
    def from_value(cls, value: object = None) -> "ReasonerSettings":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="python")
        if not isinstance(value, Mapping):
            raise TypeError("reasoner settings must be a mapping")
        normalized = dict(value)
        if "timeout" in normalized:
            if "timeout_seconds" in normalized:
                raise ValueError("use only one of timeout or timeout_seconds")
            normalized["timeout_seconds"] = normalized.pop("timeout")
        unknown = sorted(
            set(map(str, normalized))
            - {"backend", "workers", "timeout_seconds", "fallback", "worker_wire"}
        )
        if unknown:
            raise ValueError(f"unknown reasoner setting(s): {', '.join(unknown)}")
        return cls(
            backend=str(normalized.get("backend", "auto")),
            workers=cast(int, normalized.get("workers", 0)),
            timeout_seconds=cast(float | None, normalized.get("timeout_seconds")),
            fallback=cast(ReasonerFallback, normalized.get("fallback", "error")),
            worker_wire=cast(bool, normalized.get("worker_wire", False)),
        )


@dataclass(frozen=True, slots=True)
class ReasonerProvenance:
    """Path-free semantic identity and runtime-selection diagnostics."""

    requested_reasoner: str
    effective_reasoner: str
    reasoner_package_version: str
    requested_backend: str
    effective_backend: str
    backend_implementation_version: str
    backend_fallback_reason: str | None
    semantic_fallback_reason: str | None
    failure_reason: str | None
    workers: int
    timeout_seconds: float | None
    fallback_policy: ReasonerFallback
    timed_out: bool
    worker_wire: bool
    verified_wire: bool
    structural_fingerprint: str
    logical_fingerprint: str
    signature_fingerprint: str
    consumer_handoff: ConsumerHandoffProvenance | None = None
    mmap_verified: bool = False
    owl_parse_count: int | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "selection": {
                "requested": self.requested_reasoner,
                "effective": self.effective_reasoner,
                "package_version": self.reasoner_package_version,
            },
            "backend": {
                "requested": self.requested_backend,
                "effective": self.effective_backend,
                "implementation_version": self.backend_implementation_version,
                "fallback_reason": self.backend_fallback_reason,
            },
            "options": {
                "workers": self.workers,
                "timeout_seconds": self.timeout_seconds,
                "fallback": self.fallback_policy,
                "worker_wire": self.worker_wire,
            },
            "fallback_reason": self.semantic_fallback_reason,
            "failure_reason": self.failure_reason,
            "timed_out": self.timed_out,
            "verified_wire": self.verified_wire,
            "mmap_verified": self.mmap_verified,
            "owl_parse_count": self.owl_parse_count,
            "core": {
                "package_version": str(pyowl_core.__version__),
                "api_version": list(pyowl_core.API_VERSION),
                "model_schema_version": int(pyowl_core.MODEL_SCHEMA_VERSION),
                "wire_format_version": list(pyowl_core.WIRE_FORMAT_VERSION),
                "adapter_protocol_version": int(pyowl_core.ADAPTER_PROTOCOL_VERSION),
            },
            "fingerprints": {
                "structural": self.structural_fingerprint,
                "logical": self.logical_fingerprint,
                "signature": self.signature_fingerprint,
            },
        }
        if self.consumer_handoff is not None:
            result["consumer_handoff"] = self.consumer_handoff.as_dict()
        return result


@runtime_checkable
class ReasonerProtocol(Protocol):
    def direct_parents(self, iri: str) -> list[str]: ...

    def direct_children(self, iri: str) -> list[str]: ...

    def ancestors(self, iri: str) -> set[str]: ...

    def descendants(self, iri: str) -> set[str]: ...


def _fingerprint(value: object) -> str:
    return f"{getattr(value, 'algorithm')}:{getattr(value, 'schema')}:" f"{getattr(value, 'hex')}"


def _base_provenance(
    snapshot: OntologyView,
    settings: ReasonerSettings,
    *,
    requested_reasoner: str,
    effective_reasoner: str,
    package_version: str,
    effective_backend: str,
    implementation_version: str,
    backend_fallback_reason: str | None = None,
    verified_wire: bool = False,
) -> ReasonerProvenance:
    return ReasonerProvenance(
        requested_reasoner=requested_reasoner,
        effective_reasoner=effective_reasoner,
        reasoner_package_version=package_version,
        requested_backend=settings.backend,
        effective_backend=effective_backend,
        backend_implementation_version=implementation_version,
        backend_fallback_reason=backend_fallback_reason,
        semantic_fallback_reason=None,
        failure_reason=None,
        workers=settings.workers,
        timeout_seconds=settings.timeout_seconds,
        fallback_policy=settings.fallback,
        timed_out=False,
        worker_wire=settings.worker_wire,
        verified_wire=verified_wire,
        structural_fingerprint=_fingerprint(snapshot.structural_fingerprint),
        logical_fingerprint=_fingerprint(snapshot.logical_fingerprint),
        signature_fingerprint=_fingerprint(snapshot.signature_fingerprint),
    )


def _distribution_version(module: object, distribution: str) -> str:
    candidate = getattr(module, "__version__", None)
    if isinstance(candidate, str) and candidate:
        return candidate
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _consumer_handoff_from_record(value: object) -> ConsumerHandoffProvenance | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("reasoner consumer_handoff must be a mapping")
    fields = set(value)
    required = {"ingestion_path", "compiler_digest", "counters"}
    if not required.issubset(fields) or not fields.issubset(required | _HANDOFF_OPTIONAL_FIELDS):
        raise ValueError("reasoner consumer_handoff fields are incompatible")
    ingestion_path = value["ingestion_path"]
    compiler_digest = value["compiler_digest"]
    raw_counters = value["counters"]
    if not isinstance(ingestion_path, str):
        raise TypeError("reasoner ingestion path must be text")
    if compiler_digest is not None and not isinstance(compiler_digest, str):
        raise TypeError("reasoner compiler digest must be text or None")
    if not isinstance(raw_counters, Mapping) or any(
        not isinstance(name, str) for name in raw_counters
    ):
        raise TypeError("reasoner handoff counters must be a string-keyed mapping")
    return ConsumerHandoffProvenance(
        ingestion_path=ingestion_path,
        compiler_digest=compiler_digest,
        counters=tuple(sorted(cast(Mapping[str, int | bool], raw_counters).items())),
        compiler_cache_schema_version=cast(int | None, value.get("compiler_cache_schema_version")),
        ir_schema_version=cast(int | None, value.get("ir_schema_version")),
        native_abi_version=cast(int | str | None, value.get("native_abi_version")),
        implementation_version=cast(str | None, value.get("implementation_version")),
        consumer_compile_seconds=cast(float | None, value.get("consumer_compile_seconds")),
        encoded_view_publication_seconds=cast(
            float | None, value.get("encoded_view_publication_seconds")
        ),
    )


def _consumer_handoff(reasoner: object) -> ConsumerHandoffProvenance | None:
    diagnostics = getattr(reasoner, "diagnostics", None)
    if not callable(diagnostics):
        return None
    values = diagnostics()
    if not isinstance(values, Mapping):
        raise TypeError("reasoner diagnostics must be a mapping")
    ingestion_path = values.get("ingestion_path")
    if ingestion_path is None:
        return None
    if not isinstance(ingestion_path, str):
        raise TypeError("reasoner ingestion_path diagnostic must be text")
    compiler_digest = values.get("compiler_digest")
    if compiler_digest is not None and not isinstance(compiler_digest, str):
        raise TypeError("reasoner compiler_digest diagnostic must be text or None")
    counters = {
        name: cast(int | bool, values[name]) for name in sorted(_HANDOFF_COUNTERS) if name in values
    }
    record: dict[str, object] = {
        "ingestion_path": ingestion_path,
        "compiler_digest": compiler_digest,
        "counters": counters,
    }
    for name in sorted(_HANDOFF_OPTIONAL_FIELDS):
        if name in values:
            record[name] = values[name]
    return _consumer_handoff_from_record(record)


def _public_positive_int(module: object, name: str) -> int | None:
    value = getattr(module, name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"public reasoner {name} must be a positive integer")
    return value


def _public_nonempty_text(module: object, name: str) -> str | None:
    value = getattr(module, name, None)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"public reasoner {name} must be nonempty text")
    return value


def _reasoner_compiler_identity(name: str, package_version: str) -> dict[str, object]:
    result: dict[str, object] = {
        "compiler_cache_schema_version": None,
        "ir_schema_version": None,
        "native_abi_version": None,
        "compatibility_id": None,
    }
    if package_version == "not-installed":
        return result
    if name == "elk":
        indexing = import_module("pyelk.indexing")
        result["compiler_cache_schema_version"] = _public_positive_int(
            indexing, "COMPILER_SCHEMA_VERSION"
        )
        result["compatibility_id"] = _public_nonempty_text(indexing, "ELK_COMPATIBILITY_ID")
    elif name == "hermit":
        hermit = import_module("pyhermit")
        result["compiler_cache_schema_version"] = _public_positive_int(
            hermit, "COMPILER_CACHE_SCHEMA_VERSION"
        )
        result["ir_schema_version"] = _public_positive_int(hermit, "COMPILED_IR_SCHEMA_VERSION")
        result["native_abi_version"] = _public_positive_int(hermit, "NATIVE_ABI_VERSION")
    return result


def reasoner_cache_identity(
    name: str, settings: ReasonerSettings | Mapping[str, object] | None = None
) -> dict[str, object]:
    """Return import-light version/options fields for dataset cache partitioning."""

    normalized = str(name).strip().lower()
    selected = ReasonerSettings.from_value(settings)
    distribution = {"elk": "pyelk-reasoner", "hermit": "pyHermiT"}.get(normalized)
    if distribution is None:
        package_version = str(pyowl_core.__version__) if normalized == "asserted" else "plugin"
    else:
        try:
            package_version = version(distribution)
        except PackageNotFoundError:
            package_version = "not-installed"
    return {
        "selection": normalized,
        "package_version": package_version,
        "backend": selected.backend,
        "workers": selected.workers,
        "timeout_seconds": selected.timeout_seconds,
        "fallback": selected.fallback,
        "worker_wire": selected.worker_wire,
        "core_package_version": str(pyowl_core.__version__),
        "core_api_version": list(pyowl_core.API_VERSION),
        "core_model_schema_version": int(pyowl_core.MODEL_SCHEMA_VERSION),
        "core_wire_format_version": list(pyowl_core.WIRE_FORMAT_VERSION),
        "core_adapter_protocol_version": int(pyowl_core.ADAPTER_PROTOCOL_VERSION),
        "encoded_contract": encoded_contract_identity().as_dict()["core"],
        "consumer_compiler": _reasoner_compiler_identity(normalized, package_version),
        "worker_schema_version": _WORKER_SCHEMA_VERSION,
    }


def _optional_module(module_name: str, extra_label: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise ReasonerUnavailableError(
            f"Reasoner {extra_label!r} requires the optional reasoning dependencies; "
            'install Exact-OM with the "reasoning" extra.'
        ) from error


def _validate_backend(name: str, backend: str) -> None:
    choices = {
        "elk": {"auto", "python", "rust"},
        "hermit": {"auto", "python", "native", "verify"},
    }[name]
    if backend not in choices:
        rendered = ", ".join(sorted(choices))
        raise ValueError(f"{name} backend must be one of: {rendered}")


class AssertedHierarchyReasoner:
    """The default reasoner backed by shared-core asserted structural views."""

    def __init__(
        self,
        store: OwlOntologySource,
        *,
        _requested_reasoner: str = "asserted",
        _settings: ReasonerSettings | None = None,
        _fallback_reason: str | None = None,
        _timed_out: bool = False,
    ) -> None:
        self.store = store
        self.snapshot = store.owl_snapshot()
        settings = _settings or ReasonerSettings()
        provenance = _base_provenance(
            self.snapshot,
            settings,
            requested_reasoner=_requested_reasoner,
            effective_reasoner="asserted",
            package_version=str(pyowl_core.__version__),
            effective_backend="structural",
            implementation_version=str(pyowl_core.__version__),
        )
        self._provenance = replace(
            provenance,
            semantic_fallback_reason=_fallback_reason,
            failure_reason=_fallback_reason,
            timed_out=_timed_out,
        )

    @property
    def ontology(self) -> OntologyView:
        return self.snapshot

    @property
    def provenance(self) -> dict[str, object]:
        return self._provenance.as_dict()

    def direct_parents(self, iri: str) -> list[str]:
        return self.store.hierarchy.direct_parents(iri)

    def direct_children(self, iri: str) -> list[str]:
        return self.store.hierarchy.direct_children(iri)

    def ancestors(self, iri: str) -> set[str]:
        return self.store.hierarchy.ancestors(iri)

    def descendants(self, iri: str) -> set[str]:
        return self.store.hierarchy.descendants(iri)

    def close(self) -> None:
        """Match optional-adapter lifecycle without owning shared state."""


class _InferredHierarchyReasoner:
    """Common bounded conversion, explicit fallback, and lifecycle behavior."""

    def __init__(self, store: OwlOntologySource, settings: ReasonerSettings) -> None:
        self.store = store
        self.snapshot = store.owl_snapshot()
        self.settings = settings
        self._asserted = AssertedHierarchyReasoner(store)
        self._fallback_active = False
        self._lock = RLock()
        self._provenance: ReasonerProvenance

    @property
    def ontology(self) -> OntologyView:
        return self.snapshot

    @property
    def provenance(self) -> dict[str, object]:
        return self._provenance.as_dict()

    def _query(self, iri: str, *, upward: bool, direct: bool) -> set[str]:
        raise NotImplementedError

    def _handled_error(self, error: Exception) -> bool:
        return isinstance(error, TimeoutError)

    def _activate_fallback(self, error: Exception) -> None:
        reason = _failure_reason(error)
        self._fallback_active = True
        self._provenance = replace(
            self._provenance,
            effective_reasoner="asserted",
            semantic_fallback_reason=reason,
            failure_reason=reason,
            timed_out=isinstance(error, TimeoutError),
        )

    def _relatives(self, iri: str, *, upward: bool, direct: bool) -> set[str]:
        with self._lock:
            if not self._fallback_active:
                try:
                    return self._query(str(iri), upward=upward, direct=direct)
                except Exception as error:
                    if not self._handled_error(error):
                        raise
                    self._provenance = replace(
                        self._provenance,
                        failure_reason=_failure_reason(error),
                        timed_out=isinstance(error, TimeoutError),
                    )
                    if self.settings.fallback != "asserted":
                        raise
                    self._activate_fallback(error)
        if upward and direct:
            return set(self._asserted.direct_parents(iri))
        if not upward and direct:
            return set(self._asserted.direct_children(iri))
        return self._asserted.ancestors(iri) if upward else self._asserted.descendants(iri)

    def direct_parents(self, iri: str) -> list[str]:
        return sorted(self._relatives(iri, upward=True, direct=True))

    def direct_children(self, iri: str) -> list[str]:
        return sorted(self._relatives(iri, upward=False, direct=True))

    def ancestors(self, iri: str) -> set[str]:
        return self._relatives(iri, upward=True, direct=False)

    def descendants(self, iri: str) -> set[str]:
        return self._relatives(iri, upward=False, direct=False)


def _class_iri(member: object) -> str:
    iri = getattr(member, "iri", None)
    value = getattr(iri, "value", None)
    if not isinstance(value, str) or not value:
        raise TypeError("reasoner hierarchy members must be named pyowl-core classes")
    return value


def _elk_query(reasoner: Any, iri: str, *, upward: bool, direct: bool) -> set[str]:
    expression = Class(IRI(iri))
    result = (
        reasoner.superclasses(expression, direct=direct)
        if upward
        else reasoner.subclasses(expression, direct=direct)
    )
    nodes = result.require_complete()
    return {
        value
        for node in nodes
        for member in node.members
        if (value := _class_iri(member)) not in _OWL_BOUNDS
    }


def _create_elk(
    snapshot: OntologyView, settings: ReasonerSettings
) -> tuple[Any, ReasonerProvenance, type[Exception]]:
    _validate_backend("elk", settings.backend)
    pyelk = _optional_module("pyelk", "elk")
    config = pyelk.ReasonerConfig(
        backend=settings.backend,
        workers=settings.workers,
        allow_fresh_entities=True,
        unsupported="ignore",
        allow_incomplete_imports=False,
    )
    reasoner = pyelk.Reasoner(snapshot, config=config)
    if reasoner.ontology is not snapshot:
        reasoner.close()
        raise RuntimeError("pyELK did not retain Exact's shared snapshot by identity")
    backend = reasoner.backend
    provenance = _base_provenance(
        snapshot,
        settings,
        requested_reasoner="elk",
        effective_reasoner="elk",
        package_version=_distribution_version(pyelk, "pyelk-reasoner"),
        effective_backend=str(backend.name),
        implementation_version=str(backend.implementation_version),
        backend_fallback_reason=backend.fallback_reason,
    )
    provenance = replace(provenance, consumer_handoff=_consumer_handoff(reasoner))
    error_type = cast(type[Exception], import_module("pyelk.exceptions").PyElkError)
    return reasoner, provenance, error_type


class ElkHierarchyReasoner(_InferredHierarchyReasoner):
    """Exact's narrow adapter over the public pyELK facade."""

    def __init__(self, store: OwlOntologySource, settings: ReasonerSettings | None = None) -> None:
        selected = settings or ReasonerSettings()
        super().__init__(store, selected)
        self._reasoner, self._provenance, self._error_type = _create_elk(self.snapshot, selected)

    @property
    def shared_reasoner(self) -> object:
        """Identity diagnostic without exposing private reasoner IR."""

        return self._reasoner

    def _handled_error(self, error: Exception) -> bool:
        return isinstance(error, (TimeoutError, self._error_type))

    def _query(self, iri: str, *, upward: bool, direct: bool) -> set[str]:
        return _elk_query(self._reasoner, iri, upward=upward, direct=direct)

    def close(self) -> None:
        self._reasoner.close()

    def __enter__(self) -> "ElkHierarchyReasoner":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _hermit_fallback_reason(pyhermit: Any, settings: ReasonerSettings, backend: Any) -> str | None:
    if settings.backend != "auto" or backend.name != "python":
        return None
    status = pyhermit.backend_info()
    environment = status.environment_request
    if environment == "python":
        return "PYHERMIT_BACKEND selected the Python backend"
    if not status.native.available:
        return status.native.reason or "native backend unavailable"
    return None


def _hermit_query(reasoner: Any, iri: str, *, upward: bool, direct: bool) -> set[str]:
    expression = Class(IRI(iri))
    groups = (
        reasoner.superclasses(expression, direct=direct)
        if upward
        else reasoner.subclasses(expression, direct=direct)
    )
    return {
        value
        for group in groups
        for member in group
        if (value := _class_iri(member)) not in _OWL_BOUNDS
    }


def _create_hermit(
    snapshot: OntologyView, settings: ReasonerSettings
) -> tuple[Any, ReasonerProvenance, type[Exception]]:
    _validate_backend("hermit", settings.backend)
    pyhermit = _optional_module("pyhermit", "hermit")
    config = pyhermit.ReasonerConfig(
        backend=settings.backend,
        timeout=settings.timeout_seconds,
        workers=settings.workers,
    )
    reasoner = pyhermit.Reasoner(snapshot, config=config)
    if reasoner.ontology is not snapshot:
        reasoner.dispose()
        raise RuntimeError("pyHermiT did not retain Exact's shared snapshot by identity")
    backend = reasoner.backend
    provenance = _base_provenance(
        snapshot,
        settings,
        requested_reasoner="hermit",
        effective_reasoner="hermit",
        package_version=_distribution_version(pyhermit, "pyHermiT"),
        effective_backend=str(backend.name),
        implementation_version=str(backend.implementation_version),
        backend_fallback_reason=_hermit_fallback_reason(pyhermit, settings, backend),
    )
    provenance = replace(provenance, consumer_handoff=_consumer_handoff(reasoner))
    error_type = cast(type[Exception], pyhermit.PyHermiTError)
    return reasoner, provenance, error_type


class HermitHierarchyReasoner(_InferredHierarchyReasoner):
    """Exact's narrow adapter over the public pyHermiT facade."""

    def __init__(self, store: OwlOntologySource, settings: ReasonerSettings | None = None) -> None:
        selected = settings or ReasonerSettings()
        super().__init__(store, selected)
        self._reasoner, self._provenance, self._error_type = _create_hermit(self.snapshot, selected)

    @property
    def shared_reasoner(self) -> object:
        """Identity diagnostic without exposing private reasoner IR."""

        return self._reasoner

    def _handled_error(self, error: Exception) -> bool:
        return isinstance(error, (TimeoutError, self._error_type))

    def _query(self, iri: str, *, upward: bool, direct: bool) -> set[str]:
        return _hermit_query(self._reasoner, iri, upward=upward, direct=direct)

    def close(self) -> None:
        self._reasoner.dispose()

    def __enter__(self) -> "HermitHierarchyReasoner":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _worker_payload(
    reasoner_name: str,
    snapshot: OntologyView,
    settings: ReasonerSettings,
    *,
    iri: str,
    upward: bool,
    direct: bool,
) -> dict[str, object]:
    """Run one bounded query over an already-verified worker snapshot."""

    features = snapshot.capabilities.features
    if not {"wire-verified", "mmap-snapshot"}.issubset(features):
        raise ReasonerWorkerError("reasoner worker requires a verified mmap snapshot")
    worker_settings = replace(settings, worker_wire=False)
    if reasoner_name == "elk":
        reasoner, provenance, _ = _create_elk(snapshot, worker_settings)
        try:
            values = _elk_query(reasoner, iri, upward=upward, direct=direct)
        finally:
            reasoner.close()
    elif reasoner_name == "hermit":
        reasoner, provenance, _ = _create_hermit(snapshot, worker_settings)
        try:
            values = _hermit_query(reasoner, iri, upward=upward, direct=direct)
        finally:
            reasoner.dispose()
    else:  # pragma: no cover - CLI validation guards this branch.
        raise ValueError("wire worker reasoner must be elk or hermit")
    runtime: dict[str, object] = {
        "package_version": provenance.reasoner_package_version,
        "backend": provenance.effective_backend,
        "implementation_version": provenance.backend_implementation_version,
        "backend_fallback_reason": provenance.backend_fallback_reason,
    }
    if provenance.consumer_handoff is not None:
        runtime["consumer_handoff"] = provenance.consumer_handoff.as_dict()
    return {
        "schema_version": _WORKER_SCHEMA_VERSION,
        "values": sorted(values),
        "runtime": runtime,
        "fingerprints": {
            "structural": provenance.structural_fingerprint,
            "logical": provenance.logical_fingerprint,
            "signature": provenance.signature_fingerprint,
        },
        "verified_wire": True,
        "mmap_verified": True,
        "owl_parse_count": 0,
    }


def _run_wire_worker(
    reasoner_name: str,
    snapshot: OntologyView,
    settings: ReasonerSettings,
    *,
    wire_path: Path,
    iri: str,
    upward: bool,
    direct: bool,
) -> tuple[set[str], dict[str, object]]:
    command = [
        sys.executable,
        "-m",
        "exact.ontology._reasoner_worker",
        "--wire",
        os.fspath(wire_path),
        "--reasoner",
        reasoner_name,
        "--backend",
        settings.backend,
        "--workers",
        str(settings.workers),
        "--iri",
        iri,
        "--direction",
        "up" if upward else "down",
    ]
    if direct:
        command.append("--direct")
    if settings.timeout_seconds is not None:
        command.extend(("--timeout", repr(settings.timeout_seconds)))
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=settings.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ReasonerWorkerTimeoutError(
            f"{reasoner_name} wire worker exceeded {settings.timeout_seconds} seconds"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-4000:] or "worker exited without diagnostics"
        raise ReasonerWorkerError(f"{reasoner_name} wire worker failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReasonerWorkerError("reasoner worker returned invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != _WORKER_SCHEMA_VERSION:
        raise ReasonerWorkerError("reasoner worker schema version is incompatible")
    if payload.get("verified_wire") is not True:
        raise ReasonerWorkerError("reasoner worker did not verify its core wire input")
    if payload.get("mmap_verified") is not True:
        raise ReasonerWorkerError("reasoner worker did not retain a mapped core owner")
    if payload.get("owl_parse_count") != 0:
        raise ReasonerWorkerError("reasoner worker parsed an OWL document")
    expected = {
        "structural": _fingerprint(snapshot.structural_fingerprint),
        "logical": _fingerprint(snapshot.logical_fingerprint),
        "signature": _fingerprint(snapshot.signature_fingerprint),
    }
    if payload.get("fingerprints") != expected:
        raise ReasonerWorkerError("reasoner worker snapshot fingerprint mismatch")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ReasonerWorkerError("reasoner worker omitted runtime provenance")
    values = payload.get("values")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ReasonerWorkerError("reasoner worker returned malformed query values")
    if values != sorted(set(values)):
        raise ReasonerWorkerError("reasoner worker query values are not canonical")
    return set(values), runtime


class WorkerWireHierarchyReasoner(_InferredHierarchyReasoner):
    """Isolated classifier consuming a verified pyowl-core wire snapshot."""

    def __init__(
        self,
        reasoner_name: Literal["elk", "hermit"],
        store: OwlOntologySource,
        settings: ReasonerSettings,
    ) -> None:
        selected = replace(settings, worker_wire=True)
        _validate_backend(reasoner_name, selected.backend)
        super().__init__(store, selected)
        self.reasoner_name = reasoner_name
        module = _optional_module("pyelk" if reasoner_name == "elk" else "pyhermit", reasoner_name)
        distribution = "pyelk-reasoner" if reasoner_name == "elk" else "pyHermiT"
        self._provenance = _base_provenance(
            self.snapshot,
            selected,
            requested_reasoner=reasoner_name,
            effective_reasoner=reasoner_name,
            package_version=_distribution_version(module, distribution),
            effective_backend="worker-pending",
            implementation_version="worker-pending",
            verified_wire=False,
        )
        self._wire_directory = tempfile.TemporaryDirectory(prefix="exact-reasoner-")
        self._wire_path = Path(self._wire_directory.name) / "ontology.pyocore"
        self._wire_path.write_bytes(pyowl_core.encode_snapshot(self.snapshot))
        self._closed = False

    def _handled_error(self, error: Exception) -> bool:
        return isinstance(error, (ReasonerWorkerError, TimeoutError))

    def _query(self, iri: str, *, upward: bool, direct: bool) -> set[str]:
        if self._closed:
            raise ReasonerWorkerError("reasoner wire worker adapter is closed")
        values, runtime = _run_wire_worker(
            self.reasoner_name,
            self.snapshot,
            self.settings,
            wire_path=self._wire_path,
            iri=iri,
            upward=upward,
            direct=direct,
        )
        self._provenance = replace(
            self._provenance,
            reasoner_package_version=str(runtime.get("package_version", "unknown")),
            effective_backend=str(runtime.get("backend", "unknown")),
            backend_implementation_version=str(runtime.get("implementation_version", "unknown")),
            backend_fallback_reason=cast(str | None, runtime.get("backend_fallback_reason")),
            consumer_handoff=_consumer_handoff_from_record(runtime.get("consumer_handoff")),
            verified_wire=True,
            mmap_verified=True,
            owl_parse_count=0,
        )
        return values

    def close(self) -> None:
        """Release the reusable encoded wire artifact exactly once."""

        if self._closed:
            return
        self._closed = True
        self._wire_directory.cleanup()

    def __enter__(self) -> "WorkerWireHierarchyReasoner":
        if self._closed:
            raise ReasonerWorkerError("reasoner wire worker adapter is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _reasoner_entry_points() -> list[Any]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group="exact.reasoners"))
    return list(
        cast(Any, discovered).get("exact.reasoners", ())
    )  # pragma: no cover - Python <3.10.


def load_reasoner(
    name: str,
    store: OwlOntologySource,
    *,
    settings: ReasonerSettings | Mapping[str, object] | None = None,
    backend: str | None = None,
    workers: int | None = None,
    timeout: float | None = None,
    fallback: ReasonerFallback | None = None,
    worker_wire: bool | None = None,
) -> ReasonerProtocol:
    """Load an explicit built-in/shared adapter or an ``exact.reasoners`` plugin.

    A pyELK timeout automatically selects the verified-wire worker because pyELK's
    in-process facade has no cancellable deadline.  HermiT receives its timeout in
    process unless ``worker_wire`` is explicitly requested.
    """

    if not isinstance(store, OwlOntologySource):
        raise TypeError("store must be OwlOntologySource")
    selected = ReasonerSettings.from_value(settings)
    selected = ReasonerSettings(
        backend=selected.backend if backend is None else backend,
        workers=selected.workers if workers is None else workers,
        timeout_seconds=selected.timeout_seconds if timeout is None else timeout,
        fallback=selected.fallback if fallback is None else fallback,
        worker_wire=selected.worker_wire if worker_wire is None else worker_wire,
    )

    normalized = str(name).strip().lower()
    if normalized == "asserted":
        return AssertedHierarchyReasoner(store, _settings=selected)
    try:
        if normalized == "elk":
            if selected.worker_wire or selected.timeout_seconds is not None:
                return WorkerWireHierarchyReasoner("elk", store, selected)
            return ElkHierarchyReasoner(store, selected)
        if normalized == "hermit":
            if selected.worker_wire:
                return WorkerWireHierarchyReasoner("hermit", store, selected)
            return HermitHierarchyReasoner(store, selected)
    except Exception as error:
        if selected.fallback != "asserted":
            raise
        return AssertedHierarchyReasoner(
            store,
            _requested_reasoner=normalized,
            _settings=selected,
            _fallback_reason=_failure_reason(error),
            _timed_out=isinstance(error, TimeoutError),
        )

    plugins = {plugin.name: plugin for plugin in _reasoner_entry_points()}
    if normalized not in plugins:
        installed = ", ".join(sorted(plugins)) or "none"
        raise ValueError(
            f"Unknown reasoner {name!r}. Built in: asserted, elk, hermit. "
            f"Installed plugins: {installed}."
        )

    factory = plugins[normalized].load()
    if hasattr(factory, "create"):
        reasoner = factory.create(store)
    elif callable(factory):
        reasoner = factory(store)
    else:
        reasoner = factory
    required = ("direct_parents", "direct_children", "ancestors", "descendants")
    missing = [method for method in required if not callable(getattr(reasoner, method, None))]
    if missing:
        raise TypeError(f"Reasoner plugin {name!r} does not implement: {', '.join(missing)}")
    return cast(ReasonerProtocol, reasoner)


__all__ = [
    "AssertedHierarchyReasoner",
    "ElkHierarchyReasoner",
    "HermitHierarchyReasoner",
    "ReasonerFallback",
    "ReasonerProtocol",
    "ReasonerProvenance",
    "ReasonerSettings",
    "ReasonerUnavailableError",
    "ReasonerWorkerError",
    "ReasonerWorkerTimeoutError",
    "WorkerWireHierarchyReasoner",
    "load_reasoner",
    "reasoner_cache_identity",
]
