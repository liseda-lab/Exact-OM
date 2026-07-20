"""Path-free release provenance for Exact's shared OWL stack."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

import pyowl2vec_star_projector as shared_projector
import pyowl_core
from pyowl2vec_star_projector import ProjectionOptions, select_backend
from pyowl_core import OntologySnapshot, OntologyView

from exact.ontology.projection import (
    ProjectorSettings,
    encoded_contract_identity,
    projector_cache_identity,
)
from exact.ontology.view_contract import retain_ontology_view

ONTOLOGY_STACK_PROVENANCE_SCHEMA = 1
CONSUMER_HANDOFF_PROVENANCE_SCHEMA = 1
_PATH_FRAGMENT = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/)[^\s\"']+")
_OBJECT_ID = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")
_REASONER_INGESTION_PATHS = frozenset(
    {"scalar-python", "scalar-native", "scalar-wire", "encoded-native"}
)
_PROJECTOR_INGESTION_PATHS = frozenset({"scalar-python", "scalar-native", "encoded-native"})
_PROJECTOR_HANDOFF_COUNTERS = frozenset(
    {
        "encoded_buffer_bytes",
        "encoded_buffer_count",
        "encoded_compiler_gil_released",
        "encoded_detached_buffer_count",
        "encoded_indexed_buffer_count",
        "encoded_posting_bytes",
        "encoded_referenced_view_count",
        "encoded_segment_count",
        "encoded_staging_copy_bytes",
        "encoded_zero_copy_buffers",
        "materialized_scalar_rows",
    }
)
_REASONER_HANDOFF_COUNTERS = frozenset(
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
_REASONER_HANDOFF_OPTIONAL_FIELDS = frozenset(
    {
        "compiler_cache_schema_version",
        "consumer_compile_seconds",
        "encoded_view_publication_seconds",
        "implementation_version",
        "ir_schema_version",
        "native_abi_version",
    }
)
_ENCODED_ONLY_COUNTER_DEFAULTS: Mapping[str, int | bool] = {
    "encoded_buffer_count": 0,
    "encoded_buffer_bytes": 0,
    "encoded_zero_copy_buffers": 0,
    "encoded_detached_buffer_count": 0,
    "encoded_indexed_buffer_count": 0,
    "encoded_staging_copy_bytes": 0,
    "encoded_private_ir_bytes": 0,
    "encoded_segment_count": 0,
    "encoded_referenced_view_count": 0,
    "encoded_posting_bytes": 0,
    "encoded_compiler_gil_released": False,
}
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _safe_value(value: object) -> object:
    """Recursively redact machine paths, URL payloads, and Python object IDs."""

    if isinstance(value, str):
        return _OBJECT_ID.sub("<object-id>", _PATH_FRAGMENT.sub("<path>", value))
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _fingerprint(value: object) -> str:
    return f"{getattr(value, 'algorithm')}:{getattr(value, 'schema')}:" f"{getattr(value, 'hex')}"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _diagnostic_summary(snapshot: OntologyView) -> dict[str, object]:
    diagnostics = tuple(snapshot.report.diagnostics)
    encoded = json.dumps(
        [item.to_dict() for item in diagnostics],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    counts = Counter((item.severity.value, item.code) for item in diagnostics)
    return {
        "digest": _sha256(encoded),
        "count": len(diagnostics),
        "codes": [
            {"severity": severity, "code": code, "count": count}
            for (severity, code), count in sorted(counts.items())
        ],
    }


def _closure_provenance(snapshot: OntologyView) -> dict[str, object]:
    if isinstance(snapshot, pyowl_core.OntologyOverlay):
        overlay_result = _closure_provenance(snapshot.base)
        overlay_result.update(
            {
                "view_kind": "overlay",
                "complete": snapshot.is_complete,
                "overlay_depth": snapshot.depth,
                "overlay_provenance_sha256": snapshot.requested_delta.provenance_digest.hex(),
            }
        )
        return overlay_result
    if not isinstance(snapshot, OntologySnapshot):
        features = set(snapshot.capabilities.features)
        kind = (
            "composite"
            if "ontology-composite" in features
            else "overlay" if "ontology-overlay" in features else "view"
        )
        view_result: dict[str, object] = {
            "view_kind": kind,
            "complete": snapshot.is_complete,
            "manifest_available": False,
        }
        if isinstance(snapshot, pyowl_core.OntologyComposite):
            view_result.update(
                {
                    "member_count": len(snapshot.members),
                    "member_roles": dict(sorted(snapshot.member_roles.items())),
                    "composition_provenance_sha256": (snapshot.composition_provenance_digest.hex()),
                }
            )
        return view_result
    manifest = snapshot.import_manifest
    status_counts = Counter(edge.status.value for edge in manifest.edges)
    documents = [
        {
            "document_key": record.document_key,
            "source_sha256": record.source_sha256.hex(),
            "document_fingerprint": _fingerprint(record.document_fingerprint),
            "format": record.format.value,
            "status": record.status.value,
            "source_bytes": document.provenance.byte_length,
        }
        for record, document in snapshot.iter_documents()
    ]
    return {
        "view_kind": "snapshot",
        "manifest_available": True,
        "import_manifest_sha256": _sha256(manifest.canonical_bytes()),
        "resolver_configuration_sha256": manifest.resolver_configuration_fingerprint.hex(),
        "policy": manifest.policy.value,
        "offline": manifest.offline,
        "complete": manifest.is_complete,
        "document_count": len(manifest.documents),
        "import_edge_count": len(manifest.edges),
        "resolution_status_counts": dict(sorted(status_counts.items())),
        "source_documents": documents,
    }


def _core_provenance(snapshot: OntologyView) -> dict[str, object]:
    report = snapshot.report
    capabilities = snapshot.capabilities
    return {
        "package_version": str(pyowl_core.__version__),
        "api_version": list(pyowl_core.API_VERSION),
        "model_schema_version": int(pyowl_core.MODEL_SCHEMA_VERSION),
        "wire_format_version": list(pyowl_core.WIRE_FORMAT_VERSION),
        "adapter_protocol_version": int(pyowl_core.ADAPTER_PROTOCOL_VERSION),
        "backend": report.backend,
        "shared_snapshot": True,
        "shared_view": True,
        "verified_wire": "wire-verified" in capabilities.features,
        "fingerprints": {
            "structural": _fingerprint(snapshot.structural_fingerprint),
            "logical": _fingerprint(snapshot.logical_fingerprint),
            "signature": _fingerprint(snapshot.signature_fingerprint),
        },
        "closure": _closure_provenance(snapshot),
        "loader": {
            "effective_axiom_count": report.effective_axiom_count,
            "total_source_bytes": report.total_source_bytes,
            "resolution_attempts": report.resolution_attempts,
            "acquisition_cache_hits": report.acquisition_cache_hits,
            "document_cache_hits": report.document_cache_hits,
            "diagnostics": _diagnostic_summary(snapshot),
        },
    }


def _projector_provenance(
    settings: ProjectorSettings,
    projector: object,
) -> dict[str, object]:
    identity = projector_cache_identity(settings)
    options = ProjectionOptions(
        profile=settings.profile,
        include_literals=False,
        duplicates="unique",
        order="canonical",
        compatibility_state="isolated",
        backend=settings.backend,
    )
    last_report = getattr(projector, "last_report", None)
    last_projection: dict[str, object] | None
    if last_report is not None:
        last_projection = cast(dict[str, object], last_report.to_dict())
        last = cast(dict[str, object], last_projection["provenance"])
        selected = str(last["selected_backend"])
        fallback_reason = None
    else:
        last_projection = None
        try:
            selection = select_backend(settings.backend)
        except shared_projector.NativeBackendUnavailableError as error:
            selected = "unavailable"
            fallback_reason = str(error)
        else:
            selected = selection.selected
            fallback_reason = selection.fallback_reason
    return {
        **identity,
        "selection": {
            "requested": settings.backend,
            "effective": selected,
            "fallback_reason": fallback_reason,
        },
        "profile": settings.profile,
        "options": options.to_dict(),
        "edge_artifact_schema": str(shared_projector.EDGE_ARTIFACT_SCHEMA),
        "last_projection": last_projection,
    }


def _encoded_view_schemas(capabilities: object) -> dict[str, int]:
    raw = getattr(capabilities, "encoded_view_schemas", None)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("core encoded_view_schemas must be a mapping")
    result: dict[str, int] = {}
    for name, version in raw.items():
        if not isinstance(name, str) or not name:
            raise TypeError("core encoded-view schema names must be nonempty text")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise TypeError("core encoded-view schema versions must be positive integers")
        result[name] = version
    return dict(sorted(result.items()))


def _core_consumer_handoff(snapshot: OntologyView) -> dict[str, object]:
    capabilities = snapshot.capabilities
    features = set(capabilities.features)
    if "ontology-composite" in features:
        owner_kind = "composite"
    elif "ontology-overlay" in features:
        owner_kind = "overlay"
    elif "mmap-snapshot" in features:
        owner_kind = "mmap"
    elif "wire-verified" in features:
        owner_kind = "decoded"
    else:
        owner_kind = "direct"
    return {
        "encoded_contract": encoded_contract_identity().as_dict()["core"],
        "encoded_view_schemas": _encoded_view_schemas(capabilities),
        "owner_kind": owner_kind,
        "storage_backend": capabilities.backend,
    }


def _projector_consumer_handoff(projector: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "package_version": projector["package_version"],
        "compiler_cache_schema": projector["compiler_cache_schema"],
    }
    last_projection = projector.get("last_projection")
    if not isinstance(last_projection, Mapping):
        return result
    provenance = last_projection.get("provenance")
    if not isinstance(provenance, Mapping):
        return result
    for source, target in (("selected_backend", "selected_backend"),):
        value = provenance.get(source)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise TypeError(f"projector {source} diagnostic must be nonempty text")
            result[target] = value
    implementation = provenance.get("native_implementation_version")
    if implementation is not None:
        if not isinstance(implementation, str) or not implementation:
            raise TypeError("projector implementation diagnostic must be nonempty text")
        result["implementation_version"] = implementation
    ingestion = provenance.get("ingestion")
    if isinstance(ingestion, Mapping):
        path = ingestion.get("path")
        if path not in _PROJECTOR_INGESTION_PATHS:
            raise ValueError("projector ingestion path is incompatible")
        result["ingestion_path"] = path
        schema_name = ingestion.get("encoded_schema_name")
        schema_version = ingestion.get("encoded_schema_version")
        descriptor = ingestion.get("encoded_descriptor_sha256")
        encoded_identity = (schema_name, schema_version, descriptor)
        if path == "encoded-native":
            if (
                not isinstance(schema_name, str)
                or not schema_name
                or isinstance(schema_version, bool)
                or not isinstance(schema_version, int)
                or schema_version < 1
                or not isinstance(descriptor, str)
                or _SHA256_HEX.fullmatch(descriptor) is None
            ):
                raise TypeError("projector encoded ingestion identity is invalid")
        elif any(value is not None for value in encoded_identity):
            raise ValueError("scalar projector ingestion claimed an encoded identity")
        if schema_name is not None:
            result.update(
                {
                    "schema_name": schema_name,
                    "schema_version": schema_version,
                    "descriptor_sha256": descriptor,
                }
            )
        for name in ("encoded_view_publication_seconds", "consumer_compile_seconds"):
            value = ingestion.get(name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise TypeError(f"projector {name} diagnostic is invalid")
            result[name] = float(value)
        if path != "encoded-native" and "encoded_view_publication_seconds" in result:
            raise ValueError("scalar projector ingestion claimed encoded-view publication")
        counters = ingestion.get("counters")
        if counters is not None:
            if not isinstance(counters, Mapping) or any(
                not isinstance(name, str) or name not in _PROJECTOR_HANDOFF_COUNTERS
                for name in counters
            ):
                raise TypeError("projector ingestion counters are incompatible")
            bounded: dict[str, int | bool] = {}
            for name, value in counters.items():
                if name == "encoded_compiler_gil_released":
                    if type(value) is not bool:
                        raise TypeError("projector encoded compiler GIL diagnostic is invalid")
                elif type(value) is not int or value < 0:
                    raise TypeError("projector ingestion counters are invalid")
                bounded[name] = value
            if path != "encoded-native" and any(
                name in bounded and bounded[name] != expected
                for name, expected in _ENCODED_ONLY_COUNTER_DEFAULTS.items()
            ):
                raise ValueError("scalar projector ingestion claimed nonzero encoded resources")
            result["counters"] = dict(sorted(bounded.items()))
    return result


def _reasoner_consumer_handoff(reasoner: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    selection = reasoner.get("selection")
    if isinstance(selection, Mapping):
        for source, target in (
            ("effective", "reasoner"),
            ("package_version", "package_version"),
        ):
            value = selection.get(source)
            if value is not None:
                result[target] = value
    backend = reasoner.get("backend")
    if isinstance(backend, Mapping):
        for source, target in (
            ("effective", "selected_backend"),
            ("implementation_version", "implementation_version"),
        ):
            value = backend.get(source)
            if value is not None:
                result[target] = value
    handoff = reasoner.get("consumer_handoff")
    if isinstance(handoff, Mapping):
        fields = set(handoff)
        required = {"ingestion_path", "compiler_digest", "counters"}
        if not required.issubset(fields) or not fields.issubset(
            required | _REASONER_HANDOFF_OPTIONAL_FIELDS
        ):
            raise ValueError("reasoner consumer handoff fields are incompatible")
        ingestion_path = handoff["ingestion_path"]
        if ingestion_path not in _REASONER_INGESTION_PATHS:
            raise ValueError("reasoner consumer handoff ingestion path is incompatible")
        compiler_digest = handoff["compiler_digest"]
        if compiler_digest is not None and (
            not isinstance(compiler_digest, str) or _SHA256_HEX.fullmatch(compiler_digest) is None
        ):
            raise TypeError("reasoner consumer handoff compiler digest is invalid")
        counters = handoff["counters"]
        if not isinstance(counters, Mapping) or any(
            not isinstance(name, str) or name not in _REASONER_HANDOFF_COUNTERS for name in counters
        ):
            raise TypeError("reasoner consumer handoff counters are incompatible")
        for name, value in counters.items():
            if name == "encoded_compiler_gil_released":
                if not isinstance(value, bool):
                    raise TypeError("reasoner consumer handoff GIL diagnostic is invalid")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError("reasoner consumer handoff counters are invalid")
        if ingestion_path != "encoded-native" and any(
            name in counters and counters[name] != expected
            for name, expected in _ENCODED_ONLY_COUNTER_DEFAULTS.items()
        ):
            raise ValueError("scalar reasoner ingestion claimed nonzero encoded resources")
        result.update(
            {
                "ingestion_path": ingestion_path,
                "compiler_digest": compiler_digest,
                "counters": dict(sorted(counters.items())),
            }
        )
        for name in ("compiler_cache_schema_version", "ir_schema_version"):
            value = handoff.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TypeError(f"reasoner consumer handoff {name} is invalid")
            result[name] = value
        native_abi = handoff.get("native_abi_version")
        if native_abi is not None:
            if (
                isinstance(native_abi, bool)
                or not isinstance(native_abi, (int, str))
                or (isinstance(native_abi, int) and native_abi < 1)
                or (isinstance(native_abi, str) and not native_abi)
            ):
                raise TypeError("reasoner consumer handoff native ABI version is invalid")
            result["native_abi_version"] = native_abi
        implementation = handoff.get("implementation_version")
        if implementation is not None:
            if not isinstance(implementation, str) or not implementation:
                raise TypeError("reasoner consumer handoff implementation version is invalid")
            selected_implementation = result.get("implementation_version")
            if selected_implementation is not None and selected_implementation != implementation:
                raise ValueError("reasoner implementation diagnostics disagree")
            result["implementation_version"] = implementation
        for name in ("consumer_compile_seconds", "encoded_view_publication_seconds"):
            value = handoff.get(name)
            if value is None:
                continue
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise TypeError(f"reasoner consumer handoff {name} is invalid")
            result[name] = value
        if ingestion_path != "encoded-native" and "encoded_view_publication_seconds" in result:
            raise ValueError("scalar reasoner ingestion claimed encoded-view publication")
    for source, target in (
        ("verified_wire", "worker_wire_verified"),
        ("mmap_verified", "worker_mmap_verified"),
    ):
        if source not in reasoner:
            continue
        value = reasoner[source]
        if not isinstance(value, bool):
            raise TypeError(f"reasoner {source} diagnostic is invalid")
        result[target] = value
    owl_parse_count = reasoner.get("owl_parse_count")
    if owl_parse_count is not None:
        if (
            isinstance(owl_parse_count, bool)
            or not isinstance(owl_parse_count, int)
            or owl_parse_count < 0
        ):
            raise TypeError("reasoner worker OWL parse count is invalid")
        result["worker_owl_parse_count"] = owl_parse_count
    return result


def _consumer_handoff(
    snapshot: OntologyView,
    projector: Mapping[str, object],
    reasoner: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": CONSUMER_HANDOFF_PROVENANCE_SCHEMA,
        "core": _core_consumer_handoff(snapshot),
        "projector": _projector_consumer_handoff(projector),
        "reasoner": _reasoner_consumer_handoff(reasoner),
    }


def ontology_stack_provenance(
    snapshot: OntologyView,
    *,
    projector_settings: ProjectorSettings,
    projector: object,
    reasoner: dict[str, object],
) -> dict[str, Any]:
    """Describe one OWL source without paths, object IDs, or reparsing."""

    snapshot = retain_ontology_view(snapshot)
    projector_provenance = _projector_provenance(projector_settings, projector)
    payload = {
        "schema_version": ONTOLOGY_STACK_PROVENANCE_SCHEMA,
        "kind": "owl",
        "core": _core_provenance(snapshot),
        "projector": projector_provenance,
        "reasoner": reasoner,
        "consumer_handoff": _consumer_handoff(
            snapshot,
            projector_provenance,
            reasoner,
        ),
    }
    return cast(dict[str, Any], _safe_value(payload))


__all__ = [
    "CONSUMER_HANDOFF_PROVENANCE_SCHEMA",
    "ONTOLOGY_STACK_PROVENANCE_SCHEMA",
    "ontology_stack_provenance",
]
