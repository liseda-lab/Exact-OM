"""Path-free release provenance for Exact's shared OWL stack."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, cast

import pyowl2vec_star_projector as shared_projector
import pyowl_core
from pyowl2vec_star_projector import ProjectionOptions, select_backend
from pyowl_core import OntologySnapshot

from exact.ontology.projection import ProjectorSettings, projector_cache_identity

ONTOLOGY_STACK_PROVENANCE_SCHEMA = 1
_PATH_FRAGMENT = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/)[^\s\"']+")
_OBJECT_ID = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")


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


def _diagnostic_summary(snapshot: OntologySnapshot) -> dict[str, object]:
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


def _core_provenance(snapshot: OntologySnapshot) -> dict[str, object]:
    manifest = snapshot.import_manifest
    status_counts = Counter(edge.status.value for edge in manifest.edges)
    documents = []
    for record, document in snapshot.iter_documents():
        documents.append(
            {
                "document_key": record.document_key,
                "source_sha256": record.source_sha256.hex(),
                "document_fingerprint": _fingerprint(record.document_fingerprint),
                "format": record.format.value,
                "status": record.status.value,
                "source_bytes": document.provenance.byte_length,
            }
        )
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
        "verified_wire": "wire-verified" in capabilities.features,
        "fingerprints": {
            "structural": _fingerprint(snapshot.structural_fingerprint),
            "logical": _fingerprint(snapshot.logical_fingerprint),
            "signature": _fingerprint(snapshot.signature_fingerprint),
        },
        "closure": {
            "import_manifest_sha256": _sha256(manifest.canonical_bytes()),
            "resolver_configuration_sha256": (manifest.resolver_configuration_fingerprint.hex()),
            "policy": manifest.policy.value,
            "offline": manifest.offline,
            "complete": manifest.is_complete,
            "document_count": len(manifest.documents),
            "import_edge_count": len(manifest.edges),
            "resolution_status_counts": dict(sorted(status_counts.items())),
            "source_documents": documents,
        },
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


def ontology_stack_provenance(
    snapshot: OntologySnapshot,
    *,
    projector_settings: ProjectorSettings,
    projector: object,
    reasoner: dict[str, object],
) -> dict[str, Any]:
    """Describe one OWL source without paths, object IDs, or reparsing."""

    if not isinstance(snapshot, OntologySnapshot):
        raise TypeError("ontology stack provenance requires an OntologySnapshot")
    payload = {
        "schema_version": ONTOLOGY_STACK_PROVENANCE_SCHEMA,
        "kind": "owl",
        "core": _core_provenance(snapshot),
        "projector": _projector_provenance(projector_settings, projector),
        "reasoner": reasoner,
    }
    return cast(dict[str, Any], _safe_value(payload))


__all__ = ["ONTOLOGY_STACK_PROVENANCE_SCHEMA", "ontology_stack_provenance"]
