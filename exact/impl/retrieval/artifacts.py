"""Validation and fingerprinting for experiment-fitted retrieval models.

The runtime deliberately accepts only local, self-describing artifacts.  Model
training is outside the alignment runtime, and a missing artifact must never
fall through to a model-hub download.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

ArtifactKind = Literal["contrastive_encoder", "cross_encoder"]

_MANIFEST_NAMES = ("retrieval_artifact.json", "artifact.json")
_NEGATIVE_POLICIES = {
    "complete_reference",
    "confirmed_negative",
    "positive_unlabelled",
}
_REFERENCE_COMPLETENESS = {"complete", "known_incomplete", "unknown"}
_COMMON_METADATA = {
    "base_model",
    "training_pairs",
    "reference_completeness",
    "negative_policy",
    "dataset_lock",
    "seed",
}
_KIND_METADATA = {
    "contrastive_encoder": {"mining", "epochs"},
    "cross_encoder": {"epochs"},
}


@dataclass(frozen=True, slots=True)
class LocalRetrievalArtifact:
    """Validated local model directory and immutable provenance."""

    kind: ArtifactKind
    manifest_path: Path
    model_path: Path
    sha256: str
    metadata: Mapping[str, Any]

    def provenance(self) -> dict[str, Any]:
        """Return a JSON-safe artifact record for cache and pool manifests."""

        return {
            "kind": self.kind,
            "manifest_path": str(self.manifest_path),
            "model_path": str(self.model_path),
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
        }

    def assert_unchanged(self) -> None:
        """Fail if artifact bytes changed after validation."""

        current = _artifact_sha256(self.manifest_path, self.model_path)
        if current != self.sha256:
            raise ValueError(
                "Retrieval artifact changed after validation: "
                f"{self.manifest_path} ({self.sha256} != {current})"
            )


def resolve_local_retrieval_artifact(
    value: str | Path | None,
    *,
    expected_kind: ArtifactKind,
    negative_policy: str | None = None,
) -> LocalRetrievalArtifact:
    """Validate a fitted artifact without downloading or training anything.

    ``value`` may be a JSON manifest or a directory containing exactly one of
    ``retrieval_artifact.json`` and ``artifact.json``.  ``model_path`` in the
    manifest is resolved relative to that manifest and must remain inside the
    artifact directory.
    """

    if value is None or not str(value).strip():
        raise ValueError(f"{expected_kind} mode requires a local artifact path")

    supplied = Path(value).expanduser()
    if not supplied.exists():
        raise FileNotFoundError(f"Local {expected_kind} artifact does not exist: {supplied}")
    if supplied.is_symlink():
        raise ValueError(f"Retrieval artifact roots may not be symbolic links: {supplied}")
    supplied = supplied.resolve()
    manifest_path = _resolve_manifest_path(supplied)
    payload = _read_manifest(manifest_path)

    if payload.get("schema_version") != 1:
        raise ValueError(f"{manifest_path} must declare retrieval artifact schema_version=1")
    if payload.get("artifact_type") != expected_kind:
        raise ValueError(
            f"{manifest_path} has artifact_type={payload.get('artifact_type')!r}; "
            f"expected {expected_kind!r}"
        )

    missing = sorted((_COMMON_METADATA | _KIND_METADATA[expected_kind]) - set(payload))
    if missing:
        raise ValueError(f"{manifest_path} is missing required metadata: {', '.join(missing)}")
    _validate_artifact_metadata(payload, manifest_path, expected_kind)
    if payload["reference_completeness"] not in _REFERENCE_COMPLETENESS:
        raise ValueError(
            f"{manifest_path} reference_completeness must be complete, "
            "known_incomplete, or unknown"
        )
    if payload["negative_policy"] not in _NEGATIVE_POLICIES:
        raise ValueError(f"{manifest_path} declares an unsupported negative_policy")
    if (
        payload["negative_policy"] == "complete_reference"
        and payload["reference_completeness"] != "complete"
    ):
        raise ValueError(
            f"{manifest_path} cannot use complete_reference negatives with "
            f"reference_completeness={payload['reference_completeness']!r}"
        )
    if negative_policy is not None and payload["negative_policy"] != negative_policy:
        raise ValueError(
            "Configured encoder_finetune.negative_policy does not match the fitted "
            f"artifact ({negative_policy!r} != {payload['negative_policy']!r})"
        )

    artifact_root = supplied if supplied.is_dir() else manifest_path.parent
    raw_model_path = payload.get("model_path", ".")
    if not isinstance(raw_model_path, str) or not raw_model_path.strip():
        raise ValueError(f"{manifest_path} model_path must be a non-empty string")
    model_path = (manifest_path.parent / raw_model_path).resolve()
    try:
        model_path.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError(f"{manifest_path} model_path must remain inside {artifact_root}") from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Fitted model path does not exist: {model_path}")

    digest = _artifact_sha256(manifest_path, model_path)
    return LocalRetrievalArtifact(
        kind=expected_kind,
        manifest_path=manifest_path,
        model_path=model_path,
        sha256=digest,
        metadata=dict(payload),
    )


def retrieval_artifact_requirement(
    value: str | Path | None,
    *,
    expected_kind: ArtifactKind,
    negative_policy: str | None = None,
    expected_dataset_lock: str | Mapping[str, Any] | None = None,
    expected_candidate_pool_fingerprint: str | None = None,
    expected_dataset_identity: str | None = None,
) -> dict[str, Any]:
    """Return a machine-readable preflight result for an E20 artifact.

    An absent local artifact is an unavailable capability and may keep an arm
    explicitly deferred. A present but malformed or incompatibly bound
    artifact is invalid and must fail closed.
    """

    try:
        artifact = resolve_local_retrieval_artifact(
            value,
            expected_kind=expected_kind,
            negative_policy=negative_policy,
        )
    except FileNotFoundError as exc:
        return _artifact_requirement_result(
            status="deferred_unavailable",
            code="missing_fitted_retrieval_artifact",
            expected_kind=expected_kind,
            expected_dataset_identity=expected_dataset_identity,
            detail=str(exc),
        )
    except ValueError as exc:
        if value is None or not str(value).strip():
            return _artifact_requirement_result(
                status="deferred_unavailable",
                code="missing_fitted_retrieval_artifact",
                expected_kind=expected_kind,
                expected_dataset_identity=expected_dataset_identity,
                detail=str(exc),
            )
        return _artifact_requirement_result(
            status="invalid",
            code="invalid_fitted_retrieval_artifact",
            expected_kind=expected_kind,
            expected_dataset_identity=expected_dataset_identity,
            detail=str(exc),
        )

    if expected_dataset_lock is not None:
        expected_lock = _immutable_binding_digest(expected_dataset_lock, "expected_dataset_lock")
        artifact_lock = _immutable_binding_digest(
            artifact.metadata.get("dataset_lock"),
            f"{artifact.manifest_path} dataset_lock",
        )
        if artifact_lock != expected_lock:
            return _artifact_requirement_result(
                status="invalid",
                code="dataset_lock_mismatch",
                expected_kind=expected_kind,
                expected_dataset_identity=expected_dataset_identity,
                detail=f"artifact={artifact_lock}; expected={expected_lock}",
            )

    expected_pool = None
    if expected_candidate_pool_fingerprint is not None:
        expected_pool = _sha256_hex(
            expected_candidate_pool_fingerprint,
            "expected_candidate_pool_fingerprint",
        )
        declared_pool = artifact.metadata.get("candidate_pool_fingerprint")
        mining = artifact.metadata.get("mining")
        if declared_pool is None and isinstance(mining, Mapping):
            declared_pool = mining.get("candidate_pool_fingerprint")
        if declared_pool is None:
            return _artifact_requirement_result(
                status="invalid",
                code="missing_candidate_pool_binding",
                expected_kind=expected_kind,
                expected_dataset_identity=expected_dataset_identity,
                detail=f"artifact must bind candidate pool {expected_pool}",
            )
        actual_pool = _sha256_hex(
            declared_pool,
            f"{artifact.manifest_path} candidate_pool_fingerprint",
        )
        if actual_pool != expected_pool:
            return _artifact_requirement_result(
                status="invalid",
                code="candidate_pool_mismatch",
                expected_kind=expected_kind,
                expected_dataset_identity=expected_dataset_identity,
                detail=f"artifact={actual_pool}; expected={expected_pool}",
            )

    return {
        "schema_version": 1,
        "status": "ready",
        "code": "retrieval_artifact_ready",
        "capability": expected_kind,
        "expected_dataset_identity": expected_dataset_identity,
        "artifact_sha256": artifact.sha256,
        "dataset_lock_sha256": _immutable_binding_digest(
            artifact.metadata.get("dataset_lock"),
            f"{artifact.manifest_path} dataset_lock",
        ),
        "candidate_pool_fingerprint": expected_pool,
    }


def _resolve_manifest_path(supplied: Path) -> Path:
    if supplied.is_file():
        if supplied.suffix.lower() != ".json":
            raise ValueError("A retrieval artifact file must be a JSON manifest")
        return supplied
    if not supplied.is_dir():
        raise ValueError(f"Retrieval artifact must be a file or directory: {supplied}")

    manifests = [supplied / name for name in _MANIFEST_NAMES if (supplied / name).is_file()]
    if not manifests:
        names = " or ".join(_MANIFEST_NAMES)
        raise ValueError(f"Retrieval artifact directory must contain {names}: {supplied}")
    if len(manifests) > 1:
        raise ValueError(f"Retrieval artifact directory contains ambiguous manifests: {supplied}")
    if manifests[0].is_symlink():
        raise ValueError(f"Retrieval artifact manifests may not be symbolic links: {manifests[0]}")
    return manifests[0]


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid retrieval artifact JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Retrieval artifact manifest must contain a JSON object: {path}")
    return value


def _artifact_sha256(manifest_path: Path, model_path: Path) -> str:
    files = {manifest_path.resolve()}
    if model_path.is_file():
        files.add(model_path.resolve())
    elif model_path.is_dir():
        for item in model_path.rglob("*"):
            if item.is_symlink():
                raise ValueError(f"Retrieval artifacts may not contain symlinks: {item}")
            if item.is_file():
                files.add(item.resolve())
    else:
        raise ValueError(f"Fitted model path is not a regular file or directory: {model_path}")

    root = manifest_path.parent.resolve()
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Retrieval artifact file escapes {root}: {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_artifact_metadata(
    payload: Mapping[str, Any],
    manifest_path: Path,
    expected_kind: ArtifactKind,
) -> None:
    base_model = payload.get("base_model")
    if isinstance(base_model, str):
        if not base_model.strip():
            raise ValueError(f"{manifest_path} base_model must be non-empty")
    elif isinstance(base_model, Mapping):
        identifier = base_model.get("identifier", base_model.get("model_id"))
        revision = base_model.get("revision")
        if not str(identifier or "").strip() or not str(revision or "").strip():
            raise ValueError(
                f"{manifest_path} base_model mappings require identifier/model_id and revision"
            )
    else:
        raise ValueError(f"{manifest_path} base_model must be a string or mapping")

    training_pairs = payload.get("training_pairs")
    if not isinstance(training_pairs, list) or not training_pairs:
        raise ValueError(f"{manifest_path} training_pairs must be a non-empty list")
    for index, item in enumerate(training_pairs):
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(f"{manifest_path} training_pairs[{index}] must be non-empty")
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"{manifest_path} training_pairs[{index}] must be a string or mapping")
        split_role = str(item.get("split_role", item.get("reference_role", "train"))).lower()
        if split_role in {"test", "reporting", "confirm"}:
            raise ValueError(
                f"{manifest_path} training_pairs[{index}] declares forbidden split role "
                f"{split_role!r}"
            )

    _immutable_binding_digest(payload.get("dataset_lock"), f"{manifest_path} dataset_lock")
    epochs = payload.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"{manifest_path} epochs must be a positive integer")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{manifest_path} seed must be an integer")
    if expected_kind == "contrastive_encoder":
        mining = payload.get("mining")
        if not isinstance(mining, Mapping) or not mining:
            raise ValueError(f"{manifest_path} mining must be a non-empty mapping")
        required_mining = {
            "candidate_pool_fingerprint",
            "top_k",
            "max_negatives_per_source",
            "max_training_pairs",
            "exclude_known_positives",
        }
        missing_mining = sorted(required_mining - set(mining))
        if missing_mining:
            raise ValueError(
                f"{manifest_path} mining is missing required metadata: " + ", ".join(missing_mining)
            )
        _sha256_hex(
            mining.get("candidate_pool_fingerprint"),
            f"{manifest_path} mining.candidate_pool_fingerprint",
        )
        for name in ("top_k", "max_negatives_per_source", "max_training_pairs"):
            number = mining.get(name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f"{manifest_path} mining.{name} must be a positive integer")
        if mining.get("exclude_known_positives") is not True:
            raise ValueError(f"{manifest_path} mining.exclude_known_positives must be true")


def _artifact_requirement_result(
    *,
    status: str,
    code: str,
    expected_kind: ArtifactKind,
    expected_dataset_identity: str | None,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "code": code,
        "capability": expected_kind,
        "expected_dataset_identity": expected_dataset_identity,
        "detail": detail,
    }


def _immutable_binding_digest(value: Any, label: str) -> str:
    if isinstance(value, Mapping):
        value = value.get("sha256", value.get("fingerprint"))
    return _sha256_hex(value, label)


def _sha256_hex(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must contain an immutable SHA-256 digest")
    return text
