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
    if not isinstance(payload["training_pairs"], list):
        raise ValueError(f"{manifest_path} training_pairs must be a list")
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
