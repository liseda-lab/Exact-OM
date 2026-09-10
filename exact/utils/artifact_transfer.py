"""Explicit donor-head transfer without changing an artifact or reading recipient labels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from exact.utils.fitted_artifacts import fingerprint, freeze_json


def transfer_feature_contract(config):
    """Bind score-producing settings, including encoder identities, across ontology pairs."""

    def without_artifacts(value):
        if isinstance(value, dict):
            return {
                key: without_artifacts(item)
                for key, item in value.items()
                if "artifact" not in key
                and key not in {"training_reference_file_path", "training_source_labels"}
            }
        if isinstance(value, list):
            return [without_artifacts(item) for item in value]
        return value

    return without_artifacts(
        {
            "primary": config.pipeline[0].model_dump(mode="json"),
            "channels": config.matching.channels.model_dump(mode="json"),
            "fusion": config.matching.fusion.model_dump(mode="json"),
            "llm": config.llm.model_dump(mode="json"),
        }
    )


def artifact_features(payload, kind):
    if kind == "selector":
        provenance = payload["fit_provenance"]
        return {"rank": provenance["rank_features"], "accept": provenance["accept_features"]}
    if kind == "fusion":
        return payload["feature_schema"]
    if kind == "calibration":
        return ["S_pair_final"]
    raise ValueError(f"Artifact kind {kind!r} has no donor-transfer contract")


def _application(payload):
    return payload.get("fit_provenance", {}).get("application", {})


def freeze_transfer_manifest(
    path,
    artifacts,
    *,
    recipient_signature,
    feature_contract,
    score_threshold,
    entity_kinds=("class",),
):
    """Select immutable donor artifacts before recipient scoring; paths are exact, hashes authoritative."""
    if not recipient_signature or not feature_contract or not math.isfinite(score_threshold):
        raise ValueError(
            "Transfer requires a recipient signature, feature contract and fixed threshold"
        )
    entries = {}
    for kind, artifact_path in artifacts.items():
        artifact_path = Path(artifact_path).resolve()
        payload = json.loads(artifact_path.read_text())
        application = _application(payload)
        donor = payload.get("dataset_signature", application.get("dataset_signature"))
        if not donor or donor == recipient_signature:
            raise ValueError("Cross-pair transfer requires a distinct declared donor dataset")
        if sorted(application.get("entity_kinds", [])) != sorted(entity_kinds):
            raise ValueError("Donor artifact must declare matching entity kinds")
        entries[kind] = {
            "path": str(artifact_path),
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "donor_dataset_signature": donor,
            "features": artifact_features(payload, kind),
            "fit_provenance_sha256": fingerprint(payload.get("fit_provenance", {})),
        }
        if kind == "selector":
            entries[kind]["accept_threshold"] = payload["accept_threshold"]
    if not entries:
        raise ValueError("Transfer requires at least one frozen donor head")
    return freeze_json(
        path,
        {
            "schema_version": 1,
            "kind": "cross_pair_transfer",
            "artifacts": entries,
            "recipient_dataset_signature": recipient_signature,
            "entity_kinds": sorted(entity_kinds),
            "feature_contract": feature_contract,
            "score_threshold": float(score_threshold),
            "recipient_refit": False,
            "supervision_label": "cross_pair_transfer",
        },
    )


def validate_transfer(dataset):
    path = getattr(dataset, "transfer_artifact", None)
    if path is None:
        return None
    manifest = json.loads(Path(path).read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "cross_pair_transfer"
        or manifest.get("recipient_refit") is not False
    ):
        raise ValueError("Invalid immutable donor-transfer manifest")
    if manifest.get("recipient_dataset_signature") != getattr(dataset, "dataset_signature", None):
        raise ValueError("Transfer recipient dataset signature mismatch")
    kinds = [getattr(kind, "value", kind) for kind in getattr(dataset, "_entity_kinds", ("class",))]
    if sorted(kinds) != manifest.get("entity_kinds"):
        raise ValueError("Transfer entity-kind contract mismatch")
    if manifest.get("feature_contract") != getattr(dataset, "transfer_feature_contract", None):
        raise ValueError("Transfer feature-producing recipe mismatch")
    threshold = getattr(dataset, "transfer_score_threshold", None)
    if threshold is None or abs(float(threshold) - manifest["score_threshold"]) > 1e-12:
        raise ValueError("Transfer requires the unchanged fixed donor score threshold")
    for entry in manifest["artifacts"].values():
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Transferred donor artifact hash mismatch")
    return manifest


def validate_transferred_artifact(dataset, path, *, kind, features, score_threshold=None):
    manifest = validate_transfer(dataset)
    if manifest is None:
        return False
    entry = manifest["artifacts"].get(kind)
    if entry is None or Path(path).resolve() != Path(entry["path"]).resolve():
        raise ValueError(f"Active {kind} artifact is not the selected immutable donor artifact")
    if entry["features"] != features:
        raise ValueError(f"Transferred {kind} feature schema mismatch")
    if (
        score_threshold is not None
        and abs(float(score_threshold) - manifest["score_threshold"]) > 1e-12
    ):
        raise ValueError("Transfer cannot adapt the donor decision threshold")
    return True
