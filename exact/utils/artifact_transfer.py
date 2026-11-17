"""Explicit donor-head transfer without changing an artifact or reading recipient labels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import dataset_signature_for_paths, sha256_file


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
            "llm_profiles": {
                key: value.model_dump(mode="json") for key, value in config.llm_profiles.items()
            },
            "llm_routing": config.llm_routing.model_dump(mode="json"),
            "dataset": config.dataset.model_dump(mode="json"),
            "score_calibration": config.matching.calibration.model_dump(mode="json"),
            "selector": config.selector.model_dump(mode="json"),
            **(
                {"nil": config.matching.nil.model_dump(mode="json")}
                if config.matching.nil.mode != "off"
                else {}
            ),
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
    if kind == "nil" and payload.get("kind") in {"natural_nil_head", "benchmark_nil_head"}:
        return payload["feature_schema"]
    raise ValueError(f"Artifact kind {kind!r} has no donor-transfer contract")


def _application(payload):
    return payload.get("fit_provenance", {}).get("application", payload.get("application", {}))


def freeze_transfer_manifest(
    path,
    artifacts,
    *,
    recipient_signature,
    feature_contract,
    score_threshold,
    entity_kinds=("class",),
    application_provenance=None,
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
    if len({entry["donor_dataset_signature"] for entry in entries.values()}) != 1:
        raise ValueError("Transfer heads must share the selected donor ontology pair")
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
            "application_provenance": application_provenance or {},
        },
    )


def validate_transfer(dataset):
    path = getattr(dataset, "transfer_artifact", None)
    if path is None:
        return None
    manifest = json.loads(Path(path).read_text())
    if manifest.get("kind") == "cross_pair_transfer_bundle":
        binding = manifest.get("applications", {}).get(getattr(dataset, "dataset_signature", None))
        if not binding:
            raise ValueError("Transfer bundle has no frozen recipient application binding")
        application_path = Path(binding["path"])
        if sha256_file(application_path) != binding["sha256"]:
            raise ValueError("Transfer application manifest hash mismatch")
        manifest = json.loads(application_path.read_text())
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
    population = manifest.get("application_provenance", {}).get("recipient_population")
    if population and sha256_file(Path(population["path"])) != population["sha256"]:
        raise ValueError("Transfer recipient population changed after application binding")
    for side in ("donor_inputs", "recipient_inputs"):
        for entry in manifest.get("application_provenance", {}).get(side, {}).values():
            if sha256_file(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("Transfer ontology content changed after application binding")
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


def validate_transfer_config(config):
    """Validate the exact consumer configuration before constructing a dataset."""
    from types import SimpleNamespace

    if config.data.source is None or config.data.target is None:
        raise ValueError("Transfer requires explicitly bound recipient ontology paths")
    dataset = SimpleNamespace(
        transfer_artifact=config.supervision.transfer_artifact,
        dataset_signature=dataset_signature_for_paths(config.data.source, config.data.target),
        transfer_feature_contract=transfer_feature_contract(config),
        transfer_score_threshold=config.matching.threshold,
        _entity_kinds=config.matching.entity_kinds,
    )
    manifest = validate_transfer(dataset)
    if manifest is None:
        raise ValueError("Cross-pair transfer requires its bound immutable donor manifest")
    active = {
        "selector": config.selector.rerank.artifact,
        "fusion": config.matching.fusion.artifact,
        "calibration": config.matching.calibration.artifact,
        "nil": config.matching.nil.artifact,
    }
    if "nil" in manifest["artifacts"] and (
        config.matching.nil.mode != "fitted"
        or config.matching.nil.training_source_labels is not None
        or config.data.train_candidates is not None
        or config.data.refs.get("train") is not None
    ):
        raise ValueError("Transferred NIL requires a frozen head and no recipient training inputs")
    if set(manifest["artifacts"]) != {key for key, value in active.items() if value is not None}:
        raise ValueError("Transfer active fitted heads differ from its frozen donor manifest")
    for kind, entry in manifest["artifacts"].items():
        if Path(active[kind]).resolve() != Path(entry["path"]).resolve():
            raise ValueError("Transfer active artifact path differs from its frozen donor manifest")
    return manifest
