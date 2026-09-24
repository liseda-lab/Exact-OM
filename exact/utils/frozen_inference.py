"""Explicit same-pair deployment of immutable fitted heads, without fitting/evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import dataset_signature_for_paths, sha256_path


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_path(path)}


def _artifacts(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if key == "inference_artifact":
            continue
        if isinstance(value, dict):
            result.update(_artifacts(value, name))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    result.update(_artifacts(item, f"{name}.{index}"))
        elif value and "artifact" in key and isinstance(value, (str, Path)):
            result[name] = _binding(value)
    return result


def _contract(config):
    mapping = config.model_dump(mode="json", by_alias=True)
    mapping["supervision"].pop("inference_artifact", None)
    return {
        **{
            key: mapping[key]
            for key in (
                "pipeline",
                "matching",
                "candidates",
                "dataset",
                "selector",
                "llm",
                "supervision",
                "inference",
            )
        },
        "seed": config.seed,
        "population_policy": {
            key: mapping["data"][key]
            for key in ("execution_mode", "candidate_source", "candidate_provenance")
        },
    }


def _check_config(config):
    if config.data.source is None or config.data.target is None:
        raise ValueError("Frozen inference requires explicit source and target ontologies")
    if (
        config.data.refs
        or config.data.train_candidates
        or config.data.reference_role
        or config.data.track
        or config.data.descriptor
        or config.supervision.transfer_artifact
    ):
        raise ValueError(
            "Frozen inference requires explicit ontology inputs and no references or training pool"
        )
    if config.run.source_cap is not None or config.data.source_universe is None:
        raise ValueError("Frozen inference requires its uncapped public source population")
    if config.matching.relation_training_file or config.matching.nil.training_source_labels:
        raise ValueError("Frozen inference cannot consume relation/NIL training labels")
    if any(
        getattr(config.candidates, name).training is not None
        for name in ("encoder_finetune", "cross_encoder")
    ):
        raise ValueError("Frozen inference cannot fit retrieval encoders")
    for row in config.pipeline:
        if any(
            value
            for key, value in row.params.items()
            if "reference" in key or key == "training_source_labels"
        ):
            raise ValueError("Frozen inference pipeline cannot consume reference/training inputs")
        if row.params.get("use_llm_calibration") and any(
            row.params.get(key) is None for key in ("llm_calibration_a", "llm_calibration_b")
        ):
            raise ValueError(
                "Frozen inference requires already fitted LLM calibration coefficients"
            )
    if any(
        config.supervision.components.get(name, config.supervision.mode) == "auto"
        for name in (
            "retrieval",
            "fusion",
            "rerank",
            "llm",
            "accept",
            "calibration",
            "structure",
            "relation",
        )
    ):
        raise ValueError("Frozen inference requires resolved component supervision modes")
    llm = config.llm.experiment
    if (
        llm.gate.mode in {"oracle_perfect", "oracle_replay", "forced_sample"}
        or config.matching.nil.pool_miss_development_reference is not None
    ):
        raise ValueError("Diagnostic oracle/population interventions cannot be deployed")
    required = {
        "fusion": (
            config.matching.fusion.mode in {"analytic_fitted", "learned_global"},
            config.matching.fusion.artifact,
        ),
        "calibration": (
            config.matching.calibration.mode != "none",
            config.matching.calibration.artifact,
        ),
        "NIL": (config.matching.nil.mode == "fitted", config.matching.nil.artifact),
        "graph": (
            config.matching.channels.graph.mode in {"inductive", "graph_only"},
            config.matching.channels.graph.artifact,
        ),
        "relation": (
            config.matching.relation_prediction in {"learned_three_way", "semantic_then_learned"},
            config.matching.relation_artifact,
        ),
        "retrieval": (
            config.candidates.encoder_finetune.mode != "off",
            config.candidates.encoder_finetune.artifact,
        ),
        "cross encoder": (
            config.candidates.cross_encoder.mode != "off",
            config.candidates.cross_encoder.artifact,
        ),
        "exemplars": (llm.exemplars == "knn", llm.exemplar_artifact),
        "student": (llm.distill == "student", llm.distill_artifact),
        "learned gate": (llm.gate.mode == "learned", llm.gate.artifact),
    }
    selector_active = any(
        row.name == "CandidateSetSelector"
        and (
            config.selector.runtime_enabled
            if config.selector.runtime_enabled is not None
            else row.params.get("enabled", True)
        )
        for row in config.pipeline
    )
    supervised_selector = any(
        config.supervision.components.get(name, config.supervision.mode) == "supervised"
        for name in ("rerank", "accept", "calibration")
    )
    required["selector"] = (
        selector_active and supervised_selector,
        config.selector.rerank.artifact,
    )
    for name, (needed, path) in required.items():
        if needed and (path is None or not Path(path).exists()):
            raise ValueError(f"Frozen inference requires an already fitted {name} artifact")


def freeze_inference_manifest(config, path):
    _check_config(config)
    artifacts = _artifacts(config.model_dump(mode="json"))
    signature = dataset_signature_for_paths(config.data.source, config.data.target)
    for entry in artifacts.values():
        artifact = Path(entry["path"])
        if artifact.suffix == ".json":
            payload = json.loads(artifact.read_text())
            application = payload.get(
                "application", payload.get("fit_provenance", {}).get("application", {})
            )
            bound = payload.get("dataset_signature", application.get("dataset_signature"))
            if bound and bound != signature:
                raise ValueError("Frozen inference artifact belongs to a different ontology pair")
    return freeze_json(
        path,
        {
            "schema_version": 1,
            "kind": "same_pair_frozen_inference",
            "refit": False,
            "evaluation": False,
            "dataset_signature": signature,
            "entity_kinds": sorted(config.matching.entity_kinds),
            "feature_contract": _contract(config),
            "artifacts": artifacts,
            "inputs": {
                name: _binding(getattr(config.data, name))
                for name in ("source", "target", "source_universe", "candidates")
                if getattr(config.data, name) is not None
            },
        },
    )


def validate_inference_config(config, *, run_eval=False):
    path = config.supervision.inference_artifact
    if path is None:
        return None
    _check_config(config)
    manifest = json.loads(Path(path).read_text())
    if (
        run_eval
        or manifest.get("kind") != "same_pair_frozen_inference"
        or manifest.get("refit") is not False
        or manifest.get("evaluation") is not False
    ):
        raise ValueError("Frozen inference is deployment only: no fitting or evaluation")
    if manifest.get("feature_contract") != _contract(config) or manifest.get(
        "entity_kinds"
    ) != sorted(config.matching.entity_kinds):
        raise ValueError("Frozen inference feature recipe changed")
    inputs = {
        name: _binding(getattr(config.data, name))
        for name in ("source", "target", "source_universe", "candidates")
        if getattr(config.data, name) is not None
    }
    if inputs != manifest["inputs"]:
        raise ValueError("Frozen inference ontology/population binding changed")
    if _artifacts(config.model_dump(mode="json")) != manifest["artifacts"]:
        raise ValueError("Frozen inference fitted artifact hash/path changed")
    if manifest["dataset_signature"] != dataset_signature_for_paths(
        config.data.source, config.data.target
    ):
        raise ValueError("Frozen inference ontology pair identity changed")
    return manifest


def frozen_application(dataset, payload=None):
    """Admission is attached only after config/input validation before dataset construction."""
    manifest = getattr(dataset, "frozen_inference_manifest", None)
    if manifest is None:
        return False
    if manifest["dataset_signature"] != dataset.dataset_signature:
        raise ValueError("Frozen inference runtime ontology pair changed")
    if payload is not None:
        digests = getattr(dataset, "_frozen_artifact_payloads", None)
        if digests is None:
            digests = set()
            for entry in manifest["artifacts"].values():
                path = Path(entry["path"])
                if sha256_path(path) != entry["sha256"]:
                    raise ValueError("Frozen inference fitted artifact changed after admission")
                if path.suffix == ".json":
                    digests.add(fingerprint(json.loads(path.read_text())))
            dataset._frozen_artifact_payloads = digests
        if fingerprint(payload) not in digests:
            raise ValueError("Applied head is absent from the frozen inference manifest")
    return True
