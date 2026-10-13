"""Explicit G4 fit-only restart: fresh predictions, retained learned parameters.

This worker-scoped adapter changes only bound fitted artifact slots, completed-fit
training dispatch, and the reviewed synthetic-diagnostic omission. It never imports
old predictions, numerical caches, training aggregates, or inference checkpoints.
"""
from __future__ import annotations

import copy
import shutil
from contextlib import contextmanager
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

KIND = "g4_batched_fit_restart_v1"
CELLS = {
    f"{arm}/{case}-{mode}"
    for arm in ("baseline", "core")
    for case in ("D0_E03", "D1")
    for mode in ("global_alignment", "local_ranking")
}
HEAD_FIELDS = {
    "selector": "selector.rerank.artifact",
    "fusion": "matching.fusion.artifact",
    "calibration": "matching.calibration.artifact",
}


def _set(mapping, dotted, value):
    keys = dotted.split(".")
    for key in keys[:-1]:
        mapping = mapping[key]
    mapping[keys[-1]] = value


def _canonical(config):
    from exact.core.entities.configs.config import ConfigModel

    return ConfigModel.from_mapping(config, warn_v1=False).model_dump(mode="json", by_alias=True)


def _head(path, name, config):
    from exact.utils.provenance import dataset_signature_for_paths

    payload = read(path)
    provenance = payload["fit_provenance"]
    application = provenance["application"]
    expected = dataset_signature_for_paths(config["data"]["source"], config["data"]["target"])
    if application.get("dataset_signature") != expected:
        raise ValueError("Saved fitted head belongs to another ontology pair")
    if name == "fusion":
        if payload.get("mode") != config["matching"]["fusion"]["mode"]:
            raise ValueError("Saved fusion mode differs from the selected recipe")
        seed = payload.get("seed")
    else:
        if payload.get("kind") != {"selector": "fitted_selector", "calibration": "score_calibrator"}[name]:
            raise ValueError("Saved fitted head has the wrong kind")
        seed = provenance.get("seed")
        sources = set(provenance["training_sources"])
        if not sources or sources & set(application["source_ids"]):
            raise ValueError("Saved training provenance overlaps reporting sources")
    if seed != config["run"]["seed"]:
        raise ValueError("Saved fitted head seed differs")
    if name == "selector" and provenance.get("score_calibration") != config["matching"]["calibration"]["mode"]:
        raise ValueError("Saved selector calibration differs")
    if name == "selector" and (
        provenance["recipe"]["rerank"] != {
            key: value for key, value in config["selector"]["rerank"].items() if key != "artifact"
        }
        or provenance["accept_training"] != config["selector"]["accept_training"]
    ):
        raise ValueError("Saved selector training recipe differs")
    return binding(path)


def _retrieval(path, config):
    from exact.impl.retrieval.artifacts import resolve_local_retrieval_artifact
    from exact.utils.fitted_artifacts import fingerprint
    from exact.utils.provenance import dataset_signature_for_paths

    path = Path(path)
    recipe_path = path.parent / "recipe.json"
    recipe = read(recipe_path)
    application = recipe["application"]
    artifact = resolve_local_retrieval_artifact(path, expected_kind="cross_encoder")
    signature = dataset_signature_for_paths(config["data"]["source"], config["data"]["target"])
    if (
        application["dataset_signature"] != signature
        or artifact.metadata["dataset_lock"] != fingerprint(application)
        or artifact.metadata["fit_identity"] != fingerprint(recipe)
        or artifact.metadata["seed"] != recipe["seed"]
        or recipe["seed"] != config["run"]["seed"]
        or application["negative_label_policy"] != config["supervision"]["negative_label_policy"]
        or recipe["kind"] != "cross_encoder"
    ):
        raise ValueError("Saved retrieval model belongs to another case or seed")
    training = copy.deepcopy(config["candidates"]["cross_encoder"]["training"])
    model = {"identifier": training.pop("base_model"), "revision": training.pop("revision")}
    if model != recipe["base_model"] or any(recipe.get(key) != value for key, value in training.items()):
        raise ValueError("Saved retrieval training differs from the selected recipe")
    reporting = set(application["source_ids"])
    training_sources = {
        row["source"] if isinstance(row, dict) else row
        for row in artifact.metadata["training_pairs"]
    } | set(artifact.metadata.get("validation_sources", []))
    if not training_sources or reporting & training_sources:
        raise ValueError("Saved retrieval training overlaps reporting sources")
    return {
        "manifest": binding(path), "recipe": binding(recipe_path),
        "model_sha256": artifact.sha256, "model_path": str(artifact.model_path),
    }


def _one(root, pattern):
    paths = sorted(root.glob(pattern))
    if len(paths) > 1:
        raise ValueError(f"Ambiguous fitted artifact under {root}: {pattern}")
    return paths[0] if paths else None


def _source_record(path, key):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.harness import hash_payload

    report = read(path)
    arm, task = key.split("/", 1)
    if (report.get("experiment_id"), report.get("arm_id"), report.get("task_id"), report.get("seed"), report.get("stage")) != (
        "G4", arm, task, 17, "screen"
    ) or report.get("generate_rationales") is not False:
        raise ValueError("Original G4 manifest cell/seed/stage differs")
    config_path = path.parent / "_inputs/resolved.config.yaml"
    config = load_yaml_mapping(config_path)
    if hash_payload(config) != report["resolved_config_hash"]:
        raise ValueError("Original resolved config hash changed")
    if config["data"]["reference_role"] not in {"valid", "validation", "development"}:
        raise ValueError("G4 restart cannot change development reporting roles")
    return report, config_path, _canonical(config)


def prepare(old_wave, new_wave):
    """Verify the eight original recipes and freeze only reusable fitted bindings."""
    old_wave, new_wave = Path(old_wave).resolve(), Path(new_wave).resolve()
    if old_wave == new_wave or new_wave.is_relative_to(old_wave):
        raise ValueError("G4 restart requires a separate output root")
    manifest_path = new_wave / "g4-fit-restart.json"
    if manifest_path.exists():
        raise ValueError("Fit restart preparation already exists")
    recipe = read(old_wave / "recipe.json")
    repair = read(verified(recipe["repair_record"]))
    if set(repair["cells"]) != CELLS:
        raise ValueError("Fit restart requires all eight original G4 cells")
    current = old_wave / "runtime/exact-om-focused-v2/screen/runs/G4"
    records = {}
    # Validate every source before writing any new file.
    for key in sorted(CELLS):
        original_manifest = verified(repair["cells"][key]["manifest"])
        report, config_path, config = _source_record(original_manifest, key)
        source = original_manifest.parent
        if key.startswith("baseline/") and (
            report.get("status") != "complete" or report.get("extraction_complete") is not True
        ):
            raise ValueError("Saved baseline lacks a completed fitted application")
        fit_source = current / key / "seed-17" if key == "core/D0_E03-global_alignment" else source
        heads = {
            "selector": _one(fit_source / "fitting", "*/selector.json"),
            "fusion": _one(fit_source / "fitting", "*/fusion.json"),
            "calibration": _one(fit_source / "fitting", "score_calibrator.json"),
        }
        expected = {"selector"} if key.startswith("baseline/") else {"selector", "fusion", "calibration"}
        present = {name for name, path in heads.items() if path}
        complete_fit = present == expected
        if present and not complete_fit:
            raise ValueError("Partial learned-head set cannot silently enter a restart")
        if key.startswith("baseline/") and not complete_fit:
            raise ValueError("Completed baseline selector is missing")
        entry = {
            "original_manifest": binding(original_manifest), "original_config": binding(config_path),
            "canonical_original_config": config, "complete_fit": complete_fit,
            "heads": {name: {"source": _head(heads[name], name, config)} for name in sorted(present)},
            "retrieval": None,
            "action": "reuse_saved_heads_no_refit" if complete_fit else "reuse_retrieval_fit_train_missing_heads",
        }
        if key.startswith("core/"):
            retrieval = fit_source / "fitting/cross_encoder/retrieval_artifact.json"
            entry["retrieval"] = _retrieval(retrieval, config)
        if key == "core/D0_E03-global_alignment":
            checkpoints = sorted((fit_source / "checkpoints").glob("inference_*.json"))
            checkpoints = [p for p in checkpoints if "additional_models" not in p.name]
            if len(checkpoints) != 1:
                raise ValueError("Expected one stopped G4 primary checkpoint")
            checkpoint = read(checkpoints[0])
            fusion = checkpoint["checkpoint_fingerprint_payload"]["models"][0]["payload"]["pair_adaptive_channels"]["experiments"]["fusion_artifact"]
            if entry["heads"].get("fusion", {}).get("source") != {"path": fusion["path"], "sha256": fusion["sha256"]}:
                raise ValueError("Saved fitted fusion differs from the stopped primary checkpoint")
            entry["stopped_checkpoint_not_imported"] = binding(checkpoints[0])
            entry["discarded_prediction_prefix_rows"] = checkpoint["processed_examples"]
        records[key] = entry
    for key, entry in records.items():
        for name, head in entry["heads"].items():
            source = verified(head["source"])
            target = new_wave / "retained-fits" / key / (name + ".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("Fit destination already exists")
            shutil.copyfile(source, target)
            head["retained"] = binding(target)
            if head["retained"]["sha256"] != head["source"]["sha256"]:
                raise ValueError("Copied head differs from the learned parameters")
    write(manifest_path, {
        "schema_version": 1, "kind": KIND, "source_wave": str(old_wave), "cells": records,
        "prediction_restart": "all_eight_cells_from_zero", "old_predictions_imported": False,
        "old_training_aggregates_imported": False, "old_numerical_caches_imported": False,
        "controlled_perturbations": False, "cuda_graphs": False,
        "fitting_scope": "same-cell completed heads only; unfinished heads fit normally with saved retrieval models",
    }, immutable=True)
    return binding(manifest_path)


def _verify_manifest(path):
    manifest = read(path)
    if manifest.get("kind") != KIND or set(manifest.get("cells", {})) != CELLS or any(
        manifest.get(field) is not False
        for field in ("old_predictions_imported", "old_training_aggregates_imported", "old_numerical_caches_imported", "controlled_perturbations", "cuda_graphs")
    ):
        raise ValueError("Invalid fit-only G4 restart contract")
    for key, entry in manifest["cells"].items():
        original_config = verified(entry["original_config"])
        _, config_path, config = _source_record(verified(entry["original_manifest"]), key)
        if config_path != original_config:
            raise ValueError("Saved config is outside the bound original cell")
        if config != entry["canonical_original_config"]:
            raise ValueError("Original fitted recipe changed")
        expected_heads = {"selector"} if key.startswith("baseline/") else {"selector", "fusion", "calibration"}
        if (
            type(entry["complete_fit"]) is not bool
            or set(entry["heads"]) != (expected_heads if entry["complete_fit"] else set())
            or (key.startswith("baseline/") or key == "core/D0_E03-global_alignment") and not entry["complete_fit"]
            or bool(entry.get("retrieval")) != key.startswith("core/")
        ):
            raise ValueError("Missing or inconsistent fitted head set")
        for name, head in entry["heads"].items():
            source, retained = verified(head["source"]), verified(head["retained"])
            if binding(source)["sha256"] != binding(retained)["sha256"]:
                raise ValueError("Retained head differs from original")
            _head(retained, name, config)
        if entry.get("retrieval"):
            retrieval = entry["retrieval"]
            verified(retrieval["recipe"])
            if _retrieval(verified(retrieval["manifest"]), config) != retrieval:
                raise ValueError("Retained retrieval model bytes changed")
        if entry.get("stopped_checkpoint_not_imported"):
            verified(entry["stopped_checkpoint_not_imported"])
    return manifest


def resolve_cell(resolved, *, key, seed, stage, manifest):
    """Return an explicitly amended config; all unrelated resolved fields must match."""
    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.evidence_diagnostics import ordinary_inference_config

    if key not in CELLS or seed != 17 or stage != "screen":
        raise ValueError("Fit restart cannot add or replace a G4 case/seed/stage")
    entry = manifest["cells"][key]
    expected = entry["canonical_original_config"]
    # The parent may already record the reviewed diagnostic-only omission in its declaration.
    result, _ = ordinary_inference_config(resolved)
    if result != ordinary_inference_config(expected)[0]:
        raise ValueError("G4 resolved configuration differs outside the reviewed diagnostic omission: " + key)
    for name, head in entry["heads"].items():
        _set(result, HEAD_FIELDS[name], head["retained"]["path"])
    if entry.get("retrieval"):
        _set(result, "candidates.cross_encoder.artifact", entry["retrieval"]["manifest"]["path"])
        _set(result, "candidates.cross_encoder.training", None)
    if entry["complete_fit"]:
        result["data"]["train_candidates"] = None
    validated = ConfigModel.from_mapping(result, warn_v1=False)
    return validated.model_dump(mode="json", by_alias=True), validated.fingerprint()


@contextmanager
def cell_configs(manifest_path):
    """Install only for the bound prepared worker, and restore on success or failure."""
    from exact.experiments import harness

    manifest = _verify_manifest(manifest_path)
    original = harness._resolve_config

    def resolve(source, *, task, arm, stage, seed, **kwargs):
        resolved, digest, label, supervision = original(source, task=task, arm=arm, stage=stage, seed=seed, **kwargs)
        if source.config.experiment_id == "G4":
            resolved, digest = resolve_cell(resolved, key=arm.id + "/" + task.id, seed=seed, stage=stage, manifest=manifest)
        return resolved, digest, label, supervision

    harness._resolve_config = resolve
    try:
        yield manifest
    finally:
        harness._resolve_config = original
