"""G4 restarts retain same-cell parameters, never old scoring/training aggregates."""

import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.experiments.harness import hash_payload
from exact.utils.fitted_artifacts import fingerprint
from exact.utils.provenance import dataset_signature_for_paths
from tools import recover_g4_batching as restart
from tools.prepared_batch import binding, read, verified, write


def source_fixture(tmp_path):
    old, original, new = tmp_path / "stopped", tmp_path / "original", tmp_path / "restart"
    cells = {}
    for key in sorted(restart.CELLS):
        arm, task = key.split("/")
        case, mode = task.split("-", 1)
        source, target = tmp_path / (case + "-s.owl"), tmp_path / (case + "-t.owl")
        if not source.exists():
            source.write_text("source")
            target.write_text("target")
        model = ConfigModel()
        model.data.source, model.data.target = source, target
        model.data.execution_mode = mode
        model.data.reference_role = "valid"
        model.data.refs = {"train": tmp_path / "private-training.tsv", "valid": tmp_path / "development.tsv"}
        model.data.train_candidates = tmp_path / "private-candidates.tsv"
        model.run.seed = 17
        model.matching.channels.diff.controlled_perturbations = True
        config = model.model_dump(mode="json", by_alias=True)
        if arm == "core":
            config["matching"]["fusion"]["mode"] = "analytic_fitted"
            config["matching"]["calibration"]["mode"] = "platt"
            config["candidates"]["cross_encoder"].update(mode="on", training={"base_model": "test/encoder", "revision": "a" * 40})
        config = restart._canonical(config)
        run = original / key / "seed-17"
        config_path = run / "_inputs/resolved.config.yaml"
        write(config_path, config)
        manifest = run / "manifest.json"
        write(manifest, {"experiment_id": "G4", "arm_id": arm, "task_id": task, "seed": 17,
                         "stage": "screen", "generate_rationales": False,
                         "resolved_config_hash": hash_payload(config), "status": "complete" if arm == "baseline" else "failed",
                         "extraction_complete": arm == "baseline"})
        cells[key] = {"manifest": binding(manifest)}
        fit_run = old / "runtime/exact-om-focused-v2/screen/runs/G4" / key / "seed-17" if key == "core/D0_E03-global_alignment" else run
        application = {"dataset_signature": dataset_signature_for_paths(source, target),
                       "source_ids": [case + "-report"], "negative_label_policy": config["supervision"]["negative_label_policy"]}
        complete = arm == "baseline" or key == "core/D0_E03-global_alignment"
        if complete:
            for name in (["selector"] if arm == "baseline" else ["selector", "fusion", "calibration"]):
                path = fit_run / "fitting" / ("score_calibrator.json" if name == "calibration" else "fit/" + name + ".json")
                write(path, {"kind": "fitted_selector" if name == "selector" else "score_calibrator",
                             "mode": config["matching"]["fusion"]["mode"], "seed": 17,
                             "fit_provenance": {"application": application, "training_sources": [case + "-train"],
                                                "seed": 17, "score_calibration": config["matching"]["calibration"]["mode"],
                                                "accept_training": config["selector"]["accept_training"],
                                                "recipe": {"rerank": {k: v for k, v in config["selector"]["rerank"].items() if k != "artifact"}}},
                             "learned_parameters": [0.1, 0.9]})
        if arm == "core":
            root = fit_run / "fitting/cross_encoder"
            training = copy.deepcopy(config["candidates"]["cross_encoder"]["training"])
            base_model = {"identifier": training.pop("base_model"), "revision": training.pop("revision")}
            recipe = {**training, "base_model": base_model, "application": application, "seed": 17, "kind": "cross_encoder"}
            write(root / "recipe.json", recipe)
            write(root / "retrieval_artifact.json", {
                "schema_version": 1, "artifact_type": "cross_encoder", "model_path": "model",
                "base_model": base_model, "dataset_lock": fingerprint(application), "seed": 17, "epochs": training["epochs"],
                "fit_identity": fingerprint(recipe), "training_pairs": [{"source": case + "-train", "split_role": "train"}],
                "negative_policy": "confirmed_negative", "reference_completeness": "known_incomplete", "validation_sources": [case + "-fit-valid"],
            })
            (root / "model").mkdir()
            (root / "model/weights.bin").write_bytes(b"model bytes")
        if key == "core/D0_E03-global_alignment":
            write(fit_run / "checkpoints/inference_1.json", {
                "processed_examples": 9728, "checkpoint_fingerprint_payload": {"models": [{"payload": {
                    "pair_adaptive_channels": {"experiments": {"fusion_artifact": binding(fit_run / "fitting/fit/fusion.json")}}
                }}]}
            })
        write(fit_run / "fitting/training_scores.json", {"must_not_import": True})
        write(fit_run / "predictions.json", {"must_not_import": True})
    write(old / "repair.json", {"cells": cells})
    write(old / "recipe.json", {"repair_record": binding(old / "repair.json")})
    return old, original, new


def prepared(tmp_path):
    old, original, new = source_fixture(tmp_path)
    manifest_path = verified(restart.prepare(old, new))
    return old, original, new, manifest_path, restart._verify_manifest(manifest_path)


def test_prepare_reuses_only_same_cell_heads_without_reading_training_or_predictions(tmp_path, monkeypatch):
    old, _, new = source_fixture(tmp_path)
    original_open = Path.open

    def guard(path, *args, **kwargs):
        assert not path.name.startswith("private-")
        assert path.name not in {"predictions.json", "training_scores.json", "development.tsv"}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    manifest = restart._verify_manifest(verified(restart.prepare(old, new)))
    assert sum(row["complete_fit"] for row in manifest["cells"].values()) == 5
    assert len(list(new.rglob("*.json"))) == 8  # seven tiny heads plus the manifest
    assert not list(new.rglob("*.bin")) and not (new / "runtime").exists()
    for key, row in manifest["cells"].items():
        assert row["retrieval"] is None or Path(row["retrieval"]["manifest"]["path"]).is_relative_to(tmp_path)
        for head in row["heads"].values():
            assert head["source"]["sha256"] == head["retained"]["sha256"]
            assert key in head["retained"]["path"]
    assert manifest["cells"]["core/D0_E03-global_alignment"]["discarded_prediction_prefix_rows"] == 9728


def test_resolve_preserves_scientific_recipe_and_pending_training(tmp_path):
    *_, manifest = prepared(tmp_path)
    for key, row in manifest["cells"].items():
        old = row["canonical_original_config"]
        new, digest = restart.resolve_cell(old, key=key, seed=17, stage="screen", manifest=manifest)
        assert digest == ConfigModel.from_mapping(new, warn_v1=False).fingerprint()
        assert new["data"]["refs"] == old["data"]["refs"]
        assert new["supervision"] == old["supervision"]
        assert new["matching"]["channels"]["diff"]["controlled_perturbations"] is False
        if row["complete_fit"]:
            assert new["data"]["train_candidates"] is None
        else:
            assert new["data"]["train_candidates"] == old["data"]["train_candidates"]
        if row["retrieval"]:
            assert new["candidates"]["cross_encoder"]["training"] is None
            assert new["candidates"]["cross_encoder"]["artifact"] == row["retrieval"]["manifest"]["path"]
        restored = copy.deepcopy(new)
        restored["data"]["train_candidates"] = old["data"]["train_candidates"]
        restored["candidates"] = old["candidates"]
        restored["matching"]["channels"]["diff"]["controlled_perturbations"] = True
        for name in row["heads"]:
            restart._set(restored, restart.HEAD_FIELDS[name], None)
        assert restored == old


@pytest.mark.parametrize("change", ["source", "train_reference", "threshold", "mode", "seed", "stage", "key"])
def test_resolve_rejects_unapproved_scientific_changes(tmp_path, change):
    *_, manifest = prepared(tmp_path)
    key, seed, stage = "core/D0_E03-global_alignment", 17, "screen"
    config = copy.deepcopy(manifest["cells"][key]["canonical_original_config"])
    if change == "source":
        config["data"]["source"] = "another.owl"
    elif change == "train_reference":
        config["data"]["refs"]["train"] = "another.tsv"
    elif change == "threshold":
        config["matching"]["threshold"] = 0.999
    elif change == "mode":
        config["data"]["execution_mode"] = "local_ranking"
    elif change == "seed":
        seed = 18
    elif change == "stage":
        stage = "final"
    else:
        key = "core/another-global_alignment"
    with pytest.raises(ValueError, match="differs|cannot"):
        restart.resolve_cell(config, key=key, seed=seed, stage=stage, manifest=manifest)


@pytest.mark.parametrize("change", ["source_config", "retained_head", "missing_head", "model_bytes", "wrong_case", "retrieval_seed", "overlap"])
def test_manifest_rejects_missing_or_changed_fits(tmp_path, change):
    *_, manifest_path, manifest = prepared(tmp_path)
    row = manifest["cells"]["core/D0_E03-global_alignment"]
    if change == "source_config":
        Path(row["original_config"]["path"]).write_text("changed")
    elif change == "retained_head":
        Path(row["heads"]["selector"]["retained"]["path"]).write_text("changed")
    elif change == "missing_head":
        row["heads"].pop("selector")
        write(manifest_path, manifest)
    elif change == "model_bytes":
        (Path(row["retrieval"]["model_path"]) / "weights.bin").write_bytes(b"other bytes")
    else:
        config = row["canonical_original_config"]
        if change == "wrong_case":
            config["data"]["source"] = str(tmp_path / "D1-s.owl")
        elif change == "retrieval_seed":
            config["run"]["seed"] = 18
        else:
            path = Path(row["retrieval"]["manifest"]["path"])
            value = read(path)
            value["validation_sources"] = ["D0_E03-report"]
            write(path, value)
        with pytest.raises(ValueError, match="another|overlaps"):
            restart._retrieval(Path(row["retrieval"]["manifest"]["path"]), config)
        return
    with pytest.raises((ValueError, FileNotFoundError), match="changed|Missing|differs"):
        restart._verify_manifest(manifest_path)


def test_original_config_hash_fails_before_writes(tmp_path):
    old, original, new = source_fixture(tmp_path)
    path = original / "baseline/D0_E03-global_alignment/seed-17/_inputs/resolved.config.yaml"
    config = read(path)
    config["run"]["seed"] = 18
    write(path, config)
    with pytest.raises(ValueError, match="config hash"):
        restart.prepare(old, new)
    assert not new.exists()


def test_partial_completed_fit_cannot_silently_refit(tmp_path):
    old, _, new = source_fixture(tmp_path)
    head = old / "runtime/exact-om-focused-v2/screen/runs/G4/core/D0_E03-global_alignment/seed-17/fitting/fit/selector.json"
    head.unlink()
    with pytest.raises(ValueError, match="Partial learned-head"):
        restart.prepare(old, new)
    assert not new.exists()


def test_selector_recipe_is_checked_before_dispatch(tmp_path):
    *_, manifest = prepared(tmp_path)
    row = manifest["cells"]["core/D0_E03-global_alignment"]
    config = copy.deepcopy(row["canonical_original_config"])
    config["selector"]["accept_training"] = "changed"
    with pytest.raises(ValueError, match="selector training recipe"):
        restart._head(Path(row["heads"]["selector"]["retained"]["path"]), "selector", config)


def test_scoped_hook_restores_original_and_preserves_supervision(tmp_path, monkeypatch):
    from exact.experiments import harness
    *_, path, manifest = prepared(tmp_path)
    key = "core/D0_E03-global_alignment"
    config = manifest["cells"][key]["canonical_original_config"]
    supervision = {"label": "target_supervised", "components": {"rerank": "supervised"}}

    def original(*args, **kwargs):
        return copy.deepcopy(config), "old", "target_supervised", supervision
    monkeypatch.setattr(harness, "_resolve_config", original)
    args = {"task": NS(id=key.split("/")[1]), "arm": NS(id="core"), "seed": 17, "stage": "screen"}
    with pytest.raises(RuntimeError, match="stop"):
        with restart.cell_configs(path):
            result = harness._resolve_config(NS(config=NS(experiment_id="G4")), **args)
            assert result[0]["data"]["train_candidates"] is None
            assert result[2:] == ("target_supervised", supervision)
            other = harness._resolve_config(NS(config=NS(experiment_id="E07")), **args)
            assert other[0] == config and other[1] == "old"
            raise RuntimeError("stop")
    assert harness._resolve_config is original


def test_retained_heads_are_discovered_for_final_deployment(tmp_path):
    from exact.utils.frozen_inference import freeze_runtime_fitted_artifacts, inject_runtime_fitted_artifacts
    *_, manifest = prepared(tmp_path)
    row = manifest["cells"]["core/D0_E03-global_alignment"]
    config, _ = restart.resolve_cell(row["canonical_original_config"], key="core/D0_E03-global_alignment", seed=17, stage="screen", manifest=manifest)
    root = tmp_path / "new-cell"
    write(root / "config.yaml", config)
    model = NS(fusion_config=config["matching"]["fusion"])
    selector = NS(rerank_config=config["selector"]["rerank"], matching_calibration=config["matching"]["calibration"])
    trainer = NS(output_dir=root, model=model, models=[model, selector],
                 dataset=NS(dataset_signature=dataset_signature_for_paths(config["data"]["source"], config["data"]["target"]),
                            _candidate_generation_params=config["candidates"]))
    freeze_runtime_fitted_artifacts(trainer)
    receipt = root / "fitting/deployment-artifacts.json"
    result = copy.deepcopy(row["canonical_original_config"])
    inject_runtime_fitted_artifacts(result, receipt, selected_config=root / "config.yaml", source=config["data"]["source"], target=config["data"]["target"])
    assert result["selector"]["rerank"]["artifact"] == row["heads"]["selector"]["retained"]["path"]
    assert result["matching"]["fusion"]["artifact"] == row["heads"]["fusion"]["retained"]["path"]
    assert result["matching"]["calibration"]["artifact"] == row["heads"]["calibration"]["retained"]["path"]
    assert result["candidates"]["cross_encoder"]["artifact"] == row["retrieval"]["manifest"]["path"]
