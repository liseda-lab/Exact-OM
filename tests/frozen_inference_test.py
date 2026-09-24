"""Real CPU head replay with full native population, never final-phase fitting."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.experiments.public_inference import prepare_public_inference
from exact.impl.models.selector import CandidateSetSelector
from exact.impl.trainer.fitting import TrainingPoolMixin
from exact.utils.frozen_inference import validate_inference_config
from exact.utils.provenance import dataset_signature_for_paths


def test_full_native_deployment_applies_fitted_head_over_training_sources(tmp_path):
    source, target = tmp_path / "s.ofn", tmp_path / "t.ofn"
    source.write_text(
        "Ontology(" + " ".join(f"Declaration(Class(<urn:s{i}>))" for i in range(4)) + ")"
    )
    target.write_text("Ontology(Declaration(Class(<urn:yes>)) Declaration(Class(<urn:no>)))")
    signature = dataset_signature_for_paths(source, target)
    rows = pd.DataFrame(
        [
            {
                "Src": f"urn:s{i}",
                "Tgt": f"urn:{t}",
                "S_final": score,
                "s_label": score,
                "S_struct": score,
            }
            for i in range(3)
            for t, score in (("yes", 0.9), ("no", 0.2))
        ]
    )
    fitted = CandidateSetSelector(
        enabled=True,
        global_only=False,
        strategy="calibrated_rank_accept",
        calibration={"max_epochs": 3, "min_positive_sources": 1, "validation_folds": 2},
        experiment_config={"enabled": True},
        request_seed=17,
    )
    artifact = tmp_path / "head.json"
    payload = fitted.fit_training_artifact(
        rows,
        {(f"urn:s{i}", "urn:yes") for i in range(3)},
        artifact,
        application={
            "dataset_signature": signature,
            "source_ids": ["urn:s3"],
            "negative_label_policy": "complete_reference",
        },
    )
    config = ConfigModel.from_mapping(
        {
            "config_version": 2,
            "data": {
                "source": str(source),
                "target": str(target),
                "refs": {"train": "/must-not-read/train.tsv", "test": "/must-not-read/test.tsv"},
                "train_candidates": "/must-not-read/train-pool.tsv",
            },
            "selector": {
                "enabled": True,
                "runtime_enabled": True,
                "rerank": {"artifact": str(artifact)},
            },
            "supervision": {
                "mode": "label_free",
                "components": {"rerank": "supervised", "accept": "supervised"},
            },
        }
    )
    selected = tmp_path / "selected.yaml"
    selected.write_text(json.dumps(config.model_dump(mode="json")))
    plan = json.loads(
        prepare_public_inference(
            selected, tmp_path / "public", source=source, target=target, track="bioml-global"
        ).read_text()
    )
    deployed = ConfigModel.load_config(Path(plan["runs"][0]["config"]["path"]))
    assert deployed.data.refs == {} and deployed.data.train_candidates is None
    assert set(deployed.data.source_universe.read_text().splitlines()) == {
        f"urn:s{i}" for i in range(4)
    }
    manifest = validate_inference_config(deployed)
    dataset = SimpleNamespace(dataset_signature=signature, frozen_inference_manifest=manifest)
    before = artifact.read_bytes()
    deployment_rows = pd.concat([rows, rows.iloc[:2].assign(Src="urn:s3")], ignore_index=True)
    fitted._fit_rank_model = lambda *_a, **_k: pytest.fail("Final deployment must not fit")
    result = fitted(candidate_df=deployment_rows.copy(), dataset=dataset, threshold=0.5)["candidate_df"]
    assert result.selection_winner.sum() == 4
    assert set(result.selection_accept_threshold) == {payload["accept_threshold"]}
    trainer = TrainingPoolMixin()
    trainer.dataset = dataset
    trainer.fit_relation_head = lambda: pytest.fail(
        "Final deployment must not refit relation heads"
    )
    trainer.fit_training_pool()
    assert trainer._training_pool_fitted and artifact.read_bytes() == before
    # Research evaluation retains its overlap guard; deployment admission is explicit.
    with pytest.raises(ValueError, match="overlap"):
        fitted(
            candidate_df=rows.copy(),
            dataset=SimpleNamespace(dataset_signature=signature),
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="no fitting or evaluation"):
        validate_inference_config(deployed, run_eval=True)
    pool = tmp_path / "added.tsv"
    pool.write_text("Src\tTgt\nurn:s0\turn:yes\n")
    deployed.data.candidates = pool
    with pytest.raises(ValueError, match="population binding"):
        validate_inference_config(deployed)
    deployed.data.candidates = None
    deployed.selector.rerank.model = "channel_gating"
    with pytest.raises(ValueError, match="feature recipe"):
        validate_inference_config(deployed)
    deployed.selector.rerank.model = "current_linear"
    artifact.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="artifact hash"):
        validate_inference_config(deployed)


def test_missing_fitted_head_is_rejected_without_reading_training_inputs(tmp_path):
    ontology = tmp_path / "ontology.ofn"
    ontology.write_text("Ontology(Declaration(Class(<urn:c>)))")
    selected = tmp_path / "selected.yaml"
    selected.write_text(
        "config_version: 2\ndata:\n  refs:\n    train: /must-not-read/train.tsv\nselector:\n  runtime_enabled: true\nsupervision:\n  mode: label_free\n  components:\n    accept: supervised\n"
    )
    with pytest.raises(ValueError, match="already fitted selector"):
        prepare_public_inference(
            selected, tmp_path / "public", source=ontology, target=ontology, track="bioml-global"
        )
