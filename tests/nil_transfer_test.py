"""Frozen NIL heads use the existing donor manifest; recipient labels are never fit."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.impl.models.selector.nil_head import (
    fit_nil_artifact,
    validate_nil_application,
)
from exact.utils.artifact_transfer import (
    freeze_transfer_manifest,
    transfer_feature_contract,
    validate_transfer_config,
)
from exact.utils.provenance import dataset_signature_for_paths
from tests.benchmark_nil_test import benchmark_task
from tests.grouped_fitting_test import TinyScorer, selector, tiny_runner


@pytest.fixture
def nil_transfer(tmp_path):
    source, target = tmp_path / "recipient-source.owl", tmp_path / "recipient-target.owl"
    source.write_text("source")
    target.write_text("target")
    frame, labels, reference = benchmark_task()
    artifact_path = tmp_path / "donor-nil.json"
    artifact = fit_nil_artifact(
        frame,
        labels,
        reference,
        artifact_path,
        application={
            "dataset_signature": "donor",
            "source_ids": ["donor-valid"],
            "entity_kinds": ["class"],
            "nil_label_semantics": "benchmark_pool",
        },
    )
    manifest = tmp_path / "application.json"
    config = ConfigModel.from_mapping(
        {
            "config_version": 2,
            "data": {"source": str(source), "target": str(target)},
            "supervision": {"mode": "label_free", "transfer_artifact": str(manifest)},
            "matching": {
                "threshold": 0.5,
                "nil": {
                    "mode": "fitted",
                    "label_semantics": "benchmark_pool",
                    "artifact": str(artifact_path),
                },
            },
        },
        warn_v1=False,
    )
    recipe = transfer_feature_contract(config)
    freeze_transfer_manifest(
        manifest,
        {"nil": artifact_path},
        recipient_signature=dataset_signature_for_paths(source, target),
        feature_contract=recipe,
        score_threshold=0.5,
    )
    dataset = SimpleNamespace(
        dataset_signature=dataset_signature_for_paths(source, target),
        transfer_artifact=manifest,
        transfer_feature_contract=recipe,
        transfer_score_threshold=0.5,
        _entity_kinds=["class"],
    )
    report = frame.iloc[:2].copy()
    report["Src"], report["P_rank"], report["S_select"] = "recipient", [0.7, 0.3], [0.8, 0.0]
    return config, dataset, report, artifact_path, artifact


def test_frozen_nil_transfer_preserves_probabilities_and_skips_recipient_fit(
    nil_transfer, tmp_path, monkeypatch
):
    config, dataset, report, artifact_path, artifact = nil_transfer
    original = artifact_path.read_bytes()
    assert validate_transfer_config(config)["recipient_refit"] is False
    head = selector()
    head.nil_config.update(config.matching.nil.model_dump(mode="json"))
    donor = SimpleNamespace(dataset_signature="donor")
    expected = head._apply_joint_nil_ranking(report.copy(), dataset=donor)
    actual = head._apply_joint_nil_ranking(report.copy(), dataset=dataset)
    pd.testing.assert_frame_equal(actual, expected)
    validate_nil_application(dataset, artifact_path, artifact, head.nil_config)

    def forbidden(*args, **kwargs):
        raise AssertionError("Recipient fitting is forbidden")

    monkeypatch.setattr("exact.impl.models.selector.nil_head._fit", forbidden)
    runner = tiny_runner(tmp_path, dataset, TinyScorer(), head)
    runner.training_candidates_file_path = tmp_path / "must-not-read.tsv"
    runner.fit_training_pool()
    assert runner.transfer_report["recipient_refit"] is False
    assert artifact_path.read_bytes() == original
    head.nil_config["training_source_labels"] = str(tmp_path / "must-not-read-labels.tsv")
    with pytest.raises(ValueError, match="recipient training source labels"):
        runner.fit_training_pool()


def test_nil_transfer_rejects_missing_manifest_changed_recipe_and_donor_bytes(nil_transfer):
    config, dataset, report, artifact_path, artifact = nil_transfer
    head = selector()
    head.nil_config.update(config.matching.nil.model_dump(mode="json"))
    with pytest.raises(ValueError, match="dataset mismatch"):
        head._apply_joint_nil_ranking(
            report.copy(), dataset=SimpleNamespace(dataset_signature="recipient")
        )
    config.data.refs["train"] = Path("must-not-read.tsv")
    with pytest.raises(ValueError, match="no recipient training inputs"):
        validate_transfer_config(config)
    config.data.refs.clear()
    dataset.transfer_feature_contract = {"changed": True}
    with pytest.raises(ValueError, match="feature-producing recipe"):
        head._apply_joint_nil_ranking(report.copy(), dataset=dataset)
    dataset.transfer_feature_contract = transfer_feature_contract(config)
    changed = config.model_copy(deep=True)
    changed.matching.nil.label_semantics = "natural"
    assert transfer_feature_contract(changed) != transfer_feature_contract(config)
    head.nil_config["label_semantics"] = "natural"
    with pytest.raises(ValueError, match="label-semantics"):
        head._apply_joint_nil_ranking(report.copy(), dataset=dataset)
    head.nil_config["label_semantics"] = "benchmark_pool"
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_nil_application(dataset, artifact_path, artifact, head.nil_config)
