from types import SimpleNamespace

import pandas as pd
import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.impl.models.selector import CandidateSetSelector
from exact.impl.trainer.fitting import TrainingPoolMixin
from exact.utils.artifact_transfer import freeze_transfer_manifest


def _selector():
    return CandidateSetSelector(
        enabled=True,
        global_only=False,
        strategy="calibrated_rank_accept",
        calibration={"max_epochs": 8, "min_positive_sources": 1, "validation_folds": 3},
        experiment_config={"enabled": True},
        request_seed=17,
    )


def test_actual_donor_fit_applies_unchanged_to_recipient_without_labels(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "Src": f"train{i}",
                "Tgt": f"target{i}-{j}",
                "S_final": score,
                "s_label": score,
                "S_struct": score,
            }
            for i in range(9)
            for j, score in enumerate((0.9, 0.2))
        ]
    )
    donor = _selector()
    head_path = tmp_path / "donor.json"
    payload = donor.fit_training_artifact(
        rows,
        {(f"train{i}", f"target{i}-0") for i in range(9)},
        head_path,
        application={
            "dataset_signature": "donor",
            "source_ids": ["donor-report"],
            "negative_label_policy": "complete_reference",
        },
    )
    original = head_path.read_bytes()
    manifest_path = tmp_path / "transfer.json"
    freeze_transfer_manifest(
        manifest_path,
        {"selector": head_path},
        recipient_signature="recipient",
        feature_contract={"encoders": "locked-fixture"},
        score_threshold=0.5,
    )
    dataset = SimpleNamespace(
        dataset_signature="recipient",
        transfer_artifact=manifest_path,
        transfer_feature_contract={"encoders": "locked-fixture"},
        transfer_score_threshold=0.5,
        _entity_kinds=["class"],
    )
    report = rows.iloc[:2].copy()
    report["Src"] = "recipient-source"
    recipient = _selector()
    recipient.rerank_config["artifact"] = str(head_path)
    recipient._fit_rank_model = lambda *_a, **_k: pytest.fail("recipient fit is forbidden")
    output = recipient(candidate_df=report, dataset=dataset, threshold=0.5)["candidate_df"]
    assert output.selection_winner.sum() == 1
    assert set(output.selection_accept_threshold) == {payload["accept_threshold"]}
    assert head_path.read_bytes() == original
    trainer = TrainingPoolMixin()
    trainer.dataset = dataset
    trainer.training_candidates_file_path = tmp_path / "must-not-open.tsv"
    trainer.fit_training_pool()
    assert trainer.transfer_report["recipient_refit"] is False
    with pytest.raises(ValueError, match="donor decision threshold"):
        recipient(candidate_df=report, dataset=dataset, threshold=0.6)
    dataset.transfer_feature_contract = {"encoders": "changed"}
    with pytest.raises(ValueError, match="feature-producing recipe"):
        recipient(candidate_df=report, dataset=dataset, threshold=0.5)
    dataset.transfer_feature_contract = {"encoders": "locked-fixture"}
    head_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        recipient(candidate_df=report, dataset=dataset, threshold=0.5)


def test_transfer_is_explicit_and_cannot_enable_recipient_supervision():
    with pytest.raises(ValueError, match="label_free supervision"):
        ConfigModel.from_mapping(
            {"config_version": 2, "supervision": {"transfer_artifact": "x.json"}}
        )
    with pytest.raises(ValueError, match="label_free supervision"):
        ConfigModel.from_mapping(
            {
                "config_version": 2,
                "supervision": {
                    "transfer_artifact": "x.json",
                    "mode": "label_free",
                    "components": {"accept": "supervised"},
                },
            }
        )
    with pytest.raises(ValueError, match="fixed decision threshold"):
        ConfigModel.from_mapping(
            {
                "config_version": 2,
                "supervision": {"transfer_artifact": "x.json", "mode": "label_free"},
                "matching": {"calibration": {"threshold_mode": "otsu"}},
            }
        )
