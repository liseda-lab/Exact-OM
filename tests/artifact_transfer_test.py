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


def test_completed_donor_binding_two_recipients_and_relocation(tmp_path):
    import json
    from copy import deepcopy
    from pathlib import Path

    from exact.core.entities.configs.yaml_io import dump_yaml_document
    from exact.experiments import harness
    from exact.experiments.schema import ExperimentConfig
    from exact.experiments.transfer import materialize_transfer
    from exact.utils.artifact_transfer import (
        transfer_feature_contract,
        validate_transfer_config,
    )
    from exact.utils.fitted_artifacts import freeze_json
    from exact.utils.provenance import dataset_signature_for_paths

    paths = []
    for name in [
        "donor-source",
        "donor-target",
        "first-source",
        "first-target",
        "second-source",
        "second-target",
    ]:
        path = tmp_path / (name + ".owl")
        path.write_text(name)
        paths.append(path)
    donor_config = ConfigModel.load_config("exact/default_config.yaml")
    donor_config.data.source, donor_config.data.target = paths[:2]
    donor_config.run.seed = 17
    donor_config.selector.enabled = True
    donor_config.selector.runtime_enabled = True
    donor_config.selector.runtime_global_only = False
    output = (
        tmp_path
        / "fixture"
        / "screen"
        / "runs"
        / "E18"
        / "current_listwise"
        / "development"
        / "seed-17"
    )
    (output / "_inputs").mkdir(parents=True)
    (output / "_inputs" / "resolved.config.yaml").write_text(
        dump_yaml_document(donor_config.model_dump(mode="json"))
    )
    head = output / "fitting" / "cache" / "selector.json"
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
    _selector().fit_training_artifact(
        rows,
        {(f"train{i}", f"target{i}-0") for i in range(9)},
        head,
        application={
            "dataset_signature": dataset_signature_for_paths(*paths[:2]),
            "source_ids": ["donor-report"],
            "negative_label_policy": "complete_reference",
        },
    )
    original = head.read_bytes()
    tasks = []
    for index, (left, right) in enumerate([paths[2:4], paths[4:6]]):
        universe = tmp_path / f"sources-{index}.txt"
        universe.write_text(f"report-{index}\n")
        tasks.append(
            {
                "id": f"recipient-{index}",
                "split_role": "development",
                "reference_role": "valid",
                "overlay": {
                    "data": {
                        "source": str(left),
                        "target": str(right),
                        "source_universe": str(universe),
                    },
                    "supervision": {"negative_label_policy": "confirmed_negatives"},
                },
            }
        )
    declaration = {
        "schema_version": 2,
        "experiment_id": "E16",
        "title": "Bounded donor transfer",
        "base_config": str(Path("exact/default_config.yaml").resolve()),
        "screen": {"tasks": tasks, "seeds": [17]},
        "confirm": {
            "tasks": [
                {**tasks[0], "id": "reporting", "split_role": "reporting", "reference_role": "test"}
            ],
            "seeds": [17, 29, 43],
        },
        "arms": [
            {"id": "label_free", "role": "baseline", "supervision_label": "target_label_free"},
            {
                "id": "in_pair_supervised",
                "role": "candidate",
                "supervision_label": "in_pair_supervised",
            },
            {
                "id": "donor_transfer",
                "role": "candidate",
                "supervision_label": "cross_pair_transfer",
            },
        ],
        "selection": {
            "decisions": [
                {
                    "id": "transfer",
                    "baseline": "label_free",
                    "candidates": ["in_pair_supervised", "donor_transfer"],
                    "metric": "F1",
                }
            ]
        },
        "design": {
            "primary_comparison": "transfer",
            "primary_endpoint": "F1",
            "independent_unit": "source",
            "power_status": "descriptive",
            "assumptions": ["synthetic test"],
        },
        "frozen_constants": {"donor_transfer": {"producer": "E18"}},
    }
    path = tmp_path / "E16.json"
    freeze_json(path, declaration)
    source = harness.ExperimentSource(ExperimentConfig.model_validate(declaration), path)
    manifests = [
        {
            "experiment_id": "E18",
            "arm_id": "current_listwise",
            "stage": "screen",
            "seed": 17,
            "status": "complete",
            "resolved_config_hash": donor_config.fingerprint(),
            "fingerprint_payload": {"output_dir": str(output)},
        }
    ]
    selections = {
        "E18": {
            "status": "selected",
            "decisions": [{"selected_arm": "current_listwise", "baseline": "analytic"}],
        }
    }
    bound = materialize_transfer(source, None, manifests, selections)
    transfer = next(arm for arm in bound.config.arms if arm.id == "donor_transfer")
    for task in bound.config.screen.tasks:
        config = ConfigModel.from_mapping(
            harness.deep_merge(
                harness._inventory_config(bound, task, "screen").model_dump(mode="python"),
                transfer.overlay,
            )
        )
        manifest = validate_transfer_config(config)
        assert manifest["recipient_refit"] is False
        assert manifest["application_provenance"]["shared_ontology_bytes"] == []
        dataset = SimpleNamespace(
            transfer_artifact=config.supervision.transfer_artifact,
            dataset_signature=dataset_signature_for_paths(config.data.source, config.data.target),
            transfer_feature_contract=transfer_feature_contract(config),
            transfer_score_threshold=config.matching.threshold,
            _entity_kinds=["class"],
        )
        recipient = _selector()
        recipient.rerank_config["artifact"] = str(head)
        recipient._fit_rank_model = lambda *_a, **_k: pytest.fail("recipient fitting is forbidden")
        report = rows.iloc[:2].copy()
        report["Src"] = task.id
        result = recipient(
            candidate_df=report, dataset=dataset, threshold=config.matching.threshold
        )["candidate_df"]
        assert result.selection_winner.sum() == 1
        trainer = TrainingPoolMixin()
        trainer.dataset = dataset
        trainer.training_candidates_file_path = tmp_path / "must-not-open-recipient-train.tsv"
        trainer.fit_training_pool()
    assert head.read_bytes() == original
    assert materialize_transfer(source, None, manifests, selections).raw_hash() == bound.raw_hash()

    # Relocating identical recipient bytes creates a fresh application binding,
    # while the selected donor weights and their content identity stay fixed.
    moved = deepcopy(declaration)
    moved_paths = []
    for name, old in zip(["moved-source", "moved-target"], paths[2:4]):
        new = tmp_path / (name + ".owl")
        new.write_bytes(old.read_bytes())
        moved_paths.append(new)
    moved["screen"]["tasks"][0]["overlay"]["data"].update(
        source=str(moved_paths[0]), target=str(moved_paths[1])
    )
    moved_path = tmp_path / "E16-moved.json"
    freeze_json(moved_path, moved)
    relocated = materialize_transfer(
        harness.ExperimentSource(ExperimentConfig.model_validate(moved), moved_path),
        None,
        manifests,
        selections,
    )
    assert relocated.path != bound.path
    new_arm = next(arm for arm in relocated.config.arms if arm.id == "donor_transfer")
    task = relocated.config.screen.tasks[0]
    config = ConfigModel.from_mapping(
        harness.deep_merge(
            harness._inventory_config(relocated, task, "screen").model_dump(mode="python"),
            new_arm.overlay,
        )
    )
    rebound = validate_transfer_config(config)
    assert rebound["artifacts"]["selector"]["path"] == str(head)
    assert rebound["recipient_dataset_signature"] == dataset_signature_for_paths(*moved_paths)
    assert head.read_bytes() == original
    moved_paths[0].write_text("mutated")
    with pytest.raises(ValueError, match="application binding|content changed"):
        validate_transfer_config(config)
    assert (
        json.loads(Path(new_arm.overlay["supervision"]["transfer_artifact"]).read_text())["kind"]
        == "cross_pair_transfer_bundle"
    )
