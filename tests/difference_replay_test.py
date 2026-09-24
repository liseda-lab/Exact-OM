"""E24 channel interventions must reconstruct actual frozen final decisions."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from exact.experiments.difference_replay import (
    evaluate_difference_replay,
    replay_analytic_fusion,
    write_difference_replay,
)
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset


@pytest.mark.parametrize(
    "sigma", ["full", "constant_q", "no_sharpening", "suppression_only", "uniform"]
)
@pytest.mark.parametrize("string", [False, True])
def test_natural_replay_matches_real_forward_scores_and_uncertainty(sigma, string):
    model = _scorer(
        return_explanations=True,
        strsim={"enabled": string},
        diff={"enabled": True, "controlled_perturbations": True, "dump_components": True},
        fusion_config={"enabled": True, "sigma_mode": sigma},
    )
    model.attach_dataset(_TinyDataset())
    result = model(
        src_iris=["s"],
        tgt_iris=["t"],
        src_label_lists=[["heart valve"]],
        tgt_label_lists=[["heart valve"]],
    )
    evidence = result["explanations"][0]["experiment_diagnostics"]
    assert "controlled_perturbations" in evidence["difference"]
    observed = replay_analytic_fusion(model, evidence["difference_replay"]["channels"])
    for field in ("S_base", "U"):
        assert observed[field] == pytest.approx(float(result[field][0]), abs=1e-6)


def synthetic(tmp_path):
    model = _scorer(
        diff={"enabled": True, "controlled_perturbations": True, "dump_components": True},
        fusion_config={"enabled": True, "sigma_mode": "full"},
    )
    channels = [
        {"name": "label", "score": 0.5, "quality": 0.0, "active": False},
        {"name": "strsim", "score": 0.5, "quality": 0.0, "active": False},
        {"name": "diff", "score": 0.9, "quality": 1.0, "active": True},
    ]
    output = replay_analytic_fusion(model, channels)
    explanations = {
        ("s", "t"): {
            "experiment_diagnostics": {
                "difference_replay": {"channels": channels, **output},
                "difference": {
                    "controlled_perturbations": {
                        "inventory_sha256": "frozen-inventory",
                        "variants": [
                            {"name": "natural", "score": 0.9, "quality": 1.0, "active": True},
                            {
                                "name": "deletion_source",
                                "score": 0.5,
                                "quality": 0.0,
                                "active": False,
                            },
                        ],
                    }
                },
            }
        }
    }
    runner = SimpleNamespace(
        model=model,
        models=[model],
        output_dir=tmp_path,
        dataset=SimpleNamespace(eligible_source_iris=["s", "anchor", "empty"]),
    )
    frame = pd.DataFrame(
        [
            {"Src": "s", "Tgt": "t", "S_final": output["S_base"]},
            {"Src": "anchor", "Tgt": "exact", "S_final": 1.0},
        ]
    )
    policy = {
        "threshold": 0.7,
        "source_cardinality": 1,
        "target_cardinality": 1,
        "extraction": {"mode": "greedy"},
    }
    return (
        runner,
        frame,
        explanations,
        {("s", "t"), ("anchor", "exact")},
        {("anchor", "exact")},
        policy,
    )


def test_replay_preserves_anchors_and_empty_sources_and_reports_known_harm(tmp_path):
    args = synthetic(tmp_path)
    path = write_difference_replay(*args)
    payload = json.loads(path.read_text())
    assert payload["reference_labels_used"] is False
    assert payload["source_universe"] == ["anchor", "empty", "s"]
    assert payload["variants"]["deletion_source"]["emitted"] == [["anchor", "exact"]]
    reference = tmp_path / "valid.tsv"
    reference.write_text("Src\tTgt\ns\tt\n")
    cell = SimpleNamespace(
        output_dir=tmp_path,
        split_role="development",
        reference_role="valid",
        negative_label_policy="positive_unlabelled",
        resolved_config={"data": {"refs": {"valid": str(reference)}}},
    )
    report = evaluate_difference_replay(cell)
    change = json.loads(open(report["path"]).read())["variants"]["deletion_source"]
    assert change["harmed"] == 1 and change["corrected"] == 0
    assert change["unknown_label_count"] == 1
    cell.split_role = "reporting"
    with pytest.raises(ValueError, match="development diagnostics"):
        evaluate_difference_replay(cell)


def test_natural_numerical_and_emitted_mapping_guards_fail_closed(tmp_path):
    args = list(synthetic(tmp_path))
    args[2][("s", "t")]["experiment_diagnostics"]["difference_replay"]["U"] += 0.1
    with pytest.raises(ValueError, match="natural numerical"):
        write_difference_replay(*args)
    args = list(synthetic(tmp_path))
    args[3] = {("anchor", "exact")}
    with pytest.raises(ValueError, match="natural emitted mapping"):
        write_difference_replay(*args)
    args = list(synthetic(tmp_path))
    args[0].models.append(SimpleNamespace(enabled=True))
    with pytest.raises(ValueError, match="active extra heads"):
        write_difference_replay(*args)


def test_target_competition_is_replayed_over_whole_population(tmp_path):
    args = list(synthetic(tmp_path))
    runner, frame, explanations, _, protected, policy = args
    second = deepcopy(explanations[("s", "t")])
    raw = second["experiment_diagnostics"]["difference_replay"]
    raw["channels"][-1]["score"] = 0.8
    raw.update(replay_analytic_fusion(runner.model, raw["channels"]))
    variants = second["experiment_diagnostics"]["difference"]["controlled_perturbations"][
        "variants"
    ]
    for variant in variants:
        variant.update(score=0.8, quality=1.0, active=True)
    explanations[("other", "t")] = second
    frame = pd.concat(
        [frame, pd.DataFrame([{"Src": "other", "Tgt": "t", "S_final": 0.8}])], ignore_index=True
    )
    path = write_difference_replay(
        runner, frame, explanations, {("s", "t"), *protected}, protected, policy
    )
    result = json.loads(path.read_text())
    assert ["other", "t"] not in result["variants"]["natural"]["emitted"]
    assert ["other", "t"] in result["variants"]["deletion_source"]["emitted"]


@pytest.mark.parametrize(
    "sigma", ["full", "constant_q", "no_sharpening", "suppression_only", "uniform"]
)
@pytest.mark.parametrize("placement", ["off", "channel", "folded_into_lexical"])
def test_replay_mixed_active_channels_uses_real_sigma_and_fusion(monkeypatch, sigma, placement):
    model = _scorer(
        return_explanations=True,
        strsim={
            "enabled": placement != "off",
            "placement": "channel" if placement == "off" else placement,
        },
        diff={"enabled": True, "controlled_perturbations": True, "dump_components": True},
        fusion_config={"enabled": True, "sigma_mode": sigma},
    )
    model.use_lexical = True
    monkeypatch.setattr(
        model,
        "encode_labels_batch",
        lambda labels: torch.tensor(
            [[1.0, 0.0] if "heart" in label else [0.6, 0.8] for label in labels]
        ),
    )
    for method, score, quality in (
        ("_score_hierarchy_family", 0.8, 0.4),
        ("_score_similarity_channel", 0.3, 0.8),
        ("_score_difference_channel", 0.65, 0.5),
        ("_score_attribute_channel", 0.2, 0.7),
    ):
        original = getattr(model, method)

        def fixed(*args, original=original, score=score, quality=quality, **kwargs):
            payload = original(*args, **kwargs)
            payload.update(
                score=score,
                quality=quality,
                active=True,
                src_selected=[{"triple": ["s", "p", "o"], "item_id": "fixture:fact"}],
            )
            return payload

        monkeypatch.setattr(model, method, fixed)
    model.attach_dataset(_TinyDataset())
    actual = model(
        src_iris=["s"],
        tgt_iris=["t"],
        src_label_lists=[["heart valve"]],
        tgt_label_lists=[["cardiac valve"]],
    )
    raw = actual["explanations"][0]["experiment_diagnostics"]["difference_replay"]
    assert sum(bool(item["active"]) for item in raw["channels"]) >= 4
    replay = replay_analytic_fusion(model, raw["channels"])
    for field in ("S_base", "U"):
        assert replay[field] == pytest.approx(float(actual[field][0]), abs=1e-6)
