"""Replay persisted scalar transforms without the fitted model or artifact files."""

import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.impl.models.selector import CandidateSetSelector


def _linear(values, model):
    return model["bias"] + sum(
        (value - mean) / scale * weight
        for value, mean, scale, weight in zip(
            values, model["mean"], model["scale"], model["weights"]
        )
    )


def _sigmoid(value):
    return 1 / (1 + math.exp(-value))


@pytest.mark.parametrize(
    "mode,model_type",
    [
        ("analytic", "current_linear"),
        ("current_listwise", "current_linear"),
        ("current_listwise", "additive_gam"),
        ("current_listwise", "channel_gating"),
    ],
)
def test_fitted_trace_replays_rank_acceptance_and_confidence_from_serialized_records(
    tmp_path, mode, model_type
):
    selector = CandidateSetSelector(
        enabled=True,
        global_only=False,
        strategy="calibrated_rank_accept",
        calibration={"max_epochs": 8, "min_positive_sources": 1, "validation_folds": 3},
        experiment_config={
            "enabled": True,
            "emit_candidate_scores": True,
            "rerank": {"mode": mode, "model": model_type},
        },
        request_seed=17,
    )
    frame = pd.DataFrame(
        [
            {
                "Src": f"s{i}",
                "Tgt": f"t{i}-{j}",
                "S_final": score,
                "s_label": score,
                "S_struct": 0.6,
                "s_diff": 0.4,
                "s_hier": 0.5,
                "s_sim": 0.6,
            }
            for i in range(9)
            for j, score in enumerate((0.9, 0.2))
        ]
    )
    artifact = tmp_path / "head.json"
    selector.fit_training_artifact(
        frame,
        {(f"s{i}", f"t{i}-0") for i in range(9)},
        artifact,
        application={
            "dataset_signature": "report",
            "source_ids": ["report"],
            "negative_label_policy": "complete_reference",
        },
    )
    report = frame.iloc[:2].copy()
    report["Src"] = "report"
    records = [
        {"src_iri": row.Src, "tgt_iri": row.Tgt, "confidences": {}} for row in report.itertuples()
    ]
    result = selector(
        candidate_df=report,
        dataset=SimpleNamespace(dataset_signature="report"),
        results_json=records,
        threshold=0.6,
    )
    persisted = json.loads(json.dumps(result["results_json"]))
    artifact.unlink()  # The replay below has no model/artifact access.
    source = next(
        record["selector_explanation"]["source_decision"]
        for record in persisted
        if "source_decision" in record["selector_explanation"]
    )
    acceptance = source["accept"]
    values = [acceptance["features"][name] for name in acceptance["feature_names"]]
    probability = _sigmoid(_linear(values, acceptance["model"]))
    assert probability == pytest.approx(acceptance["probability"])
    accept = probability >= source["accept_threshold"] and not source["displayed_none"]
    utilities = []
    for record in persisted:
        trace = record["selector_explanation"]
        rank = trace["rank"]
        utility = (
            _sigmoid(rank["features"][rank["feature_names"][0]])
            if rank["model"]["model_type"] == "analytic"
            else _linear(rank["basis"], rank["model"])
        )
        assert utility == pytest.approx(trace["logit"])
        assert trace["bias"] + sum(trace["contributions"].values()) == pytest.approx(utility)
        utilities.append(utility)
    masses = [
        math.exp((value - max(utilities)) / source["rank_temperature"]) for value in utilities
    ]
    probabilities = [value / sum(masses) for value in masses]
    winner = max(range(len(utilities)), key=utilities.__getitem__)
    assert persisted[winner]["tgt_iri"] == source["winner"]
    for index, record in enumerate(persisted):
        pair_probability = probability
        if index != winner:
            pair_probability = min(
                probability * probabilities[index] / max(probabilities[winner], source["eps"]),
                max(0.0, probability - source["eps"]),
            )
        score = (
            pair_probability
            if source["score_mode"] == "p_match"
            else min(
                1.0,
                max(0.0, source["score_threshold"] + pair_probability - source["accept_threshold"]),
            )
        )
        score = score if accept and (index == winner or source["emit_candidate_scores"]) else 0.0
        assert score == pytest.approx(record["confidences"]["S_select"])


def test_calibrator_parameters_and_raw_score_survive_sync(tmp_path):
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "score_calibrator",
                "calibrator": {"mode": "platt", "slope": 2.0, "intercept": -1.0},
            }
        )
    )
    selector = CandidateSetSelector(
        enabled=True,
        global_only=False,
        matching_calibration={"mode": "platt", "artifact": str(artifact)},
    )
    frame = pd.DataFrame([{"Src": "s", "Tgt": "t", "S_final": 0.8, "S_pair_final": 0.8}])
    frame = selector._apply_matching_score_calibration(frame)
    frame["Q_pool_miss"] = 0.2
    records = [{"src_iri": "s", "tgt_iri": "t", "confidences": {"S_pair_final": 0.8}}]
    selector._sync_results_json(frame, records)
    record = json.loads(json.dumps(records[0]))
    params = record["score_calibration"]["parameters"]
    assert _sigmoid(
        params["slope"] * record["score_calibration"]["input"] + params["intercept"]
    ) == pytest.approx(record["confidences"]["S_pair_final"])
    assert record["confidences"]["S_pair_pre_calibration"] == 0.8
    assert record["confidences"]["Q_pool_miss"] == 0.2
