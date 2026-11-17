"""TRAIN and application must use identical feature meanings and evidence."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from exact.impl.models.selector.feature_contract import (
    SPECIFICITY_FEATURES,
    apply_feature_contract,
    feature_contract,
    restore_specificity_metadata,
)
from exact.impl.models.selector.fusion_fitting import fusion_scores
from tests.fitting_storage_retention_test import runner_at
from tests.grouped_fitting_test import selector, training_rows


def feature_rows(model, frame):
    frame = frame.copy()
    frame["S_pair_final"] = frame.S_final
    return model._rank_feature_rows(frame, {}, {})


def test_missing_retrieval_metadata_cannot_change_feature_meaning_at_application():
    model = selector()
    train = training_rows()
    contract = feature_contract(train)
    application = train.assign(cand_sim=[0.001, 0.999] * 9, cand_sim_prob=0.8)
    before = feature_rows(model, apply_feature_contract(train, contract))
    after = feature_rows(model, apply_feature_contract(application, contract))
    assert before == after
    position = model.RANK_FEATURE_NAMES.index("cand_sim")
    assert all(row[position] == 0.0 for row in before.values())
    assert "cand_sim" in contract["unavailable_metadata"]


def test_supplied_retrieval_metadata_is_required_and_does_not_change_gap_schema():
    model = selector()
    train = training_rows().assign(cand_sim=[0.95, 0.1] * 9)
    contract = feature_contract(train)
    with pytest.raises(ValueError, match="requires finite feature cand_sim"):
        apply_feature_contract(train.drop(columns="cand_sim"), contract)
    with pytest.raises(ValueError, match="finite"):
        feature_contract(train.assign(cand_sim=float("nan")))
    with_score = feature_rows(model, apply_feature_contract(train, contract))
    without_score = feature_rows(model, train.drop(columns="cand_sim"))
    gap = model.RANK_FEATURE_NAMES.index("channel_gap")
    candidate = model.RANK_FEATURE_NAMES.index("cand_sim")
    assert with_score[0][candidate] == 0.95
    assert [row[gap] for row in with_score.values()] == [row[gap] for row in without_score.values()]


def test_fixed_rank_gap_missing_channel_is_zero_not_a_changed_denominator():
    model = selector()
    partial = training_rows().drop(columns=["S_struct", "s_hier", "s_sim", "s_attr"])
    absent = feature_rows(model, partial)
    explicit = feature_rows(model, partial.assign(S_struct=0.0, s_hier=0.0, s_sim=0.0, s_attr=0.0))
    gap = model.RANK_FEATURE_NAMES.index("channel_gap")
    assert absent == explicit
    assert absent[0][gap] == pytest.approx((0.9 - 0.2) / 5)


def test_resident_evidence_and_specificity_are_learned_and_replayed(tmp_path):
    model = selector()
    train = training_rows()
    train["selector_evidence_items"] = [
        [{"key": f"target:{row.Tgt}", "strength": 0.9 if row.Tgt.endswith("0") else 0.1}]
        for row in train.itertuples()
    ]
    for name in SPECIFICITY_FEATURES:
        train[name] = 0.0
    train["tgt_obj_ic_mean"] = [0.0, 1.0] * 9
    artifact = tmp_path / "selector.json"
    payload = model.fit_training_artifact(
        train,
        {(f"s{i}", f"t{i}-0") for i in range(9)},
        artifact,
        application={"negative_label_policy": "complete_reference", "source_ids": ["report"]},
    )
    assert payload["fit_provenance"]["fitting_recipe_version"] == 3
    distinct = model.RANK_FEATURE_NAMES.index("distinctive")
    safety = model.RANK_FEATURE_NAMES.index("safety")
    assert payload["rank_model"]["mean"][distinct] > 0.0
    assert payload["rank_model"]["weights"][distinct] != 0.0
    assert payload["rank_model"]["scale"][safety] != 1.0
    report = train.iloc[:2].assign(Src="report")
    records = [
        {
            "src_iri": row.Src,
            "tgt_iri": row.Tgt,
            "resident_evidence_schema": 1,
            "selector_evidence_items": row.selector_evidence_items,
        }
        for row in report.itertuples()
    ]
    applied = model(candidate_df=report, results_json=records, dataset=SimpleNamespace())[
        "candidate_df"
    ]
    explanations = [json.loads(value)["rank"]["features"] for value in applied.selector_explanation]
    assert [value["distinctive"] for value in explanations] == pytest.approx([1.0, 1 / 9])
    assert explanations[0]["safety"] != explanations[1]["safety"]
    # Old heads must never be accepted under corrected feature semantics.
    payload["fit_provenance"]["fitting_recipe_version"] = 2
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="refit required"):
        model(candidate_df=report, dataset=SimpleNamespace())


def test_legacy_numerical_rows_hydrate_only_missing_metadata_without_changing_scores():
    frame = training_rows().iloc[:2].copy()
    original = frame.copy(deep=True)
    calls = []

    def get_features(pairs):
        calls.append(pairs.copy())
        return pairs.assign(**{name: [0.2, 0.8] for name in SPECIFICITY_FEATURES})

    dataset = SimpleNamespace(_pair_adaptive_metric_columns=True, get_features=get_features)
    enriched = restore_specificity_metadata(frame, dataset)
    pd.testing.assert_frame_equal(enriched[frame.columns], original)
    pd.testing.assert_frame_equal(frame, original)
    assert set(calls[0].columns) == {"Src", "Tgt"}
    assert enriched.src_obj_ic_mean.tolist() == [0.2, 0.8]
    assert restore_specificity_metadata(enriched, dataset) is enriched
    assert len(calls) == 1
    dataset.get_features = lambda pairs: get_features(pairs).iloc[::-1]
    with pytest.raises(ValueError, match="identity or order"):
        restore_specificity_metadata(frame, dataset)


def test_fitted_neutral_values_replayed_for_every_inactive_selector_channel():
    names = ["label", "sim_obj", "diff", "attr_aux", "strsim", "hier__is_a", "lex", "struct"]
    scores = torch.tensor([[0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
    active = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    tau = torch.tensor(0.42, dtype=torch.float64)
    params = (tau, torch.tensor(1.2), torch.ones(len(names)))
    result = fusion_scores(
        (scores, active, active), names, params, mode="analytic_fitted", return_components=True
    )
    assert result["S_final"].item() == pytest.approx(0.8)
    for name in ("s_sim", "s_diff", "s_attr", "s_strsim", "s_hier", "S_struct"):
        assert result[name].item() == pytest.approx(0.42)
    # Retained active evidence is unchanged, even when its score equals old tau.
    active[0, names.index("sim_obj")] = 1.0
    result = fusion_scores(
        (scores, active, active), names, params, mode="analytic_fitted", return_components=True
    )
    assert result["s_sim"].item() == 0.5


def test_cached_training_metadata_recovery_does_not_rescore_or_rewrite_aggregate(tmp_path):
    original = runner_at(tmp_path)
    original.fit_training_pool(batch_size=5)
    aggregate = next(tmp_path.glob("fitting/*/training_scores.json"))
    before = aggregate.read_bytes()
    assert "src_obj_ic_mean" not in json.loads(before)["rows"][0]
    resumed = runner_at(tmp_path)
    resumed.dataset._pair_adaptive_metric_columns = lambda: list(SPECIFICITY_FEATURES)
    original_features = resumed.dataset.get_features
    resumed.dataset.get_features = lambda frame: original_features(frame).assign(
        **{name: 0.2 for name in SPECIFICITY_FEATURES}
    )
    head = resumed.models[1]
    head.rerank_config["artifact"] = str(tmp_path / "corrected-selector.json")
    resumed.fit_training_pool(batch_size=5)
    assert resumed.model.calls == []
    assert aggregate.read_bytes() == before
    fitted = json.loads((tmp_path / "corrected-selector.json").read_text())
    assert set(SPECIFICITY_FEATURES) <= set(
        fitted["fit_provenance"]["feature_contract"]["available_metadata"]
    )
