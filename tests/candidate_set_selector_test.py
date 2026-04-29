import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch


def _load_selector_module():
    module_path = Path(__file__).resolve().parents[1] / "exact" / "impl" / "models" / "candidate_set_selector.py"
    spec = importlib.util.spec_from_file_location("candidate_set_selector_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "exact" / "impl" / "trainer" / "semantic_runner.py"
    spec = importlib.util.spec_from_file_location("semantic_runner_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dataset_module_or_skip():
    module_path = Path(__file__).resolve().parents[1] / "exact" / "core" / "contracts" / "dataset.py"
    spec = importlib.util.spec_from_file_location("dataset_contract_module", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"Dataset contract import unavailable in this environment: {exc}")
    return module


def _base_df():
    return pd.DataFrame(
        {
            "Src": ["s1", "s1"],
            "Tgt": ["t_good", "t_bad"],
            "S_final": [0.84, 0.86],
            "s_label": [0.83, 0.86],
            "S_struct": [0.92, 0.74],
            "s_hier": [0.90, 0.72],
            "s_sim": [0.80, 0.80],
            "s_attr": [0.92, 0.55],
            "s_diff": [0.96, 0.96],
            "cand_sim_prob": [0.49, 0.51],
            "src_obj_ic_mean": [0.5, 0.5],
            "tgt_obj_ic_mean": [0.5, 0.5],
        }
    )


def _selector_training_df(include_candidate_miss=False):
    rows = [
        {
            "Src": "s_train",
            "Tgt": "t_gold_train",
            "S_final": 0.70,
            "s_label": 0.96,
            "S_struct": 0.96,
            "s_hier": 0.96,
            "s_sim": 0.95,
            "s_attr": 0.95,
            "s_diff": 0.98,
            "cand_sim": 0.70,
            "cand_sim_prob": 0.45,
            "src_obj_ic_mean": 0.5,
            "tgt_obj_ic_mean": 0.5,
        },
        {
            "Src": "s_train",
            "Tgt": "t_bad_train",
            "S_final": 0.79,
            "s_label": 0.10,
            "S_struct": 0.12,
            "s_hier": 0.12,
            "s_sim": 0.12,
            "s_attr": 0.10,
            "s_diff": 0.45,
            "cand_sim": 0.79,
            "cand_sim_prob": 0.55,
            "src_obj_ic_mean": 0.5,
            "tgt_obj_ic_mean": 0.5,
        },
        {
            "Src": "s_eval",
            "Tgt": "t_gold_eval",
            "S_final": 0.71,
            "s_label": 0.95,
            "S_struct": 0.95,
            "s_hier": 0.95,
            "s_sim": 0.94,
            "s_attr": 0.94,
            "s_diff": 0.98,
            "cand_sim": 0.71,
            "cand_sim_prob": 0.46,
            "src_obj_ic_mean": 0.5,
            "tgt_obj_ic_mean": 0.5,
        },
        {
            "Src": "s_eval",
            "Tgt": "t_bad_eval",
            "S_final": 0.80,
            "s_label": 0.11,
            "S_struct": 0.13,
            "s_hier": 0.13,
            "s_sim": 0.13,
            "s_attr": 0.11,
            "s_diff": 0.45,
            "cand_sim": 0.80,
            "cand_sim_prob": 0.54,
            "src_obj_ic_mean": 0.5,
            "tgt_obj_ic_mean": 0.5,
        },
    ]
    if include_candidate_miss:
        rows.extend(
            [
                {
                    "Src": "s_missing",
                    "Tgt": "t_false_a",
                    "S_final": 0.82,
                    "s_label": 0.05,
                    "S_struct": 0.08,
                    "s_hier": 0.08,
                    "s_sim": 0.08,
                    "s_attr": 0.05,
                    "s_diff": 0.40,
                    "cand_sim": 0.82,
                    "cand_sim_prob": 0.52,
                    "src_obj_ic_mean": 0.5,
                    "tgt_obj_ic_mean": 0.5,
                },
                {
                    "Src": "s_missing",
                    "Tgt": "t_false_b",
                    "S_final": 0.81,
                    "s_label": 0.04,
                    "S_struct": 0.07,
                    "s_hier": 0.07,
                    "s_sim": 0.07,
                    "s_attr": 0.04,
                    "s_diff": 0.40,
                    "cand_sim": 0.81,
                    "cand_sim_prob": 0.48,
                    "src_obj_ic_mean": 0.5,
                    "tgt_obj_ic_mean": 0.5,
                },
            ]
        )
    return pd.DataFrame(rows)


def _kfold_selector_training_df(n_sources=6):
    rows = []
    for idx in range(n_sources):
        src = f"s{idx}"
        rows.extend(
            [
                {
                    "Src": src,
                    "Tgt": f"t_gold_{idx}",
                    "S_final": 0.70,
                    "s_label": 0.96,
                    "S_struct": 0.96,
                    "s_hier": 0.96,
                    "s_sim": 0.95,
                    "s_attr": 0.95,
                    "s_diff": 0.98,
                    "cand_sim": 0.70,
                    "cand_sim_prob": 0.45,
                    "src_obj_ic_mean": 0.5,
                    "tgt_obj_ic_mean": 0.5,
                },
                {
                    "Src": src,
                    "Tgt": f"t_bad_{idx}",
                    "S_final": 0.80,
                    "s_label": 0.10,
                    "S_struct": 0.12,
                    "s_hier": 0.12,
                    "s_sim": 0.12,
                    "s_attr": 0.10,
                    "s_diff": 0.45,
                    "cand_sim": 0.80,
                    "cand_sim_prob": 0.55,
                    "src_obj_ic_mean": 0.5,
                    "tgt_obj_ic_mean": 0.5,
                },
            ]
        )
    return pd.DataFrame(rows)


def _write_training_reference(tmp_path, pairs):
    path = tmp_path / "train.tsv"
    pd.DataFrame(pairs, columns=["Src", "Tgt"]).to_csv(path, sep="\t", index=False)
    return path


def _threshold_decisions(prob_labels):
    return {
        f"s{idx}": {
            "p_match": prob,
            "label": label,
            "sample_weight": 1.0,
        }
        for idx, (prob, label) in enumerate(prob_labels)
    }


def _record(src, tgt, target_attribute=None):
    target_items = []
    if target_attribute:
        target_items.append(
            {
                "item_id": f"{tgt}-attr",
                "property": "definition",
                "value": target_attribute,
                "weight": 1.0,
            }
        )
    return {
        "src_iri": src,
        "tgt_iri": tgt,
        "confidences": {"S_final": 0.5},
        "prediction": {},
        "attributes": {"source": [], "target": target_items},
        "cross_side_provenance": {},
    }


def test_distinctive_evidence_can_beat_slightly_higher_pairwise_score():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        enabled=True,
        use_no_match=False,
        llm={"enabled": False},
        support_weight=0.1,
    )
    records = [
        _record("s1", "t_good", "decisive candidate-specific definition"),
        _record("s1", "t_bad"),
    ]

    out = selector.forward(_base_df(), results_json=records)
    scored = out["candidate_df"].set_index("Tgt")

    assert scored.loc["t_good", "S_select"] > scored.loc["t_bad", "S_select"]
    assert scored.loc["t_good", "S_pair_final"] == 0.84
    assert scored.loc["t_good", "S_final"] == scored.loc["t_good", "S_select"]
    assert records[0]["confidences"]["S_pair_final"] == 0.84
    assert records[0]["confidences"]["S_final"] == records[0]["confidences"]["S_select"]


def test_no_match_wins_for_low_close_generic_candidates():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        enabled=True,
        llm={"enabled": False},
        support_weight=1.0,
    )
    df = _base_df()
    df["S_final"] = [0.32, 0.33]

    out = selector.forward(df, results_json=[_record("s1", "t_good"), _record("s1", "t_bad")])

    assert out["candidate_df"]["selection_abstained"].all()
    assert (out["candidate_df"]["selection_no_match_prob"] > out["candidate_df"]["S_select"]).all()


def test_local_alignment_is_noop():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(enabled=True)
    df = _base_df()

    out = selector.forward(df, local_alignment=True)

    assert list(out["candidate_df"]["S_final"]) == list(df["S_final"])
    assert "S_select" not in out["candidate_df"].columns


def test_shared_evidence_has_no_distinctive_gain():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(enabled=True, llm={"enabled": False})
    df = _base_df()
    records = [
        _record("s1", "t_good", "shared generic evidence"),
        _record("s1", "t_bad", "shared generic evidence"),
    ]

    lookup = selector._record_lookup(records)
    scores = selector._distinctive_scores(df, lookup)

    assert scores[0] == 0.0
    assert scores[1] == 0.0


def test_llm_equivalent_choice_boosts_ambiguous_candidate():
    module = _load_selector_module()

    class FakePrimary:
        def candidate_set_select(self, prompt):
            return '{"winner": "t_good", "relation": "equivalent", "confidence": 1.0, "decisive_evidence": "x", "rejected": {}}'

    selector = module.CandidateSetSelector(
        enabled=True,
        use_no_match=False,
        support_weight=1.0,
        llm={
            "enabled": True,
            "ambiguity_margin": 1.0,
        },
    )
    df = _base_df()

    out = selector.forward(df, primary_model=FakePrimary(), results_json=[_record("s1", "t_good"), _record("s1", "t_bad")])
    scored = out["candidate_df"].set_index("Tgt")

    assert scored.loc["t_good", "S_select"] > scored.loc["t_bad", "S_select"]
    assert scored["selection_llm_used"].all()


def test_calibrated_selector_learns_rank_accept_from_training_reference(tmp_path):
    module = _load_selector_module()
    train_ref = _write_training_reference(
        tmp_path,
        [("s_train", "t_gold_train"), ("s_eval", "t_gold_eval")],
    )
    selector = module.CandidateSetSelector(
        enabled=True,
        strategy="calibrated_rank_accept",
        use_no_match=True,
        llm={"enabled": False},
        training_reference_file_path=train_ref,
        calibration={
            "min_positive_sources": 1,
            "background_negative_weight": 0.0,
            "background_negative_weight_grid": [0.0],
            "validation_fraction": 0.0,
            "max_epochs": 80,
        },
    )

    out = selector.forward(_selector_training_df(), threshold=0.7)
    scored = out["candidate_df"].set_index(["Src", "Tgt"])

    assert scored.loc[("s_eval", "t_gold_eval"), "selection_winner"]
    assert scored.loc[("s_eval", "t_gold_eval"), "S_select"] == scored.loc[("s_eval", "t_gold_eval"), "P_match"]
    assert scored.loc[("s_eval", "t_gold_eval"), "P_match"] >= scored.loc[("s_eval", "t_gold_eval"), "selection_accept_threshold"]
    assert scored.loc[("s_eval", "t_bad_eval"), "S_select"] == 0.0
    assert scored.loc[("s_eval", "t_gold_eval"), "P_rank"] > scored.loc[("s_eval", "t_bad_eval"), "P_rank"]


def test_calibrated_selector_uses_source_disjoint_oof_calibration(tmp_path):
    module = _load_selector_module()
    pairs = [(f"s{idx}", f"t_gold_{idx}") for idx in range(6)]
    train_ref = _write_training_reference(tmp_path, pairs)
    selector = module.CandidateSetSelector(
        enabled=True,
        strategy="calibrated_rank_accept",
        use_no_match=True,
        llm={"enabled": False},
        training_reference_file_path=train_ref,
        request_seed=7,
        calibration={
            "min_positive_sources": 1,
            "background_negative_weight": 0.0,
            "background_negative_weight_grid": [0.0],
            "validation_fraction": 0.2,
            "validation_folds": 3,
            "max_epochs": 40,
        },
    )

    out = selector.forward(_kfold_selector_training_df(), threshold=0.7)
    scored = out["candidate_df"].set_index(["Src", "Tgt"])

    assert selector._calibration_meta["validation_split"]["strategy"] == "kfold"
    assert selector._calibration_meta["validation_split"]["n_folds"] == 3
    assert selector._calibration_meta["threshold_metrics"]["validation_scope"] == "final_refit"
    assert selector._calibration_meta["oof_threshold_metrics"]["validation_scope"] == "oof"
    assert selector._calibration_meta["oof_accept_threshold"] is not None
    assert (
        selector._calibration_meta["accept_threshold"]
        == selector._calibration_meta["threshold_metrics"]["selected_threshold"]
    )
    assert "final_refit_selected_sources" in selector._calibration_meta["threshold_metrics"]
    assert selector._calibration_meta["n_calibration_train_reference_pairs"] == 6
    assert selector._calibration_meta["n_validation_reference_pairs"] == 6
    assert scored.loc[("s0", "t_gold_0"), "P_match"] >= 0.0


def test_calibrated_selector_fits_inside_no_grad_predict_context(tmp_path):
    module = _load_selector_module()
    train_ref = _write_training_reference(tmp_path, [("s_train", "t_gold_train")])
    selector = module.CandidateSetSelector(
        enabled=True,
        strategy="calibrated_rank_accept",
        use_no_match=True,
        llm={"enabled": False},
        training_reference_file_path=train_ref,
        calibration={
            "min_positive_sources": 1,
            "background_negative_weight": 0.0,
            "max_epochs": 10,
        },
    )

    with torch.no_grad():
        out = selector.forward(_selector_training_df(), threshold=0.7)

    assert not out["candidate_df"].empty
    assert "P_match" in out["candidate_df"].columns


def test_accept_threshold_f1_preserves_precision_oriented_selection():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        calibration={
            "accept_objective": "f1",
            "threshold_grid_step": 0.05,
        },
    )
    decisions = _threshold_decisions(
        [
            (0.90, 1.0),
            (0.45, 1.0),
            (0.44, 1.0),
            (0.80, 0.0),
            (0.70, 0.0),
            (0.60, 0.0),
            (0.50, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
        ]
    )

    threshold, metrics = selector._tune_accept_threshold(decisions)

    assert threshold == 0.9
    assert metrics["accept_objective"] == "f1"
    assert metrics["P"] == 1.0
    assert metrics["R"] == 1.0 / 3.0
    assert metrics["fallback_to_f1"] is False


def test_accept_threshold_f_beta_prefers_recall_when_configured():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        calibration={
            "accept_objective": "f_beta",
            "f_beta": 1.5,
            "threshold_grid_step": 0.05,
        },
    )
    decisions = _threshold_decisions(
        [
            (0.90, 1.0),
            (0.45, 1.0),
            (0.44, 1.0),
            (0.80, 0.0),
            (0.70, 0.0),
            (0.60, 0.0),
            (0.50, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
            (0.43, 0.0),
        ]
    )

    threshold, metrics = selector._tune_accept_threshold(decisions)

    assert threshold == 0.45
    assert metrics["accept_objective"] == "f_beta"
    assert metrics["R"] > metrics["best_f1"]["R"]


def test_accept_threshold_recall_at_precision_maximizes_recall_under_floor():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        calibration={
            "accept_objective": "recall_at_precision",
            "min_precision": 0.6,
            "threshold_grid_step": 0.1,
        },
    )
    decisions = _threshold_decisions(
        [
            (0.90, 1.0),
            (0.60, 1.0),
            (0.40, 1.0),
            (0.80, 0.0),
            (0.50, 0.0),
        ]
    )

    threshold, metrics = selector._tune_accept_threshold(decisions)

    assert threshold == 0.4
    assert metrics["P"] >= 0.6
    assert metrics["R"] == 1.0
    assert metrics["fallback_to_f1"] is False


def test_accept_threshold_recall_at_precision_falls_back_to_f1_when_floor_impossible():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        calibration={
            "accept_objective": "recall_at_precision",
            "min_precision": 1.0,
            "threshold_grid_step": 0.1,
        },
    )
    decisions = _threshold_decisions([(0.80, 1.0), (0.90, 0.0)])

    threshold, metrics = selector._tune_accept_threshold(decisions)

    assert threshold == metrics["best_f1"]["threshold"]
    assert metrics["fallback_to_f1"] is True
    assert metrics["recall_at_precision"] == {}


def test_calibrated_llm_trigger_ignores_rejected_evidence_disagreement():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(llm={"enabled": True})

    assert not selector._should_use_llm_calibrated(
        acceptance_margin=0.10,
        rank_margin=0.90,
        primary_model=object(),
        accepted=False,
        evidence_support=0.90,
        evidence_support_floor=0.82,
    )


def test_calibrated_llm_trigger_allows_accepted_near_boundary():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(llm={"enabled": True})

    assert selector._should_use_llm_calibrated(
        acceptance_margin=0.01,
        rank_margin=0.90,
        primary_model=object(),
        accepted=True,
        evidence_support=0.70,
        evidence_support_floor=0.82,
    )


def test_calibrated_llm_trigger_ignores_low_evidence_rejection():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(llm={"enabled": True})

    assert not selector._should_use_llm_calibrated(
        acceptance_margin=0.10,
        rank_margin=0.10,
        primary_model=object(),
        accepted=False,
        evidence_support=0.80,
        evidence_support_floor=0.82,
    )


def test_calibrated_llm_trigger_ignores_far_rejected_evidence_disagreement():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(llm={"enabled": True})

    assert not selector._should_use_llm_calibrated(
        acceptance_margin=0.20,
        rank_margin=0.90,
        primary_model=object(),
        accepted=False,
        evidence_support=0.90,
        evidence_support_floor=0.82,
    )


def test_evidence_support_uses_broad_agreement_and_penalizes_conflict():
    module = _load_selector_module()
    selector = module.CandidateSetSelector()

    strong = selector._evidence_support_from_features(
        [0.0, 0.92, 0.94, 0.0, 0.0, 0.15, 0.0, 0.93, 0.91, 0.5]
    )
    conflicted = selector._evidence_support_from_features(
        [0.0, 0.92, 0.94, 0.0, 0.0, 0.15, 0.0, 0.93, 0.91, 0.25]
    )

    assert strong > 0.82
    assert conflicted < strong


def test_calibrated_selector_treats_train_candidate_miss_as_abstention(tmp_path):
    module = _load_selector_module()
    train_ref = _write_training_reference(
        tmp_path,
        [("s_train", "t_gold_train"), ("s_missing", "t_true_missing")],
    )
    selector = module.CandidateSetSelector(
        enabled=True,
        strategy="calibrated_rank_accept",
        use_no_match=True,
        llm={"enabled": False},
        training_reference_file_path=train_ref,
        calibration={
            "min_positive_sources": 1,
            "background_negative_weight": 0.0,
            "max_epochs": 120,
        },
    )

    out = selector.forward(_selector_training_df(include_candidate_miss=True), threshold=0.7)
    scored = out["candidate_df"]
    missing = scored[scored["Src"] == "s_missing"]

    assert missing["selection_abstained"].all()
    assert (missing["S_select"] == 0.0).all()
    assert missing["P_match"].max() < missing["selection_accept_threshold"].max()


def test_calibrated_selector_excludes_exact_prefiltered_train_sources(tmp_path):
    module = _load_selector_module()
    train_ref = _write_training_reference(
        tmp_path,
        [("s_train", "t_gold_train"), ("s_missing", "t_true_missing")],
    )
    dataset = SimpleNamespace(
        dataframe=pd.DataFrame(
            {
                "Src": ["s_missing"],
                "Tgt": ["t_true_missing"],
                "prefiltered": [True],
                "Scores": [1.0],
            }
        )
    )
    selector = module.CandidateSetSelector(
        enabled=True,
        strategy="calibrated_rank_accept",
        use_no_match=True,
        llm={"enabled": False},
        training_reference_file_path=train_ref,
        calibration={
            "min_positive_sources": 1,
            "background_negative_weight": 1.0,
            "max_epochs": 60,
        },
    )

    selector.forward(_selector_training_df(include_candidate_miss=True), dataset=dataset, threshold=0.7)

    assert selector._calibration_meta["n_reference_pairs"] == 2
    assert selector._calibration_meta["n_calibration_reference_pairs"] == 1
    assert selector._calibration_meta["n_exact_prefiltered_reference_pairs"] == 1
    assert selector._calibration_meta["n_exact_prefiltered_reference_sources"] == 1


def test_exact_prefiltered_sources_are_acceptance_hard_negatives():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        strategy="calibrated_rank_accept",
        calibration={
            "background_negative_weight": 1.0,
            "exact_prefiltered_source_policy": "hard_negative",
            "exact_prefiltered_negative_weight": 0.35,
        },
    )
    df = _selector_training_df(include_candidate_miss=True)
    df["S_pair_final"] = df["S_final"]
    rank_features = selector._rank_feature_rows(df, distinctive={}, reciprocity={})
    utilities = {idx: float(df.at[idx, "S_pair_final"]) for idx in df.index}

    decisions = selector._source_decisions(
        df,
        utilities,
        rank_features,
        distinctive={},
        ref_pairs={("s_train", "t_gold_train")},
        exact_prefiltered_sources={"s_missing"},
    )

    assert decisions["s_missing"]["sample_weight"] == 0.35
    assert decisions["s_missing"]["label"] == 0.0


def test_exact_prefiltered_sources_can_be_excluded_from_acceptance():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        strategy="calibrated_rank_accept",
        calibration={
            "exact_prefiltered_source_policy": "exclude",
            "exact_prefiltered_negative_weight": 1.0,
        },
    )
    df = _selector_training_df(include_candidate_miss=True)
    df["S_pair_final"] = df["S_final"]
    rank_features = selector._rank_feature_rows(df, distinctive={}, reciprocity={})
    utilities = {idx: float(df.at[idx, "S_pair_final"]) for idx in df.index}

    decisions = selector._source_decisions(
        df,
        utilities,
        rank_features,
        distinctive={},
        ref_pairs={("s_train", "t_gold_train")},
        exact_prefiltered_sources={"s_missing"},
    )

    assert decisions["s_missing"]["sample_weight"] == 0.0
    assert decisions["s_missing"]["label"] == 0.0


def test_exact_prefiltered_pair_reader_accepts_score_column():
    module = _load_selector_module()
    selector = module.CandidateSetSelector()
    dataset_df = pd.DataFrame(
        {
            "Src": ["s_missing", "s_missing"],
            "Tgt": ["t_true_missing", "t_false_missing"],
            "prefiltered": [True, True],
            "Score": [1.0, 0.0],
        }
    )

    assert selector._exact_prefiltered_pairs_from_dataframe(dataset_df) == {
        ("s_missing", "t_true_missing")
    }


def test_calibrated_llm_arbitration_has_no_prompt_cap(tmp_path):
    module = _load_selector_module()
    train_ref = _write_training_reference(
        tmp_path,
        [("s_train", "t_gold_train"), ("s_eval", "t_gold_eval")],
    )

    class FakePrimary:
        def __init__(self):
            self.calls = 0

        def candidate_set_select(self, prompt):
            self.calls += 1
            return '{"winner": "C1", "relation": "equivalent", "confidence": 1.0, "decisive_evidence": "x", "rejected": {}}'

    primary = FakePrimary()
    selector = module.CandidateSetSelector(
        enabled=True,
        strategy="calibrated_rank_accept",
        use_no_match=True,
        training_reference_file_path=train_ref,
        calibration={
            "min_positive_sources": 1,
            "background_negative_weight": 0.0,
            "background_negative_weight_grid": [0.0],
            "validation_fraction": 0.0,
            "max_epochs": 40,
        },
        llm={
            "enabled": True,
            "max_prompts": 0,
            "max_prompt_fraction": 0.0,
            "trigger_acceptance_margin": 1.0,
            "trigger_rank_margin": 1.0,
            "min_confidence": 0.0,
        },
    )

    out = selector.forward(_selector_training_df(), primary_model=primary, threshold=0.7)

    assert primary.calls >= 2
    assert out["candidate_df"].groupby("Src")["selection_llm_used"].any().all()


def test_legacy_selector_weights_map_to_support_weight():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        enabled=True,
        weights={
            "absolute_support": 0.25,
            "candidate_competition": 0.25,
            "distinctive_evidence": 0.25,
            "equivalence_safety": 0.25,
        },
    )

    assert selector.support_weight == 0.25
    assert selector.no_match_threshold == 0.55
    assert selector.llm["max_candidates"] == 5


def test_selector_llm_defaults_off_but_can_be_enabled_explicitly():
    module = _load_selector_module()

    default_selector = module.CandidateSetSelector()
    enabled_selector = module.CandidateSetSelector(llm={"enabled": True})

    assert default_selector.llm["enabled"] is False
    assert default_selector.llm["mode"] == "off"
    assert enabled_selector.llm["enabled"] is True
    assert enabled_selector.llm["mode"] == "veto"


def test_runner_applies_additional_model_with_results_json():
    runner_module = _load_runner_module()
    runner = object.__new__(runner_module.SemanticAlignmentRunner)

    class ExtraModel:
        def eval(self):
            return self

        def forward(self, candidate_df, results_json=None, local_alignment=False, **kwargs):
            df = candidate_df.copy()
            df["S_final"] = [0.9, 0.1]
            if results_json:
                for rec, score in zip(results_json, df["S_final"]):
                    rec["confidences"]["S_final"] = float(score)
            return {"candidate_df": df, "results_json": results_json}

    runner._models = [SimpleNamespace(), ExtraModel()]
    runner._model = runner._models[0]
    runner._dataset = SimpleNamespace(
        dataframe=pd.DataFrame(
            {
                "Src": ["s1", "s1"],
                "Tgt": ["t1", "t2"],
                "cand_sim": [0.8, 0.7],
                "cand_sim_prob": [0.53, 0.47],
            }
        )
    )
    runner._candidate_rows = [
        {"Src": "s1", "Tgt": "t1", "S_final": 0.2},
        {"Src": "s1", "Tgt": "t2", "S_final": 0.8},
    ]
    runner._results_json = [
        {"src_iri": "s1", "tgt_iri": "t1", "confidences": {"S_final": 0.2}, "prediction": {}},
        {"src_iri": "s1", "tgt_iri": "t2", "confidences": {"S_final": 0.8}, "prediction": {}},
    ]
    runner.log = lambda *args, **kwargs: None

    df = runner._build_candidate_dataframe()
    out = runner._apply_additional_models(df, results_json=runner.results_json)

    assert list(out["S_final"]) == [0.9, 0.1]
    assert "cand_sim_prob" in out.columns
    assert runner.results_json[0]["confidences"]["S_final"] == 0.9


def test_validation_split_is_deterministic_and_source_disjoint():
    module = _load_selector_module()
    pairs = {(f"s{i}", f"t{i}") for i in range(10)}
    selector_a = module.CandidateSetSelector(
        request_seed=13,
        calibration={"validation_fraction": 0.3, "min_positive_sources": 2},
    )
    selector_b = module.CandidateSetSelector(
        request_seed=13,
        calibration={"validation_fraction": 0.3, "min_positive_sources": 2},
    )

    train_a, val_a, meta_a = selector_a._split_reference_pairs_by_source(pairs)
    train_b, val_b, meta_b = selector_b._split_reference_pairs_by_source(pairs)

    assert train_a == train_b
    assert val_a == val_b
    assert meta_a["held_out"] is True
    assert {src for src, _ in train_a}.isdisjoint({src for src, _ in val_a})


def test_equal_f1_threshold_tie_chooses_higher_precision_then_threshold():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(
        calibration={"accept_objective": "f1", "threshold_grid_step": 0.1}
    )
    decisions = _threshold_decisions(
        [
            (0.90, 1.0),
            (0.80, 0.0),
            (0.70, 0.0),
            (0.60, 1.0),
        ]
    )

    threshold, metrics = selector._tune_accept_threshold(decisions)

    assert threshold == 0.9
    assert metrics["P"] == 1.0
    assert metrics["F1"] == pytest.approx(2.0 / 3.0)


def test_p_match_score_mode_keeps_raw_monotonic_scores():
    module = _load_selector_module()
    selector = module.CandidateSetSelector(score_mode="p_match")

    assert selector._final_selector_score(0.95, accept_threshold=0.50, score_threshold=0.70) == 0.95
    assert selector._final_selector_score(0.72, accept_threshold=0.70, score_threshold=0.70) == 0.72
    assert selector._final_selector_score(0.80, accept_threshold=0.70, score_threshold=0.70) > selector._final_selector_score(0.72, accept_threshold=0.70, score_threshold=0.70)


def test_calibrated_llm_veto_cannot_switch_winner(tmp_path):
    module = _load_selector_module()
    train_ref = _write_training_reference(
        tmp_path,
        [("s_train", "t_gold_train"), ("s_eval", "t_gold_eval")],
    )

    class FakePrimary:
        def candidate_set_select(self, prompt):
            return '{"winner": "C1", "relation": "equivalent", "confidence": 1.0, "decisive_evidence": "x", "rejected": {}}'

    selector = module.CandidateSetSelector(
        enabled=True,
        strategy="calibrated_rank_accept",
        use_no_match=True,
        training_reference_file_path=train_ref,
        calibration={
            "min_positive_sources": 1,
            "background_negative_weight": 0.0,
            "background_negative_weight_grid": [0.0],
            "validation_fraction": 0.0,
            "max_epochs": 40,
        },
        llm={
            "enabled": True,
            "mode": "veto",
            "trigger_acceptance_margin": 1.0,
            "trigger_rank_margin": 1.0,
            "min_confidence": 0.0,
        },
    )

    out = selector.forward(_selector_training_df(), primary_model=FakePrimary(), threshold=0.7)
    scored = out["candidate_df"].set_index(["Src", "Tgt"])

    assert scored.loc[("s_eval", "t_gold_eval"), "selection_winner"]
    assert not scored.loc[("s_eval", "t_bad_eval"), "selection_winner"]
    assert scored.loc[("s_eval", "t_gold_eval"), "selection_llm_used"]
    assert scored.loc[("s_eval", "t_gold_eval"), "selection_reason"] != "llm"


def test_runner_uses_selector_accept_threshold_for_raw_p_match_scores():
    runner_module = _load_runner_module()
    candidate_df = pd.DataFrame(
        {
            "Src": ["s1", "s2"],
            "Tgt": ["t1", "t2"],
            "S_final": [0.52, 0.49],
            "selection_accept_threshold": [0.5, 0.5],
        }
    )

    assert runner_module.SemanticAlignmentRunner._effective_alignment_threshold(candidate_df, 0.7) == 0.5


def test_target_conflict_resolver_preserves_exact_mapping():
    from exact.core.entities.mappings import EntityMapping

    exact = EntityMapping("s_exact", "t_shared", score=1.0)
    learned = EntityMapping("s_learned", "t_shared", score=0.99)
    other = EntityMapping("s_other", "t_other", score=0.80)

    filtered = EntityMapping.filter_top_n_target_entity_mappings(
        [learned, other, exact],
        1,
        protected_pairs={("s_exact", "t_shared")},
    )

    assert ("s_exact", "t_shared") in EntityMapping.as_tuples(filtered)
    assert ("s_learned", "t_shared") not in EntityMapping.as_tuples(filtered)
    assert ("s_other", "t_other") in EntityMapping.as_tuples(filtered)


def test_target_conflict_validation_metric_can_prefer_resolver():
    module = _load_selector_module()
    selector = module.CandidateSetSelector()
    decisions = {
        "s_exact_conflict": {
            "winner_pair": ("s_exact_conflict", "t_exact"),
            "p_match": 0.9,
            "label": 0.0,
            "has_reference": True,
            "sample_weight": 1.0,
        },
        "s_good": {
            "winner_pair": ("s_good", "t_good"),
            "p_match": 0.8,
            "label": 1.0,
            "has_reference": True,
            "sample_weight": 1.0,
        },
    }

    no_resolver = selector._decision_metrics_at_threshold(decisions, 0.5)
    resolver = selector._metrics_with_target_conflict(
        decisions=decisions,
        threshold=0.5,
        target_cardinality=1,
        protected_exact_pairs={("s_exact", "t_exact")},
    )

    assert resolver["FP"] < no_resolver["FP"]
    assert resolver["F1"] > no_resolver["F1"]


def test_ignored_alignment_classes_are_filtered_from_candidate_rows():
    module = _load_dataset_module_or_skip()

    class DummyDataset(module.IDataset):
        def __getitem__(self, idx):
            raise IndexError(idx)

        def __len__(self):
            return 0

        def get_features(self, df):
            return df

        def plot_feature_distributions(self, *args, **kwargs):
            return None

        def log_sanity_examples(self, *args, **kwargs):
            return None

    dataset = object.__new__(DummyDataset)
    dataset._filter_ignored_alignment_classes = True
    dataset._source_ignored_alignment_classes = {"s_drop"}
    dataset._target_ignored_alignment_classes = {"t_drop"}
    dataset.log = lambda *args, **kwargs: None
    df = pd.DataFrame(
        {
            "Src": ["s_keep", "s_drop", "s_keep"],
            "Tgt": ["t_keep", "t_keep", "t_drop"],
            "Label": [0, 0, 0],
        }
    )

    filtered = dataset._filter_candidates_ignored_classes(df)

    assert filtered[["Src", "Tgt"]].values.tolist() == [["s_keep", "t_keep"]]
