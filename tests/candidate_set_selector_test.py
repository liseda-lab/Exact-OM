import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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


def _write_training_reference(tmp_path, pairs):
    path = tmp_path / "train.tsv"
    pd.DataFrame(pairs, columns=["Src", "Tgt"]).to_csv(path, sep="\t", index=False)
    return path


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
            "max_epochs": 80,
        },
    )

    out = selector.forward(_selector_training_df(), threshold=0.7)
    scored = out["candidate_df"].set_index(["Src", "Tgt"])

    assert scored.loc[("s_eval", "t_gold_eval"), "selection_winner"]
    assert scored.loc[("s_eval", "t_gold_eval"), "S_select"] >= 0.7
    assert scored.loc[("s_eval", "t_bad_eval"), "S_select"] == 0.0
    assert scored.loc[("s_eval", "t_gold_eval"), "P_rank"] > scored.loc[("s_eval", "t_bad_eval"), "P_rank"]


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


def test_calibrated_llm_arbitration_has_no_prompt_cap(tmp_path):
    module = _load_selector_module()
    train_ref = _write_training_reference(tmp_path, [("s_train", "t_gold_train")])

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

    assert primary.calls == 2
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
