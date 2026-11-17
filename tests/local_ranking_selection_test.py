"""Local ranking preserves evidence ordering across global acceptance decisions."""

import ast
import json
import math

import pandas as pd
import pytest
import torch

from exact.core.contracts.model import IModel
from exact.core.entities.configs.dataset import DatasetMask
from exact.impl.models.selector import CandidateSetSelector
from exact.impl.models.selector.feature_contract import feature_contract
from exact.impl.trainer import SemanticAlignmentRunner


def _frame():
    return pd.DataFrame(
        {
            "Src": ["source"] * 3,
            "Tgt": ["a", "c", "b"],
            "S_final": [0.9, 0.3, 0.2],
            "s_label": [0.1, 0.4, 0.9],
        }
    )


def _fitted(tmp_path, accepted=True, **kwargs):
    """Fixed synthetic weights exercise real artifact inference without training."""
    selector = CandidateSetSelector(
        enabled=True,
        global_only=False,
        strategy="calibrated_rank_accept",
        experiment_config={"enabled": True, "emit_candidate_scores": False},
        **kwargs,
    )
    rank_count, accept_count = len(selector.RANK_FEATURE_NAMES), len(selector.ACCEPT_FEATURE_NAMES)
    rank_weights = [0.0] * rank_count
    rank_weights[selector.RANK_FEATURE_NAMES.index("s_label")] = 3.0
    path = tmp_path / "selector.json"
    path.write_text(
        json.dumps(
            {
                "kind": "fitted_selector",
                "schema_version": 1,
                "fit_identity": "synthetic-fixed-head",
                "explanation_schema": "feature_additive",
                "oof_metrics": {},
                "accept_threshold": 0.5,
                "rank_model": {
                    "model_type": "current_linear",
                    "mean": [0.0] * rank_count,
                    "scale": [1.0] * rank_count,
                    "weights": rank_weights,
                    "bias": 0.0,
                },
                "accept_model": {
                    "mean": [0.0] * accept_count,
                    "scale": [1.0] * accept_count,
                    "weights": [0.0] * accept_count,
                    "bias": 5.0 if accepted else -5.0,
                },
                "fit_provenance": {
                    "fitting_recipe_version": 3,
                    "feature_contract": feature_contract(_frame()),
                    "rank_features": selector.RANK_FEATURE_NAMES,
                    "accept_features": selector.ACCEPT_FEATURE_NAMES,
                    "score_calibration": "none",
                    "application": {},
                    "training_sources": ["train"],
                    "recipe": {
                        "rerank": {
                            key: value
                            for key, value in selector.rerank_config.items()
                            if key != "artifact"
                        }
                    },
                },
            }
        )
    )
    selector.rerank_config["artifact"] = str(path)
    return selector


@pytest.mark.parametrize("accepted", [True, False])
def test_local_fitted_rank_retains_nonwinners_and_rejected_sources(tmp_path, accepted):
    selector = _fitted(tmp_path, accepted)
    records = [
        {"src_iri": "source", "tgt_iri": target, "confidences": {}} for target in _frame().Tgt
    ]
    global_frame = selector.forward(_frame(), threshold=0.7)["candidate_df"]
    local = selector.forward(_frame(), local_alignment=True, threshold=0.7, results_json=records)[
        "candidate_df"
    ]

    assert local.sort_values("S_pair_final", ascending=False).Tgt.tolist() == ["a", "c", "b"]
    assert local.sort_values("S_final", ascending=False).Tgt.tolist() == ["b", "c", "a"]
    assert (local.S_final > 0).all()
    assert local.S_final.tolist() == pytest.approx(local.P_rank.tolist())
    assert local.S_rank.tolist() == pytest.approx(local.P_rank.tolist())
    assert local.S_select.tolist() == global_frame.S_select.tolist()
    assert local.P_match.tolist() == global_frame.P_match.tolist()
    assert local.selection_abstained.tolist() == [not accepted] * 3
    assert global_frame.S_final.tolist() == global_frame.S_select.tolist()
    assert (global_frame.S_final > 0).sum() == int(accepted)
    if accepted:
        assert global_frame.loc[2, "S_final"] == pytest.approx(1 / (1 + math.exp(-5)))
    for row, record in zip(local.itertuples(), records):
        assert record["confidences"]["S_final"] == row.S_final
        assert record["confidences"]["S_rank"] == row.S_rank
        assert record["selector_explanation"]["output"]["S_select"] == row.S_select


def test_explicit_nil_joint_scale_remains_separate_from_conditional_ranking(tmp_path):
    selector = _fitted(tmp_path, False, nil_config={"mode": "accept_model"})
    local = selector.forward(_frame(), local_alignment=True)["candidate_df"]
    assert local.selection_abstained.all()
    assert (local.S_select == 0).all()
    assert local.S_rank.sum() == pytest.approx(1.0)
    assert local.S_final.tolist() == pytest.approx(local.Q_match.tolist())
    assert local.S_final.sum() + local.Q_nil.iloc[0] == pytest.approx(1.0)
    assert local.S_final.tolist() != local.S_rank.tolist()
    assert local.sort_values("S_final", ascending=False).Tgt.tolist() == ["b", "c", "a"]


@pytest.mark.parametrize("mode", ["heuristic", "score_partition", "reciprocal_consensus"])
def test_other_selector_strategies_keep_complete_local_rank(mode):
    selector = CandidateSetSelector(
        enabled=True,
        global_only=False,
        no_match_threshold=0.0,
        experiment_config={
            "enabled": mode != "heuristic",
            "label_free_mode": mode if mode != "heuristic" else "current_fallback",
        },
    )
    result = selector.forward(_frame(), local_alignment=True, threshold=0.99)["candidate_df"]
    assert (result.S_final > 0).all()
    assert result.S_final.tolist() == pytest.approx(result.P_rank.tolist())
    assert result.Tgt.tolist() == _frame().Tgt.tolist()


class _Dataset:
    dataset_signature = "local-ranking-regression"
    cache_fingerprint = "fixed-synthetic-candidates"

    def __init__(self):
        self.dataframe = pd.DataFrame(
            {
                "Src": ["source"] * 4,
                "Tgt": ["a", "c", "b", "anchor"],
                "Scores": [0.0, 0.0, 0.0, 1.0],
                DatasetMask.prefiltered: [False, False, False, True],
            }
        )

    def __len__(self):
        return 3

    def __getitem__(self, index):
        target = self.dataframe.iloc[index].Tgt
        return {
            "src_iri": "source",
            "tgt_iri": target,
            "src_labels": ["source"],
            "tgt_labels": [target],
        }


class _LocalSelectionScorer(IModel):
    def __init__(self, source_choice="__NONE__", **kwargs):
        super().__init__()
        self.source_choice = source_choice

    def forward(self, *, tgt_iris, **kwargs):
        values = {row.Tgt: row for row in _frame().itertuples()}
        output = {
            "S_final": torch.tensor([values[target].S_final for target in tgt_iris]),
            "s_label": torch.tensor([values[target].s_label for target in tgt_iris]),
        }
        if self.source_choice is not None:
            # Global comparative abstention must not erase a local ranking.
            output["llm_source_choice"] = [self.source_choice] * len(tgt_iris)
        return output


def test_runner_and_writer_keep_all_local_candidates_and_exact_anchors(tmp_path):
    selector = _fitted(tmp_path, False)
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(), model=_LocalSelectionScorer, device=torch.device("cpu"), output_dir=tmp_path / "run"
    )
    runner._models.append(selector)
    predictions, _ = runner.predict(
        local_alignment=True,
        threshold=0.99,
        cardinality=1,
        target_cardinality=1,
        batch_size=3,
        num_workers=0,
        enable_checkpoints=False,
        audit_shards_enabled=False,
        cache_persist_policy="never",
        log_every=100,
    )
    assert {mapping.tail for mapping in predictions} == {"a", "b", "c"}
    assert all(mapping.score > 0 for mapping in predictions)
    assert runner.last_effective_threshold is None
    assert runner._decision_policy["source_cardinality"] is None
    assert runner._decision_policy["target_cardinality"] is None
    predictions = runner.apply_prefilter(predictions)
    assert [
        (mapping.tail, mapping.score) for mapping in predictions if mapping.tail == "anchor"
    ] == [("anchor", 1.0)]
    pool = tmp_path / "pool.tsv"
    pool.write_text("SrcEntity\tTgtEntity\tTgtCandidates\nsource\t\t['c', 'anchor', 'a', 'b']\n")
    paths = runner.save_results(
        predictions,
        candidates_one2many_path=pool,
        save_json=False,
        save_csv=False,
        save_stats_csv=False,
        paper_audit=True,
    )
    saved = ast.literal_eval(pd.read_csv(paths["alignment_tsv"], sep="\t").iloc[0].TgtCandidates)
    assert [target for target, _ in saved] == ["c", "anchor", "a", "b"]
    assert [target for target, _ in sorted(saved, key=lambda item: (-item[1], item[0]))] == [
        "anchor",
        "b",
        "c",
        "a",
    ]
    source = json.loads(paths["source_decisions_json"].read_text())["records"][0]
    assert all(row["emitted"] for row in source["candidates"])
    assert all("S_rank" in row for row in source["candidates"] if not row["protected_exact"])


def test_local_postprocessing_contract_does_not_invalidate_primary_or_global_checkpoints(tmp_path):
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(), model=_LocalSelectionScorer, device=torch.device("cpu"), output_dir=tmp_path
    )
    before = runner._build_checkpoint_fingerprint_payload()
    local = runner._postprocess_fingerprint_payload(DatasetMask.inference, True, None, None)
    global_ = runner._postprocess_fingerprint_payload(DatasetMask.inference, False, None, None)
    assert local.pop("local_ranking_contract") == 1
    assert "local_ranking_contract" not in global_
    local["local_alignment"] = False
    assert local == global_
    assert before == runner._build_checkpoint_fingerprint_payload()


@pytest.mark.parametrize("local", [True, False])
def test_baseline_runner_threshold_is_global_only(tmp_path, local):
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(),
        model=_LocalSelectionScorer,
        model_params={"source_choice": None},
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    predictions, _ = runner.predict(
        local_alignment=local,
        threshold=0.7,
        cardinality=1,
        batch_size=3,
        num_workers=0,
        enable_checkpoints=False,
        cache_persist_policy="never",
        log_every=100,
    )
    assert {mapping.tail for mapping in predictions} == ({"a", "b", "c"} if local else {"a"})
    assert runner.last_effective_threshold == (None if local else 0.7)


def test_local_ties_follow_iri_order_after_writer_and_evaluation_join(tmp_path):
    from exact.core.actions.evaluation import materialize_local_ranking_inputs
    from exact.io.writers.tsv import LocalTsvWriter

    selector = _fitted(tmp_path)
    pool, reference = tmp_path / "pool.tsv", tmp_path / "reference.tsv"
    pool.write_text("SrcEntity\tTgtEntity\tTgtCandidates\nsource\t\t['c', 'a', 'b']\n")
    reference.write_text("SrcEntity\tTgtEntity\tRelation\nsource\tb\t=\n")
    frame = _frame()
    frame.loc[frame.Tgt == "c", "s_label"] = 0.9
    for i, order in enumerate((frame, frame.iloc[::-1])):
        scored = selector.forward(order, local_alignment=True)["candidate_df"]
        pairs = list(zip(scored.Tgt, scored.S_final))
        score_path = tmp_path / f"scores-{i}.tsv"
        LocalTsvWriter().write(
            pd.DataFrame(
                [
                    {
                        "SrcEntity": "source",
                        "TgtEntity": "",
                        "TgtCandidates": pairs,
                    }
                ]
            ),
            score_path,
        )
        ranked, _ = materialize_local_ranking_inputs(
            score_path, pool, reference, tmp_path / f"eval-{i}"
        )
        candidates = ast.literal_eval(pd.read_csv(ranked, sep="\t").iloc[0].TgtCandidates)
        assert [target for target, _ in candidates] == ["b", "c", "a"]
        assert candidates[0][1] == candidates[1][1]
