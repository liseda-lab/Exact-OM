import builtins
import json
import math

import pandas as pd
import pytest

from exact.core.entities.mappings.entity import EntityMapping
from exact.impl.extraction import extract_global_alignment
from exact.impl.models.selector import CandidateSetSelector
from exact.impl.models.selector.calibration_helpers import (
    brier_score,
    expected_calibration_error,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    knee_threshold,
    otsu_threshold,
    score_calibrator_from_dict,
)
from exact.impl.models.selector.llm_gate import (
    analytical_oracle_ceiling,
    fit_forced_sample_artifact,
    fit_quantile_gate_artifact,
)
from exact.impl.models.selector.nil_ranking import joint_nil_distribution
from exact.impl.models.semantic_llm import (
    LISTWISE_NONE_KEY,
    aggregate_listwise_call_probabilities,
    build_listwise_call_plan,
    build_listwise_decision_prompt,
    categorical_probabilities_from_logprobs,
    transform_listwise_probabilities,
)


def _pairs(result):
    return {(mapping.head, mapping.tail) for mapping in result.mappings}


def _collision_graph():
    return [
        EntityMapping("s1", "t1", score=0.90),
        EntityMapping("s1", "t2", score=0.80),
        EntityMapping("s2", "t1", score=0.85),
        EntityMapping("s2", "t2", score=0.10),
    ]


def test_extraction_strategies_are_distinct_and_deterministic():
    mappings = _collision_graph()
    greedy = extract_global_alignment(mappings, mode="greedy", threshold=0.0)
    mutual = extract_global_alignment(mappings, mode="mutual_best", threshold=0.0)
    stable = extract_global_alignment(mappings, mode="stable_marriage", threshold=0.0)
    assignment = extract_global_alignment(mappings, mode="assignment", threshold=0.0)

    assert _pairs(greedy) == {("s1", "t1")}
    assert _pairs(mutual) == {("s1", "t1")}
    assert _pairs(stable) == {("s1", "t1"), ("s2", "t2")}
    assert _pairs(assignment) == {("s1", "t2"), ("s2", "t1")}
    assert _pairs(
        extract_global_alignment(list(reversed(mappings)), mode="assignment", threshold=0.0)
    ) == _pairs(assignment)


def test_extraction_preassigns_exact_and_caps_assignment_components():
    mappings = _collision_graph()
    protected = extract_global_alignment(
        mappings,
        mode="assignment",
        threshold=0.95,
        protected_pairs={("s1", "t2")},
    )
    assert _pairs(protected) == {("s1", "t2")}

    capped = extract_global_alignment(
        mappings,
        mode="assignment",
        threshold=0.0,
        assignment_component_cap=3,
    )
    assert _pairs(capped) == {("s1", "t1")}
    assert capped.diagnostics["assignment_fallback_components"] == 1


@pytest.mark.parametrize(
    "protected_pairs",
    [
        {("s1", "t1"), ("s1", "t2")},
        {("s1", "t1"), ("s2", "t1")},
    ],
)
def test_extraction_rejects_conflicting_protected_exact_matches(protected_pairs):
    with pytest.raises(ValueError, match="Conflicting protected exact matches"):
        extract_global_alignment(
            _collision_graph(),
            mode="assignment",
            protected_pairs=protected_pairs,
        )


def test_assignment_reports_missing_scipy_but_cap_fallback_does_not_import_it(monkeypatch):
    real_import = builtins.__import__

    def import_without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("simulated missing scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_scipy)
    capped = extract_global_alignment(
        _collision_graph(), mode="assignment", assignment_component_cap=3
    )
    assert capped.diagnostics["assignment_fallback_components"] == 1
    with pytest.raises(RuntimeError, match="requires SciPy"):
        extract_global_alignment(
            _collision_graph(),
            mode="assignment",
            assignment_component_cap=100,
        )


def test_calibration_and_distribution_threshold_primitives():
    scores = [0.05, 0.15, 0.80, 0.95]
    labels = [0, 0, 1, 1]
    platt = fit_platt_calibrator(scores, labels)
    isotonic = fit_isotonic_calibrator(scores, labels)

    assert platt.predict_one(0.9) > platt.predict_one(0.1)
    iso_predictions = isotonic.predict(scores)
    assert iso_predictions == sorted(iso_predictions)
    assert otsu_threshold(scores) == pytest.approx(0.475)
    assert knee_threshold(scores) == pytest.approx(0.475)
    assert brier_score([0.1, 0.9], [0, 1]) == pytest.approx(0.01)
    assert expected_calibration_error([0.1, 0.9], [0, 1], bins=2) == pytest.approx(0.1)
    restored = score_calibrator_from_dict(platt.to_dict(), expected_mode="platt")
    assert restored.predict_one(0.9) == pytest.approx(platt.predict_one(0.9))


def test_score_calibration_artifact_is_applied_and_missing_artifact_fails_closed(tmp_path):
    artifact = tmp_path / "platt.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "score_calibrator",
                "calibrator": {"mode": "platt", "slope": 10.0, "intercept": -5.0},
                "fit_provenance": {"split_role": "training_oof"},
            }
        ),
        encoding="utf-8",
    )
    selector = CandidateSetSelector(
        enabled=True,
        experiment_config=_experiment_config(label_free_mode="score_partition"),
        matching_calibration={
            "mode": "platt",
            "threshold_mode": "fixed",
            "artifact": artifact,
        },
    )
    result = selector.forward(_label_free_frame(), threshold=0.5)["candidate_df"]

    assert result["S_pair_pre_calibration"].tolist() == pytest.approx(
        _label_free_frame()["S_final"].tolist()
    )
    assert result.iloc[0]["S_pair_final"] > result.iloc[0]["S_pair_pre_calibration"]
    assert selector._calibration_meta["score_calibration"]["artifact"]["sha256"]
    with pytest.raises(ValueError, match="requires an immutable fitted artifact"):
        CandidateSetSelector(matching_calibration={"mode": "isotonic"})


def _experiment_config(**overrides):
    config = {
        "enabled": True,
        "emit_candidate_scores": False,
        "accept_model": "logistic",
        "accept_training": "winner_only",
        "label_free_mode": "current_fallback",
        "tuning": {"count_reference_miss_as": "fp_fn"},
        "rerank": {
            "mode": "current",
            "model": "current_linear",
            "features": "current",
            "artifact": None,
        },
    }
    config.update(overrides)
    return config


def _label_free_frame():
    return pd.DataFrame(
        {
            "Src": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "Tgt": ["t1", "t2", "t1", "t2", "t3", "t4"],
            "S_final": [0.95, 0.20, 0.90, 0.85, 0.80, 0.70],
            "s_label": [0.95, 0.10, 0.90, 0.85, 0.80, 0.70],
            "S_struct": [0.95, 0.10, 0.90, 0.85, 0.80, 0.70],
            "cand_sim": [0.95, 0.10, 0.90, 0.85, 0.80, 0.70],
            "s_diff": [0.90] * 6,
        }
    )


def test_score_partition_emits_all_candidate_scores_only_when_requested():
    config = _experiment_config(
        emit_candidate_scores=True,
        label_free_mode="score_partition",
    )
    selector = CandidateSetSelector(
        enabled=True,
        use_no_match=False,
        experiment_config=config,
        matching_calibration={"mode": "none", "threshold_mode": "fixed"},
    )
    result = selector.forward(_label_free_frame(), threshold=0.9)["candidate_df"]

    s1 = result[result["Src"] == "s1"].set_index("Tgt")
    s3 = result[result["Src"] == "s3"]
    assert s1.loc["t1", "selection_winner"]
    assert s1.loc["t2", "S_select"] == pytest.approx(0.20)
    assert (s3["S_select"] == 0.0).all()
    assert selector._calibration_meta["target_labels_used"] is False


def test_reciprocal_consensus_requires_margin_channels_and_reciprocity():
    selector = CandidateSetSelector(
        enabled=True,
        use_no_match=False,
        experiment_config=_experiment_config(label_free_mode="reciprocal_consensus"),
    )
    result = selector.forward(_label_free_frame(), threshold=None)["candidate_df"]
    winners = result[result["selection_winner"]]

    assert list(zip(winners["Src"], winners["Tgt"])) == [("s1", "t1")]
    assert bool(winners.iloc[0]["selection_reciprocal"])
    assert int(winners.iloc[0]["selection_channel_agreement"]) == 3
    assert selector._calibration_meta["target_labels_used"] is False


def test_nil_joint_scale_uses_probabilities_and_is_attached_to_every_candidate():
    distribution = joint_nil_distribution([0.8, 0.2], 0.4)
    assert sum(distribution.real_q) + distribution.nil_q == pytest.approx(1.0)
    assert distribution.real_q[0] > distribution.nil_q > distribution.real_q[1]

    selector = CandidateSetSelector(
        enabled=True,
        use_no_match=True,
        nil_config={"mode": "heuristic", "ranking_scale": "joint_accept_probability"},
    )
    frame = _label_free_frame().query("Src == 's1'").reset_index(drop=True)
    result = selector.forward(frame)["candidate_df"]
    assert result["P_match"].nunique() == 2
    assert result["P_nil"].nunique() == 1
    assert result["Q_match"].sum() + result.iloc[0]["Q_nil"] == pytest.approx(1.0)
    assert result["P_rank"].tolist() == pytest.approx(result["Q_match"].tolist())
    assert selector._calibration_meta["nil"]["applied"] is True


def test_reference_miss_count_mode_and_analytic_rerank_dispatch():
    wrong_winner = {
        "s": {
            "p_match": 0.9,
            "label": 0.0,
            "has_reference": True,
            "sample_weight": 1.0,
        }
    }
    fp_fn = CandidateSetSelector(
        experiment_config=_experiment_config(tuning={"count_reference_miss_as": "fp_fn"})
    )
    fp_only = CandidateSetSelector(
        experiment_config=_experiment_config(tuning={"count_reference_miss_as": "fp"})
    )
    assert fp_fn._decision_metrics_at_threshold(wrong_winner, 0.5)["FN"] == 1.0
    assert fp_only._decision_metrics_at_threshold(wrong_winner, 0.5)["FN"] == 0.0

    analytic = CandidateSetSelector(
        experiment_config=_experiment_config(
            rerank={
                "mode": "analytic",
                "model": "current_linear",
                "features": "current",
                "artifact": None,
            }
        )
    )
    model = analytic._fit_rank_model({}, [])
    assert model == {"model_type": "analytic"}
    assert analytic._score_rank_model([math.log(0.8 / 0.2)], model) == pytest.approx(0.8)


def test_unimplemented_selector_arms_fail_explicitly():
    with pytest.raises(NotImplementedError, match="pseudo_label_accept"):
        CandidateSetSelector(
            experiment_config=_experiment_config(label_free_mode="pseudo_label_accept")
        )
    with pytest.raises(NotImplementedError, match="Rerank objective"):
        CandidateSetSelector(
            experiment_config=_experiment_config(
                rerank={
                    "mode": "listwise_nil",
                    "model": "current_linear",
                    "features": "current",
                    "artifact": None,
                }
            )
        )


def test_listwise_prompt_and_probability_transforms_are_pure_and_complete():
    prompt, index_by_label = build_listwise_decision_prompt(
        "source", "summary", ["candidate one", "candidate two"]
    )
    assert "{A, B, Z}" in prompt["user"]
    assert index_by_label == {"A": 0, "B": 1, "Z": None}

    categorical = categorical_probabilities_from_logprobs(
        {"A": math.log(0.6), "B": math.log(0.3), "Z": math.log(0.1)},
        2,
    )
    assert sum(categorical.values()) == pytest.approx(1.0)
    assert transform_listwise_probabilities(categorical, 2, "raw_joint")["A"] == pytest.approx(0.6)
    assert transform_listwise_probabilities(categorical, 2, "conditional_real")[
        "A"
    ] == pytest.approx(2.0 / 3.0)
    assert transform_listwise_probabilities(categorical, 2, "pairwise_vs_none")[
        "A"
    ] == pytest.approx(6.0 / 7.0)
    assert transform_listwise_probabilities(categorical, 2, "max_normalized")["A"] == pytest.approx(
        1.0
    )


def test_listwise_sc_plan_is_six_calls_and_aggregation_is_order_invariant():
    candidates = [
        {"target_iri": f"t{index}", "score": 1.0 - (index / 10), "brief": f"brief {index}"}
        for index in range(1, 7)
    ]
    plan = build_listwise_call_plan(
        source_iri="s1",
        source_label="source",
        source_summary="summary",
        candidates=candidates,
        mode="listwise_sc",
        request_seed=17,
    )
    replay = build_listwise_call_plan(
        source_iri="s1",
        source_label="source",
        source_summary="summary",
        candidates=list(reversed(candidates)),
        mode="listwise_sc",
        request_seed=17,
    )
    assert plan == replay
    assert len(plan.calls) == 6
    assert {call.temperature for call in plan.calls} == {0.7}
    assert len({call.seed for call in plan.calls}) == 6
    assert plan.overflow_candidate_ids == ("t6",)
    assert plan.calls[3].candidate_ids == tuple(reversed(plan.calls[0].candidate_ids))

    call_probabilities = []
    for call in plan.calls:
        labels = [chr(ord("A") + index) for index in range(len(call.candidate_ids))]
        probabilities = {
            label: (0.7 if target == "t2" else 0.2 / (len(labels) - 1))
            for label, target in zip(labels, call.candidate_ids)
        }
        probabilities["Z"] = 0.1
        call_probabilities.append(probabilities)
    aggregate = aggregate_listwise_call_probabilities(
        plan,
        call_probabilities,
        probability_mode="raw_joint",
    )
    assert aggregate.call_count == 6
    assert aggregate.categorical["t2"] == pytest.approx(0.7)
    assert aggregate.categorical[LISTWISE_NONE_KEY] == pytest.approx(0.1)
    assert aggregate.pair_probabilities["t2"] == pytest.approx(0.7)


def test_quantile_forced_and_oracle_gate_helpers_are_exact_and_no_call():
    rows = [
        {"U": 0.8, "source_iri": "s2", "target_iri": "t"},
        {"U": 0.8, "source_iri": "s1", "target_iri": "t"},
        {"U": 0.1, "source_iri": "s3", "target_iri": "t"},
    ]
    quantile = fit_quantile_gate_artifact(
        rows,
        fraction=1.0 / 3.0,
        task_id="task",
        entity_kind="class",
    )
    assert quantile["selected_count"] == 1
    assert quantile["pairs"] == [["s1", "t"]]
    assert quantile["boundary_ids"]["excluded_first"] == ["s2", "t"]

    forced = fit_forced_sample_artifact(
        rows,
        sample_size=2,
        seed=9,
        task_id="task",
        entity_kind="class",
    )
    assert forced == fit_forced_sample_artifact(
        list(reversed(rows)),
        sample_size=2,
        seed=9,
        task_id="task",
        entity_kind="class",
    )
    assert len(forced["pairs"]) == 2

    oracle = analytical_oracle_ceiling([0.9, 0.8, 0.2], [1, 0, 1], threshold=0.5)
    assert oracle.routed == (False, True, True)
    assert oracle.corrected_scores == pytest.approx((0.9, 0.0, 1.0))
    assert oracle.llm_invocations == 0
    assert oracle.deployable is False
