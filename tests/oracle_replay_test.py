import pandas as pd
import pytest

from exact.impl.models.selector.oracle_replay import (
    observed_response_oracle,
    perfect_intervention,
)


def test_cached_oracle_uses_actual_corrections_and_shared_perfect_population():
    frame = pd.DataFrame(
        [
            {"Src": source, "Tgt": target, "S_base": score, "U": 1.0}
            for source in ("correction", "harm", "unseen", "unknown")
            for target, score in (("wrong", 0.8), ("right", 0.7))
        ]
    )
    frame.loc[(frame.Src == "harm") & (frame.Tgt == "right"), "S_base"] = 0.9
    references = {(source, "right") for source in ("correction", "harm", "unseen")}
    cached = [
        {
            "source": "correction",
            "valid": True,
            "pair_probabilities": {"wrong": 0.01, "right": 0.98},
            "choice": "right",
        },
        {
            "source": "harm",
            "valid": True,
            "pair_probabilities": {"wrong": 0.98, "right": 0.01},
            "choice": "wrong",
        },
    ]
    observed = observed_response_oracle(
        frame, cached, references, budget=2, negative_label_policy="complete_reference"
    )
    assert observed["selected_sources"] == ["correction"]
    assert {row["source"]: row["net_correction"] for row in observed["counterfactuals"]} == {
        "correction": 1,
        "harm": -1,
    }
    assert {row["source"] for row in observed["excluded_sources"]} == {"unknown", "unseen"}
    assert observed["no_llm_invocations"] and not observed["deployable"]
    perfect = perfect_intervention(
        frame,
        references,
        observed["selected_sources"],
        displayed_candidates=observed["decision_probs"],
        negative_label_policy="complete_reference",
    )
    assert perfect["selected_sources"] == observed["selected_sources"]
    assert perfect["decision_probs"] == {"correction": {"wrong": 0.0, "right": 1.0}}
    assert perfect["counterfactuals"][0]["net_correction"] == 1


def test_oracle_refuses_unknown_negatives_and_unfrozen_candidates():
    frame = pd.DataFrame(
        {
            "Src": ["s", "s"],
            "Tgt": ["p", "u"],
            "S_base": [0.8, 0.7],
            "U": [1.0, 1.0],
            "confirmed_label": [1.0, None],
        }
    )
    response = [
        {"source": "s", "valid": True, "pair_probabilities": {"p": 0.9, "u": 0.01}, "choice": "p"}
    ]
    result = observed_response_oracle(
        frame, response, {("s", "p")}, budget=1, negative_label_policy="confirmed_negatives"
    )
    assert not result["selected_sources"]
    with pytest.raises(ValueError, match="positive-unlabelled"):
        observed_response_oracle(
            frame, response, {("s", "p")}, budget=1, negative_label_policy="unknown"
        )
    response[0]["pair_probabilities"]["outside"] = 0.1
    with pytest.raises(ValueError, match="frozen"):
        observed_response_oracle(
            frame, response, {("s", "p")}, budget=1, negative_label_policy="complete_reference"
        )


def test_benchmark_outcomes_do_not_invent_training_labels_or_drop_unlabelled_sources(monkeypatch):
    from exact.impl.models.selector import oracle_replay

    frame = pd.DataFrame(
        [
            {"Src": source, "Tgt": target, "S_base": score, "U": 1.0, "confirmed_label": None}
            for source in ("correction", "no_reference", "unobserved")
            for target, score in (("a", 0.8), ("b", 0.7))
        ]
    )
    original = frame.copy(deep=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("Benchmark labels must never enter the training-label path")

    monkeypatch.setattr(oracle_replay, "safe_training_labels", unexpected)
    cached = [
        {"source": "correction", "valid": True, "pair_probabilities": {"a": 0.01, "b": 0.98}},
        {"source": "no_reference", "valid": True, "pair_probabilities": {"a": 0.01, "b": 0.01}},
    ]
    result = observed_response_oracle(
        frame,
        cached,
        {("correction", "b")},
        budget=2,
        negative_label_policy="unknown",
        outcome_semantics="benchmark_reference",
    )
    assert result["selected_sources"] == ["correction", "no_reference"]
    assert result["negative_label_policy"] == "unknown"
    assert result["no_training_use"] and not result["training_use_permitted"]
    assert result["global_f1_optimality_claim"] is False
    assert result["probability_semantics"] == "observed_cached_model_probability"
    assert result["reference_abstention_semantics"] == "no_recorded_positive_not_natural_nil"
    assert result["excluded_sources"] == [
        {"source": "unobserved", "reason": "no_valid_observed_response"}
    ]
    assert len(result["population_rows"]) == len(frame)
    assert all("confirmed_label" not in row for row in result["population_rows"])
    outcomes = {row["source"]: row for row in result["counterfactuals"]}
    assert outcomes["no_reference"]["reference_abstention"] is True
    assert outcomes["no_reference"]["intervention_choice"] is None
    assert all(row["net_reference_gain"] == 1 for row in outcomes.values())
    assert all(
        "baseline_correct" not in row and "net_correction" not in row for row in outcomes.values()
    )
    pd.testing.assert_frame_equal(frame, original)


def test_benchmark_perfect_cannot_insert_missing_gold_or_claim_global_optimality():
    frame = pd.DataFrame(
        {"Src": ["s", "s"], "Tgt": ["a", "b"], "S_base": [0.8, 0.7], "U": [1.0, 1.0]}
    )
    result = perfect_intervention(
        frame,
        {("s", "outside")},
        ["s"],
        displayed_candidates={"s": {"a": 0.1}},
        negative_label_policy="unknown",
        outcome_semantics="benchmark_reference",
    )
    assert result["decision_probs"] == {"s": {"a": 0.0}}
    assert result["population_rows"] == frame.to_dict("records")
    # The unsupported positive is never inserted. An untouched candidate may still win.
    assert result["counterfactuals"][0]["intervention_choice"] == "b"
    assert result["counterfactuals"][0]["intervention_reference_match"] is False
    assert result["counterfactuals"][0]["reference_abstention"] is False
    assert result["probability_semantics"] == "synthetic_reference_membership"
    assert result["global_f1_optimality_claim"] is False
    assert result["negative_label_policy"] == "unknown"


@pytest.mark.parametrize(
    "response",
    [
        {"source": "outside", "valid": True, "pair_probabilities": {"a": 0.5}},
        {"source": "s", "valid": True, "pair_probabilities": {"outside": 0.5}},
        {"source": "s", "valid": True, "pair_probabilities": {"a": float("nan")}},
        {"source": "s", "valid": True, "pair_probabilities": {"a": float("inf")}},
        {"source": "s", "valid": True, "pair_probabilities": {"a": 1.1}},
    ],
)
def test_benchmark_rejects_unobserved_support_and_invalid_probabilities(response):
    frame = pd.DataFrame({"Src": ["s"], "Tgt": ["a"], "S_base": [0.8], "U": [1.0]})
    with pytest.raises(ValueError, match="frozen candidate population"):
        observed_response_oracle(
            frame,
            [response],
            set(),
            budget=1,
            negative_label_policy="unknown",
            outcome_semantics="benchmark_reference",
        )


@pytest.mark.parametrize("support", [{}, {"s": ["a"]}, {"s": {"a": float("nan")}}])
def test_benchmark_perfect_requires_actual_observed_probability_support(support):
    frame = pd.DataFrame({"Src": ["s"], "Tgt": ["a"], "S_base": [0.8], "U": [1.0]})
    with pytest.raises(ValueError, match="observed"):
        perfect_intervention(
            frame,
            set(),
            ["s"],
            displayed_candidates=support,
            negative_label_policy="unknown",
            outcome_semantics="benchmark_reference",
        )


def test_benchmark_ignored_pair_semantics_fail_closed_and_budget_is_preserved():
    frame = pd.DataFrame({"Src": ["s"], "Tgt": ["a"], "S_base": [0.8], "U": [1.0]})
    options = {"negative_label_policy": "unknown", "outcome_semantics": "benchmark_reference"}
    with pytest.raises(ValueError, match="ignored-pair"):
        observed_response_oracle(frame, [], set(), budget=1, ignored_pairs={("s", "a")}, **options)
    with pytest.raises(ValueError, match="ignored-pair"):
        perfect_intervention(
            frame, set(), [], displayed_candidates={}, ignored_pairs={("s", "a")}, **options
        )
    with pytest.raises(ValueError, match="200"):
        observed_response_oracle(frame, [], set(), budget=201, **options)
    result = observed_response_oracle(
        frame,
        [{"source": "s", "valid": True, "pair_probabilities": {"a": 0.0}}],
        set(),
        budget=0,
        **options,
    )
    assert result["selected_sources"] == []
    assert result["population_rows"] == frame.to_dict("records")
