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
