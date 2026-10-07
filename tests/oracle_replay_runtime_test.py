import json

import pandas as pd
import pytest

from exact.impl.models.selector.oracle_replay import (
    observed_response_oracle,
    perfect_intervention,
)
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset


@pytest.mark.parametrize("outcome_semantics", ["verified_labels", "benchmark_reference"])
def test_cached_observed_and_fixed_perfect_replay_never_call_hosted_roles(
    tmp_path, monkeypatch, outcome_semantics
):
    dataset = _TinyDataset()
    inputs = {
        "src_iris": ["s", "s"],
        "tgt_iris": ["a", "t"],
        "src_label_lists": [["s"], ["s"]],
        "tgt_label_lists": [["a"], ["t"]],
    }
    base = _scorer(
        llm_experiment_config={
            "enabled": True,
            "gate": {"mode": "off"},
            "fusion_weight": "constant",
            "constant_weight": 0.5,
        }
    )
    base.attach_dataset(dataset)
    raw = base(**inputs)
    population = pd.DataFrame(
        {
            "Src": ["s", "s"],
            "Tgt": ["a", "t"],
            "S_base": raw["S_base"].tolist(),
            "U": raw["U"].tolist(),
        }
    )
    outcome_options = {
        "outcome_semantics": outcome_semantics,
        "negative_label_policy": (
            "unknown" if outcome_semantics == "benchmark_reference" else "complete_reference"
        ),
        "teacher_binding": {
            "dataset_signature": dataset.dataset_signature,
            "reference_role": "valid",
        },
    }
    observed = observed_response_oracle(
        population,
        [{"source": "s", "valid": True, "pair_probabilities": {"a": 0.1, "t": 0.9}, "choice": "t"}],
        {("s", "t")},
        budget=1,
        fusion_weight="constant",
        constant_weight=0.5,
        **outcome_options,
    )
    perfect = perfect_intervention(
        population,
        {("s", "t")},
        observed["selected_sources"],
        displayed_candidates=observed["decision_probs"],
        fusion_weight="constant",
        constant_weight=0.5,
        **outcome_options,
    )
    for state in (observed, perfect):
        path = tmp_path / (state["mode"] + ".json")
        path.write_text(json.dumps(state))
        scorer = _scorer(
            return_explanations=True,
            force_llm_summaries=True,
            llm_experiment_config={
                "enabled": True,
                "gate": {"mode": state["mode"], "artifact": str(path)},
                "fusion_weight": "constant",
                "constant_weight": 0.5,
            },
        )
        scorer.attach_dataset(dataset)
        scorer.use_llm = True

        def unexpected(*args, **kwargs):
            raise AssertionError("Oracle replay attempted a hosted call")

        for method in (
            "generate_pair_briefs_batched",
            "llm_grouped_decision_probs",
            "llm_yesno_probs_batched",
        ):
            monkeypatch.setattr(scorer, method, unexpected)
        result = scorer(**inputs)
        scorer.generate_llm_rationales = True
        assert scorer.generate_final_rationales_for_records([{}]) == [""]
        expected = [
            (float(before) + state["decision_probs"]["s"][target]) * 0.5
            for before, target in zip(raw["S_base"], ["a", "t"])
        ]
        assert result["S_final"].tolist() == pytest.approx(expected)
        assert result["oracle_diagnostic"]["llm_invocations"] == 0
        assert result["oracle_diagnostic"]["deployable"] is False
        corrupted = json.loads(json.dumps(state))
        corrupted["population_rows"][0]["S_base"] += 0.1
        path.write_text(json.dumps(corrupted))
        other = _scorer(
            llm_experiment_config={
                "enabled": True,
                "gate": {"mode": state["mode"], "artifact": str(path)},
                "fusion_weight": "constant",
            }
        )
        other.attach_dataset(dataset)
        with pytest.raises(ValueError, match="frozen response"):
            other(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome_semantics", None),
        ("protocol", "observed_cached"),
        ("protocol", "benchmark_reference_perfect_fixed_sources"),
        ("no_training_use", None),
        ("no_training_use", False),
        ("training_use_permitted", True),
        ("global_f1_optimality_claim", True),
        ("ignored_reference_pairs", [["s", "a"]]),
        ("ignored_pair_policy", None),
        ("teacher_binding", None),
        ("teacher_binding", {"reference_role": "test"}),
        ("teacher_binding", {"reference_role": "final"}),
        ("teacher_binding", {"reference_role": "train"}),
    ],
)
def test_benchmark_runtime_rejects_missing_or_incompatible_diagnostic_contract(
    tmp_path, monkeypatch, field, value
):
    from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer

    frame = pd.DataFrame({"Src": ["s"], "Tgt": ["a"], "S_base": [0.8], "U": [1.0]})
    state = observed_response_oracle(
        frame,
        [{"source": "s", "valid": True, "pair_probabilities": {"a": 0.0}}],
        set(),
        budget=1,
        negative_label_policy="unknown",
        outcome_semantics="benchmark_reference",
        teacher_binding={"reference_role": "valid"},
    )
    if value is None:
        state.pop(field, None)
    else:
        state[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(state))

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid oracle artifact attempted a hosted call")

    for method in (
        "generate_pair_briefs_batched",
        "llm_grouped_decision_probs",
        "llm_yesno_probs_batched",
    ):
        monkeypatch.setattr(PairAdaptiveSemanticScorer, method, unexpected)
    with pytest.raises(ValueError, match="[Oo]racle"):
        scorer = _scorer(
            llm_experiment_config={
                "enabled": True,
                "gate": {"mode": "oracle_replay", "artifact": str(path)},
            }
        )
        scorer.attach_dataset(_TinyDataset())
