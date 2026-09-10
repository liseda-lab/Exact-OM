import json

import pandas as pd
import pytest

from exact.impl.models.selector.oracle_replay import (
    observed_response_oracle,
    perfect_intervention,
)
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset


def test_cached_observed_and_fixed_perfect_replay_never_call_hosted_roles(tmp_path, monkeypatch):
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
    observed = observed_response_oracle(
        population,
        [{"source": "s", "valid": True, "pair_probabilities": {"a": 0.1, "t": 0.9}, "choice": "t"}],
        {("s", "t")},
        budget=1,
        negative_label_policy="complete_reference",
        fusion_weight="constant",
        constant_weight=0.5,
        teacher_binding={"dataset_signature": dataset.dataset_signature},
    )
    perfect = perfect_intervention(
        population,
        {("s", "t")},
        observed["selected_sources"],
        displayed_candidates=observed["decision_probs"],
        negative_label_policy="complete_reference",
        fusion_weight="constant",
        constant_weight=0.5,
        teacher_binding={"dataset_signature": dataset.dataset_signature},
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
