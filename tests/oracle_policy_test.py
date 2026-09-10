import json
from pathlib import Path

import pytest

from exact.experiments.oracle_policy import build_oracle_artifacts
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset


def forced_trace(tmp_path):
    sources = [source for source in ("good", "harm", "nil") for _ in range(2)]
    targets = ["a", "t"] * 3
    inputs = {
        "src_iris": sources,
        "tgt_iris": targets,
        "src_label_lists": [[source] for source in sources],
        "tgt_label_lists": [[target] for target in targets],
    }
    model = _scorer()
    model.attach_dataset(_TinyDataset())
    base = model(**inputs)
    rows = []
    for offset, source in enumerate(("good", "harm", "nil")):
        probabilities = {"a": 0.1, "t": 0.1 if source == "nil" else 0.9}
        judgment = {
            "source": source,
            "valid": True,
            "pair_probabilities": probabilities,
            "choice": "__NONE__" if source == "nil" else "t",
        }
        candidates = [
            {
                "target": target,
                "S_base": float(base["S_base"][offset * 2 + index]),
                "U": float(base["U"][offset * 2 + index]),
                "p_llm": probabilities[target],
                "llm_gate_invoked": True,
                "llm_grouped_decision": json.dumps(judgment),
            }
            for index, target in enumerate(("a", "t"))
        ]
        rows.append({"Src": source, "candidates": candidates})
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe_status": "declared",
        "dataset_signature": "tiny-signature",
        "source_universe": ["good", "harm", "nil"],
        "records": rows,
        "policy": {
            "threshold": 0.5,
            "llm": {
                "beta": 0.8,
                "pair_threshold": 0.5,
                "gate": {"mode": "forced_sample"},
                "forced_sample": {
                    "selected_sources": ["good", "harm", "nil"],
                    "population_fingerprint": "frozen-fixture",
                },
            },
        },
    }
    path = tmp_path / "source_decisions.json"
    path.write_text(json.dumps(trace))
    return path, inputs


def test_cached_producer_freezes_trust_sources_but_budget_selects_observed_oracle(
    tmp_path, monkeypatch
):
    trace, inputs = forced_trace(tmp_path)
    refs = [{"SrcEntity": "good", "TgtEntity": "t"}, {"SrcEntity": "harm", "TgtEntity": "a"}]
    result = build_oracle_artifacts(
        trace,
        refs,
        tmp_path / "policies",
        reference_role="development",
        negative_label_policy="complete_reference",
        budget=2,
        nil_sources=["nil"],
    )
    assert set(result["artifacts"]) == {
        "trust_shipped",
        "trust_constant",
        "trust_source",
        "oracle_observed",
        "oracle_perfect",
    }
    states = {
        name: json.loads(Path(path).read_text()) for name, path in result["artifacts"].items()
    }
    for name in ("trust_shipped", "trust_constant", "trust_source"):
        assert states[name]["selected_sources"] == ["good", "harm", "nil"]
    assert states["oracle_observed"]["selected_sources"] == ["good", "nil"]
    assert states["oracle_perfect"]["selected_sources"] == ["good", "nil"]
    assert (
        states["oracle_observed"]["decision_probs"].keys()
        == states["oracle_perfect"]["decision_probs"].keys()
    )
    for name, state in states.items():
        scorer = _scorer(
            llm_experiment_config={
                "enabled": True,
                "gate": {"mode": state["mode"], "artifact": result["artifacts"][name]},
                "fusion_weight": state["fixed_fusion"]["fusion_weight"],
                "constant_weight": 0.5,
                "decision": {"mode": "listwise" if name == "trust_source" else "binary"},
            }
        )
        scorer.attach_dataset(_TinyDataset())
        scorer.use_llm = True

        def forbidden(*args, **kwargs):
            raise AssertionError("Cached policy attempted a hosted call")

        for method in (
            "llm_grouped_decision_probs",
            "llm_yesno_probs_batched",
            "generate_pair_briefs_batched",
        ):
            monkeypatch.setattr(scorer, method, forbidden)
        output = scorer(**inputs)
        assert output["oracle_diagnostic"]["llm_invocations"] == 0
        if name == "trust_constant":
            assert output["S_final"].tolist() == pytest.approx([0.3, 0.7, 0.3, 0.7, 0.3, 0.3])
    assert result == build_oracle_artifacts(
        trace,
        refs,
        tmp_path / "policies",
        reference_role="development",
        negative_label_policy="complete_reference",
        budget=2,
        nil_sources=["nil"],
    )


def test_unknown_outcomes_allow_fixed_trust_but_never_outcome_selected_oracle(tmp_path):
    trace, _ = forced_trace(tmp_path)
    result = build_oracle_artifacts(
        trace,
        [],
        tmp_path / "policies",
        reference_role="development",
        negative_label_policy="unknown",
    )
    assert set(result["artifacts"]) == {"trust_shipped", "trust_constant", "trust_source"}
    assert set(result["unavailable"]) == {"oracle_observed", "oracle_perfect"}
    with pytest.raises(ValueError, match="development/diagnostic"):
        build_oracle_artifacts(
            trace, [], tmp_path, reference_role="test", negative_label_policy="unknown"
        )


def test_binary_cache_does_not_invent_a_comparative_choice(tmp_path):
    path, _ = forced_trace(tmp_path)
    trace = json.loads(path.read_text())
    for source in trace["records"]:
        for candidate in source["candidates"]:
            candidate.pop("llm_grouped_decision")
    path.write_text(json.dumps(trace))
    result = build_oracle_artifacts(
        path,
        [],
        tmp_path / "policies",
        reference_role="diagnostic",
        negative_label_policy="unknown",
    )
    assert set(result["artifacts"]) == {"trust_shipped", "trust_constant"}
    assert "comparative choices" in result["unavailable"]["trust_source"]
