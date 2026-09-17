"""Replay numerical and final source decisions using only serialized tiny fixtures."""

import json
import math

import pandas as pd
import pytest
import torch

from exact.core.entities.configs.dataset import DatasetMask
from exact.core.entities.mappings import EntityMapping
from exact.impl.trainer import SemanticAlignmentRunner
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset
from tests.trainer_run_layout_test import _Dataset, _Model


@pytest.mark.parametrize(
    "decision,weight,gate",
    [
        ("binary", "constant", "analytic"),
        ("binary", "beta_u", "analytic"),
        ("listwise", "constant", "analytic"),
        ("listwise", "source_first", "analytic"),
        ("binary", "constant", "off"),
    ],
)
def test_pair_explanations_reconstruct_every_llm_mixture_without_encoder(
    monkeypatch, tmp_path, decision, weight, gate
):
    scorer = _scorer(
        return_explanations=True,
        llm_experiment_config={
            "enabled": True,
            "gate": {"mode": gate, "threshold": 0.0},
            "fusion_weight": weight,
            "constant_weight": 0.4,
            "decision": {"mode": decision, "evidence": "structured_packet"},
        },
    )
    scorer.use_lexical = scorer.use_llm = True
    scorer.attach_dataset(_TinyDataset())
    monkeypatch.setattr(
        scorer,
        "_score_label_channel",
        lambda *args: (torch.tensor([0.8]), torch.tensor([0.7]), [("s", "t")], [{}]),
    )
    calls = []

    def binary(*args):
        calls.append("binary")
        return torch.tensor([0.1])

    def grouped(*args):
        calls.append("listwise")
        return (
            torch.tensor([0.1]),
            torch.tensor([True]),
            [{"source": "s", "choice": "__NONE__", "valid": True}],
        )

    monkeypatch.setattr(scorer, "llm_yesno_probs_batched", binary)
    monkeypatch.setattr(scorer, "llm_grouped_decision_probs", grouped)
    result = scorer(
        src_iris=["s"], tgt_iris=["t"], src_label_lists=[["s"]], tgt_label_lists=[["t"]]
    )
    record = json.loads(json.dumps(result["explanations"][0]))
    replay = record["reconstruction"]
    restored = replay["baseline"] + sum(
        record["contributions"][key] for key in replay["component_names"]
    )
    assert restored == pytest.approx(float(result["S_final"][0]), abs=1e-6)
    assert calls == ([] if gate == "off" else [decision])
    # Downstream rank/accept probabilities must not replace the pair confidence
    # against which additive pair explanations are checked.
    record["confidences"]["S_final"] = 0.02
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(), model=_Model, device=torch.device("cpu"), output_dir=tmp_path
    )
    runner.results_json.append(record)
    stats = runner._compute_run_stats(runner._make_summary_dataframe([record]))
    assert stats["explanation_reconstruction"]["failed_rows"] == 0


@pytest.mark.parametrize(
    "exact_scores", [{"Scores": 1.0}, {"Score": 1.0}, {"Scores": 1.0, "Score": 0.1}]
)
@pytest.mark.parametrize("exact_already_present", [False, True])
def test_source_trace_is_after_target_cardinality_and_keeps_empty_and_exact_sources(
    tmp_path, exact_scores, exact_already_present
):
    dataset = _Dataset()
    dataset.eligible_source_iris = ["empty", "exact", "s1", "s2"]
    dataset.dataframe = pd.DataFrame(
        {
            "Src": ["exact", "s1", "s2"],
            "Tgt": ["t0", "t1", "t1"],
            **{column: [score, 0.9, 0.8] for column, score in exact_scores.items()},
            DatasetMask.prefiltered: [True, False, False],
        }
    )
    runner = SemanticAlignmentRunner(
        dataset=dataset, model=_Model, device=torch.device("cpu"), output_dir=tmp_path
    )
    runner._final_candidate_frame = pd.DataFrame(
        {
            "Src": ["s1", "s2"],
            "Tgt": ["t1", "t1"],
            "S_final": [0.9, 0.8],
        }
    )
    if exact_already_present:
        runner._final_candidate_frame.loc[len(runner._final_candidate_frame)] = ["exact", "t0", 1.0]
    runner._decision_policy = {
        "threshold": 0.7,
        "source_cardinality": 1,
        "target_cardinality": 1,
        "extraction": {"mode": "greedy"},
    }
    predictions = runner.apply_prefilter(
        [EntityMapping("s1", "t1", score=0.9), EntityMapping("s2", "t1", score=0.8)],
        threshold=0.7,
        cardinality=1,
        target_cardinality=1,
    )
    paths = runner.save_results(predictions, output_formats=["tsv-global"])
    trace = json.loads(paths["source_decisions_json"].read_text())
    rows = {row["Src"]: row for row in trace["records"]}
    assert trace["source_universe"] == dataset.eligible_source_iris
    assert trace["reference_labels_used"] is False
    assert rows["empty"]["empty_candidate_pool"]
    exact = rows["exact"]["candidates"]
    assert len(exact) == 1
    assert exact[0]["protected_exact"]
    assert exact[0]["threshold_positive"]
    assert exact[0]["S_final"] == 1.0
    assert exact[0]["emitted"]
    assert next(item.score for item in predictions if item.head == "exact") == 1.0
    assert rows["s1"]["emitted_targets"] == ["t1"]
    assert rows["s2"]["emitted_targets"] == []
    assert rows["s2"]["candidates"][0]["reason"] == "cardinality_or_extraction"
    assert rows["s2"]["competing_sources_by_target"] == {"t1": ["s1"]}


@pytest.mark.parametrize("alias,column", [("src_iri", "Src"), ("tgt_iri", "Tgt")])
@pytest.mark.parametrize("canonical_value", ["same", None])
def test_source_trace_coalesces_consistent_identity_aliases(
    tmp_path, alias, column, canonical_value
):
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(), model=_Model, device=torch.device("cpu"), output_dir=tmp_path
    )
    frame = pd.DataFrame([{"Src": "source", "Tgt": "target", "S_final": 0.9}])
    frame[alias] = frame[column]
    if canonical_value is None:
        frame[column] = None
    runner._final_candidate_frame = frame
    original = frame.copy(deep=True)
    paths = runner.save_results(
        [EntityMapping("source", "target", score=0.9)], output_formats=["tsv-global"]
    )
    record = json.loads(paths["source_decisions_json"].read_text())["records"][0]
    assert record["Src"] == "source"
    assert record["emitted_targets"] == ["target"]
    assert record["candidates"][0]["S_final"] == 0.9
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("alias,column", [("src_iri", "Src"), ("tgt_iri", "Tgt")])
def test_source_trace_rejects_conflicting_identity_aliases(tmp_path, alias, column):
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(), model=_Model, device=torch.device("cpu"), output_dir=tmp_path
    )
    runner._final_candidate_frame = pd.DataFrame(
        [{"Src": "source", "Tgt": "target", alias: "different", "S_final": 0.9}]
    )
    with pytest.raises(
        ValueError, match=f"Conflicting source-decision columns: {column} and {alias}"
    ):
        runner._write_source_decisions([], pd.DataFrame())
    assert not (tmp_path / "source_decisions.json").exists()


@pytest.mark.parametrize("quality", ["candidate_margin", "entropy"])
def test_lexical_controls_replay_from_serialized_distinct_candidates_and_labels(
    monkeypatch, quality
):
    scorer = _scorer(
        return_explanations=True,
        lex={"enabled": True, "quality": quality, "deduplicate_labels": True},
    )
    scorer.use_lexical = True
    data = _TinyDataset()
    data.dataframe = pd.DataFrame({"Src": ["s", "s"], "Tgt": ["t", "tt"]})
    scorer.attach_dataset(data)
    vectors = {"source": [1.0, 0.0], "same": [1.0, 0.0], "close": [0.8, 0.6]}
    monkeypatch.setattr(
        scorer,
        "encode_labels_batch",
        lambda labels: torch.tensor([vectors[label] for label in labels]),
    )
    output = scorer(
        src_iris=["s", "s"],
        tgt_iris=["t", "tt"],
        src_label_lists=[["source", "source"]] * 2,
        tgt_label_lists=[["same", "same", "close"], ["close", "close"]],
    )
    rows = json.loads(json.dumps(output["explanations"]))
    for row in rows:
        diagnostic = row["experiment_diagnostics"]["quality_components"]["label"]
        assert diagnostic["raw_source_label_count"] == 2
        assert diagnostic["source_label_count"] == 1
        if quality == "candidate_margin":
            ordered = sorted(diagnostic["candidate_scores"].values(), reverse=True)
            expected = ordered[0] - ordered[1]
        elif diagnostic["entropy_quality_defined"]:
            values = torch.tensor(diagnostic["top_m_similarities"])
            probabilities = torch.softmax(values / diagnostic["entropy_temperature"], dim=0)
            expected = 1 + float((probabilities * probabilities.log()).sum()) / math.log(
                len(values)
            )
        else:
            expected = 0.0
        assert diagnostic["selected"] == pytest.approx(expected, abs=1e-6)
        replay = row["reconstruction"]
        assert replay["baseline"] + sum(
            row["contributions"][key] for key in replay["component_names"]
        ) == pytest.approx(replay["score"], abs=1e-6)
