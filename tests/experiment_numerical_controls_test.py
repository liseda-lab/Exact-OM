from __future__ import annotations

import json
from itertools import permutations

import pytest
import torch

from exact.core.entities.mappings.entity import EntityMapping
from exact.impl.extraction import extract_global_alignment
from exact.impl.models.selector.llm_gate import select_inference_gate_artifact
from tests.pair_adaptive_experiments_test import _TinyDataset, _scorer


def test_accepted_assignment_retains_strong_edge_hidden_by_legacy_objective():
    edges = [
        EntityMapping("s1", "t1", score=0.90),
        EntityMapping("s1", "t2", score=0.69),
        EntityMapping("s2", "t1", score=0.69),
    ]
    for order in permutations(edges):
        result = extract_global_alignment(order, mode="assignment_accepted_utility", threshold=0.70)
        assert [(edge.head, edge.tail) for edge in result.mappings] == [("s1", "t1")]
        assert result.diagnostics["threshold_removed"] == 2
        assert result.diagnostics["unmatched_sources"] == 1
    assert extract_global_alignment(edges, mode="assignment_legacy", threshold=0.70).mappings == []


def test_assignment_threshold_is_unmatched_utility_and_cap_fallback_is_greedy():
    # Raw sums prefer two weak edges; accepted utility prefers the strong edge.
    edges = [
        EntityMapping("s1", "t1", score=0.95),
        EntityMapping("s1", "t2", score=0.72),
        EntityMapping("s2", "t1", score=0.72),
    ]
    result = extract_global_alignment(edges, mode="assignment", threshold=0.7)
    assert [(edge.head, edge.tail) for edge in result.mappings] == [("s1", "t1")]
    capped = extract_global_alignment(
        edges, mode="assignment", threshold=0.7, assignment_component_cap=1
    )
    assert capped.mappings == extract_global_alignment(edges, mode="greedy", threshold=0.7).mappings
    assert capped.diagnostics["assignment_fallback_components"] == 1
    with pytest.raises(ValueError, match="one-to-one"):
        extract_global_alignment(edges, mode="assignment", source_cardinality=2)


@pytest.mark.parametrize("enabled", [False, True])
def test_off_control_disables_forced_briefs_decisions_and_final_rationales(monkeypatch, enabled):
    scorer = _scorer(
        force_llm_summaries=True,
        generate_llm_rationales=True,
        llm_experiment_config={"enabled": enabled, "gate": {"mode": "off"}},
    )
    scorer.use_llm = True

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled role invoked")

    for method in (
        "generate_pair_briefs_batched",
        "llm_yesno_probs_batched",
        "generate_rationales_batched",
    ):
        monkeypatch.setattr(scorer, method, forbidden)
    scorer.attach_dataset(_TinyDataset())
    result = scorer(
        src_iris=["s"], tgt_iris=["t"], src_label_lists=[["s"]], tgt_label_lists=[["t"]]
    )
    assert torch.equal(result["S_final"], result["S_base"])
    assert result["batch_pair_adaptive_stats"]["brief_requested_pairs"] == 0
    assert result["batch_pair_adaptive_stats"]["decision_requested_pairs"] == 0
    assert scorer.generate_final_rationales_for_records([{}]) == [""]


def test_analytic_fitted_neutral_parameters_replay_nested_scores_and_explanations(
    tmp_path, monkeypatch
):
    path = tmp_path / "fusion.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "analytic_fitted",
                "dataset_signature": "tiny-signature",
                "candidate_pool_fingerprint": "tiny-pool",
                "dataset_lock_sha256": "tiny-lock",
                "seed": 17,
                "feature_schema": ["score", "quality", "active"],
                "negative_label_policy": "complete_reference",
                "parameters": {
                    "tau": 0.5,
                    "gamma": 2.0,
                    "multipliers": {"label": 1.0, "sim_obj": 1.0, "diff": 1.0, "attr_aux": 1.0},
                },
            }
        )
    )
    scorers = [
        _scorer(return_explanations=True),
        _scorer(
            return_explanations=True,
            fusion_config={"enabled": True, "mode": "analytic_fitted", "artifact": path},
        ),
    ]
    results = []
    for scorer in scorers:
        scorer.use_lexical = True
        scorer.attach_dataset(_TinyDataset())
        monkeypatch.setattr(
            scorer,
            "_score_label_channel",
            lambda *args: (torch.tensor([0.86]), torch.tensor([0.7]), [("s", "t")], [{}]),
        )
        for method, score, quality in [
            ("_score_similarity_channel", 0.68, 0.9),
            ("_score_difference_channel", 0.12, 0.4),
        ]:
            original = getattr(scorer, method)

            def score_channel(*args, original=original, score=score, quality=quality, **kwargs):
                payload = original(*args, **kwargs)
                payload.update(
                    score=score, quality=quality, src_selected=[{"triple": ["s", "r", "v"]}]
                )
                return payload

            monkeypatch.setattr(scorer, method, score_channel)
        results.append(
            scorer(src_iris=["s"], tgt_iris=["t"], src_label_lists=[["s"]], tgt_label_lists=[["t"]])
        )
    for field in ("S_base", "S_final", "w_struct", "I_label", "I_struct", "Q_struct"):
        torch.testing.assert_close(results[0][field], results[1][field], rtol=0, atol=0)
    assert (
        results[0]["explanations"][0]["contributions"]
        == results[1]["explanations"][0]["contributions"]
    )


def test_missingness_has_no_contradiction_and_explicit_rules_survive_perturbations():
    scorer = _scorer(
        diff={
            "enabled": True,
            "formulation": "missingness_aware",
            "incompatibilities": [
                {
                    "property_iri": "urn:role",
                    "source_object": "urn:left",
                    "target_object": "urn:right",
                    "semantic_rule": "exclusive_values",
                    "evidence_id": "ontology:axiom-1",
                }
            ],
        }
    )
    scorer.use_context = True
    left = {
        "triple": ["s", "role", "left"],
        "rel_iri": "urn:role",
        "object_iri": "urn:left",
        "score": 0.8,
    }
    right = {
        "triple": ["t", "role", "right"],
        "rel_iri": "urn:role",
        "object_iri": "urn:right",
        "score": 0.8,
    }
    irrelevant = {"triple": ["t", "unrelated", "value"], "score": 0.8}
    for source, target in [([], []), ([left], []), ([], [right]), ([left], [irrelevant])]:
        result = scorer._score_difference_channel(source, target)
        assert (result["score"], result["quality"]) == (0.5, 0.0)
        assert all(row["state"] == "unobserved" for row in result["evidence_states"])
    baseline = scorer._score_difference_channel([left], [right])
    assert (baseline["score"], baseline["quality"]) == (0.0, 0.8)
    for source, target in [
        ([left, left], [right]),
        ([left], [right, irrelevant]),
        ([right], [left]),
    ]:
        result = scorer._score_difference_channel(source, target)
        assert (result["score"], result["quality"]) == (baseline["score"], baseline["quality"])
        assert (
            len([row for row in result["evidence_states"] if row["state"] == "contradicted"]) == 2
        )
    assert (
        baseline["src_selected"][0]["contradiction_semantics"][0]["evidence_id"]
        == "ontology:axiom-1"
    )


def test_source_fraction_replays_current_population_across_pair_batches(tmp_path):
    rows = [
        {"source_iri": source, "target_iri": target, "score": score}
        for source, target, score in [
            ("s1", "t1", 0.9),
            ("s1", "t2", 0.8),
            ("s2", "t1", 0.7),
            ("s3", "t3", 0.7),
        ]
    ]
    artifact = select_inference_gate_artifact(
        rows,
        mode="source_top_fraction",
        fraction=0.5,
        dataset_signature="tiny-signature",
        source_universe=["empty"],
    )
    assert artifact["selected_sources"] == ["s1", "s2"]
    assert artifact["no_candidate_sources"] == ["empty"]
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(artifact))
    scorer = _scorer(
        llm_experiment_config={
            "enabled": True,
            "gate": {"mode": "source_top_fraction", "quantile_fraction": 0.5, "artifact": path},
        }
    )

    def route(row):
        tensor = torch.tensor([0.5])
        return scorer._llm_gate_mask(
            U_ind=tensor,
            U_dis=tensor,
            U=tensor,
            S_base=tensor,
            q_label=tensor,
            Q_struct=tensor,
            src_iris=[row["source_iri"]],
            tgt_iris=[row["target_iri"]],
            label=None,
        )[0].item()

    assert [route(row) for row in reversed(rows)] == [False, True, True, True]
    with pytest.raises(ValueError, match="population differs"):
        route({"source_iri": "development-source", "target_iri": "dev-target"})


def test_duplicate_label_control_records_counts_and_preserves_selected_quality(monkeypatch):
    scorer = _scorer(lex={"enabled": True, "deduplicate_labels": True})
    scorer.use_lexical = True
    monkeypatch.setattr(
        scorer, "encode_labels_batch", lambda labels: torch.tensor([[1.0, 0.0] for _ in labels])
    )
    singleton = scorer._score_label_channel([["Heart"]], [["heart"]])
    duplicate = scorer._score_label_channel([["Heart", "heart", " Heart "]], [["heart"]])
    torch.testing.assert_close(singleton[0], duplicate[0])
    torch.testing.assert_close(singleton[1], duplicate[1])
    assert duplicate[3][0]["raw_source_label_count"] == 3
    assert duplicate[3][0]["source_label_count"] == 1
    assert duplicate[3][0]["top_m_similarities"] == [1.0]


def test_structured_packet_excludes_numeric_authority_and_marks_missingness():
    scorer = _scorer(
        llm_experiment_config={"enabled": True, "decision": {"evidence": "structured_packet"}}
    )
    fact = {
        "triple": ["source", "relation", "target"],
        "item_id": "fact:1",
        "score": 0.123,
        "quality": 0.321,
    }
    packet = scorer._build_evidence_packet(
        "s", "t", {}, {"src_selected": [fact]}, {"src_selected": [fact]}, {}
    )
    payload = json.loads(packet)
    assert len(payload["facts"]) == 1
    assert payload["facts"][0]["item_id"] == "fact:1"
    assert "score" not in packet and "quality" not in packet
    assert "unobserved" in payload["unknowns"]
