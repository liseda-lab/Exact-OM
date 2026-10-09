from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from exact.core.entities.kinds import EntityKind
from exact.impl.models.pair_adaptive_experiments import (
    JsonExperimentArtifact,
    abbreviation_similarity,
    isub_similarity,
    jaro_winkler_similarity,
    token_set_similarity,
)
from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer


def _scorer(**kwargs) -> PairAdaptiveSemanticScorer:
    return PairAdaptiveSemanticScorer(
        use_lexical=False,
        use_context=False,
        use_llm=False,
        llm_model_name=None,
        persist_cache_to_disk=False,
        device="cpu",
        **kwargs,
    )


class _TinySource:
    def __init__(self, parents: dict[str, list[str]], entities: list[str]) -> None:
        self.parents = parents
        self._entities = entities

    def direct_parents(self, iri: str, kind: EntityKind) -> list[str]:
        return list(self.parents.get(iri, []))

    def direct_children(self, iri: str, kind: EntityKind) -> list[str]:
        return [child for child, parents in self.parents.items() if iri in parents]

    def entities(self, kind: EntityKind) -> list[str]:
        return list(self._entities)


class _TinyDataset:
    dataset_signature = "tiny-signature"
    hierarchical_relation_families: dict[str, dict] = {}

    def __init__(self) -> None:
        self.source = _TinySource({"s": ["sa"], "ss": ["sa"]}, ["s", "ss", "sa"])
        self.target = _TinySource({"t": ["ta"], "tt": ["ta"]}, ["t", "tt", "ta"])
        self.exact_matches = pd.DataFrame({"Src": ["sa", "ss"], "Tgt": ["ta", "tt"]})

    def entity_kind_for(self, iri: str, side: str, warn_unknown: bool = False) -> EntityKind:
        return EntityKind.CLASS

    def has_entity_features_cached(self, iri: str, side: str) -> bool:
        return False

    def get_entity_features(self, iri: str, side: str) -> dict:
        label = "chronic obstructive pulmonary disease" if side == "src" else "copd"
        return {
            "labels": [label],
            "hierarchy": {},
            "object_triples": [],
            "attributes": [],
        }


def test_binary_exemplar_inference_retains_frozen_candidate_cap(monkeypatch):
    scorer = _scorer()
    scorer.attach_dataset(_TinyDataset())
    scorer.use_llm = True
    scorer.llm_experiment_enabled = True
    scorer.llm_experiment_config["exemplars"] = "knn"
    scorer.llm_experiment_config["decision"]["evidence"] = "structured_packet"
    scorer.llm_experiment_config["gate"] = {"mode": "analytic", "threshold": 0.0}
    calls = []

    def binary(sources, targets, *args):
        calls.append((sources, targets))
        return torch.full((len(targets),), 0.9), [{"source": "s", "exemplar_sources": ["train"]}]

    monkeypatch.setattr(scorer, "llm_binary_decision_probs", binary)
    monkeypatch.setattr(
        scorer, "llm_yesno_probs_batched", lambda *args: pytest.fail("Exemplar path was bypassed")
    )
    result = scorer(
        src_iris=["s"] * 6,
        tgt_iris=["t5", "t4", "t3", "t2", "t1", "t0"],
        src_label_lists=[["source"]] * 6,
        tgt_label_lists=[["target"]] * 6,
    )
    assert calls == [(["s"] * 5, ["t4", "t3", "t2", "t1", "t0"])]
    assert result["S_final"][0] == result["S_base"][0]
    assert result["llm_grouped_decisions"][0]["exemplar_sources"] == ["train"]


def test_string_metrics_and_abbreviation_are_bounded_and_conservative() -> None:
    assert isub_similarity("Heart Valve", "heart valve") == pytest.approx(1.0)
    assert jaro_winkler_similarity("martha", "marhta") == pytest.approx(0.961, abs=0.002)
    assert token_set_similarity("alpha beta", "alpha beta gamma delta") < 1.0
    assert abbreviation_similarity("COPD", "chronic obstructive pulmonary disease") == 1.0
    assert abbreviation_similarity("AB", "unrelated concept") == 0.0


def test_string_channel_runs_without_encoder_and_preserves_subsignal_provenance() -> None:
    scorer = _scorer(strsim={"enabled": True, "placement": "channel", "abbreviation": "initialism"})
    scores, qualities, payloads = scorer._score_string_channel(
        [["chronic obstructive pulmonary disease"]], [["COPD"]]
    )
    assert scores.tolist() == pytest.approx([1.0])
    assert qualities.tolist() == pytest.approx([1.0])
    assert payloads[0]["winner"] == "abbreviation"
    assert payloads[0]["components"]["abbreviation"] == pytest.approx(1.0)


def test_lexical_quality_variants_are_explicit_and_non_degenerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = {
        "exact": [1.0, 0.0],
        "alternative": [0.0, 1.0],
        "target": [1.0, 0.0],
    }

    entropy = _scorer(lex={"enabled": True, "quality": "entropy", "entropy_top_m": 5})
    entropy.use_lexical = True
    monkeypatch.setattr(
        entropy,
        "encode_labels_batch",
        lambda labels: torch.tensor([vectors[label] for label in labels]),
    )
    _, entropy_quality, _, payloads = entropy._score_label_channel(
        [["exact", "alternative"]], [["target"]]
    )
    assert 0.0 < entropy_quality.item() < 1.0
    assert payloads[0]["mode"] == "entropy"
    assert payloads[0]["margin"] > payloads[0]["entropy"]
    _, singleton_quality, _, singleton_payloads = entropy._score_label_channel(
        [["exact"]], [["target"]]
    )
    assert singleton_quality.item() == pytest.approx(0.0)
    assert singleton_payloads[0]["entropy_raw"] == pytest.approx(0.0)
    assert singleton_payloads[0]["entropy_quality_defined"] is False

    constant = _scorer(lex={"enabled": True, "quality": "constant"})
    constant.use_lexical = True
    monkeypatch.setattr(
        constant,
        "encode_labels_batch",
        lambda labels: torch.tensor([vectors[label] for label in labels]),
    )
    _, constant_quality, _, _ = constant._score_label_channel(
        [["exact", "alternative"]], [["target"]]
    )
    assert constant_quality.item() == pytest.approx(1.0)

    agreement = _scorer(lex={"enabled": True, "quality": "encoder_agreement"})
    agreement.use_lexical = True
    monkeypatch.setattr(
        agreement,
        "encode_labels_batch",
        lambda labels: torch.tensor([vectors[label] for label in labels]),
    )
    with pytest.raises(ValueError, match="requires the context encoder"):
        agreement._score_label_channel([["exact"]], [["target"]])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("full", 0.25 * 0.3**2),
        ("constant_q", 0.3**2),
        ("no_sharpening", 0.25),
        ("suppression_only", 1.0),
        ("uniform", 1.0),
    ],
)
def test_sigma_decomposition_modes_retain_inactivity_suppression(
    mode: str, expected: float
) -> None:
    scorer = _scorer(
        fusion_config={
            "enabled": True,
            "mode": "analytic_shipped",
            "sigma_mode": mode,
            "tau": 0.5,
            "gamma": 2.0,
            "beta": 0.8,
        }
    )
    authority = scorer._sigma_authority(
        torch.tensor([0.8]),
        torch.tensor([0.25]),
        torch.tensor([True]),
        channel="label",
    )
    inactive = scorer._sigma_authority(
        torch.tensor([0.8]),
        torch.tensor([0.25]),
        torch.tensor([False]),
        channel="label",
    )
    assert authority.item() == pytest.approx(expected)
    if mode == "uniform":
        assert inactive.item() == pytest.approx(1.0)
    elif mode != "full":
        assert inactive.item() == pytest.approx(0.0)


def test_fusion_llm_pivot_maps_to_scorer_tau_llm() -> None:
    scorer = _scorer(
        fusion_config={
            "enabled": True,
            "mode": "analytic_shipped",
            "llm_pivot": 0.35,
        }
    )
    assert scorer.tau_LLM == pytest.approx(0.35)


def test_difference_formulations_and_empty_pool_diagnostics() -> None:
    src = [
        {"triple": ("s", "r1", "a"), "score": 1.0},
        {"triple": ("s", "r2", "b"), "score": 1.0},
    ]
    tgt = [{"triple": ("t", "r1", "a"), "score": 1.0}]
    support = torch.tensor([[1.0], [0.0]])

    normalised = _scorer(
        diff={"enabled": True, "formulation": "normalised", "dump_components": True}
    )
    normalised.use_context = True
    normalised_payload = normalised._score_difference_channel(src, tgt, support)
    assert normalised_payload["score"] == pytest.approx(0.75)
    assert normalised_payload["unsupported_mass_src"] == pytest.approx(1.0)
    assert normalised_payload["c_x"] == pytest.approx(0.5)

    absolute = _scorer(diff={"enabled": True, "formulation": "absolute"})
    absolute.use_context = True
    absolute_payload = absolute._score_difference_channel(src, tgt, support)
    assert absolute_payload["diff_absolute"] == pytest.approx(0.25)
    assert absolute_payload["score"] == pytest.approx(0.75)

    asymmetric = _scorer(
        diff={"enabled": True, "formulation": "asymmetric", "relation_interpretation": "<"}
    )
    asymmetric.use_context = True
    assert asymmetric._score_difference_channel(src, tgt, support)["score"] == pytest.approx(1.0)

    empty = absolute._score_difference_channel([], tgt, torch.zeros((0, 1)))
    assert empty["score"] == pytest.approx(0.5)
    assert empty["diff_pivot_reason"] == "empty_source"
    both_empty = absolute._score_difference_channel([], [], torch.zeros((0, 0)))
    assert both_empty["quality"] == 0.0
    assert both_empty["diff_pivot_reason"] == "empty_both"

    off = _scorer(diff={"enabled": True, "formulation": "off"})
    off.use_context = True
    assert off._score_difference_channel(src, tgt, support)["diff_pivot_reason"] == "channel_off"


def test_signed_identifier_attributes_can_contribute_below_pivot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _scorer(
        attr={
            "enabled": True,
            "polarity": "signed",
            "bank": "attrs_only",
            "signed_property_allowlist": ["urn:test:identifier"],
        }
    )
    scorer.use_context = True
    monkeypatch.setattr(
        scorer,
        "_encode_context_matrix",
        lambda left, right: torch.full((len(left), len(right)), 0.9),
    )
    payload = scorer._score_attribute_channel(
        [
            {
                "prop": "identifier",
                "prop_iri": "urn:test:identifier",
                "identifier_namespace": "test",
                "identifier_exclusive": True,
                "annotation_semantics_evidence_id": "fixture:exclusive-test-id",
                "identifier_normalized": "A1",
                "value": "A-1",
                "text": "identifier A-1",
            }
        ],
        [
            {
                "prop": "code",
                "prop_iri": "urn:test:identifier",
                "identifier_namespace": "test",
                "identifier_exclusive": True,
                "annotation_semantics_evidence_id": "fixture:exclusive-test-id",
                "identifier_normalized": "B9",
                "value": "B-9",
                "text": "code B-9",
            }
        ],
        ["source"],
        ["target"],
        {},
        {},
    )
    assert payload["bank"] == "attrs_only"
    assert payload["identifier_disagreement"] == 1.0
    assert payload["identifier_comparable_groups"] == 1
    assert payload["score"] < scorer.tau


def test_hierarchy_overlap_uses_exact_anchors_and_reports_sibling_conflict() -> None:
    scorer = _scorer(
        hier={
            "enabled": True,
            "mode": "labels_overlap",
            "siblings": True,
            "depth": 2,
            "overlap_weight": 0.5,
        }
    )
    scorer.attach_dataset(_TinyDataset())
    overlap, coverage, sibling_conflict, sibling_coverage = scorer._hierarchy_anchor_terms("s", "t")
    assert overlap == pytest.approx(1.0)
    assert coverage == pytest.approx(1.0)
    assert sibling_coverage == pytest.approx(1.0)
    assert sibling_conflict == pytest.approx(0.0)


def test_gate_instrumentation_transfers_cutoff_instead_of_development_pair_ids(tmp_path) -> None:
    path = tmp_path / "quantile-gate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "quantile",
                "dataset_signature": "tiny-signature",
                "task_id": "tiny-task",
                "entity_kind": "class",
                "fraction": 0.5,
                "row_count": 2,
                "selected_count": 1,
                "threshold": 0.8,
                "pairs": [["s2", "t2"]],
                "boundary_ids": {
                    "selected_last": ["s2", "t2"],
                    "excluded_first": ["s1", "t1"],
                },
                "tie_rule": "(-U, source_iri, target_iri)",
            }
        ),
        encoding="utf-8",
    )
    scorer = _scorer(
        llm_experiment_config={
            "enabled": True,
            "gate": {
                "mode": "quantile",
                "threshold": 0.7,
                "quantile_fraction": 0.5,
                "artifact": path,
            },
        }
    )
    mask, rows = scorer._llm_gate_mask(
        U_ind=torch.tensor([0.9, 0.1]),
        U_dis=torch.tensor([0.1, 0.0]),
        U=torch.tensor([0.9, 0.1]),
        S_base=torch.tensor([0.9, 0.5]),
        q_label=torch.tensor([1.0, 0.5]),
        Q_struct=torch.tensor([0.2, 0.5]),
        src_iris=["s1", "s2"],
        tgt_iris=["t1", "t2"],
        label=None,
    )
    assert mask.tolist() == [True, False]
    assert rows[0]["threshold_source"] == "development_numeric_cutoff"
    assert rows[1]["would_route"] is False
    assert rows[1]["U_ind"] == pytest.approx(0.1)
    assert rows[1]["fitted_cutoff"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"exemplars": "knn", "exemplar_count": 2}, "immutable training artifact"),
        ({"distill": "student", "distill_artifact": "student.json"}, "immutable training artifact"),
        ({"gate": {"mode": "router"}}, "legacy 'router' linear head"),
    ],
)
def test_unavailable_llm_arms_fail_closed(config: dict, message: str) -> None:
    expected = NotImplementedError if config.get("gate", {}).get("mode") == "router" else ValueError
    with pytest.raises(expected, match=message):
        _scorer(llm_experiment_config={"enabled": True, **config})


def test_quantile_gate_requires_fitted_selection_artifact() -> None:
    with pytest.raises(ValueError, match="requires an immutable selection artifact"):
        _scorer(
            llm_experiment_config={
                "enabled": True,
                "gate": {"mode": "quantile", "quantile_fraction": 0.05},
            }
        )


def test_off_gate_routes_no_pairs() -> None:
    scorer = _scorer(llm_experiment_config={"enabled": True, "gate": {"mode": "off"}})
    mask, rows = scorer._llm_gate_mask(
        U_ind=torch.tensor([1.0, 0.0]),
        U_dis=torch.tensor([0.0, 1.0]),
        U=torch.tensor([1.0, 1.0]),
        S_base=torch.tensor([0.9, 0.1]),
        q_label=torch.tensor([1.0, 0.0]),
        Q_struct=torch.tensor([0.0, 1.0]),
        src_iris=["s1", "s2"],
        tgt_iris=["t1", "t2"],
        label=None,
    )
    assert mask.tolist() == [False, False]
    assert all(row["mode"] == "off" and row["llm_disabled"] for row in rows)


def test_oracle_is_analytical_non_deployable_and_never_calls_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = PairAdaptiveSemanticScorer(
        use_lexical=False,
        use_context=False,
        use_llm=True,
        llm_model_name="fixture-llm",
        force_llm_summaries=True,
        persist_cache_to_disk=False,
        device="cpu",
        return_explanations=True,
        strsim={"enabled": True, "placement": "channel", "abbreviation": "initialism"},
        llm_experiment_config={"enabled": True, "gate": {"mode": "oracle"}},
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("oracle must not invoke any LLM backend")

    monkeypatch.setattr(scorer, "generate_pair_briefs_batched", unexpected_call)
    monkeypatch.setattr(scorer, "llm_yesno_probs_batched", unexpected_call)
    scorer.attach_dataset(_TinyDataset())
    result = scorer(
        src_iris=["s"],
        tgt_iris=["t"],
        src_label_lists=[["chronic obstructive pulmonary disease"]],
        tgt_label_lists=[["COPD"]],
        label=[0.0],
    )
    assert result["S_base"].item() == pytest.approx(1.0)
    assert result["S_final"].item() == pytest.approx(0.0)
    assert result["need_llm"].tolist() == [False]
    assert result["batch_pair_adaptive_stats"]["llm_gated_pairs"] == 1
    assert result["batch_pair_adaptive_stats"]["decision_requested_pairs"] == 0
    assert result["oracle_diagnostic"] == {
        "oracle_only": True,
        "deployable": False,
        "llm_invocations": 0,
        "routed": [True],
        "baseline_predictions": [True],
        "labels": [False],
    }
    assert result["llm_gate_diagnostics"][0]["invoked"] is False
    assert result["llm_gate_diagnostics"][0]["oracle_adjustment"] == pytest.approx(-1.0)
    contributions = result["explanations"][0]["contributions"]
    assert contributions["C_oracle"] == pytest.approx(-1.0)
    reconstructed = sum(
        contributions[name] for name in ("C_label", "C_strsim", "C_struct", "C_llm", "C_oracle")
    )
    assert reconstructed == pytest.approx(result["S_final"].item() - scorer.tau)


def test_unimplemented_graph_arm_fails_closed_instead_of_running_baseline() -> None:
    with pytest.raises(NotImplementedError, match="refusing to execute"):
        _scorer(graph={"mode": "transductive", "artifact": "graph.json"})


def test_json_artifact_validation_is_data_only_and_dataset_locked(tmp_path) -> None:
    path = tmp_path / "fusion.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "learned_global",
                "dataset_signature": "tiny-signature",
                "weights": {"label": 1.0, "diff": 0.5},
            }
        ),
        encoding="utf-8",
    )
    artifact = JsonExperimentArtifact.load(path, expected_mode="learned_global", kind="fusion")
    artifact.validate_dataset("tiny-signature", required=True)
    assert artifact.provenance["sha256"]
    with pytest.raises(ValueError, match="dataset_signature mismatch"):
        artifact.validate_dataset("other")


def test_analytic_fitted_artifact_applies_constants_and_channel_multipliers(tmp_path) -> None:
    path = tmp_path / "analytic-fitted.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "analytic_fitted",
                "dataset_signature": "tiny-signature",
                "candidate_pool_fingerprint": "tiny-pool",
                "dataset_lock_sha256": "tiny-lock",
                "seed": 7,
                "feature_schema": ["score", "quality", "active"],
                "negative_label_policy": "complete_reference",
                "parameters": {
                    "tau": 0.4,
                    "gamma": 1.0,
                    "multipliers": {"label": 1.5, "diff": 0.5, "default": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )
    scorer = _scorer(
        fusion_config={
            "enabled": True,
            "mode": "analytic_fitted",
            "scope": "global",
            "artifact": path,
        }
    )
    scorer.attach_dataset(_TinyDataset())
    assert (scorer.tau, scorer.gamma, scorer.beta) == pytest.approx((0.4, 1.0, 0.8))
    authority = scorer._sigma_authority(
        torch.tensor([0.8]),
        torch.tensor([0.5]),
        torch.tensor([True]),
        channel="label",
    )
    assert authority.item() == pytest.approx(0.3)


def test_end_to_end_string_channel_reconstructs_explanation_contributions() -> None:
    scorer = _scorer(
        strsim={"enabled": True, "placement": "channel", "abbreviation": "initialism"},
        return_explanations=True,
    )
    scorer.attach_dataset(_TinyDataset())
    result = scorer(
        src_iris=["s"],
        tgt_iris=["t"],
        src_label_lists=[["chronic obstructive pulmonary disease"]],
        tgt_label_lists=[["COPD"]],
    )
    assert result["S_final"].item() == pytest.approx(1.0)
    explanation = result["explanations"][0]
    contributions = explanation["contributions"]
    reconstructed = (
        contributions["C_label"]
        + contributions["C_strsim"]
        + contributions["C_struct"]
        + contributions["C_llm"]
    )
    assert reconstructed == pytest.approx(result["S_final"].item() - scorer.tau)
    assert explanation["experiment_diagnostics"]["string_similarity"]["winner"] == ("abbreviation")


@pytest.mark.parametrize("n_pairs", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels", ["lexical", "structural", "both", "neither"])
def test_channel_fusion_preserves_fp32_output_with_mixed_precision_embeddings(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, channels: str, n_pairs: int
) -> None:
    scorer = _scorer(return_explanations=True)
    scorer.use_lexical = channels in {"lexical", "both"}
    scorer.use_context = channels in {"structural", "both"}
    dataset = _TinyDataset()
    features = dataset.get_entity_features

    def entity_features(iri: str, side: str) -> dict:
        result = features(iri, side)
        result["hierarchy"] = {
            "is_a": [{"triple": (iri, "subClassOf", "parent"), "specificity": 0.5}]
        }
        return result

    monkeypatch.setattr(dataset, "get_entity_features", entity_features)
    monkeypatch.setattr(
        scorer,
        "encode_labels_batch",
        lambda labels: torch.tensor(
            [[1.0, 0.0] if label == "source" else [0.6, 0.8] for label in labels],
            dtype=dtype,
        ),
    )
    monkeypatch.setattr(
        scorer,
        "_encode_label_matrix",
        lambda left, right: torch.full((len(left), len(right)), 0.875, dtype=dtype),
    )
    monkeypatch.setattr(scorer, "_context_similarity_from_sentences", lambda *args: 0.875)
    scorer.attach_dataset(dataset)
    result = scorer(
        src_iris=["s"] * n_pairs,
        tgt_iris=["t"] * n_pairs,
        src_label_lists=[["source"]] * n_pairs,
        tgt_label_lists=[["target"]] * n_pairs,
    )
    assert result["S_base"].dtype == torch.float32
    assert torch.equal(result["S_base"], result["S_final"])
    assert torch.isfinite(result["S_final"]).all()
    if channels in {"lexical", "both"}:
        assert result["s_label"].dtype == dtype
    if channels == "lexical":
        assert torch.equal(result["S_base"], result["s_label"].float())
        assert result["w_struct"][0].item() == 0.0
    elif channels == "structural":
        assert result["S_base"][0].item() == pytest.approx(0.875)
        assert result["w_struct"][0].item() == 1.0
    elif channels == "both":
        assert result["s_label"][0].item() < result["S_base"][0].item() < 0.875
        assert 0.0 < result["w_struct"][0].item() < 1.0
    else:
        assert result["S_base"][0].item() == scorer.tau
    contributions = result["explanations"][0]["contributions"]
    assert sum(
        contributions[key] for key in ("C_label", "C_strsim", "C_struct", "C_llm")
    ) == pytest.approx(result["S_final"][0].item() - scorer.tau, abs=1.0e-7)


class _EvidenceDataset(_TinyDataset):
    hierarchical_relation_families = {"part_of": {}}

    def get_entity_features(self, iri, side):
        width = 0 if iri.endswith("missing") else (1 if iri.endswith("short") else 3)
        facts = [{"triple": (iri, "relation" if i < 2 else "other", f"tail {i % 2}"),
                  "score": 0.5, "subject_iri": iri, "object_iri": f"urn:tail:{i}",
                  "rel_iri": "urn:relation"} for i in range(width)]
        hierarchy = [{"triple": (iri, "is_a", f"tail {i % 2}"), "specificity": 0.5,
                      "subject_iri": iri, "object_iri": f"urn:ancestor:{i}"}
                     for i in range(width)]
        attrs = [{"prop": "definition", "prop_iri": "urn:definition", "value": iri,
                  "text": f"definition {iri}", "entity_iri": iri}] if width else []
        return {"labels": [iri, iri], "hierarchy": {"is_a": hierarchy, "part_of": hierarchy[:1]},
                "object_triples": facts, "attributes": attrs}


def _evidence_scorer(monkeypatch, **kwargs):
    import hashlib
    scorer = _scorer(return_explanations=True, **kwargs)
    scorer.use_context = scorer.use_lexical = True
    scorer.max_hierarchy_triples_per_family = 2
    scorer.max_object_triples = 2
    scorer.similarity_per_relation_cap = 1
    scorer.max_input_tokens_hier = 6
    scorer.max_input_tokens_sim = 4
    scorer.attach_dataset(_EvidenceDataset())
    calls = {"labels": [], "contexts": []}
    def encode(texts, kind):
        calls[kind].append(list(texts))
        out = torch.zeros((len(texts), 4))
        for index, text in enumerate(texts):
            out[index, hashlib.sha256(text.encode()).digest()[0] % 4] = 1.0
        return out
    monkeypatch.setattr(scorer, "encode_labels_batch", lambda texts: encode(texts, "labels"))
    monkeypatch.setattr(scorer, "encode_contexts_batch", lambda texts: encode(texts, "contexts"))
    return scorer, calls


def _evidence_forward(scorer):
    sources = ["source", "source", "source", "source_short", "source_missing"]
    targets = ["target", "target_short", "other", "target", "target_missing"]
    return scorer(src_iris=sources, tgt_iris=targets,
                  src_label_lists=[[s, s] for s in sources],
                  tgt_label_lists=[[t, t] for t in targets])


@pytest.mark.parametrize("block", [2, 32])
@pytest.mark.parametrize("bank", ["full", "attrs_labels", "attrs_only"])
def test_staged_channels_preserve_scores_evidence_routes_and_ties(monkeypatch, block, bank):
    monkeypatch.setenv("EXACT_EVIDENCE_PREFETCH", "0")
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_BATCHING", "0")
    scorer, calls = _evidence_scorer(monkeypatch, attr={"enabled": True, "bank": bank})
    legacy = _evidence_forward(scorer)
    calls["contexts"].clear()
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_BATCHING", "1")
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_BLOCK_PAIRS", str(block))
    batched = _evidence_forward(scorer)
    for key, value in legacy.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, batched[key], rtol=0, atol=0, equal_nan=True)
    assert legacy["explanations"] == batched["explanations"]
    assert legacy["llm_evidence_packets"] == batched["llm_evidence_packets"]
    assert torch.equal(torch.argsort(legacy["S_final"], stable=True), torch.argsort(batched["S_final"], stable=True))
    assert any(len(batch) > 2 for batch in calls["contexts"])
    assert all(len(batch) == len(set(batch)) for batch in calls["contexts"])
    assert getattr(scorer, "_prepared_evidence_result", None) is None


@pytest.mark.parametrize("formulation", ["off", "normalised", "absolute", "asymmetric", "missingness_aware"])
def test_ordinary_inference_omits_synthetic_diagnostics_without_natural_changes(monkeypatch, formulation):
    import json
    import exact.experiments.evidence_diagnostics as diagnostics
    scorer, _ = _evidence_scorer(monkeypatch, diff={"enabled": True, "formulation": formulation,
        "relation_interpretation": "<" if formulation == "asymmetric" else None,
        "dump_components": True, "controlled_perturbations": True})
    with_diagnostics = _evidence_forward(scorer)
    packets = [row["experiment_diagnostics"]["difference"]["controlled_perturbations"]
               for row in with_diagnostics["explanations"]]
    assert all(len(packet["variants"]) == 10 for packet in packets)
    scorer.diff_config["controlled_perturbations"] = False
    def forbidden(*args, **kwargs):
        pytest.fail("ordinary inference invoked synthetic diagnostic replay")
    monkeypatch.setattr(diagnostics, "difference_perturbation_inventory", forbidden)
    monkeypatch.setattr(diagnostics, "replay_difference_diagnostics", forbidden)
    ordinary = _evidence_forward(scorer)
    for key, value in with_diagnostics.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, ordinary[key], rtol=0, atol=0, equal_nan=True)
    def natural(value):
        if isinstance(value, dict):
            return {key: natural(item) for key, item in value.items()
                    if key not in {"controlled_perturbations", "difference_replay"}}
        if isinstance(value, list):
            return [natural(item) for item in value]
        return value
    assert natural(with_diagnostics["explanations"]) == natural(ordinary["explanations"])
    assert len(json.dumps(ordinary["explanations"])) < len(json.dumps(with_diagnostics["explanations"]))


def test_prepared_payload_scope_restores_after_failure():
    from types import SimpleNamespace
    from exact.impl.models.pair_adaptive_batch import call_prepared
    scorer = SimpleNamespace(_prepared_evidence_result=("outer", {}))
    def fail():
        raise RuntimeError("fixture failure")
    with pytest.raises(RuntimeError, match="fixture failure"):
        call_prepared(scorer, "inner", fail, {"score": 0.5})
    assert scorer._prepared_evidence_result == ("outer", {})


def test_context_batch_oom_reduces_batch_without_losing_texts(monkeypatch):
    from types import SimpleNamespace
    from exact.impl.models.pair_adaptive_batch import _encode_unique
    seen = []
    def encode(texts):
        if len(texts) > 2:
            raise torch.cuda.OutOfMemoryError("fixture pressure")
        seen.extend(texts)
        return torch.tensor([[1.0, 0.0] for _ in texts])
    scorer = SimpleNamespace(encode_contexts_batch=encode)
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_TEXT_BATCH", "64")
    vectors = _encode_unique(scorer, ["a", "b", "a", "c", "d", "e"], "similarity")
    assert seen == ["a", "b", "c", "d", "e"]
    assert list(vectors) == seen


def test_ordinary_diagnostic_amendment_is_explicit_and_does_not_mutate_e24():
    from exact.experiments.evidence_diagnostics import ordinary_inference_config
    original = {"matching":{"channels":{"diff":{"enabled":True,"formulation":"off",
                "dump_components":True,"controlled_perturbations":True}}},
                "pipeline":[{"name":"PairAdaptiveSemanticScorer", "params":{
                    "diff":{"controlled_perturbations":True,"formulation":"off"}}}]}
    resolved, receipt = ordinary_inference_config(original)
    assert original["matching"]["channels"]["diff"]["controlled_perturbations"] is True
    assert resolved["matching"]["channels"]["diff"] == {"enabled":True,"formulation":"off",
                                          "dump_components":True,"controlled_perturbations":False}
    assert resolved["pipeline"][0]["params"]["diff"]["controlled_perturbations"] is False
    assert receipt["synthetic_replays_per_pair_avoided"] == 10
    assert receipt["natural_evidence_preserved"] is True
