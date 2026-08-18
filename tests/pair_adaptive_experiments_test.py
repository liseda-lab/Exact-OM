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
    assert absolute._score_difference_channel(src, tgt, support)["score"] == pytest.approx(
        1.0 - 1.0 / 2.0**0.5
    )

    asymmetric = _scorer(diff={"enabled": True, "formulation": "asymmetric"})
    asymmetric.use_context = True
    assert asymmetric._score_difference_channel(src, tgt, support)["score"] == pytest.approx(1.0)

    empty = absolute._score_difference_channel([], tgt, torch.zeros((0, 1)))
    assert empty["quality"] == 0.0
    assert empty["diff_pivot_reason"] == "empty_source"

    off = _scorer(diff={"enabled": True, "formulation": "off"})
    off.use_context = True
    assert off._score_difference_channel(src, tgt, support)["diff_pivot_reason"] == "channel_off"


def test_signed_identifier_attributes_can_contribute_below_pivot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _scorer(attr={"enabled": True, "polarity": "signed", "bank": "attrs_only"})
    scorer.use_context = True
    monkeypatch.setattr(
        scorer,
        "_encode_context_matrix",
        lambda left, right: torch.full((len(left), len(right)), 0.9),
    )
    payload = scorer._score_attribute_channel(
        [{"prop": "identifier", "value": "A-1", "text": "identifier A-1"}],
        [{"prop": "code", "value": "B-9", "text": "code B-9"}],
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


def test_gate_instrumentation_uses_frozen_threshold_without_batch_quantiles() -> None:
    scorer = _scorer(
        llm_experiment_config={
            "enabled": True,
            "gate": {"mode": "quantile", "threshold": 0.7, "quantile_fraction": 0.05},
        }
    )
    mask, rows = scorer._llm_gate_mask(
        U_ind=torch.tensor([0.2, 0.8]),
        U_dis=torch.tensor([0.1, 0.3]),
        U=torch.tensor([0.2, 0.8]),
        S_base=torch.tensor([0.9, 0.5]),
        q_label=torch.tensor([1.0, 0.5]),
        Q_struct=torch.tensor([0.2, 0.5]),
        src_iris=["s1", "s2"],
        tgt_iris=["t1", "t2"],
        label=None,
    )
    assert mask.tolist() == [False, True]
    assert rows[0]["threshold_source"] == "config_frozen"
    assert rows[1]["would_route"] is True
    assert rows[1]["U_ind"] == pytest.approx(0.8)


def test_unimplemented_graph_arm_fails_closed_instead_of_running_baseline() -> None:
    with pytest.raises(NotImplementedError, match="refusing to execute"):
        _scorer(graph={"mode": "inductive", "artifact": "graph.json"})


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
                "parameters": {
                    "tau": 0.4,
                    "gamma": 1.0,
                    "beta": 0.25,
                    "multipliers": {"label": 2.0, "default": 0.5},
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
    assert (scorer.tau, scorer.gamma, scorer.beta) == pytest.approx((0.4, 1.0, 0.25))
    authority = scorer._sigma_authority(
        torch.tensor([0.8]),
        torch.tensor([0.5]),
        torch.tensor([True]),
        channel="label",
    )
    assert authority.item() == pytest.approx(0.4)


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
