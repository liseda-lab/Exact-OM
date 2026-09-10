"""Strict, default-off configuration models for methodology experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from exact.core.entities.configs.strict import StrictConfigModel


class ExtractionConfig(StrictConfigModel):
    """Post-selector global extraction used by E01."""

    mode: Literal[
        "greedy",
        "mutual_best",
        "assignment",
        "stable_marriage",
        "assignment_accepted_utility",
        "assignment_legacy",
    ] = Field(
        "greedy", description="Global extraction strategy; greedy preserves shipped behavior."
    )
    assignment_component_cap: int = Field(
        500,
        ge=1,
        description="Connected-component size above which assignment falls back to threshold-first greedy.",
    )


class AnchorRescoringConfig(StrictConfigModel):
    """Anchor-guided structural second pass used by E02."""

    mode: Literal["off", "one_pass", "iterate"] = Field(
        "off", description="Anchor-rescoring mode; off preserves shipped behavior."
    )
    threshold: float = Field(
        0.95, ge=0.0, le=1.0, description="Base-score threshold for predicted anchors."
    )
    exact_policy: Optional[Literal["hard", "soft"]] = None
    source: Literal["exact", "trusted", "predicted"] = "predicted"
    trusted_file: Optional[Path] = None
    rule_artifact: Optional[Path] = None
    margin: float = Field(0.1, ge=0.0, le=1.0)
    corruption_fraction: float = Field(0.0, ge=0.0, le=0.05)
    diagnostic: bool = False

    @field_validator("corruption_fraction")
    @classmethod
    def validate_corruption_fraction(cls, value: float) -> float:
        if value not in {0.0, 0.01, 0.05}:
            raise ValueError("Anchor corruption must be 0%, 1%, or 5%")
        return value

    @model_validator(mode="after")
    def corruption_is_diagnostic(self):
        if self.corruption_fraction and not self.diagnostic:
            raise ValueError("Anchor corruption must be explicitly diagnostic")
        return self

    exact_only: bool = Field(False, description="Restrict the anchor set to exact matches.")
    max_passes: int = Field(2, ge=1, le=2, description="Maximum bounded anchor-rescoring passes.")


class ScoreCalibrationConfig(StrictConfigModel):
    """Score calibration and target-label-free threshold controls from E03."""

    mode: Literal["none", "platt", "isotonic"] = Field(
        "none", description="Calibration applied to final pair scores."
    )
    threshold_mode: Literal["fixed", "otsu", "knee"] = Field(
        "fixed", description="Acceptance-threshold source."
    )
    artifact: Optional[Path] = Field(
        None, description="Optional immutable fitted calibrator artifact."
    )


class NilConfig(StrictConfigModel):
    """NIL/abstention controls from E04."""

    mode: Literal["off", "accept_model", "heuristic", "fitted"] = Field(
        "off", description="NIL scoring mode; off preserves non-NIL behavior."
    )
    artifact: Optional[Path] = None
    training_source_labels: Optional[Path] = None
    pool_miss_development_reference: Optional[Path] = None
    label_semantics: Literal["unknown", "natural"] = "unknown"
    ranking_scale: Literal["joint_accept_probability"] = Field(
        "joint_accept_probability", description="Common scale for real candidates and NIL."
    )


class FusionExperimentConfig(StrictConfigModel):
    """Canonical experiment surface for E10/E19/E24/E26 score fusion."""

    enabled: bool = Field(False, description="Enable experiment-side fusion controls.")
    mode: Literal["analytic_shipped", "analytic_fitted", "learned_global", "learned_adaptive"] = (
        Field("analytic_shipped", description="Evidence-fusion weight provider.")
    )
    scope: Literal["global", "per_kind", "per_profile"] = Field(
        "global", description="Scope of fitted fusion parameters."
    )
    sigma_mode: Literal["full", "constant_q", "no_sharpening", "suppression_only", "uniform"] = (
        Field("full", description="E26 decomposition of quality, sharpening, and suppression.")
    )
    tau: float = Field(0.5, ge=0.0, le=1.0, description="Neutral score pivot when enabled.")
    llm_pivot: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Canonical E10 mapping to PairAdaptiveSemanticScorer.params.tau_LLM.",
    )
    gamma: float = Field(2.0, ge=0.0, description="Informative-deviation exponent when enabled.")
    beta: float = Field(0.8, ge=0.0, description="LLM contribution scale when enabled.")
    artifact: Optional[Path] = Field(None, description="Optional immutable fitted fusion artifact.")


class StringSimilarityChannelConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable the E06 scoring-time string channel.")
    placement: Literal["channel", "folded_into_lexical"] = Field(
        "channel", description="Treat the signal independently or fold it into lexical evidence."
    )
    abbreviation: Literal["off", "initialism"] = Field(
        "off", description="Optional conservative abbreviation sub-signal."
    )
    isub_weight: float = Field(1.0, ge=0.0)
    jaro_winkler_weight: float = Field(1.0, ge=0.0)
    token_set_weight: float = Field(1.0, ge=0.0)


class AttributeChannelExperimentConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable E08 attribute-channel variants.")
    polarity: Literal["support_only", "signed"] = Field("support_only")
    bank: Literal["full", "attrs_labels", "attrs_only"] = Field("full")
    signed_property_allowlist: List[str] = Field(default_factory=list)
    provenance_dedup: bool = False


class HierarchyChannelExperimentConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable E09 hierarchy-semantic variants.")
    mode: Literal["labels", "labels_overlap", "off"] = Field("labels")
    siblings: bool = Field(False)
    depth: Optional[int] = Field(2, ge=1, description="Ancestor depth; null requests full closure.")
    overlap_weight: float = Field(0.5, ge=0.0, le=1.0)


class DifferenceIncompatibility(StrictConfigModel):
    """A pinned semantic incompatibility between two ontology object values."""

    property_iri: str = Field(min_length=1)
    source_object: str = Field(min_length=1)
    target_object: str = Field(min_length=1)
    semantic_rule: Literal["disjoint_objects", "exclusive_values"]
    evidence_id: str = Field(min_length=1)


class DifferenceChannelExperimentConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable E24 contrastive-channel diagnostics/variants.")
    formulation: Literal["normalised", "absolute", "asymmetric", "missingness_aware", "off"] = (
        Field("normalised")
    )
    dump_components: bool = Field(False, description="Persist pivot-reason diagnostics.")
    incompatibilities: List[DifferenceIncompatibility] = Field(default_factory=list)
    relation_interpretation: Optional[Literal["<", ">"]] = None


class LexicalChannelExperimentConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable E26 lexical-quality variants.")
    quality: Literal["margin", "candidate_margin", "entropy", "encoder_agreement", "constant"] = (
        Field("margin")
    )
    entropy_top_m: int = Field(5, ge=2)
    entropy_temperature: float = Field(1.0, gt=0.0)
    deduplicate_labels: bool = False


class EvidenceGroupConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable kind-specific experiment evidence switches.")
    labels: bool = True
    annotations: bool = True
    signature: bool = True
    hierarchy: bool = True
    characteristics: bool = True
    usage: bool = True
    types: bool = True
    relations: bool = True
    literals: bool = True
    anchors: bool = False
    relations_shuffled: bool = False


class GraphChannelExperimentConfig(StrictConfigModel):
    mode: Literal["off", "features", "inductive", "transductive", "graph_only"] = Field("off")
    artifact: Optional[Path] = None
    dump_profile: bool = False
    shuffled: bool = False
    hierarchy_removal: float = Field(0.0, ge=0.0, le=1.0)
    negative_label_policy: Optional[Literal["complete_reference", "confirmed_negatives"]] = None

    @field_validator("hierarchy_removal")
    @classmethod
    def validate_hierarchy_removal(cls, value: float) -> float:
        if value not in {0.0, 0.5, 1.0}:
            raise ValueError("Hierarchy removal must be 0%, 50%, or 100%")
        return value


class SelectorTuningConfig(StrictConfigModel):
    count_reference_miss_as: Literal["fp_fn", "fp"] = Field("fp_fn")


class SelectorRerankConfig(StrictConfigModel):
    mode: Literal[
        "current", "analytic", "current_listwise", "pointwise", "pairwise", "listwise_nil"
    ] = Field("current")
    model: Literal["current_linear", "channel_gating", "additive_gam", "gbdt_monotonic"] = Field(
        "current_linear"
    )
    features: Literal["current", "extended"] = Field("current")
    artifact: Optional[Path] = None


class SelectorExperimentConfig(StrictConfigModel):
    """Canonical selector surface for E01/E03/E04/E10/E15/E18."""

    enabled: bool = Field(False, description="Enable experiment-side selector controls.")
    runtime_enabled: Optional[bool] = Field(
        None, description="Optional experiment override for the CandidateSetSelector stage."
    )
    runtime_global_only: Optional[bool] = None
    emit_candidate_scores: bool = False
    accept_model: Literal["logistic", "gbdt_monotonic"] = Field("logistic")
    accept_training: Literal["winner_only", "winner_plus_runnerup"] = Field("winner_only")
    label_free_mode: Literal[
        "current_fallback", "score_partition", "reciprocal_consensus", "pseudo_label_accept"
    ] = Field("current_fallback")
    tuning: SelectorTuningConfig = Field(default=SelectorTuningConfig.model_validate({}))
    rerank: SelectorRerankConfig = Field(default=SelectorRerankConfig.model_validate({}))


class LLMDecisionExperimentConfig(StrictConfigModel):
    mode: Literal["binary", "listwise", "listwise_sc"] = Field("binary")
    probability: Literal["raw_joint", "conditional_real", "pairwise_vs_none", "max_normalized"] = (
        Field("raw_joint")
    )
    listwise_max_candidates: int = Field(5, ge=2, le=5)
    evidence: Literal["generated_brief", "scored_packet", "structured_packet"] = "generated_brief"
    brief_max_tokens: Literal[64, 256] = 64
    permutations: Literal[1, 2] = 1
    output: Literal["categorical", "hard_choice"] = "categorical"
    max_evidence_packets: Literal[0, 2] = 0
    samples_per_permutation: Literal[3] = 3


class LLMGateExperimentConfig(StrictConfigModel):
    mode: Literal[
        "off",
        "analytic",
        "quantile",
        "transferred_threshold",
        "source_top_fraction",
        "pair_top_fraction",
        "forced_sample",
        "oracle",
        "oracle_perfect",
        "oracle_replay",
        "learned",
    ] = Field("analytic")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    quantile_fraction: float = Field(0.05, gt=0.0, le=1.0)
    strata_artifact: Optional[Path] = Field(
        None, description="Frozen forced-source strata with permitted development label provenance."
    )
    forced_sample_size: int = Field(
        0, ge=0, le=200, description="Maximum frozen development source groups, not candidate rows."
    )
    artifact: Optional[Path] = None


class LLMExperimentConfig(StrictConfigModel):
    """Canonical LLM experiment controls for E07/E21/E25."""

    enabled: bool = Field(False, description="Enable experiment-side LLM controls.")
    decision: LLMDecisionExperimentConfig = Field(
        default=LLMDecisionExperimentConfig.model_validate({})
    )
    gate: LLMGateExperimentConfig = Field(default=LLMGateExperimentConfig.model_validate({}))
    exemplars: Literal["off", "knn"] = Field("off")
    exemplar_count: int = Field(0, ge=0, le=3)
    exemplar_artifact: Optional[Path] = None
    distill: Literal["off", "student"] = Field("off")
    distill_artifact: Optional[Path] = None
    student_training: Literal["gold_only", "gold_teacher"] = "gold_teacher"
    teacher_source_cap: int = Field(200, ge=1, le=200)
    outcome_policy: Literal["unknown", "complete_sources"] = "unknown"
    fusion_weight: Literal["beta_u", "constant", "source_first", "learned"] = Field("beta_u")
    constant_weight: float = Field(0.5, ge=0.0, le=1.0)
    fusion_artifact: Optional[Path] = None

    @model_validator(mode="after")
    def validate_exemplar_count(self) -> "LLMExperimentConfig":
        if self.exemplars == "knn" and self.exemplar_count < 1:
            raise ValueError("knn exemplars require exemplar_count between 1 and 3")
        return self


class AdaptiveKConfig(StrictConfigModel):
    enabled: bool = False
    criterion: Literal["gap", "entropy"] = Field("gap")
    gap: float = Field(0.05, ge=0.0, le=1.0)
    entropy: float = Field(0.75, ge=0.0, le=1.0)
    k_min: int = Field(20, ge=1)
    k_max: int = Field(20, ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "AdaptiveKConfig":
        if self.k_max < self.k_min:
            raise ValueError("adaptive_k.k_max must be greater than or equal to k_min")
        return self


class RetrievalTrainingConfig(StrictConfigModel):
    base_model: str
    revision: str
    epochs: int = Field(3, ge=1, le=3)
    max_steps: int = Field(1000, ge=1, le=10000)
    batch_size: int = Field(4, ge=1)
    accumulation: int = Field(8, ge=1)
    checkpoint_steps: int = Field(25, ge=1, le=100)
    patience: int = Field(3, ge=1, le=3)


class EncoderFinetuneConfig(StrictConfigModel):
    training: Optional[RetrievalTrainingConfig] = None
    mode: Literal["off", "contrastive"] = Field("off")
    artifact: Optional[Path] = None
    negative_policy: Literal["complete_reference", "confirmed_negative", "positive_unlabelled"] = (
        Field("complete_reference")
    )


class CrossEncoderConfig(StrictConfigModel):
    training: Optional[RetrievalTrainingConfig] = None
    mode: Literal["off", "on"] = Field("off")
    artifact: Optional[Path] = None
    top_k: int = Field(20, ge=1)


class CandidateMultiViewConfig(StrictConfigModel):
    mode: Literal["labels", "labels_types", "labels_relations"] = Field("labels")


class SupervisionAutoPolicyConfig(StrictConfigModel):
    kind: Literal["current", "threshold", "profile_rule"] = Field("current")
    artifact: Optional[Path] = None
    fallback: Literal["label_free"] = Field("label_free")


class EffectiveTrainingUnitConfig(StrictConfigModel):
    unit: str
    minimum: int = Field(ge=0)


class SupervisionConfig(StrictConfigModel):
    transfer_artifact: Optional[Path] = None

    negative_label_policy: Literal["unknown", "complete_reference", "confirmed_negatives"] = (
        "unknown"
    )
    label_budget: Optional[int] = Field(None, ge=1)
    label_selection: Literal["passive", "uncertainty"] = "passive"

    mode: Literal["auto", "supervised", "label_free"] = Field("auto")
    components: dict[
        Literal[
            "retrieval",
            "fusion",
            "rerank",
            "llm",
            "accept",
            "calibration",
            "structure",
            "relation",
        ],
        Literal["auto", "supervised", "label_free"],
    ] = Field(default_factory=dict)
    auto_policy: SupervisionAutoPolicyConfig = Field(
        default=SupervisionAutoPolicyConfig.model_validate({})
    )
    min_effective_training_units: dict[str, EffectiveTrainingUnitConfig] = Field(
        default_factory=dict
    )
    artifacts: dict[str, Path] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transfer_mode(self):
        if self.transfer_artifact is not None and (
            self.mode != "label_free"
            or any(mode != "label_free" for mode in self.components.values())
        ):
            raise ValueError(
                "Donor transfer requires label_free supervision for every recipient component"
            )
        return self

    def resolve_component(
        self,
        component: str,
        *,
        training_available: bool,
        profile_binding: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Resolve one component without silently consuming unavailable labels."""

        components: dict[str, str] = {
            str(key): str(value) for key, value in self.components.items()
        }
        requested = components.get(component, self.mode)
        if requested == "supervised":
            if not training_available:
                raise ValueError(
                    f"supervision.components.{component}=supervised requires a "
                    "usable training reference"
                )
            return "supervised", "explicit_override"
        if requested == "label_free":
            return "label_free", "explicit_override_ignores_training"

        policy = self.auto_policy.kind
        if not training_available:
            return "label_free", "auto_no_training_reference"
        if policy == "current":
            # Preserve the shipped path: the selector rank/accept/calibration
            # fit and LLM calibration may consume the in-pair train split.
            supervised = {"rerank", "accept", "calibration", "llm"}
            return (
                ("supervised", "auto_current_compatibility")
                if component in supervised
                else ("label_free", "auto_current_compatibility")
            )
        if policy == "threshold":
            threshold = self.min_effective_training_units.get(component)
            if threshold is None:
                return "label_free", "auto_threshold_missing_definition"
            return (
                "label_free",
                f"auto_threshold_requires_{threshold.minimum}_{threshold.unit}",
            )
        if self.auto_policy.artifact is None:
            raise ValueError("profile_rule auto policy requires an immutable artifact")
        payload = json.loads(self.auto_policy.artifact.read_text())
        if payload.get("schema_version") != 1 or payload.get("kind") != "supervision_count_policy":
            raise ValueError("Invalid frozen supervision policy")
        units_path = self.artifacts.get("training_units")
        if units_path is None:
            return "label_free", "auto_profile_missing_effective_units"
        units = json.loads(units_path.read_text())
        if payload["binding"] != units.get("binding"):
            raise ValueError("Supervision policy effective-unit binding mismatch")
        if not profile_binding:
            return "label_free", "auto_profile_missing_runtime_binding"
        if any(payload["binding"].get(key) != value for key, value in profile_binding.items()):
            raise ValueError("Supervision policy runtime binding mismatch")
        rule = payload.get("components", {}).get(component, {})
        minimum = rule.get("minimum_groups")
        count = int(units.get("component_units", {}).get(component, 0))
        definition = units.get("component_definitions", {}).get(component)
        if minimum is None or definition != payload.get("count_definition") or count < int(minimum):
            return "label_free", "auto_profile_label_free_fallback"
        return "supervised", "auto_profile_frozen_count_crossover"


__all__ = [
    "AdaptiveKConfig",
    "AnchorRescoringConfig",
    "AttributeChannelExperimentConfig",
    "CandidateMultiViewConfig",
    "CrossEncoderConfig",
    "DifferenceChannelExperimentConfig",
    "EffectiveTrainingUnitConfig",
    "EncoderFinetuneConfig",
    "EvidenceGroupConfig",
    "ExtractionConfig",
    "FusionExperimentConfig",
    "GraphChannelExperimentConfig",
    "HierarchyChannelExperimentConfig",
    "LLMExperimentConfig",
    "LexicalChannelExperimentConfig",
    "NilConfig",
    "ScoreCalibrationConfig",
    "SelectorExperimentConfig",
    "StringSimilarityChannelConfig",
    "SupervisionConfig",
]
