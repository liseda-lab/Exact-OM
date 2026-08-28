"""Strict, default-off configuration models for methodology experiments."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, model_validator

from exact.core.entities.configs.strict import StrictConfigModel


class ExtractionConfig(StrictConfigModel):
    """Post-selector global extraction used by E01."""

    mode: Literal["greedy", "mutual_best", "assignment", "stable_marriage"] = Field(
        "greedy", description="Global extraction strategy; greedy preserves shipped behavior."
    )
    assignment_component_cap: int = Field(
        500,
        ge=1,
        description="Connected-component size above which assignment falls back to mutual-best.",
    )


class AnchorRescoringConfig(StrictConfigModel):
    """Anchor-guided structural second pass used by E02."""

    mode: Literal["off", "one_pass", "iterate"] = Field(
        "off", description="Anchor-rescoring mode; off preserves shipped behavior."
    )
    threshold: float = Field(
        0.95, ge=0.0, le=1.0, description="Base-score threshold for predicted anchors."
    )
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

    mode: Literal["off", "accept_model", "heuristic"] = Field(
        "off", description="NIL scoring mode; off preserves non-NIL behavior."
    )
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


class HierarchyChannelExperimentConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable E09 hierarchy-semantic variants.")
    mode: Literal["labels", "labels_overlap"] = Field("labels")
    siblings: bool = Field(False)
    depth: Optional[int] = Field(2, ge=1, description="Ancestor depth; null requests full closure.")
    overlap_weight: float = Field(0.5, ge=0.0, le=1.0)


class DifferenceChannelExperimentConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable E24 contrastive-channel diagnostics/variants.")
    formulation: Literal["normalised", "absolute", "asymmetric", "off"] = Field("normalised")
    dump_components: bool = Field(False, description="Persist pivot-reason diagnostics.")


class LexicalChannelExperimentConfig(StrictConfigModel):
    enabled: bool = Field(False, description="Enable E26 lexical-quality variants.")
    quality: Literal["margin", "entropy", "encoder_agreement", "constant"] = Field("margin")
    entropy_top_m: int = Field(5, ge=2)


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
    mode: Literal["off", "inductive", "transductive", "graph_only"] = Field("off")
    artifact: Optional[Path] = None
    dump_profile: bool = False


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
    listwise_max_candidates: int = Field(5, ge=2, le=26)
    permutations: Literal[2] = 2
    samples_per_permutation: Literal[3] = 3


class LLMGateExperimentConfig(StrictConfigModel):
    mode: Literal["off", "analytic", "quantile", "forced_sample", "oracle", "learned"] = Field(
        "analytic"
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    quantile_fraction: float = Field(0.05, gt=0.0, le=1.0)
    forced_sample_size: int = Field(0, ge=0)
    artifact: Optional[Path] = None


class LLMExperimentConfig(StrictConfigModel):
    """Canonical LLM experiment controls for E07/E21/E25."""

    enabled: bool = Field(False, description="Enable experiment-side LLM controls.")
    decision: LLMDecisionExperimentConfig = Field(
        default=LLMDecisionExperimentConfig.model_validate({})
    )
    gate: LLMGateExperimentConfig = Field(default=LLMGateExperimentConfig.model_validate({}))
    exemplars: Literal["off", "knn"] = Field("off")
    exemplar_count: int = Field(0, ge=0)
    distill: Literal["off", "student"] = Field("off")
    distill_artifact: Optional[Path] = None
    fusion_weight: Literal["beta_u", "learned"] = Field("beta_u")
    fusion_artifact: Optional[Path] = None


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


class EncoderFinetuneConfig(StrictConfigModel):
    mode: Literal["off", "contrastive"] = Field("off")
    artifact: Optional[Path] = None
    negative_policy: Literal["complete_reference", "confirmed_negative", "positive_unlabelled"] = (
        Field("complete_reference")
    )


class CrossEncoderConfig(StrictConfigModel):
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

    def resolve_component(
        self,
        component: str,
        *,
        training_available: bool,
    ) -> tuple[str, str]:
        """Resolve one component without silently consuming unavailable labels."""

        requested = str(self.components.get(component, self.mode))
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
        raise NotImplementedError(
            "profile_rule supervision resolution requires E22's fitted observable-profile "
            "runtime, which is not implemented"
        )


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
