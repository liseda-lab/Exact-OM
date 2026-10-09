"""Strict, separately versioned XR-2.1 protocols; archived XR-2 files stay untouched."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from exact.repair.records import canonical_hash
from exact.repair.semantic_fidelity import _strict_json

Text = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(ge=0)]
PositiveCount = Annotated[int, Field(ge=1)]
Amount = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveAmount = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictSection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Identity(StrictSection):
    implementation_revision: Literal[
        "exact-repair/review-corrections/v2", "exact-repair/preliminary-corrections/v1"
    ]
    research_revision: Literal["XR-2.1"]
    record_schema: Literal["exact-repair/records/v3"]
    feature_schema: Literal["exact-repair/observable-features/v3"]
    label_schema: Literal[
        "exact-repair/teacher-cache/v3",
        "exact-repair/fidelity-training/v3.3",
        "exact-repair/mixed-targets/v3.3",
        "exact-repair/symbolic-plus-fidelity/v1",
    ]
    code_hash: Text
    dirty_hash: Text
    run_id: Text
    parent_run_id: str | None
    purpose: Literal["smoke", "development", "confirmatory"]
    execution_authorized: bool
    automatic_campaign_resumption: Literal[False]


class Inputs(StrictSection):
    ontology_manifest: Text
    import_manifest: Text
    alignment_manifest: Text
    evidence_manifest: Text
    relation_interpretation: Text
    licensing_manifest: Text
    availability_manifest: Text
    matcher_source: Text
    split_parent_manifest: Text
    evidence_cutoff: Text
    defer_production_matcher: bool


class Policy(StrictSection):
    public_signature_manifest: Text
    private_signature_manifest: Text
    source_exception_mode: Literal["none", "proved_source_incoherence"]
    source_proof_manifest: Text
    hard_query_manifest: Text
    activated_obligations: bool
    eligibility_manifest: Text
    authorship_manifest: Text


class CorpusEvidence(StrictSection):
    simulator: Literal["label_jaccard_plus_noise_v2"]
    noise_std: Amount
    misleading_label_fraction: Fraction
    latent_corruption_features: Literal[False]
    counterbalance_provenance: Literal[True]


class SplitCounts(StrictSection):
    train: Count
    development: Count
    test: Count


class Corpus(StrictSection):
    families: Annotated[list[Text], Field(min_length=1)]
    groups_per_family: SplitCounts
    corruptions_per_group: PositiveCount
    clean_controls_per_group: Count
    generated_objects: Annotated[list[PositiveCount], Field(min_length=1)]
    split_seed: Count
    split_unit: Literal["clean_structural_parent"]
    composition_holdout: Text
    evidence: CorpusEvidence


class GenerationStage(StrictSection):
    classes_per_side: PositiveCount
    properties_per_side: PositiveCount
    endpoints_per_side: PositiveCount
    max_depth: Count
    max_constructors: Count
    candidate_cap: PositiveCount
    draws_per_object: Count


class Generation(StrictSection):
    grammar_version: Text
    templates: list[Text]
    constructors: list[Text]
    omission_mode: Literal["none", "final_candidate", "missing_vocabulary", "grammar_unavailable"]
    omission_manifest: Text
    stages: Annotated[list[GenerationStage], Field(min_length=1, max_length=3)]
    max_total_draws_per_object: Count
    elementary_guarantee: Literal["reserve_before_complex"]
    tie_rule: Literal["canonical_candidate_id"]
    execution_schedule: Literal["one_stage", "staged_verified_repair"] = "one_stage"
    elementary_seconds: PositiveAmount = 30.0

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        result = handler(self)
        for name in ("execution_schedule", "elementary_seconds"):
            if name not in self.model_fields_set:
                result.pop(name, None)
        return result


class Circuit(StrictSection):
    components: PositiveCount
    partition: Literal["family_template"]
    vtree: Literal["right_linear", "balanced", "grouped"]
    gc_policy: Literal["protected_roots"]
    gc_version: Text
    allocated_node_limit: PositiveCount
    live_node_limit: PositiveCount
    reachable_node_limit: PositiveCount
    element_limit: PositiveCount
    rss_mb: PositiveCount
    call_seconds: PositiveAmount
    aggregate_seconds: PositiveAmount
    cache_directory: Text
    cache_version: Text
    cache_reuse: Literal["integrity_checked_immutable"]
    semantic_constraints: Literal["proved_immutable_context"]
    proof_dependencies: Text
    distribution_identity: Text


class DeclaredGraphSchema(StrictSection):
    schema_id: Literal["exact-repair/declared-graph-schema/v1"] = Field(alias="schema")
    language: Literal["pyowl-core/public-structural-ast/v1"]
    language_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    node_types: Annotated[list[Text], Field(min_length=1)]
    edge_types: Annotated[
        list[Annotated[list[Text], Field(min_length=3, max_length=3)]], Field(min_length=1)
    ]

    @model_validator(mode="after")
    def supported_language(self):
        from exact.repair.graph_schema import declared_metadata

        declared_metadata(self.model_dump(by_alias=True))
        return self


class Model(StrictSection):
    graph_schema: DeclaredGraphSchema | None = None
    backbone: Literal["hgt", "rgcn", "no_graph"]
    readout: Literal["target_candidate_attention"]
    hidden_width: PositiveCount
    layers: PositiveCount
    attention_heads: PositiveCount
    dropout: Fraction
    context_hops: Count
    max_nodes: PositiveCount
    max_edges: PositiveCount
    max_supports: Count
    max_text_tokens: Count
    feature_masks: Literal["explicit_missingness/v3"]
    pair_selector: Literal["exact-repair/interaction-selection/v3"]
    max_pair_neighbors: Count
    max_object_pairs: Count
    max_candidate_factors: Count
    proposal_context: Literal["independent", "selected_other_actions"]
    unary_benefit: bool
    pair_benefit: bool
    plan_risk: bool
    preparation_schema: Literal["exact-repair/effective-preparation/v3.1"]
    support_admission_policy: Literal["atomic-complete-support/v3.1"]
    support_enabled: bool = False
    support_target: Literal["qualified_witness_violation/v1"] = "qualified_witness_violation/v1"
    cost_predictor: Literal[False]

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        result = handler(self)
        if self.graph_schema is None:
            result.pop("graph_schema", None)
        return result


class Teacher(StrictSection):
    target_schema: Literal["exact-repair/semantic-target/v3.1"]
    aggregation: Literal["weighted-family-means/v1"]
    consequence_manifest: Text
    typed_nonvacuity: Literal[True]
    temperature: PositiveAmount
    max_queries: PositiveCount
    family_weights: dict[Text, Amount]
    enumeration_ceiling: PositiveCount
    deadline_seconds: PositiveAmount
    per_query_masks: Literal[True]
    eligibility: Literal["complete_exact_or_explicit_sampled"]


class Collection(StrictSection):
    schedule_version: Literal["exact-repair/attempt-schedule/v1"]
    plan_quotas: dict[Literal["utility", "proposal", "diversity", "quartet", "uniform"], Count]
    quartet_budget_unit: Literal["assignment_attempts"]
    rounds: Count
    cases_per_round: Count
    plan_attempts_per_case: Count
    utility_attempts: Count
    diversity_attempts: Count
    quartet_attempts: Count
    proposal_attempts: Count
    generator_fraction: Fraction
    exploration_fraction: Fraction
    diversity_rule: Text
    quartet_context: Literal["same_background_all_four_feasible"]
    deduplication: Literal["canonical_assignment_with_origins"]
    retries: Count


class Losses(StrictSection):
    target_basis: Literal["symbolic", "ai_weak", "mixed", "symbolic_plus_llm"]
    scale_alignment: Text
    mixture_symbolic_weight: Fraction
    anchors: Text
    benefit_weight: Amount
    ranking_weight: Amount
    proposal_weight: Amount
    interaction_loss_weight: Amount
    risk_loss_weight: Amount
    support_loss_weight: Amount = 0.2
    sampled_proposal_loss_weight: Amount
    masks: Literal["complete_same_basis_per_loss"]
    normalization: Literal["eligible_terms"]
    sampled_target_approximation: Literal["sample_conditioned_ranking_imitation"]
    loss_contract: Literal["provenance_additive/v1"] | None = None
    weak_anchor_weight: Amount = 0.0
    weak_comparison_weight: Amount = 0.0

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        result = handler(self)
        for name in ("loss_contract", "weak_anchor_weight", "weak_comparison_weight"):
            if name not in self.model_fields_set:
                result.pop(name, None)
        return result


class PlanRatingAggregation(StrictSection):
    revision: Literal["per_criterion_median/v1"]
    minimum_ratings: PositiveCount
    max_criterion_range: Fraction


class LLMBudget(StrictSection):
    calls: Count
    tokens: Count
    cost_usd: Amount
    wall_seconds: Amount


class LLMLabels(StrictSection):
    enabled: bool
    execution_authorized: bool
    teacher_profile: str | None
    evaluator_profile: str | None
    teacher_role: Literal["repair_semantic_teacher"]
    evaluator_role: Literal["repair_semantic_evaluator"]
    ledger_directory: Text
    rubric_version: Text
    prompt_version: Text
    evidence_schema: Text
    model_manifest: Text
    criterion_applicability: Text
    tie_policy: Literal["separate_tie_loss"]
    abstention_policy: Literal["mask_global_targets_keep_denominator"]
    teacher_budget: LLMBudget
    evaluator_budget: LLMBudget
    aggregate_budget: LLMBudget
    cache_policy: Literal["dependency_bound_raw_revalidation"]
    independent_evaluator: bool
    annotation_manifest: str | None
    aggregation_revision: Literal["exact-repair/semantic-fidelity-aggregate/v3.3"]
    development_use_policy: Literal["independent_evaluation", "development_selection/v1"] = (
        "independent_evaluation"
    )
    plan_rating_aggregation: PlanRatingAggregation | None = None
    post_decode_annotation_manifest: str | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        result = handler(self)
        for name in (
            "development_use_policy",
            "plan_rating_aggregation",
            "post_decode_annotation_manifest",
        ):
            if name not in self.model_fields_set:
                result.pop(name, None)
        return result


class Objective(StrictSection):
    calibration_identity: Text
    edit_weights: dict[Text, Amount]
    integer_scale: PositiveCount
    rounding: Literal["half_even"]
    pair_factor_bound: Amount
    frozen_epoch_rule: Literal["invalidate_on_pool_or_coefficient_change"]


class Selection(StrictSection):
    shortlist_size: PositiveCount
    utility_window: Count
    construction_seconds: PositiveAmount
    tie_rule: Literal["canonical_assignment_id"]
    matched_utility_first_control: Literal[True]
    risk_ordering: bool
    support_cuts: Literal["proved_presence_only"]
    pending_ledger: Literal["unknown_and_untested_retain_upper_bounds"]
    exact_upper_bounds: Literal[True]
    retry_budget: Count
    max_master_solves: PositiveCount
    max_candidate_checks: PositiveCount
    stopping: Literal["first_incumbent", "optimality"]


class Reasoning(StrictSection):
    capability_matrix: Text
    routing: Literal["capability_before_cost"]
    detector_rules: Text
    support_budget: Count
    query_coverage: Literal["all_required"]
    session_cache_policy: Literal["immutable_dependencies"]
    unknown_causes: list[Text]


class Resources(StrictSection):
    case_wall_seconds: PositiveAmount
    case_cpu_seconds: PositiveAmount
    case_rss_mb: PositiveCount
    diagnosis_seconds: PositiveAmount
    generation_seconds: PositiveAmount
    verification_seconds: PositiveAmount
    startup_seconds: PositiveAmount
    cleanup_grace_seconds: PositiveAmount
    stage_wall_seconds: dict[Text, PositiveAmount]
    stage_cpu_seconds: dict[Text, PositiveAmount]
    stage_rss_mb: dict[Text, PositiveCount]
    concurrency: PositiveCount
    campaign_wall_seconds: PositiveAmount
    campaign_cpu_seconds: PositiveAmount
    campaign_gpu_hours: Amount
    allocated_gpus: Count
    campaign_cost_usd: Amount
    supervise_startup_and_transfer: Literal[True]


class Training(StrictSection):
    recovery_revision: Literal["exact-phase-resume/v3.1"]
    revision: Literal["v3"]
    optimizer: Literal["adamw"]
    dtype: Literal["float32"]
    device: Literal["cpu", "cuda"]
    threads: PositiveCount
    generation_stage: Count
    smooth_l1_beta: PositiveAmount
    ranking_temperature: PositiveAmount
    max_repair_pairs_per_case: PositiveCount
    min_dev_improvement: Amount
    learning_rate: PositiveAmount
    weight_decay: Amount
    gradient_clip: PositiveAmount
    batch_size: PositiveCount
    max_epochs: PositiveCount
    evaluate_every: PositiveCount
    patience: PositiveCount
    seeds: list[Count]
    sampled_assignments: Count
    active_round_every_epochs: PositiveCount
    minimum_generated_coverage: Fraction
    quality_uncertainty_z: Amount
    missing_label_fallback: Literal["stop", "exploratory"]
    checkpoint_criterion: Literal["generated_pool_verified_quality_effort"]
    proposal_loss: Literal["exact_and_sample_conditioned"]
    benefit_loss: Literal["anchored_value_rank_quartet_risk"]
    development_epochs: list[PositiveCount] | None = None
    development_case_ids: list[Text] | None = None
    patience_enabled: bool | None = None
    max_full_development_evaluations: PositiveCount | None = None
    final_development_reserve_seconds: Amount | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        result = handler(self)
        for name in (
            "development_epochs",
            "development_case_ids",
            "patience_enabled",
            "max_full_development_evaluations",
            "final_development_reserve_seconds",
        ):
            if name not in self.model_fields_set:
                result.pop(name, None)
        return result


class Evaluation(StrictSection):
    cohorts: list[Text]
    train_manifest: Text
    development_manifest: Text
    test_manifest: Text
    fresh_confirmatory_parents: bool
    metrics: list[Text]
    accounting: Literal["four_baseline_full_preprocess_search_query_costs"]
    cache_policy: Text
    checkpoint_rule: Literal["development_generated_pool"]
    independent_judge: Text
    grouping: Literal["parent_and_ontology_pair"]
    interval_method: Text
    all_scheduled_case_denominator: Literal[True]


class RepairProtocolV3(StrictSection):
    schema_id: Literal["exact-repair/protocol/v3"] = Field(alias="schema")
    identity: Identity
    input: Inputs
    corpus: Corpus
    policy: Policy
    generation: Generation
    circuit: Circuit
    model: Model
    teacher: Teacher
    collection: Collection
    losses: Losses
    llm_labels: LLMLabels
    objective: Objective
    selection: Selection
    reasoning: Reasoning
    resources: Resources
    training: Training
    evaluation: Evaluation

    @model_validator(mode="after")
    def cross_fields(self):
        target_schema = {
            "symbolic": "exact-repair/teacher-cache/v3",
            "ai_weak": "exact-repair/fidelity-training/v3.3",
            "mixed": "exact-repair/mixed-targets/v3.3",
            "symbolic_plus_llm": "exact-repair/symbolic-plus-fidelity/v1",
        }[self.losses.target_basis]
        if self.identity.label_schema != target_schema:
            raise ValueError("Declared label schema does not match the target basis")
        if self.losses.target_basis == "symbolic_plus_llm":
            if (
                self.losses.loss_contract != "provenance_additive/v1"
                or self.losses.weak_anchor_weight + self.losses.weak_comparison_weight <= 0
            ):
                raise ValueError("Combined supervision requires explicit additive weak losses")
            if self.llm_labels.plan_rating_aggregation is None:
                raise ValueError("Combined supervision requires a frozen plan-rating aggregation")
        if self.model.hidden_width % self.model.attention_heads or self.model.dropout >= 1:
            raise ValueError("Invalid graph width/heads/dropout")
        collection = self.collection
        if set(collection.plan_quotas) != {
            "utility",
            "proposal",
            "diversity",
            "quartet",
            "uniform",
        }:
            raise ValueError("Explicit quotas must cover every complete-plan stratum")
        if sum(collection.plan_quotas.values()) != collection.plan_attempts_per_case:
            raise ValueError("Complete-plan quotas must exactly equal the plan-attempt budget")
        if collection.plan_quotas["quartet"] % 4:
            raise ValueError("Quartet allocation counts complete four-assignment groups")
        for name in ("utility", "proposal", "diversity", "quartet"):
            if getattr(collection, name + "_attempts") != collection.plan_quotas[name]:
                raise ValueError("Legacy attempt declarations contradict explicit plan quotas")
        denominator = collection.plan_attempts_per_case
        for field, stratum in (
            ("generator_fraction", "proposal"),
            ("exploration_fraction", "uniform"),
        ):
            expected = collection.plan_quotas[stratum] / denominator if denominator else 0.0
            if abs(getattr(collection, field) - expected) > 1e-12:
                raise ValueError("Legacy fractions contradict explicit plan quotas")
        r = self.resources
        if (
            max(
                self.circuit.call_seconds,
                self.circuit.aggregate_seconds,
                self.selection.construction_seconds,
                r.diagnosis_seconds,
                r.generation_seconds,
                r.verification_seconds,
                r.startup_seconds,
            )
            + r.cleanup_grace_seconds
            > r.case_wall_seconds
        ):
            raise ValueError("Call allocation plus cleanup exceeds whole-case budget")
        if self.circuit.aggregate_seconds > r.generation_seconds:
            raise ValueError("Compilation aggregate exceeds generation allocation")
        if self.circuit.rss_mb > r.case_rss_mb:
            raise ValueError("Compiler RSS exceeds case RSS")
        if (
            sum(r.stage_wall_seconds.values()) > r.campaign_wall_seconds
            or sum(r.stage_cpu_seconds.values()) > r.campaign_cpu_seconds
        ):
            raise ValueError("Stage allocations exceed campaign budget")
        if not set(r.stage_wall_seconds) == set(r.stage_cpu_seconds) == set(r.stage_rss_mb):
            raise ValueError("Stage wall/CPU/RSS allocations must describe the same stages")
        for stage in self.generation.stages:
            if stage.max_depth > stage.max_constructors:
                raise ValueError("Grammar depth exceeds constructor bound")
            if stage.candidate_cap < 2 * stage.endpoints_per_side + 2:
                raise ValueError(
                    "Candidate cap cannot reserve declared equivalence elementary alternatives"
                )
        for previous, current in zip(self.generation.stages, self.generation.stages[1:]):
            if any(
                getattr(current, field) < getattr(previous, field)
                for field in GenerationStage.model_fields
            ):
                raise ValueError(
                    "Progressive stages must preserve nested language and resource bounds"
                )
        if self.training.generation_stage >= len(self.generation.stages):
            raise ValueError("Selected training generation stage is unavailable")
        if self.training.patience > self.training.max_epochs:
            raise ValueError("Patience exceeds epoch budget")
        training = self.training
        if training.development_epochs is not None:
            epochs = training.development_epochs
            if not epochs or epochs != sorted(set(epochs)) or epochs[-1] > training.max_epochs:
                raise ValueError(
                    "DEV epochs must be distinct, ordered and within the fitting budget"
                )
            if (
                training.max_full_development_evaluations is None
                or len(epochs) > training.max_full_development_evaluations
            ):
                raise ValueError("DEV schedule exceeds its explicit pass count")
        if training.development_case_ids is not None:
            if not training.development_case_ids or len(set(training.development_case_ids)) != len(
                training.development_case_ids
            ):
                raise ValueError("DEV case schedule must be nonempty and unique")
        if (
            self.generation.execution_schedule == "staged_verified_repair"
            and self.generation.elementary_seconds + r.cleanup_grace_seconds > r.case_wall_seconds
        ):
            raise ValueError("Elementary stage exceeds case budget")
        if (
            self.identity.purpose == "confirmatory"
            and not self.evaluation.fresh_confirmatory_parents
        ):
            raise ValueError("XR-2 pilot parents are not untouched confirmatory data")
        llm = self.llm_labels
        if llm.enabled:
            if not all(
                (
                    llm.execution_authorized,
                    llm.teacher_profile,
                    llm.evaluator_profile,
                    llm.annotation_manifest,
                )
            ):
                raise ValueError(
                    "Enabled LLM labels require authorized manifest and explicit profiles"
                )
            if llm.independent_evaluator and llm.teacher_profile == llm.evaluator_profile:
                raise ValueError("Independent evaluator profile equals teacher")
        elif self.losses.target_basis != "symbolic":
            raise ValueError("AI/mixed targets require an explicit annotation arm")
        for name in ("calls", "tokens", "cost_usd", "wall_seconds"):
            if getattr(llm.teacher_budget, name) + getattr(llm.evaluator_budget, name) > getattr(
                llm.aggregate_budget, name
            ):
                raise ValueError("Annotation role allocations exceed aggregate cap")
        return self

    @property
    def resolved_hash(self) -> str:
        return canonical_hash(self.model_dump(by_alias=True))


def load_protocol_v3(path: Path, *, for_execution: bool = False) -> RepairProtocolV3:
    """Resolve only explicit v3 inheritance; then strictly validate every field."""

    def merge(base, override):
        result = dict(base)
        for key, value in override.items():
            result[key] = (
                merge(result[key], value)
                if isinstance(value, dict) and isinstance(result.get(key), dict)
                else value
            )
        return result

    def resolve(current: Path, seen: frozenset[Path]):
        current = current.resolve()
        if current in seen:
            raise ValueError("Cyclic v3 protocol inheritance")
        value = _strict_json(current.read_text())
        if not isinstance(value, dict) or value.get("schema") != "exact-repair/protocol/v3":
            raise ValueError("Every inherited protocol must explicitly declare v3")
        parent = value.pop("extends", None)
        if parent is not None:
            if not isinstance(parent, str) or not parent:
                raise ValueError("Inheritance path must be nonempty")
            value = merge(resolve(current.parent / parent, seen | {current}), value)
        return value

    result = RepairProtocolV3.model_validate(resolve(Path(path), frozenset()))
    if for_execution:

        def unresolved(value):
            if isinstance(value, dict):
                return any(unresolved(v) for v in value.values())
            if isinstance(value, list):
                return any(unresolved(v) for v in value)
            return isinstance(value, str) and value.startswith("UNFROZEN")

        if not result.identity.execution_authorized or unresolved(result.model_dump(by_alias=True)):
            raise PermissionError(
                "Freeze code/input manifests and authorize this separate run before execution"
            )
    return result


def training_projection_v3(protocol: RepairProtocolV3) -> dict[str, Any]:
    """Explicit argument projection for the existing trainer, retaining v3 identity.

    The selected development generation stage is frozen, not a claim that every
    expansion was searched. This projection is never serialized as a v2 protocol.
    """
    p = protocol.model_dump(by_alias=True)
    stage = p["generation"]["stages"][p["training"]["generation_stage"]]
    t, g, r, loss = p["training"], p["model"], p["resources"], p["losses"]
    required_stages = {"corpus", "label", "train"}
    if not required_stages <= set(r["stage_wall_seconds"]):
        raise ValueError("Training requires explicit corpus/label/train stage allocations")
    if set(p["teacher"]["family_weights"]) != {"desired", "unwanted"}:
        raise ValueError("Trainer supports explicit desired/unwanted family weights")
    result = dict(p)
    result["resolved_v3_protocol_hash"] = protocol.resolved_hash
    result["data"] = p["corpus"]
    result["experiments"] = {"composition_holdout": p["corpus"]["composition_holdout"]}
    result["graph"] = {
        **g,
        "encoder": "none" if g["backbone"] == "no_graph" else g["backbone"],
        "max_explanations": g["max_supports"],
        "pair_factor_limit_per_object": g["max_pair_neighbors"],
        "pair_max_pairs": g["max_object_pairs"],
        "pair_max_factors": g["max_candidate_factors"],
    }
    result["training"] = {
        **t,
        "batch_cases": t["batch_size"],
        "clip_gradient_norm": t["gradient_clip"],
        "held_out_supervision": False,
        "checkpoint_selection": t["checkpoint_criterion"],
        "development_decode_every_epochs": t["evaluate_every"],
        "benefit_loss_weight": loss["benefit_weight"],
        "ranking_loss_weight": loss["ranking_weight"],
        "proposal_loss_weight": loss["proposal_weight"],
        **{
            k: loss[k]
            for k in (
                "interaction_loss_weight",
                "risk_loss_weight",
                "support_loss_weight",
                "sampled_proposal_loss_weight",
            )
        },
    }
    result["teacher"] = {
        **p["teacher"],
        "max_assignments": p["teacher"]["enumeration_ceiling"],
        "case_deadline_seconds": p["teacher"]["deadline_seconds"],
        "desired_family_weight": p["teacher"]["family_weights"]["desired"],
        "false_positive_weight": p["teacher"]["family_weights"]["unwanted"],
    }
    result["circuit"] = {
        **p["circuit"],
        "max_draws_per_object": stage["draws_per_object"],
        "compile_seconds": p["circuit"]["call_seconds"],
        "max_nodes": p["circuit"]["allocated_node_limit"],
    }
    result["grammar"] = {**stage, "retrieval_hops": g["context_hops"]}
    result["actions"] = {"max_states": {"generated": stage["candidate_cap"]}}
    result["preferences"] = {"cost_weights": p["objective"]["edit_weights"]}
    result["solver"] = {"quantization_scale": p["objective"]["integer_scale"]}
    result["resources"] = {
        **r,
        "stage_deadline_seconds": r["stage_wall_seconds"],
        "verification_call_seconds": r["verification_seconds"],
        "run_deadline_seconds": r["case_wall_seconds"],
        "memory_mb": r["case_rss_mb"],
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    loaded = load_protocol_v3(args.path)
    print(
        json.dumps(
            {
                "schema": loaded.schema_id,
                "resolved_hash": loaded.resolved_hash,
                "run_id": loaded.identity.run_id,
                "execution_authorized": loaded.identity.execution_authorized,
                "scope": "static configuration validation; no jobs submitted",
            },
            sort_keys=True,
        )
    )
