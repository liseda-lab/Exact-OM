"""Versioned additive contracts for partial navigation and description presentation."""

from typing import Any, Literal

from pydantic import Field

from .contracts import WireModel


class CompletenessResult(WireModel):
    complete: bool
    reasons: list[dict[str, Any]]


class ConsistencyResult(CompletenessResult):
    value: bool


class PartialInferenceMetadata(WireModel):
    contract_version: Literal["exact-partial-hierarchy/1"]
    basis: Literal["elk_inferred_partial"]
    status: Literal["partial"]
    complete: Literal[False]
    completion: Literal["partial_for_recorded_ELK_procedure"]
    reason: str
    class_count: int = Field(ge=0)
    unrepresented_class_count: int = Field(ge=0)
    equivalence_group_count: int = Field(ge=1)
    reported_direct_edge_count: int = Field(ge=0)
    served_direct_edge_count: int | None = Field(default=None, ge=0)
    reasoner: Literal["pyelk"]
    versions: dict[str, str]
    options: dict[str, Any]
    supported_kinds: list[Literal["class"]]
    consistency: ConsistencyResult
    classification: CompletenessResult
    unsupported_constructs: list[dict[str, Any]]
    input_sha256: str
    artifact_sha256: str
    admission_identity: str
    policy_hash: str
    scope: str
    source_imports_complete: bool
    source_documents: list[str]
    semantics: Literal["direct_named_class_sub_to_super_edges_between_equivalence_groups"]
    directness: Literal["direct_in_returned_partial_taxonomy_only"]
    positive_edge_guarantee: Literal["returned_by_pinned_ELK_supported_projection"]
    full_ontology_entailment: Literal["unknown_without_exported_justification"]
    negative_entailment: Literal["unknown"]
    satisfiability: Literal["unknown"]
    justifications: Literal["not_exported"]
    top: list[str]
    bottom: list[str]


class DescriptionPresentation(WireModel):
    contract_version: Literal["exact-description-presentation/1"]
    kind: Literal[
        "reviewed_profile",
        "necessary_conditions",
        "available_facts",
        "reviewed_comparison",
        "paired_context",
        "source_only_profile",
        "target_only_profile",
        "unavailable",
    ]
    covered_sides: list[Literal["source", "target"]]
    claim_sides: dict[str, list[Literal["source", "target"]]]
    relation_asserted: Literal[False]
    scope_note: str
    render_revision: Literal["typed-readable/3"] | None
    review_receipt_sha256: str | None
