"""Validated resource models extending the frozen explanation contract primitives."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from .contracts import Availability, EntityRef, Page, Scope, Score, WireModel


class ResourceModel(WireModel):
    """Validate shared semantics while retaining documented additive resource metadata."""

    model_config = ConfigDict(extra="allow", frozen=True, allow_inf_nan=False)


class IriTerm(WireModel):
    """An exact named term."""

    term_type: Literal["iri"]
    iri: str


class LiteralTerm(WireModel):
    """Lossless literal identity, including lexical spelling, datatype and language."""

    term_type: Literal["literal"]
    lexical_form: str
    datatype: str | None
    language: str | None


class ExpressionTerm(WireModel):
    """A lossless structured expression loaded through explicit axiom detail."""

    term_type: Literal["expression_ref"]
    expression_id: str


Term = Annotated[IriTerm | LiteralTerm | ExpressionTerm, Field(discriminator="term_type")]


class Origin(ResourceModel):
    """Original asserted axiom provenance, distinct from interpretation rules."""

    document_sha256: str
    axiom_id: str | None
    source_span: dict[str, Any] | None
    provenance_status: str


class Fact(ResourceModel):
    """Ontology fact with semantic identity and explicit interpretation basis."""

    fact_id: str
    subject: EntityRef
    predicate_iri: str | None
    value: Term | None
    axiom_ref: str | None
    interpretation: Literal[
        "asserted",
        "structurally_derived",
        "reasoner_inferred",
        "projected",
        "matcher_comparison",
        "generated",
    ]
    origins: list[Origin]
    premise_ids: list[str]
    derivation_id: str | None


class Label(ResourceModel):
    """Display text never replaces an entity's typed identity."""

    text: str
    language: str | None
    predicate_iri: str


class EntityContextResponse(ResourceModel):
    """Independent summary with bounded original-fact categories."""

    artifact_type: Literal["entity_context"]
    contract_version: Literal["exact-explain/1.0"]
    entity: EntityRef
    labels: list[Label]
    definitions: Page[Fact]
    synonyms: Page[Fact]
    parents: Page[Fact]
    capabilities: dict[str, Any]
    context_scope: Scope
    fixture_provenance: str


class StageEvent(ResourceModel):
    """Bounded finite pipeline event with recorded or explicitly absent stage evidence."""

    stage: Literal[
        "retrieval",
        "prefilter",
        "pair_scoring",
        "llm_signal",
        "selection",
        "threshold",
        "cardinality",
        "extraction",
        "relation_typing",
        "repair",
    ]
    status: Literal["completed", "not_run", "not_recorded", "failed"]
    outcome: Literal[
        "retained", "rejected", "abstained", "selected", "not_selected", "not_applicable", "unknown"
    ]
    reason_code: str
    scores: list[Score]
    competitor_pair_ids: list[str]
    artifact_refs: list[str]
    implementation_id: str | None
    provenance_status: Literal[
        "recorded", "derived_from_saved_artifacts", "recomputed", "unavailable"
    ]


class PairResponse(ResourceModel):
    """Immutable pair binding, original decisions and actual persisted membership."""

    artifact_type: Literal["pair_decision_trace"]
    contract_version: Literal["exact-explain/1.0"]
    run_id: str
    pair_id: str
    source: EntityRef
    target: EntityRef
    events: list[StageEvent]
    saved_alignment_member: bool | None
    relation: (
        Literal["equivalent", "source_narrower", "source_broader", "not_equivalent", "unresolved"]
        | None
    )
    relation_basis: Literal[
        "default_convention", "graph_closure_with_anchors", "recorded_other", "unavailable"
    ]
    assumption_refs: list[str]
    fixture_provenance: str


class HierarchyPage(Page[dict[str, Any]]):
    """Bounded topology preserving edges to already returned nodes."""

    nodes: list[EntityRef] = Field(default_factory=list)
    basis: str | None = None
    node_budget_reached: bool = False


class GenerationManifest(ResourceModel):
    """Frozen generation dependencies and actual provider identity, excluding secrets."""

    ontology_context_hashes: list[str]
    visibility_policy_hash: str
    packet_hash: str
    prompt_hash: str
    output_schema: str
    requested_model: str
    returned_model: str | None
    provider: str | None
    parameters_hash: str
    language: str
    response_hash: str | None
    status: Literal["pending", "dispatched", "response_saved", "validated", "failed", "ambiguous"]


class GeneratedExplanationResponse(ResourceModel):
    """Prepared generated artifact, with grounding and generation failure represented separately."""

    artifact_type: Literal["generated_explanation"]
    contract_version: Literal["exact-explain/1.0"]
    explanation_id: str
    task: Literal["entity_profile", "pair_comparison"]
    entities: list[EntityRef]
    claims: list[dict[str, Any]]
    grounding_status: Literal["validated", "unverified", "rejected"]
    manifest: GenerationManifest
    limitations: list[str]
    fixture_provenance: str


class OrdinalRanks(WireModel):
    """Ordinals are separate from scores and retain their ordering/tie provenance."""

    candidate_joint_rank: int | None = Field(default=None, ge=1)
    nil_rank: int | None = Field(default=None, ge=1)
    joint_ordering: str | None = None
    joint_tie_rule: str | None = None
    retrieval_rank: int | None = Field(default=None, ge=1)
    retrieval_ordering: str | None = None
    retrieval_tie_rule: str | None = None
    retrieval_provenance: str | None = None


class EvidenceCounts(WireModel):
    """Available features refer to the declared cache scope, not ontology completeness."""

    side: Literal["source", "target"]
    channel: str
    family: str | None
    scope: str
    available_count: int = Field(ge=0)
    ontology_available_count: int | None = Field(default=None, ge=0)
    eligible_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    selection_limit: int = Field(ge=0)
    omission_reason: str | None

    @model_validator(mode="after")
    def check_counts(self):
        if self.selected_count > self.eligible_count or self.eligible_count > self.available_count:
            raise ValueError("Selected/eligible evidence exceeds available scoped features")
        return self


class VisibleEvidenceCounts(WireModel):
    """Physical policy export reveals only its visible evidence count."""

    visible_selected: int = Field(ge=0)
    scope: Literal["prepared_visibility_policy"]


class MembershipProvenance(WireModel):
    """Final membership is supported by the saved alignment, never a score threshold."""

    status: Literal["derived_from_saved_artifacts", "unavailable"]
    artifact_hash: str | None


class Candidate(WireModel):
    """A typed candidate summary without graphs or an unqualified raw scorer payload."""

    pair_id: str
    source: EntityRef
    target: EntityRef
    # Accepted from indexed legacy adapters, but never serialized onto the public wire.
    values: dict[str, str | float | bool] = Field(default_factory=dict, exclude=True)
    scores: list[Score]
    retrieval_channels: list[str]
    ordinal_ranks: OrdinalRanks
    evidence_counts: list[EvidenceCounts] | VisibleEvidenceCounts
    nil: dict[str, str | float | bool]
    saved_alignment_member: bool | None
    membership_provenance: MembershipProvenance
    status: Availability
    decision_status: Availability
    evidence_status: Availability | None = None

    @model_validator(mode="after")
    def check_scores_and_membership(self):
        if any(s.name in {"nil_rank", "candidate_joint_rank", "cand_rank"} for s in self.scores):
            raise ValueError("Ordinal ranks cannot be represented as scores")
        if self.saved_alignment_member is not None and (
            self.membership_provenance.status != "derived_from_saved_artifacts"
            or not self.membership_provenance.artifact_hash
        ):
            raise ValueError("Known final membership requires a saved alignment artifact")
        return self


class FeatureDerivation(ResourceModel):
    """Projection rules remain distinct from the original asserted ontology facts."""

    provider: str
    version: str
    rules: list[str]
    premises: list[str]
    premises_unavailable: bool


class SelectedEvidence(WireModel):
    """A selected matcher feature and its many-to-many links to original ontology facts."""

    evidence_id: str
    channel: str
    side: Literal["source", "target"]
    role: str
    entity: EntityRef
    feature_id: str | None
    fact_ids: list[str]
    source_axiom_refs: list[str]
    axiom_origins: list[dict[str, JsonValue]]
    semantic_terms: dict[str, JsonValue]
    historical_item_alias: str | None
    display: dict[str, JsonValue]
    values: dict[str, float]
    interpretation: Literal[
        "asserted",
        "structurally_derived",
        "reasoner_inferred",
        "projected",
        "matcher_comparison",
        "generated",
    ]
    derivation: FeatureDerivation | None = None
    provenance_status: str
    status: Availability
    reason: str | None

    @model_validator(mode="after")
    def check_fact_links(self):
        if self.status == "available" and (not self.feature_id or not self.fact_ids):
            raise ValueError(
                "Available selected evidence requires semantic and original-fact links"
            )
        return self


class OwlExpression(ResourceModel):
    """Typed OWL constructor; constructor-specific fields remain lossless additive JSON."""

    type: str = Field(min_length=1)


class AxiomOrigin(ResourceModel):
    """Exact native source occurrence, including an optional original source span."""

    document_key: str
    source_sha256: str
    span: dict[str, JsonValue] | None = None


class AxiomInterpretation(WireModel):
    """Explicit assertion/derivation basis and frozen ontology scope."""

    kind: Literal["asserted", "structurally_derived", "reasoner_inferred", "projected"]
    scope: str


class AxiomRendering(WireModel):
    """A faithful display rendering may be incomplete without losing the original AST."""

    status: Availability
    text: str


class AxiomResponse(ResourceModel):
    """Original axiom identity, native structured form, rendering and visibility status."""

    id: str
    fact_id: str
    axiom_id: str
    ontology_version_id: str
    category: str
    interpretation: AxiomInterpretation
    availability: Availability
    ast: OwlExpression
    origins: list[AxiomOrigin]
    original_axiom_digest: str
    original_syntax: str | None
    original_format: str
    original_availability: Availability = "available"
    rendering: AxiomRendering

    @model_validator(mode="after")
    def check_original_identity(self):
        if self.id != self.fact_id or self.id != self.axiom_id:
            raise ValueError("Axiom and fact references must identify the same stored assertion")
        if self.original_syntax is None and self.original_availability == "available":
            raise ValueError("Missing original bytes require an explicit availability state")
        return self
