// Wire types for the exact-explain/1.0 read API. They mirror exact_inspect/contracts.py and
// exact_inspect/models.py; the backend owns interpretation, the frontend only presents it.

export type EntityKind = "class" | "object_property" | "data_property" | "individual";

export type Availability =
  | "available"
  | "absent_in_scope"
  | "not_exported"
  | "unavailable_source"
  | "unresolved_import"
  | "unsupported"
  | "filtered"
  | "partial"
  | "failed"
  | "not_requested"
  | "not_run";

export interface EntityRef {
  ontology_version_id: string;
  iri: string;
  kind: EntityKind;
}

export interface Scope {
  ontology_version_id: string;
  context_revision: string;
  basis: string;
  visibility_policy_hash: string;
  filter_id: string;
}

export interface Page<T> {
  items: T[];
  returned_count: number;
  total_count: number | null;
  next_cursor: string | null;
  truncated: boolean;
  scope: Scope;
  status: Availability;
  reason: string | null;
}

export interface Health {
  status: "available" | "not_requested";
  contract_version: string;
  profile: "local_app" | "public_demo";
  package_id: string | null;
  capabilities: Record<string, string>;
}

export interface OntologyMeta {
  ontology_version_id: string;
  name: string | null;
  scope: string | null;
  capabilities: {
    categories?: string[];
    kinds?: string[];
    provider?: string;
    reasoner_inferred?: { status: string; reason?: string } | string;
    source_spans?: boolean;
    typed_expressions?: boolean;
  };
  completeness: {
    axiom_count?: number;
    entity_count?: number;
    category_counts?: Record<string, number>;
    domain_completeness?: string;
    extraction?: string;
    imports_complete?: boolean;
    scope?: string;
  };
  source_root_sha256: string | null;
  source_derivation: {
    status: string;
    receipt_hash: string | null;
    original_source_sha256: string | null;
  } | null;
}

export interface RunSummary {
  run_id: string;
  revision: string | null;
  source_ontology_version_id: string;
  target_ontology_version_id: string;
  status: string | null;
  counts: Record<string, number | string>;
}

export interface PreferredLabel {
  status: Availability | string;
  value: string | null;
  language?: string | null;
  fact_id?: string;
  language_fallback?: boolean;
}

export interface LabelItem {
  iri?: string;
  entity: EntityRef | null;
  preferred_label: PreferredLabel;
}

export interface SearchItem {
  entity: EntityRef;
  preferred_label: PreferredLabel;
  alignment_eligible?: boolean | null;
  alignment_eligibility_status?: string;
}

export type Term =
  | { term_type: "iri"; iri: string }
  | { term_type: "literal"; lexical_form: string; datatype: string | null; language: string | null }
  | { term_type: "expression_ref"; expression_id: string };

export interface Origin {
  document_sha256: string;
  axiom_id: string | null;
  source_span: { start: number; end: number; unit: string } | null;
  provenance_status: string;
}

export interface Fact {
  id?: string;
  fact_id: string;
  subject: EntityRef;
  predicate_iri: string | null;
  value: Term | null;
  axiom_ref: string | null;
  axiom_id?: string;
  interpretation: string;
  origins: Origin[];
  premise_ids: string[];
  derivation_id: string | null;
  category?: string;
  qualifiers?: unknown[];
  synonym_scope?: string | null;
  availability?: Availability;
  reason?: string;
  /** The hierarchy edge a parent fact projects to; `id` is the edge identity used by continuations. */
  hierarchy_projection?: { id?: string; child: EntityRef; parent: EntityRef; basis: string };
}

export interface EntityContext {
  artifact_type: "entity_context";
  entity: EntityRef;
  ontology_version_id: string;
  preferred_label: PreferredLabel;
  labels: { text: string; language: string | null; predicate_iri: string }[];
  definitions: Page<Fact>;
  synonyms: Page<Fact>;
  parents: Page<Fact>;
  categories: Record<string, Page<Fact>>;
  alignment_eligible: boolean | null;
  alignment_eligibility_status: string;
  capabilities: Record<string, string>;
  completeness: {
    scope: string;
    extraction: string;
    imports_complete: boolean;
    domain_completeness: string;
  };
  context_scope: Scope;
}

export interface HierarchyEdge {
  id: string;
  axiom_id: string;
  basis: string;
  category: string;
  child: EntityRef;
  parent: EntityRef;
  relation: string;
  interpretation: { kind: string; rule?: string | null; provider?: string; scope?: string };
}

export interface HierarchyPage extends Page<HierarchyEdge> {
  nodes: EntityRef[];
  basis: string | null;
  node_budget_reached: boolean;
  /** Partial ELK pages: the focused entity's equivalence group (null when not represented). */
  entity_semantics?: { equivalence_group_id: string; equivalent_entity_count: number; is_top: boolean; is_bottom: boolean } | null;
  navigation_contract?: string;
}

/** A typed entry point (exact-hierarchy/1, exact-partial-hierarchy/1). */
export interface EntryPointItem {
  entity: EntityRef;
  entry_kind: "named_root" | "component_entry" | "entity_listing" | string;
  component_size?: number | null;
  preferred_label?: { status: string; value: string | null; language?: string | null } | null;
}

export interface EntryPointPage extends Page<EntryPointItem> {
  basis: string;
  default_entity: EntityRef | null;
  navigation_contract?: string;
}

/** A recorded matcher comparison (exact-evidence-metadata/1); never an original ontology fact. */
export interface EvidenceBridge {
  bridge_id: string;
  level: "channel_comparison" | "feature_correspondence";
  status: "recorded" | "withheld";
  channel: string;
  source_channel?: string | null;
  target_channel?: string | null;
  source_family?: string | null;
  target_family?: string | null;
  comparison_basis?: string | null;
  recorded_score?: number | null;
  source: EntityRef;
  target: EntityRef;
  source_evidence_ids: string[];
  target_evidence_ids: string[];
  source_fact_ids: string[];
  target_fact_ids: string[];
  artifact_refs: string[];
  interpretation: "matcher_comparison";
}

export interface PairEvidenceMetadata {
  version: "exact-evidence-metadata/1";
  status: "selected" | "none_selected" | "not_exported" | "request_failed";
  counts: { selected: number; resolved: number; unresolved: number; not_exported: number; withheld: number; graph_supported: number; graph_unsupported: number; graph_not_recorded: number };
  bridge_status: "recorded" | "not_recorded" | "withheld";
  bridges: EvidenceBridge[];
  comparison_resolution?: { recorded: number; resolved: number; unresolved: number; ambiguous: number; unsupported: number } | null;
  scope?: string;
}

export interface Score {
  name: string;
  stage: string;
  value: number;
  meaning: string;
  range: Record<string, number> | null;
  calibration_status: "not_established" | "validated" | "not_applicable";
  calibration_artifact: string | null;
}

export interface OrdinalRanks {
  candidate_joint_rank: number | null;
  nil_rank: number | null;
  joint_ordering: string | null;
  joint_tie_rule: string | null;
  retrieval_rank: number | null;
  retrieval_ordering: string | null;
  retrieval_tie_rule: string | null;
  retrieval_provenance: string | null;
}

export interface Candidate {
  pair_id: string;
  source: EntityRef;
  target: EntityRef;
  scores: Score[];
  retrieval_channels: string[];
  ordinal_ranks: OrdinalRanks;
  evidence_counts: unknown;
  nil: Record<string, string | number | boolean>;
  saved_alignment_member: boolean | null;
  membership_provenance: { status: string; artifact_hash: string | null };
  status: Availability;
  decision_status: Availability;
  evidence_status?: Availability | null;
}

export interface SourceSummary {
  entity_id: string;
  entity: EntityRef;
  source_decision?: Record<string, unknown>;
}

export interface StageEvent {
  stage: string;
  status: "completed" | "not_run" | "not_recorded" | "failed";
  outcome: string;
  reason_code: string;
  scores: Score[];
  competitor_pair_ids: string[];
  artifact_refs: string[];
  implementation_id: string | null;
  provenance_status: string;
}

export interface PairTrace {
  artifact_type: "pair_decision_trace";
  run_id: string;
  pair_id: string;
  source: EntityRef;
  target: EntityRef;
  events: StageEvent[];
  saved_alignment_member: boolean | null;
  relation: "equivalent" | "source_narrower" | "source_broader" | "not_equivalent" | "unresolved" | null;
  relation_basis: "default_convention" | "graph_closure_with_anchors" | "recorded_other" | "unavailable";
  assumption_refs: string[];
  /** Recorded completeness and matcher comparisons for the pair's selected evidence. */
  evidence_metadata?: PairEvidenceMetadata | null;
}

export interface SelectedEvidence {
  evidence_id: string;
  channel: string;
  side: "source" | "target";
  role: string;
  entity: EntityRef;
  feature_id: string | null;
  fact_ids: string[];
  source_axiom_refs: string[];
  semantic_terms: Record<string, unknown>;
  historical_item_alias: string | null;
  display: Record<string, unknown>;
  values: Record<string, number>;
  interpretation: string;
  provenance_status: string;
  status: Availability;
  reason: string | null;
}

export interface Claim {
  claim_id?: string;
  text: string;
  fact_ids: string[];
  category: string;
}

export interface GeneratedExplanation {
  artifact_type: "generated_explanation";
  explanation_id: string;
  task: "entity_profile" | "pair_comparison";
  entities: EntityRef[];
  claims: Claim[];
  grounding_status: "validated" | "unverified" | "rejected";
  limitations: string[];
  manifest: {
    requested_model: string;
    returned_model: string | null;
    provider: string | null;
    status: string;
    language: string;
  };
  /** exact-description/2: what kind of preparation produced the text and how it is supported. */
  preparation_kind?: "grounded_description" | "fact_extract" | "factual_fallback" | null;
  support_status?: "typed_template_support" | "exact_support" | "fallback" | "independently_reviewed" | null;
  support_reasons?: string[];
  /** exact-description-presentation/1: how to show it; claim sides are relative to `entities`. */
  presentation?: DescriptionPresentation | null;
  independent_review?: { reviewer_kind?: string; reviewer_id?: string; contract_version?: string } | null;
  /** The admitted facts the text was prepared from; only cited labels of referenced entities are read. */
  support_packet?: { facts?: { fact_id: string; subject?: EntityRef }[]; selection?: { referenced_labels?: { fact_id: string; subject?: EntityRef }[] } } | null;
}

export type PresentationKind = "reviewed_profile" | "necessary_conditions" | "available_facts" | "reviewed_comparison" | "paired_context" | "source_only_profile" | "target_only_profile" | "unavailable";

export interface DescriptionPresentation {
  contract_version: "exact-description-presentation/1";
  kind: PresentationKind;
  /** "source" is the artifact's first entity and "target" its second, whatever the workspace column. */
  covered_sides: ("source" | "target")[];
  claim_sides: Record<string, ("source" | "target")[]>;
  relation_asserted: false;
  scope_note: string;
  render_revision: string | null;
  review_receipt_sha256: string | null;
}

export interface ExplanationSummary {
  explanation_id: string;
  task: "entity_profile" | "pair_comparison";
  entities: EntityRef[];
  grounding_status: string;
  generation_status: string;
}

export interface OwlNode {
  type: string;
  [key: string]: unknown;
}

export interface Axiom {
  id: string;
  fact_id: string;
  axiom_id: string;
  ontology_version_id: string;
  category: string;
  interpretation: { kind: string; scope: string };
  availability: Availability;
  ast: OwlNode;
  origins?: { document_key: string; source_sha256: string; span: Record<string, unknown> | null }[];
  original_axiom_digest?: string;
  /** Canonical stored bytes (base64); never shown as human-readable syntax. */
  original_syntax?: string | null;
  original_availability: Availability;
  original_format: string;
  rendering: { status: Availability; text: string };
}

export interface BundleListItem {
  package_id: string;
  capabilities: Record<string, string>;
  contract_version: string;
  audience: "local" | "development_demo";
  policy_id: string;
  counts: { ontologies: number; runs: number; explanations: number; jobs: number; artifacts: number };
  artifact_bytes: number;
  owned_library_copy: boolean;
}

export interface ImportJob {
  job_id: string;
  status: "pending" | "uploading" | "validating" | "available" | "failed" | "cancelled" | "interrupted";
  received_bytes: number;
  cancel_requested: boolean;
  package_id: string | null;
  reason: string | null;
}
