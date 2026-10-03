// Wire types for the exact-study/1.0 participant API (exact_inspect/study/models.py). The
// server owns stage, assignment, condition and revision; the client only renders them.

import type { EntityRef } from "@/lib/types";

export type Stage = "welcome" | "setup" | "background" | "practice" | "tutorial" | "case" | "consultation" | "final" | "paused" | "closed" | "completed";
export type ResponseType = "ranked_candidates" | "none_of_these" | "insufficient_evidence";
export type Condition = "explanation" | "ontology_baseline";

export interface Question {
  id: string;
  label: string;
  options: Record<string, string> | null;
  multiple: boolean;
  required: boolean;
  show_if: Record<string, unknown> | null;
  matrix: Record<string, string> | null;
  /** exact-study/2.0: explicit presentation order of option and matrix-row codes (16 B5). */
  option_order?: string[];
  row_order?: string[];
}

export interface Forms {
  version: string;
  background: Question[];
  consultation: Question[];
  final: Question[];
  experience_note?: string;
}

export interface RankingResponse {
  case_id: string;
  presentation_id: string;
  response_type: ResponseType | null;
  ranked_candidate_ids: string[];
  revision: number;
  workflow_state: "draft" | "submitted";
  saved_at: string;
  submitted_at: string | null;
}

export interface QuestionnaireResponse {
  form_version: string;
  answers: Record<string, unknown>;
  submitted: boolean;
  answer_states: Record<string, "answered" | "skipped" | "not_answered">;
  saved_at: string;
}

export interface SetupReceipt {
  protege_installed: boolean;
  source_opened: boolean;
  target_opened: boolean;
  practice_source_located: boolean;
  practice_definition_parents_inspected: boolean;
  protege_version: string | null;
  completed_tutorial_steps: number[];
}

export interface PublicAsset {
  asset_id: string;
  sha256: string;
  size_bytes: number;
  kind: "ontology";
  media_type: string;
  policy_hash?: string;
  // Proposed exact-study/2.0 metadata (16 B2). Absent in v1 publications: roles stay unknown.
  title?: string | null;
  role?: "source" | "target" | "both" | null;
  ontology_version_id?: string | null;
  version_label?: string | null;
  license_note?: string | null;
  information_notice?: string | null;
}

export interface PracticeCase {
  practice_id: string;
  kind: "simple" | "complex" | "partial_ranking" | "none_of_these";
  title: string;
  instructions: string;
  synthetic: true;
  source: { label: string; description: string };
  candidates: { candidate_id: string; label: string; description: string }[];
}

export interface StudyState extends StudyStateV2Fields {
  artifact_type: "study_state";
  study_revision: string;
  session_id: string;
  revision: number;
  stage: Stage;
  assignment_id: string | null;
  current_case_id: string | null;
  current_presentation_id: string | null;
  completed_cases: number;
  assigned_case_count: number;
  ranking: RankingResponse | null;
  consultation: Record<string, unknown> | null;
  consent: { information_version: string; accepted: boolean; acknowledged_at: string } | null;
  /** v1: installation self-report receipt; v2: tool-neutral setup draft/submission. */
  setup: SetupReceipt | SetupV2 | null;
  questionnaires: Record<string, QuestionnaireResponse>;
  information_version: string;
  information_text: string;
  consent_text: string;
  instructions: string;
  setup_instructions: string;
  tutorial_steps: string[];
  practice_cases?: PracticeCase[];
  forms: Forms;
  ontology_resources: PublicAsset[];
  synthetic: boolean;
  gap_recovery: string;
  contract_version?: string;
  /** Proposed exact-study/2.0: event types and scopes the service accepts (16 B7). */
  telemetry?: { event_types?: string[]; scopes?: string[] };
}

export interface StudyCandidate {
  candidate_id: string;
  entity: EntityRef;
  label: string;
  score: number;
  score_meaning: string;
  display_position: number;
}

export interface StudyCase {
  artifact_type: "study_case";
  study_revision: string;
  case_id: string;
  presentation_id: string;
  condition: Condition;
  source: EntityRef;
  source_label: string;
  candidates: StudyCandidate[];
  ontology_resource_ids: string[];
  package_version: string;
  explanation_refs: string[];
  /** Proposed exact-study/2.0: participant-safe workspace scope for the active explanation case (16 B1). */
  workspace?: { scope_id: string } | null;
}

export interface FactOrigin {
  document_sha256: string;
  axiom_id: string | null;
  source_span: { start: number; end: number; unit: string } | null;
  provenance_status: string;
}

export interface OriginalFact {
  fact_id: string;
  subject: EntityRef;
  predicate_iri: string | null;
  category: string;
  value: {
    term_type: "literal" | "iri" | "expression_ref";
    expression_id?: string | null;
    lexical_form?: string | null;
    datatype?: string | null;
    language?: string | null;
    iri?: string | null;
    ast?: Record<string, unknown> | null;
  };
  qualifiers?: unknown[];
  axiom_ref: string;
  document_sha256?: string | null;
  origins?: FactOrigin[];
  interpretation: "asserted" | "structurally_derived" | "reasoner_inferred" | "projected" | string;
  premises?: string[];
  derivation?: string | null;
}

export interface GroundedClaim {
  claim_id: string;
  text: string;
  fact_ids: string[];
  category: string;
  grounding: "exact_extract" | "semantic_template";
  scoped_entities: EntityRef[];
  packet_fact_ids?: string[];
  packet_fact_subjects?: EntityRef[];
  reviewer_receipt?: string | null;
  generation_manifest_sha256?: string;
}

export interface StudyHierarchyEdge {
  child: EntityRef;
  parent: EntityRef;
  basis: "literal_asserted" | "structural_navigation" | "reasoner_inferred";
  fact_ids: string[];
  derivation?: string | null;
}

export interface StudyEvidenceLink {
  evidence_id: string;
  candidate_id: string;
  fact_ids: string[];
  channel: string;
  role: "source" | "target" | "comparison";
  interpretation: "projected" | "matcher_comparison" | string;
  scores?: { name: string; value: number; meaning?: string }[];
  saved_value?: number | null;
  value_meaning?: string | null;
  status: "available" | "not_exported" | "unresolved" | string;
}

export interface ExplanationResource {
  artifact_type: "study_explanation";
  contract_version?: string;
  policy_hash?: string;
  entities: EntityRef[];
  facts: OriginalFact[];
  referenced_labels?: OriginalFact[];
  referenced_labels_truncated?: boolean;
  entity_profiles: GroundedClaim[];
  pair_comparison: GroundedClaim[];
  hierarchy: StudyHierarchyEdge[];
  evidence: StudyEvidenceLink[];
  limitations: string[];
  capabilities: Record<string, string>;
}

/** Event types the exact-study/1.0 service accepts. */
export const V1_EVENT_TYPES = [
  "case_ready",
  "candidate_inspected",
  "rank_add",
  "rank_remove",
  "rank_move",
  "keep_initial_order",
  "response_type_change",
  "hierarchy_expand",
  "hierarchy_collapse",
  "definition_open",
  "axiom_open",
  "evidence_open",
  "table_open",
  "graph_open",
  "comparison_open",
  "graph_zoom",
  "graph_fit",
  "external_resource_link",
  "pause",
  "resume",
  "visibility",
  "submit",
  "revision",
] as const;

/** Proposed exact-study/2.0 additions (16 B7); sent only when the service declares them. */
export const V2_EVENT_TYPES = ["tab_open", "hierarchy_navigate", "search_select", "graph_reset", "graph_edge_open", "copy_iri", "resources_open", "help_open"] as const;

export type EventType = (typeof V1_EVENT_TYPES)[number] | (typeof V2_EVENT_TYPES)[number];

// ---------------------------------------------------------------------------------------
// Proposed exact-study/2.0 participant contract (specs 14–16). These are the frontend's
// data needs for B0, not implemented backend routes; a v1 service never sends them.
// ---------------------------------------------------------------------------------------

export type ResourceAccess = "not_checked" | "available" | "needs_help";

export interface SetupV2 {
  setup_version: string;
  instructions_acknowledged: boolean;
  external_inspection_optional_understood: boolean;
  resource_access: ResourceAccess;
  /** Optional, descriptive only: methods the participant is familiar with or expects to use. */
  familiar_methods: string[];
  submitted: boolean;
  saved_at?: string;
}

export type RequirementAction =
  | "inspect_other_candidate"
  | "return_to_candidate"
  | "search_entity"
  | "navigate_parent"
  | "navigate_child"
  | "return_to_compared"
  | "open_citation"
  | "open_original_axiom"
  | "locate_in_evidence_list"
  | "inspect_graph_or_list"
  | "change_graph_view"
  | "rank_with_details_open"
  | "add_rank"
  | "move_rank"
  | "remove_rank"
  | "undo_rank"
  | "keep_initial_order"
  | "check_partial_ranking"
  | "choose_none"
  | "choose_insufficient"
  | "copy_iri"
  | "locate_downloads"
  | "report_multiple_methods"
  | "report_no_methods";

export interface LessonRequirement {
  requirement_id: string;
  action: RequirementAction;
  label: string;
  /** An accessible equivalent satisfies the same requirement (e.g. list instead of graph). */
  alternatives?: RequirementAction[];
}

export interface TutorialLesson {
  lesson_id: string;
  title: string;
  /** Short instructions tied to actual controls. */
  steps: string[];
  view: "explanation" | "baseline";
  requirements: LessonRequirement[];
  optional?: boolean;
}

export interface AssessmentOption {
  code: string;
  label: string;
}

export interface AssessmentItem {
  question_id: string;
  title: string;
  prompt: string;
  kind: "single" | "multiple" | "match" | "match_and_single";
  options?: AssessmentOption[];
  rows?: { row_id: string; label: string }[];
  row_options?: AssessmentOption[];
  part_b?: { prompt: string; options: AssessmentOption[] };
  lesson_id: string;
}

export interface AssessmentResponse {
  choice?: string;
  choices?: string[];
  matches?: Record<string, string>;
  part_b?: string;
}

export interface TutorialCase {
  source: EntityRef;
  source_label: string;
  candidates: StudyCandidate[];
  explanation_refs: string[];
  ontology_resources: PublicAsset[];
}

/** Participant-visible tutorial definition; grading rules stay on the server. */
export interface TutorialPublic {
  tutorial_id: string;
  version: string;
  hash: string;
  synthetic: true;
  intro: string;
  lessons: TutorialLesson[];
  assessment: AssessmentItem[];
  case: TutorialCase;
}

export interface AttemptReceipt {
  attempt_id: string;
  question_id: string;
  response: AssessmentResponse;
  correct: boolean;
  feedback: string;
  revisit_lesson_id: string | null;
  submitted_at: string;
}

export interface TutorialProgress {
  tutorial_version: string;
  current_lesson_id: string | null;
  completed_requirements: string[];
  practice: Record<string, { response_type: ResponseType | null; ranked_candidate_ids: string[] }>;
  assessment_drafts: Record<string, AssessmentResponse>;
  attempts: AttemptReceipt[];
  passed_items: string[];
  outstanding: string[];
  help_opened: number;
  completed_at: string | null;
}

export type ConsultationMethod = "protege" | "other_editor" | "plain_files" | "other_resource" | "queries_scripts" | "reasoner" | "other_method";
export type ResourceScope = "supplied_only" | "different_or_additional" | "unsure";

export interface ConsultationDraftV2 {
  presentation_id: string;
  form_version: string;
  consulted_external_ontologies: boolean | null;
  methods: ConsultationMethod[];
  other_editor: string | null;
  other_resource: string | null;
  other_method: string | null;
  resource_scope: ResourceScope | null;
  saved_at?: string;
}

export interface ConsultationReceiptV2 extends ConsultationDraftV2 {
  case_id: string;
  submitted_at: string;
}

/** Fields a v2 StudyState adds; all optional so a v1 state type-checks unchanged. */
export interface StudyStateV2Fields {
  tutorial?: TutorialPublic | null;
  tutorial_progress?: TutorialProgress | null;
  consultation_draft?: ConsultationDraftV2 | null;
  /** The participant's previous submitted consultation, offered only for explicit reuse. */
  previous_consultation?: ConsultationReceiptV2 | null;
  protocol_versions?: Record<string, string>;
}

export function isSetupV2(setup: StudyState["setup"]): setup is SetupV2 {
  return Boolean(setup && "setup_version" in setup);
}

export function legacySetup(state: StudyState): SetupReceipt | null {
  return state.setup && !isSetupV2(state.setup) ? state.setup : null;
}
