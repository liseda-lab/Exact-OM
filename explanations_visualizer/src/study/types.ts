// Wire types for the exact-study/1.0 participant API (exact_inspect/study/models.py). The
// server owns stage, assignment, condition and revision; the client only renders them.

import type { EntityRef } from "@/lib/types";

export type Stage = "welcome" | "setup" | "background" | "practice" | "case" | "consultation" | "final" | "paused" | "closed" | "completed";
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
}

export interface StudyState {
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
  setup: SetupReceipt | null;
  questionnaires: Record<string, QuestionnaireResponse>;
  information_version: string;
  information_text: string;
  consent_text: string;
  instructions: string;
  setup_instructions: string;
  tutorial_steps: string[];
  forms: Forms;
  ontology_resources: PublicAsset[];
  synthetic: boolean;
  gap_recovery: string;
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
}

export interface OriginalFact {
  fact_id: string;
  subject: EntityRef;
  predicate_iri: string | null;
  category: string;
  value: {
    term_type: "literal" | "iri" | "expression_ref";
    lexical_form?: string | null;
    language?: string | null;
    iri?: string | null;
    ast?: Record<string, unknown> | null;
  };
  axiom_ref: string;
  interpretation: string;
}

export interface GroundedClaim {
  claim_id: string;
  text: string;
  fact_ids: string[];
  category: string;
  grounding: "exact_extract" | "semantic_template";
  scoped_entities: EntityRef[];
}

export interface StudyHierarchyEdge {
  child: EntityRef;
  parent: EntityRef;
  basis: string;
  fact_ids: string[];
}

export interface StudyEvidenceLink {
  evidence_id: string;
  candidate_id: string;
  fact_ids: string[];
  channel: string;
  role: "source" | "target" | "comparison";
  interpretation: string;
  status: string;
}

export interface ExplanationResource {
  artifact_type: "study_explanation";
  entities: EntityRef[];
  facts: OriginalFact[];
  entity_profiles: GroundedClaim[];
  pair_comparison: GroundedClaim[];
  hierarchy: StudyHierarchyEdge[];
  evidence: StudyEvidenceLink[];
  limitations: string[];
  capabilities: Record<string, string>;
}

export type EventType =
  | "case_ready"
  | "candidate_inspected"
  | "rank_add"
  | "rank_remove"
  | "rank_move"
  | "keep_initial_order"
  | "response_type_change"
  | "hierarchy_expand"
  | "hierarchy_collapse"
  | "definition_open"
  | "axiom_open"
  | "evidence_open"
  | "table_open"
  | "graph_open"
  | "comparison_open"
  | "graph_zoom"
  | "graph_fit"
  | "external_resource_link"
  | "pause"
  | "resume"
  | "visibility"
  | "submit"
  | "revision";
