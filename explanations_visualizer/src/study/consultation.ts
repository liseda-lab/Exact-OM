// Per-case report rules shared by the v2/v3 form and its tests (14 decision 5, 16 B4, 22 B06).
// Pure module (type-only imports) so it runs under Node's test runner.

import type { ConsultationDraftV2, ConsultationMethod } from "./types";

export type ConsultationAnswer = Omit<ConsultationDraftV2, "presentation_id" | "form_version" | "consultation_version" | "saved_at">;

export const EMPTY_CONSULTATION: ConsultationAnswer = {
  consulted_external_ontologies: null,
  methods: [],
  other_editor: null,
  other_resource: null,
  other_method: null,
  resource_scope: null,
};

/** study-methods/3: the only codes a consultation/3 report can carry. */
export const METHODS_V3: ConsultationMethod[] = ["protege", "plain_files", "other_method"];

/**
 * No clears method details and scope; Yes keeps only names whose method is selected.
 * consultation/3 (`v3`) carries exactly its own fields: no editor/resource names.
 */
export function normalizeConsultation(answer: ConsultationAnswer, v3 = false): ConsultationAnswer {
  const keep = (method: ConsultationMethod, value: string | null | undefined) => (answer.methods.includes(method) && value?.trim() ? value.trim().slice(0, 160) : null);
  if (v3) {
    if (answer.consulted_external_ontologies !== true) return { consulted_external_ontologies: answer.consulted_external_ontologies, methods: [], other_method: null, resource_scope: null };
    const methods = Array.from(new Set(answer.methods)).filter((method) => METHODS_V3.includes(method));
    return { consulted_external_ontologies: true, methods, other_method: methods.includes("other_method") ? answer.other_method?.trim().slice(0, 160) || null : null, resource_scope: answer.resource_scope };
  }
  if (answer.consulted_external_ontologies === false) return { ...EMPTY_CONSULTATION, consulted_external_ontologies: false };
  return {
    ...answer,
    methods: Array.from(new Set(answer.methods)),
    other_editor: keep("other_editor", answer.other_editor),
    other_resource: keep("other_resource", answer.other_resource),
    other_method: keep("other_method", answer.other_method),
  };
}

/** A final report needs Yes with at least one method, or No. A selected v3 Other needs a name. Drafts may be incomplete. */
export function consultationProblem(answer: ConsultationAnswer, v3 = false): { field: "consulted" | "methods" | "other_method"; message: string } | null {
  if (answer.consulted_external_ontologies === null) return { field: "consulted", message: "Please answer Yes or No." };
  if (answer.consulted_external_ontologies && !answer.methods.length) return { field: "methods", message: "Select at least one method you used, or answer No." };
  if (v3 && answer.consulted_external_ontologies && answer.methods.includes("other_method") && !answer.other_method?.trim()) return { field: "other_method", message: "Name the other methods you used, or clear Other method." };
  return null;
}
