// Per-case report rules shared by the v2 form and its tests (14 decision 5, 16 B4).
// Pure module (type-only imports) so it runs under Node's test runner.

import type { ConsultationDraftV2, ConsultationMethod } from "./types";

export type ConsultationAnswer = Omit<ConsultationDraftV2, "presentation_id" | "form_version" | "saved_at">;

export const EMPTY_CONSULTATION: ConsultationAnswer = {
  consulted_external_ontologies: null,
  methods: [],
  other_editor: null,
  other_resource: null,
  other_method: null,
  resource_scope: null,
};

/** No clears method details and scope; Yes keeps only names whose method is selected. */
export function normalizeConsultation(answer: ConsultationAnswer): ConsultationAnswer {
  if (answer.consulted_external_ontologies === false) return { ...EMPTY_CONSULTATION, consulted_external_ontologies: false };
  const keep = (method: ConsultationMethod, value: string | null) => (answer.methods.includes(method) && value?.trim() ? value.trim().slice(0, 160) : null);
  return {
    ...answer,
    methods: Array.from(new Set(answer.methods)),
    other_editor: keep("other_editor", answer.other_editor),
    other_resource: keep("other_resource", answer.other_resource),
    other_method: keep("other_method", answer.other_method),
  };
}

/** A final report needs Yes with at least one method, or No. Drafts may be incomplete. */
export function consultationProblem(answer: ConsultationAnswer): string | null {
  if (answer.consulted_external_ontologies === null) return "Please answer Yes or No.";
  if (answer.consulted_external_ontologies && !answer.methods.length) return "Select at least one method you used, or answer No.";
  return null;
}
