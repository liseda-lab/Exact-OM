// Declared presentation order of the legacy exact-study/1.0 questionnaires, copied from
// exact_inspect/study/forms.py (Python declaration order). Frozen v1 forms are stored with
// canonical sorted keys, so their option objects arrive alphabetically; this versioned table
// restores the declared order for display only. Answers and codes are never changed. The
// backend test tests/explanation_frontend_form_order_test.py keeps it equal to forms.py.
// v2 publications declare `option_order`/`row_order` explicitly and do not use this table.

export const LEGACY_FORM_ORDER: Record<string, Record<string, { options?: string[]; rows?: string[] }>> = /* json */ {
  "exact-study-forms/1": {
    "cs_experience_years": {"options": ["none", "less_than_1", "1_2", "3_4", "5_9", "10_plus", "prefer_not_to_say"]},
    "health_bio_experience_years": {"options": ["none", "less_than_1", "1_2", "3_4", "5_9", "10_plus", "prefer_not_to_say"]},
    "semantic_web_experience": {"options": ["yes", "no", "prefer_not_to_say"]},
    "clinical_experience": {"options": ["yes", "no", "prefer_not_to_say"]},
    "bioinformatics_experience": {"options": ["yes", "no", "prefer_not_to_say"]},
    "doid_familiarity": {"options": ["unheard", "heard", "used", "prefer_not_to_say"]},
    "doid_contexts": {"options": ["research", "clinical", "teaching", "curation", "other", "prefer_not_to_say"]},
    "ncit_familiarity": {"options": ["unheard", "heard", "used", "prefer_not_to_say"]},
    "ncit_contexts": {"options": ["research", "clinical", "teaching", "curation", "other", "prefer_not_to_say"]},
    "roles": {"options": ["undergraduate", "postgraduate", "researcher", "senior_researcher", "clinician", "other", "prefer_not_to_say"]},
    "correspondence_confidence": {"options": ["not_at_all", "slightly", "moderately", "highly", "prefer_not_to_say"]},
    "ontology_activities": {"options": ["annotated", "software", "curation", "developed", "other", "never_used", "prefer_not_to_say"]},
    "protege_experience": {"options": ["never", "occasionally", "regularly", "prefer_not_to_say"]},
    "consulted_external_ontologies": {"options": ["true", "false"]},
    "methods": {"options": ["protege", "other_editor", "plain_files", "other_resource"]},
    "component_usefulness": {"options": ["not_helpful", "slightly", "moderately", "very", "cannot_judge"], "rows": ["original_context", "entity_description", "hierarchy", "evidence_table", "evidence_graph", "pair_comparison"]},
    "most_helpful_components": {"options": ["original_context", "entity_description", "hierarchy", "evidence_table", "evidence_graph", "pair_comparison", "all_equally", "none_helpful", "cannot_judge"]},
    "workflow_preference": {"options": ["explanation", "ontology_baseline", "no_preference", "cannot_judge"]},
    "mental_effort": {"options": ["very_low", "low", "moderate", "high", "very_high", "cannot_judge"], "rows": ["explanation", "ontology_baseline"]}
  }
};
