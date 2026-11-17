// Plain words for the backend's interpretation of a record. The words state where a
// record comes from; they never upgrade a derived or projected record to an assertion.

export const INTERPRETATION_TEXT: Record<string, string> = {
  asserted: "Asserted in the ontology",
  structurally_derived: "Structurally derived from an asserted axiom",
  reasoner_inferred: "Inferred by a reasoner",
  projected: "Projected by Exact from an original fact",
  matcher_comparison: "Recorded matcher comparison",
};

export function interpretationText(kind: string | null | undefined): string {
  if (!kind) return "Interpretation not recorded";
  return INTERPRETATION_TEXT[kind] ?? kind.replace(/_/g, " ");
}
