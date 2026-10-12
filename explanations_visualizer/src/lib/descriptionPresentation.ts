// How a prepared description or comparison is presented (23 C03, 21 F05). The backend declares
// what each artifact is (`exact-description/2` preparation and support, and
// `exact-description-presentation/1`); the frontend never infers it from category names or
// strips text. Reviewed prose, typed necessary conditions, quoted excerpts, one-sided
// information and fallback facts keep distinct labels. Pure module for Node's test runner.

import type { EntityRef, GeneratedExplanation } from "./types";

export type Side = "source" | "target";
export type Tone = "generated" | "derived" | "original";

export interface PresentationView {
  title: string;
  /** Compact provenance indicator. */
  tag: { text: string; tone: Tone };
  /** One line saying what this is, when it is not plain reviewed prose. */
  intro: string | null;
  /** Comparison claims grouped by what they assert (reviewed comparisons and legacy artifacts). */
  grouped: boolean;
  /** Claims laid out per side (paired excerpts, one-sided information). */
  bySide: boolean;
  /** The artifact declares that nothing usable was prepared. */
  unavailable: boolean;
  /** Claim text is exact original wording (quotation styling; "original" labels apply). */
  quote: boolean;
}

/**
 * Where an artifact's claim wording comes from, from its declared preparation and support only
 * (never from claim categories or the layout): exact original text, a fixed reading of original
 * axioms, or generated prose. Reviewed generated prose is never presented as a quotation.
 */
export function textOrigin(explanation: GeneratedExplanation): "original" | "derived" | "generated" {
  if (explanation.support_status === "independently_reviewed") return "generated";
  if (explanation.preparation_kind === "fact_extract" || explanation.preparation_kind === "factual_fallback") return "original";
  if (explanation.support_status === "exact_support" || explanation.support_status === "fallback") return "original";
  if (explanation.support_status === "typed_template_support") return "derived";
  return "generated";
}

function tagFor(explanation: GeneratedExplanation): PresentationView["tag"] {
  switch (explanation.support_status) {
    case "independently_reviewed":
      return { text: "Generated · reviewed", tone: "generated" };
    case "typed_template_support":
      // A typed rendering either reads necessary conditions from axioms or quotes a definition.
      return explanation.preparation_kind === "fact_extract" ? { text: "Original excerpts", tone: "original" } : { text: "From original axioms", tone: "derived" };
    case "exact_support":
      return { text: "Original quotes", tone: "original" };
    case "fallback":
      return { text: "Original facts", tone: "original" };
    default:
      return { text: "Generated", tone: "generated" };
  }
}

const sideWord = (side: Side) => (side === "source" ? "source" : "target");

/** The view of an entity profile. */
export function profileView(explanation: GeneratedExplanation): PresentationView {
  const kind = explanation.presentation?.kind ?? null;
  const tag = tagFor(explanation);
  const base = { tag, grouped: false, bySide: false, unavailable: false, quote: textOrigin(explanation) === "original" };
  switch (kind) {
    case "reviewed_profile":
      return { ...base, title: "Description", intro: null };
    case "necessary_conditions":
      return { ...base, title: "Necessary conditions", intro: "Read from this class's original axioms; a summary of what they require, not a definition." };
    case "available_facts":
      return { ...base, title: "Available facts", intro: "Original statements recorded for this entity; no description was written from them." };
    case "unavailable":
      return { ...base, title: "Description", intro: null, unavailable: true };
    default:
      // Historical artifacts without presentation metadata keep their previous presentation.
      return { ...base, title: explanation.preparation_kind === "fact_extract" || explanation.preparation_kind === "factual_fallback" ? "Available facts" : "Generated description", intro: null };
  }
}

/** The view of a pair comparison; `coveredWorkspaceSides` says which concept one-sided text describes. */
export function comparisonView(explanation: GeneratedExplanation, coveredWorkspaceSides: Side[]): PresentationView {
  const kind = explanation.presentation?.kind ?? null;
  const tag = tagFor(explanation);
  const origin = textOrigin(explanation);
  const base = { tag, grouped: false, bySide: false, unavailable: false, quote: origin === "original" };
  switch (kind) {
    case "reviewed_comparison":
      return { ...base, title: "How the two descriptions compare", intro: null, grouped: true };
    case "paired_context":
      // The layout pairs the two concepts; what the text is comes from its preparation (R1).
      if (origin === "original") return { ...base, title: "Definitions side by side", intro: "Quoted from each ontology. Showing them together does not say whether they match.", bySide: true };
      if (origin === "derived") return { ...base, title: "Original facts side by side", intro: "Fixed readings of each concept's original axioms. Showing them together does not say whether they match.", bySide: true };
      return { ...base, title: "Descriptions side by side", intro: "Generated descriptions of each concept, checked against the cited records. Showing them together does not say whether they match.", bySide: true };
    case "source_only_profile":
    case "target_only_profile": {
      const side = coveredWorkspaceSides[0];
      return { ...base, title: "One-sided information", intro: side ? `Only the ${sideWord(side)} concept is described here; this is not a difference between the two.` : "Only one concept is described here; this is not a difference between the two.", bySide: true };
    }
    case "necessary_conditions":
    case "available_facts":
      return { ...base, title: kind === "necessary_conditions" ? "Necessary conditions" : "Available facts", intro: null, bySide: true };
    case "unavailable":
      return { ...base, title: "How the two descriptions compare", intro: null, unavailable: true };
    default:
      return { ...base, title: "How the two descriptions compare", intro: null, grouped: true };
  }
}

const same = (a: EntityRef, b: EntityRef) => a.ontology_version_id === b.ontology_version_id && a.iri === b.iri && a.kind === b.kind;

/**
 * Workspace side of each artifact role. An artifact's "source" is its first entity and "target"
 * its second, which may sit in either workspace column; roles are resolved by exact identity
 * (then by ontology when the columns are different ontologies), never by position.
 */
export function roleSides(explanation: GeneratedExplanation, workspace: { source: EntityRef; target: EntityRef }, sideOfOntology: (ontology: string) => Side | null): Record<Side, Side | null> {
  const resolve = (entity: EntityRef | undefined): Side | null => {
    if (!entity) return null;
    if (same(entity, workspace.source)) return "source";
    if (same(entity, workspace.target)) return "target";
    return sideOfOntology(entity.ontology_version_id);
  };
  return { source: resolve(explanation.entities[0]), target: resolve(explanation.entities[1]) };
}

/** Workspace sides each claim speaks about, from the declared claim sides. */
export function claimSides(explanation: GeneratedExplanation, roles: Record<Side, Side | null>): Record<string, Side[]> {
  const result: Record<string, Side[]> = {};
  for (const [claimId, sides] of Object.entries(explanation.presentation?.claim_sides ?? {})) {
    result[claimId] = Array.from(new Set(sides.map((role) => roles[role]).filter((side): side is Side => Boolean(side))));
  }
  return result;
}

/** Covered workspace sides, in workspace order. */
export function coveredSides(explanation: GeneratedExplanation, roles: Record<Side, Side | null>): Side[] {
  const covered = new Set((explanation.presentation?.covered_sides ?? []).map((role) => roles[role]).filter((side): side is Side => Boolean(side)));
  return (["source", "target"] as Side[]).filter((side) => covered.has(side));
}

/**
 * Citations to the labels of entities a typed rendering names (for example a parent class in a
 * necessary condition). They are not in the subject's own context, so the artifact's support
 * packet names whose label each one is.
 */
export function referencedLabelSubjects(explanation: GeneratedExplanation): Map<string, EntityRef> {
  const result = new Map<string, EntityRef>();
  for (const fact of explanation.support_packet?.selection?.referenced_labels ?? []) if (fact.subject) result.set(fact.fact_id, fact.subject);
  return result;
}

/**
 * The exact typed subject of every record the artifact cites, from its own support packet: the
 * admitted facts and the labels of referenced entities (R2). A citation is opened about this
 * entity, never about the profile's or comparison's focal entity by default.
 */
export function citedSubjects(explanation: GeneratedExplanation): Map<string, EntityRef> {
  const result = new Map<string, EntityRef>();
  for (const fact of explanation.support_packet?.facts ?? []) if (fact.subject && !result.has(fact.fact_id)) result.set(fact.fact_id, fact.subject);
  for (const fact of explanation.support_packet?.selection?.referenced_labels ?? []) if (fact.subject && !result.has(fact.fact_id)) result.set(fact.fact_id, fact.subject);
  return result;
}
