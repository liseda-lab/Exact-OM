// Hierarchy modes and their per-ontology availability (21 F06, 23 C02). One rule for every
// workspace host: asserted and structural navigation are prepared with each context; the
// complete-only reasoner and the partial ELK taxonomy are advertised per ontology. A failed or
// absent complete-only reasoner never disables an advertised partial ELK result.
// Pure module (type-only imports) for Node's test runner.

import type { EntityKind, EntityRef, EntryPointPage } from "../types";
import type { Basis, OntologyNavigation, PartialInference, WorkspaceCapabilities } from "./types";

export const PARTIAL_BASIS: Basis = "elk_inferred_partial";

export interface BasisOption {
  value: Basis;
  label: string;
  /** One short meaning shown when the mode is selected. */
  note: string;
}

export const BASIS_OPTIONS: BasisOption[] = [
  { value: "literal_asserted", label: "Asserted", note: "Relationships written directly in the ontology. Direct parents and children." },
  { value: "structural_navigation", label: "Structural", note: "Navigation extracted from class expressions by a documented rule. Direct links only." },
  { value: "reasoner_inferred", label: "Inferred", note: "Relationships computed by the recorded reasoner for the whole ontology." },
  { value: "elk_inferred_partial", label: "ELK inferred · partial", note: "Relationships the ELK reasoner returned for the parts of this ontology it supports. A missing link is unknown, not absent." },
];

/** Normalize the main API's per-ontology capabilities (`ontology.capabilities`). */
export function navigationFromOntology(capabilities: Record<string, unknown> | undefined): OntologyNavigation {
  const caps = capabilities ?? {};
  const navigation = (caps.navigation ?? null) as { entry_points?: string } | null;
  const complete = caps.reasoner_inferred as { status?: string; reason?: string | null } | string | undefined;
  const partial = (caps.elk_inferred_partial ?? null) as PartialInference | null;
  return {
    entryPoints: navigation?.entry_points ?? null,
    inference: typeof complete === "string" ? { status: complete } : complete?.status ? { status: complete.status, reason: complete.reason ?? null } : null,
    partial: partial && typeof partial.status === "string" ? partial : null,
  };
}

/** Normalize one study scope's `navigation[ontologyVersion]` entry. */
export function navigationFromScope(entry: Record<string, unknown> | undefined): OntologyNavigation {
  const value = entry ?? {};
  const inference = value.inference as { status?: string; reason?: string | null } | undefined;
  const partial = (value.partial_inference ?? null) as PartialInference | null;
  return {
    entryPoints: typeof value.entry_points === "string" ? value.entry_points : null,
    inference: inference?.status ? { status: inference.status, reason: inference.reason ?? null } : null,
    partial: partial && typeof partial.status === "string" ? partial : null,
  };
}

/** Whether a mode can be shown for an entity kind, and why not. */
export function basisAvailability(capabilities: WorkspaceCapabilities, ontology: string, kind: EntityKind, basis: Basis): { available: boolean; reason: string | null } {
  const navigation = capabilities.navigationByOntology?.[ontology] ?? null;
  if (basis === "literal_asserted" || basis === "structural_navigation") {
    return capabilities.bases.includes(basis) ? { available: true, reason: null } : { available: false, reason: "This mode was not prepared for this view." };
  }
  if (basis === "reasoner_inferred") {
    const status = navigation?.inference?.status ?? (capabilities.bases.includes(basis) ? "available" : null);
    return status === "available" && capabilities.bases.includes(basis) ? { available: true, reason: null } : { available: false, reason: "Complete inferred relationships were not prepared for this ontology." };
  }
  const partial = navigation?.partial ?? null;
  if (!capabilities.bases.includes(basis) || partial?.status !== "partial") return { available: false, reason: "ELK results were not prepared for this ontology." };
  if (!(partial.supported_kinds ?? ["class"]).includes(kind)) return { available: false, reason: "ELK results cover classes only." };
  return { available: true, reason: null };
}

/**
 * The modes to offer: asserted, structural and partial ELK when prepared; the complete-only
 * mode only when it is actually available (a failed complete reasoner is not shown as a
 * disabled competitor of the usable partial result). With neither inferred result, one
 * disabled "Inferred" option explains why.
 */
export function offeredBases(capabilities: WorkspaceCapabilities, ontology: string): Basis[] {
  const navigation = capabilities.navigationByOntology?.[ontology];
  const complete = capabilities.bases.includes("reasoner_inferred") && (navigation?.inference?.status ?? "available") === "available";
  const partial = capabilities.bases.includes("elk_inferred_partial") && navigation?.partial?.status === "partial";
  return ["literal_asserted", "structural_navigation", ...(complete || !partial ? (["reasoner_inferred"] as Basis[]) : []), ...(partial ? (["elk_inferred_partial"] as Basis[]) : [])];
}

const FEATURE_NAMES: Record<string, string> = {
  OBJECT_EXACT_CARDINALITY: "exact cardinality restrictions",
  OBJECT_MIN_CARDINALITY: "minimum cardinality restrictions",
  OBJECT_MAX_CARDINALITY: "maximum cardinality restrictions",
  OBJECT_UNION_OF_POSITIVE: "unions (OR) in class definitions",
  DATATYPE: "datatypes",
  DATATYPE_DEFINITION: "datatype definitions",
};

/** Plain-language unsupported constructs of a partial result, de-duplicated across tasks. */
export function unsupportedConstructs(partial: PartialInference): string[] {
  const names = new Set<string>();
  for (const reason of partial.unsupported_constructs ?? []) {
    for (const feature of reason.features ?? []) names.add(FEATURE_NAMES[feature] ?? feature.toLowerCase().replace(/_/g, " "));
  }
  return Array.from(names);
}

/** How a browser without an explicit, remembered or pinned focus starts (HR4, 21 F06). */
export type EntryStart = { kind: "curated"; entity: EntityRef } | { kind: "single"; entity: EntityRef } | { kind: "list" };

/**
 * A declared curated default applies only when the service says so (`selection.kind` and a
 * navigation-only assertion) for this exact ontology version and kind on an available page;
 * `default_entity` alone is the first prepared entry and is never treated as curated. Older
 * pages without `selection` keep the one-root rule: a single entry and nothing more to load.
 */
export function entryStart(page: EntryPointPage | null | undefined, ontology: string, kind: EntityKind): EntryStart {
  if (!page || page.status !== "available") return { kind: "list" };
  const selection = page.selection;
  const fallback = (): EntryStart => (page.items.length === 1 && !page.next_cursor && page.items[0].entity.ontology_version_id === ontology ? { kind: "single", entity: page.items[0].entity } : { kind: "list" });
  if (!selection || selection.kind !== "curated_default") return fallback();
  const entity = page.default_entity;
  const navigationOnly = selection.assertion === undefined || selection.assertion === "navigation_only";
  const inScope = !page.scope?.ontology_version_id || page.scope.ontology_version_id === ontology;
  if (!navigationOnly || !inScope || !entity || entity.ontology_version_id !== ontology || entity.kind !== kind) return fallback();
  return { kind: "curated", entity };
}
