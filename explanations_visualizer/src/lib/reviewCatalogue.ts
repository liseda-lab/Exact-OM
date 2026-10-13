// Prepared examples for the exploration app's first entry (spec 24 HR2). The development review
// catalogue (`GET /api/v1/review-bundle`, exact-review-about/1) names typed sources and targets;
// pairs are resolved through the run's own candidate list, never derived from labels, and the
// catalogue's target order is not a matcher rank. This module is exploration-only: the study's
// public About metadata deliberately omits `cases`, and no participant surface reads it.
// Pure module (type-only imports) for Node's test runner.

import type { EntityKind, EntityRef } from "./types";

export interface ReviewCase {
  case_id: string;
  source: EntityRef;
  targets: EntityRef[];
}

export interface ReviewCatalogue {
  title: string;
  source_run_id: string;
  presented_pairs: number | null;
  cases: ReviewCase[];
}

/** Why prepared examples are not offered; each is shown, never recast as an empty catalogue. */
export type CatalogueFailure = "incompatible" | "invalid" | "denied" | "unreachable" | "error";

export type CatalogueState =
  | { status: "loading" }
  | { status: "available"; catalogue: ReviewCatalogue }
  /** The service declares no catalogue (or predates it): ordinary browsing. */
  | { status: "absent" }
  | { status: "failed"; reason: CatalogueFailure; error: unknown };

const KINDS: EntityKind[] = ["class", "object_property", "data_property", "individual"];

function entity(value: unknown): EntityRef | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (typeof record.ontology_version_id !== "string" || !record.ontology_version_id) return null;
  if (typeof record.iri !== "string" || !record.iri) return null;
  if (typeof record.kind !== "string" || !KINDS.includes(record.kind as EntityKind)) return null;
  return { ontology_version_id: record.ontology_version_id, iri: record.iri, kind: record.kind as EntityKind };
}

export class IncompatibleCatalogue extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IncompatibleCatalogue";
  }
}

/** Validate the exact-review-about/1 fields this view uses; anything else is incompatible. */
export function parseCatalogue(value: unknown): ReviewCatalogue {
  const record = (value ?? {}) as Record<string, unknown>;
  if (record.schema_version !== "exact-review-about/1") throw new IncompatibleCatalogue("unsupported catalogue schema");
  if (typeof record.source_run_id !== "string" || !record.source_run_id) throw new IncompatibleCatalogue("catalogue names no run");
  if (!Array.isArray(record.cases)) throw new IncompatibleCatalogue("catalogue cases are missing");
  const cases: ReviewCase[] = [];
  const seen = new Set<string>();
  for (const raw of record.cases) {
    const item = (raw ?? {}) as Record<string, unknown>;
    const source = entity(item.source);
    const targets = Array.isArray(item.targets) ? item.targets.map(entity) : [];
    if (typeof item.case_id !== "string" || !source || !targets.length || targets.some((target) => !target)) throw new IncompatibleCatalogue("a catalogue case is malformed");
    if (seen.has(item.case_id)) throw new IncompatibleCatalogue("duplicate catalogue case");
    seen.add(item.case_id);
    cases.push({ case_id: item.case_id, source, targets: targets as EntityRef[] });
  }
  return {
    title: typeof record.title === "string" ? record.title : "Prepared examples",
    source_run_id: record.source_run_id,
    presented_pairs: typeof record.presented_pairs === "number" ? record.presented_pairs : null,
    cases,
  };
}

/**
 * A failed catalogue read. Only the service's declared absence (`review_bundle_not_available`,
 * or a plain `not_found` from a service without this route) means ordinary browsing.
 */
export function catalogueStateFromError(error: unknown): CatalogueState {
  if (error instanceof IncompatibleCatalogue) return { status: "failed", reason: "incompatible", error };
  const api = error as { status?: number; code?: string } | null;
  if (api && api.status === 404 && (api.code === "review_bundle_not_available" || api.code === "not_found")) return { status: "absent" };
  if (api?.code === "review_bundle_invalid") return { status: "failed", reason: "invalid", error };
  if (api?.status === 401 || api?.status === 403) return { status: "failed", reason: "denied", error };
  if (api?.code === "network_unreachable") return { status: "failed", reason: "unreachable", error };
  return { status: "failed", reason: "error", error };
}

export const sameEntity = (a: EntityRef, b: EntityRef) => a.ontology_version_id === b.ontology_version_id && a.iri === b.iri && a.kind === b.kind;

/** The catalogue case whose typed source this is, if any. */
export function caseForSource(catalogue: ReviewCatalogue | null, source: EntityRef | null): ReviewCase | null {
  if (!catalogue || !source) return null;
  return catalogue.cases.find((item) => sameEntity(item.source, source)) ?? null;
}

/**
 * The example's pair among this source's loaded candidates: the first, in the order the app
 * shows (the matcher's recorded order), whose typed target is one of the case's targets.
 */
export function examplePair<T extends { pair_id: string; target: EntityRef }>(ordered: T[], item: ReviewCase): T | null {
  return ordered.find((candidate) => item.targets.some((target) => sameEntity(target, candidate.target))) ?? null;
}

/** A deliberate choice kept for one exact package and run only (spec 24 HR2 §3). */
export interface SavedChoice {
  package_id: string;
  run_id: string;
  source: string;
  source_kind: EntityKind;
  pair: string | null;
}

export function savedChoiceFor(raw: string | null, packageId: string | null, runId: string | null): SavedChoice | null {
  if (!raw || !packageId || !runId) return null;
  try {
    const value = JSON.parse(raw) as Partial<SavedChoice>;
    if (value.package_id !== packageId || value.run_id !== runId) return null;
    if (typeof value.source !== "string" || !value.source) return null;
    const kind = KINDS.includes(value.source_kind as EntityKind) ? (value.source_kind as EntityKind) : "class";
    return { package_id: packageId, run_id: runId, source: value.source, source_kind: kind, pair: typeof value.pair === "string" && value.pair ? value.pair : null };
  } catch {
    return null;
  }
}

export type Landing =
  | { kind: "keep" }
  | { kind: "resolve_pair" }
  | { kind: "saved"; choice: SavedChoice }
  | { kind: "example"; item: ReviewCase; run_id: string }
  | { kind: "first_source"; source: EntityRef }
  | { kind: "wait" };

/**
 * Where a visit without an explicit source starts. Priority (spec 24 HR2 §3): an explicit URL →
 * a deliberate choice saved for this exact package and run → a prepared example (unless the
 * visitor already chose All sources) → the run's first source. A loading catalogue is waited for
 * only until `waited`; a failed one is shown and ordinary browsing continues.
 */
export function landingChoice(input: {
  explicitSource: boolean;
  explicitPair: boolean;
  saved: SavedChoice | null;
  catalogue: CatalogueState;
  /** Runs in the open bundle; a catalogue for another run offers no example here. */
  runIds: string[];
  listMode: "examples" | "all" | null;
  waited: boolean;
  firstSource: EntityRef | null;
}): Landing {
  if (input.explicitSource) return { kind: "keep" };
  if (input.explicitPair) return { kind: "resolve_pair" };
  if (input.saved) return { kind: "saved", choice: input.saved };
  if (input.listMode !== "all") {
    if (input.catalogue.status === "loading" && !input.waited) return { kind: "wait" };
    if (input.catalogue.status === "available" && input.runIds.includes(input.catalogue.catalogue.source_run_id) && input.catalogue.catalogue.cases.length)
      return { kind: "example", item: input.catalogue.catalogue.cases[0], run_id: input.catalogue.catalogue.source_run_id };
  }
  return input.firstSource ? { kind: "first_source", source: input.firstSource } : { kind: "wait" };
}
