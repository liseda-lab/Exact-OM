// Exploration workspace over the exact-explain/1.0 read API. The planned participant-safe
// study workspace (16 B1) mirrors these response shapes under its own scoped prefix, so the
// same implementation can serve it by changing `base` once those routes exist.

import { ApiError, getJson } from "../api";
import type { Axiom, EntityContext, EntityRef, ExplanationSummary, GeneratedExplanation, HierarchyPage, Page, SearchItem, SelectedEvidence } from "../types";
import type { Basis, EvidenceBundle, ExplanationResult, FactRef, PairScope, ResolvedFact, WorkspaceCapabilities, WorkspaceSource } from "./types";

const axiomCache = new Map<string, Promise<Axiom>>();

export function loadAxiom(base: string, ontology: string, factId: string, signal?: AbortSignal): Promise<Axiom> {
  const key = `${base}|${ontology}|${factId}`;
  if (!axiomCache.has(key)) {
    const promise = getJson<Axiom>(`${base}/axioms/${encodeURIComponent(factId)}`, { ontology_version_id: ontology }, signal);
    promise.catch(() => axiomCache.delete(key));
    axiomCache.set(key, promise);
  }
  return axiomCache.get(key)!;
}

function notFound(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 404 || error.code === "not_found");
}

const iriValue = (node: unknown): string | null => {
  const record = node as { type?: string; value?: unknown; iri?: unknown } | null;
  if (!record || typeof record !== "object") return null;
  if (record.type === "IRI" && typeof record.value === "string") return record.value;
  return iriValue(record.iri);
};

/** The entity a stored axiom is about, read from its own structure (never guessed from text). */
function axiomSubject(ast: Record<string, unknown>, ontology: string): FactRef["subject"] {
  const named = (node: unknown, kind: "class" | "individual" | "object_property" | "data_property") => {
    const iri = iriValue(node);
    return iri ? { ontology_version_id: ontology, iri, kind } : null;
  };
  switch (ast.type) {
    case "AnnotationAssertion":
      return named(ast.subject, "class");
    case "SubClassOf":
      return (ast.sub_class as { type?: string })?.type === "Class" ? named(ast.sub_class, "class") : null;
    case "ClassAssertion":
      return named(ast.individual, "individual");
    default:
      return null;
  }
}

/** Describe a stored axiom as a resolved original record without reinterpreting it. */
export function resolvedFromAxiom(ref: FactRef, axiom: Axiom): ResolvedFact {
  const ast = axiom.ast as Record<string, unknown>;
  const subject = ref.subject ?? axiomSubject(ast, axiom.ontology_version_id);
  const value = ast.type === "AnnotationAssertion" ? (ast.value as Record<string, unknown> | undefined) : undefined;
  const property = ast.type === "AnnotationAssertion" ? (ast.property as { iri?: { value?: string } } | undefined)?.iri?.value ?? null : null;
  return {
    ref,
    status: axiom.availability === "available" ? "available" : axiom.availability,
    reason: null,
    ontology: axiom.ontology_version_id,
    category: axiom.category,
    interpretation: axiom.interpretation?.kind ?? null,
    predicateIri: property,
    subjects: subject ? [{ entity: subject, side: null }] : [],
    literal: value?.type === "Literal"
      ? { text: String(value.lexical_form ?? ""), language: (value.language as string | null) ?? null, datatype: null }
      : null,
    axiom,
    reconstructed: false,
    origins: (axiom.origins ?? []).map((origin) => ({
      document: origin.source_sha256,
      span: origin.span && typeof origin.span === "object" && "start" in origin.span ? `${String(origin.span.start)}–${String(origin.span.end)}` : null,
      status: "recorded",
    })),
    qualifiers: Array.isArray(ast.annotations) ? (ast.annotations as unknown[]) : [],
    derivation: null,
  };
}

export function unresolvedFact(ref: FactRef, status: ResolvedFact["status"], reason: string): ResolvedFact {
  return {
    ref, status, reason, ontology: null, category: null, interpretation: null, predicateIri: null,
    subjects: ref.subject ? [{ entity: ref.subject, side: null }] : [], literal: null, axiom: null,
    reconstructed: false, origins: [], qualifiers: [], derivation: null,
  };
}

function entityParams(entity: EntityRef, prefix = "") {
  return {
    [`${prefix}ontology_version_id`]: entity.ontology_version_id,
    [`${prefix}iri`]: entity.iri,
    [`${prefix}kind`]: entity.kind,
  };
}

export function createApiSource(options: {
  key: string;
  base?: string;
  ontologyName: (id: string) => string | null;
  reasonerStatus?: (id: string) => string;
}): WorkspaceSource {
  const base = options.base ?? "/api/v1";
  const capabilities: WorkspaceCapabilities = {
    search: "available",
    navigation: "available",
    bases: ["literal_asserted", "structural_navigation", "reasoner_inferred"],
    reasoner: "per_ontology",
    graphExpansion: true,
    profiles: "available",
    comparison: "available",
    evidence: "available",
    scopeNote: null,
    limitations: [],
    synthetic: false,
  };

  const explanation = async (task: "entity_profile" | "pair_comparison", entities: EntityRef[], signal?: AbortSignal): Promise<ExplanationResult> => {
    const [subject, counterpart] = entities;
    const params: Record<string, string> = { ...entityParams(subject), task };
    if (task === "pair_comparison" && counterpart) Object.assign(params, entityParams(counterpart, "counterpart_"));
    const page = await getJson<Page<ExplanationSummary>>(`${base}/explanations`, { ...params, limit: 100 }, signal);
    if (!page.items.length) return { status: "not_requested" };
    const chosen = page.items.find((item) => item.grounding_status === "validated") ?? page.items[0];
    if (chosen.grounding_status !== "validated") return { status: "unverified" };
    try {
      const item = await getJson<GeneratedExplanation>(`${base}/explanations/${encodeURIComponent(chosen.explanation_id)}`, undefined, signal);
      return {
        status: "available",
        explanation: item,
        factOntologies: Array.from(new Set(entities.map((entity) => entity.ontology_version_id))),
        provenance: {
          kind: "manifest",
          requestedModel: item.manifest.requested_model,
          returnedModel: item.manifest.returned_model,
          provider: item.manifest.provider,
          language: item.manifest.language,
          grounding: item.grounding_status,
        },
      };
    } catch (error) {
      if (error instanceof ApiError && error.code === "explanation_unverified") return { status: "unverified" };
      throw error;
    }
  };

  const fact = async (ref: FactRef, signal?: AbortSignal): Promise<ResolvedFact> => {
    for (const ontology of ref.ontologies) {
      try {
        return resolvedFromAxiom(ref, await loadAxiom(base, ontology, ref.factId, signal));
      } catch (error) {
        if (notFound(error)) continue;
        if (error instanceof ApiError && error.code === "axiom_detail_too_large") {
          return { ...unresolvedFact(ref, "partial", "This record is too large to show inline; it remains in the ontology file."), ontology };
        }
        throw error;
      }
    }
    return unresolvedFact(ref, "not_exported", "This record is not available in the open bundle. It may be withheld by the bundle's information policy.");
  };

  const evidence = async (pair: PairScope, signal?: AbortSignal): Promise<EvidenceBundle> => {
    if (!pair.runId || !pair.pairId) return { items: [], total: 0, status: "not_exported", reason: "This pair has no saved matcher record.", axioms: {} };
    const items: SelectedEvidence[] = [];
    let cursor: string | null = null;
    let first: Page<SelectedEvidence> | null = null;
    for (let guard = 0; guard < 10; guard += 1) {
      const page: Page<SelectedEvidence> = await getJson<Page<SelectedEvidence>>(`${base}/runs/${encodeURIComponent(pair.runId)}/pair-evidence`, { pair_id: pair.pairId, limit: 100, cursor }, signal);
      first = first ?? page;
      items.push(...page.items);
      if (!page.next_cursor) break;
      cursor = page.next_cursor;
    }
    // Reading order: source side first, then by channel; identities stay stable.
    items.sort((a, b) => (a.side === b.side ? a.channel.localeCompare(b.channel) || a.evidence_id.localeCompare(b.evidence_id) : a.side === "source" ? -1 : 1));
    const axioms: Record<string, Axiom | null> = {};
    await Promise.all(
      items.flatMap((item) =>
        item.fact_ids.map(async (factId) => {
          try {
            axioms[factId] = await loadAxiom(base, item.entity.ontology_version_id, factId, signal);
          } catch {
            axioms[factId] = null;
          }
        }),
      ),
    );
    const truncated = Boolean(first?.next_cursor) && items.length < (first?.total_count ?? items.length);
    return { items, total: first?.total_count ?? items.length, status: truncated ? "partial" : first?.status ?? "not_exported", reason: first?.reason ?? null, axioms };
  };

  return {
    key: options.key,
    kind: "exploration",
    capabilities,
    ontologyName: options.ontologyName,
    entityContext: (entity, signal) => getJson<EntityContext>(`${base}/entity-context`, entityParams(entity), signal),
    fact,
    explanation,
    hierarchy: (entity, direction, basis: Basis, cursor, signal) =>
      getJson<HierarchyPage>(`${base}/hierarchy`, { ...entityParams(entity), direction, basis, limit: 50, cursor }, signal),
    search: (ontology, term, cursor, signal) => getJson<Page<SearchItem>>(`${base}/entities`, { ontology_version_id: ontology, term, limit: 20, cursor }, signal),
    evidence,
    labels: null,
  };
}
