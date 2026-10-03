// Workspace over frozen study resources: a participant's current explanation case (v1
// publications) or the synthetic tutorial. Only data the server already sent is used; the
// planned scoped workspace routes (16 B1) replace this for complete navigation.

import type { EntityRef } from "../types";
import { comparisonFor, entityContextFrom, evidenceFor, hierarchyFor, labelOf, profileFor, resolveFact, searchIn, type ResourceIndex } from "./resourceIndex";
import type { Basis, WorkspaceSource } from "./types";

export function createResourceSource(options: {
  key: string;
  kind: "study_resource" | "tutorial";
  index: ResourceIndex;
  /** `complete`: the resource holds the whole (synthetic) ontology; `prepared`: one case's bounded view. */
  navigation: "complete" | "prepared";
  scopeNote: string;
  ontologyName: (id: string) => string | null;
  synthetic?: boolean;
}): WorkspaceSource {
  const { index } = options;
  const complete = options.navigation === "complete";
  const bases = Array.from(new Set<Basis>(["literal_asserted", ...index.hierarchy.map((edge) => edge.basis)]));
  const reasoner = bases.includes("reasoner_inferred") ? "available" : "not_run";
  return {
    key: options.key,
    kind: options.kind,
    capabilities: {
      search: complete ? "available" : "partial",
      navigation: complete ? "available" : "partial",
      bases,
      reasoner,
      graphExpansion: complete,
      profiles: index.capabilities.profiles ?? (index.profiles.length ? "available" : "not_requested"),
      comparison: index.capabilities.comparison ?? (index.comparisons.length ? "available" : "not_requested"),
      evidence: index.capabilities.evidence ?? (index.evidence.length ? "available" : "not_exported"),
      scopeNote: options.scopeNote,
      limitations: index.limitations,
      synthetic: Boolean(options.synthetic),
    },
    ontologyName: options.ontologyName,
    entityContext: async (entity: EntityRef) => entityContextFrom(index, entity, options.scopeNote),
    fact: async (ref) => resolveFact(index, ref),
    explanation: async (task, entities) => (task === "entity_profile" ? profileFor(index, entities[0], options.synthetic) : comparisonFor(index, entities[0], entities[1], options.synthetic)),
    hierarchy: async (entity, direction, basis) => hierarchyFor(index, entity, direction, basis, options.navigation),
    search: async (ontology, term) => searchIn(index, ontology, term, !complete),
    evidence: async (pair) => evidenceFor(index, pair.source, pair.target, pair.candidateId ?? null),
    labels: (ontology) => (iri) => {
      const value = labelOf(index, ontology, iri);
      return value ? { status: "available", value } : { status: "not_included", value: null };
    },
  };
}
