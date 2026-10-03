// The shared ontology/explanation workspace reads through one data-source interface. The
// exploration app, a participant's explanation case and the synthetic tutorial each supply
// their own implementation; the components never know which product they are in. A source
// that cannot do something says so through its capabilities and availability statuses;
// it never returns an empty answer for missing preparation.

import type { Availability, Axiom, EntityContext, EntityKind, EntityRef, GeneratedExplanation, HierarchyPage, Page, SearchItem, SelectedEvidence } from "../types";

export type Basis = "literal_asserted" | "structural_navigation" | "reasoner_inferred";
export type Side = "source" | "target";

export interface EvidenceBundle {
  items: SelectedEvidence[];
  total: number | null;
  status: string;
  reason: string | null;
  axioms: Record<string, Axiom | null>;
}

/** A cited or displayed original record: always a fact identity plus, when known, its scope. */
export interface FactRef {
  factId: string;
  /** Ontologies to try, most likely first; a citation's explanation names its entities. */
  ontologies: string[];
  subject?: EntityRef | null;
}

export interface FactSubject {
  entity: EntityRef;
  side: Side | null;
}

export interface ResolvedFact {
  ref: FactRef;
  status: Availability | "available";
  reason: string | null;
  ontology: string | null;
  category: string | null;
  interpretation: string | null;
  predicateIri: string | null;
  /** Every typed subject this original record is stored under in the active scope. */
  subjects: FactSubject[];
  /** Exact literal text when the record is a literal annotation. */
  literal: { text: string; language: string | null; datatype: string | null } | null;
  /** Typed AST for display; `reconstructed` when built from a stored (subject, predicate, value). */
  axiom: Axiom | null;
  reconstructed: boolean;
  /** Where the record occurs in its original document, when recorded. */
  origins: { document: string; span: string | null; status: string }[];
  qualifiers: unknown[];
  derivation: string | null;
}

export interface ExplanationProvenance {
  kind: "manifest" | "study";
  requestedModel?: string;
  returnedModel?: string | null;
  provider?: string | null;
  language?: string;
  grounding: string;
  manifestHashes?: string[];
  claimGrounding?: Record<string, string>;
  synthetic?: boolean;
}

export type ExplanationResult =
  | { status: "available"; explanation: GeneratedExplanation; provenance: ExplanationProvenance; factOntologies: string[] }
  | { status: "unverified" }
  | { status: "not_requested"; reason?: string }
  | { status: "not_exported"; reason: string };

export interface WorkspaceCapabilities {
  /** Search over the admitted ontologies; `partial` = only prepared entities are searchable. */
  search: "available" | "partial" | "unsupported";
  /** Parent/child navigation beyond the compared entities. */
  navigation: "available" | "partial" | "unsupported";
  bases: Basis[];
  reasoner: string;
  graphExpansion: boolean;
  profiles: string;
  comparison: string;
  evidence: string;
  /** Shown wherever the scope of the information matters. */
  scopeNote: string | null;
  limitations: string[];
  synthetic: boolean;
}

export interface PairScope {
  source: EntityRef;
  target: EntityRef;
  /** Exploration run and pair, when the pair is a saved matcher candidate. */
  runId?: string | null;
  pairId?: string | null;
  /** Study candidate identity within the frozen case. */
  candidateId?: string | null;
}

export interface WorkspaceSource {
  /** Cache namespace: product, package/publication, policy and resource version. */
  key: string;
  kind: "exploration" | "study_resource" | "tutorial";
  capabilities: WorkspaceCapabilities;
  ontologyName: (ontologyVersionId: string) => string | null;
  entityContext: (entity: EntityRef, signal?: AbortSignal) => Promise<EntityContext>;
  fact: (ref: FactRef, signal?: AbortSignal) => Promise<ResolvedFact>;
  explanation: (task: "entity_profile" | "pair_comparison", entities: EntityRef[], signal?: AbortSignal) => Promise<ExplanationResult>;
  hierarchy: (entity: EntityRef, direction: "parents" | "children", basis: Basis, cursor: string | null, signal?: AbortSignal) => Promise<HierarchyPage>;
  search: (ontology: string, term: string, cursor: string | null, signal?: AbortSignal) => Promise<Page<SearchItem>>;
  evidence: (pair: PairScope, signal?: AbortSignal) => Promise<EvidenceBundle>;
  /** Local label lookup; null means the shared remote label cache applies. */
  labels: ((ontology: string) => (iri: string) => { status: "loading" | "available" | "absent" | "failed" | "not_included"; value: string | null } | undefined) | null;
}

/** Typed actions the workspace reports; the shell maps them to telemetry or lesson evidence. */
export type WorkspaceAction =
  | { type: "tab_open"; tab: string }
  | { type: "citation_open"; factId: string }
  | { type: "axiom_open"; factId: string }
  | { type: "hierarchy_expand"; side: Side; iri: string }
  | { type: "hierarchy_collapse"; side: Side; iri: string }
  | { type: "hierarchy_focus"; side: Side; iri: string; kind: EntityKind; via: "parent" | "child" | "search" | "card" | "return" }
  | { type: "search"; side: Side }
  | { type: "evidence_select"; evidenceId: string; from: "graph" | "list" }
  | { type: "graph_edge"; edgeKind: string; evidenceId: string | null }
  | { type: "graph_zoom" }
  | { type: "graph_fit" }
  | { type: "graph_reset" }
  | { type: "locate_fact"; factId: string; inList: boolean }
  | { type: "copy_iri"; side: Side };
