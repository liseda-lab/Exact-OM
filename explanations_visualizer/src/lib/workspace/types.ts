// The shared ontology/explanation workspace reads through one data-source interface. The
// exploration app, a participant's explanation case and the synthetic tutorial each supply
// their own implementation; the components never know which product they are in. A source
// that cannot do something says so through its capabilities and availability statuses;
// it never returns an empty answer for missing preparation.

import type { Availability, Axiom, EntityContext, EntityKind, EntityRef, EntryPointPage, Fact, GeneratedExplanation, HierarchyPage, Page, PairEvidenceMetadata, SearchItem, SelectedEvidence } from "../types";

/**
 * Hierarchy bases. `reasoner_inferred` is the complete-only contract (may be failed or absent);
 * `elk_inferred_partial` is the separately admitted partial ELK taxonomy (23 C02): positive,
 * direct-in-the-returned-result edges only, where a missing relationship is unknown.
 */
export type Basis = "literal_asserted" | "structural_navigation" | "reasoner_inferred" | "elk_inferred_partial";

/** exact-partial-hierarchy/1 metadata for one ontology's ELK result. Usable when `status` is `partial`. */
export interface PartialInference {
  status: string;
  reason?: string | null;
  reasoner?: string;
  versions?: Record<string, string>;
  supported_kinds?: string[];
  completion?: string;
  directness?: string;
  consistency?: { value?: boolean | null; complete?: boolean };
  unsupported_constructs?: { constructors?: string[]; features?: string[]; polarities?: string[]; task?: string }[];
  negative_entailment?: string;
  full_ontology_entailment?: string;
  justifications?: string;
  satisfiability?: string;
  scope?: string;
  source_imports_complete?: boolean;
  served_direct_edge_count?: number;
  unrepresented_class_count?: number;
  class_count?: number;
  artifact_sha256?: string;
}

/** What one ontology's prepared navigation offers, normalized from the main API or a study scope. */
export interface OntologyNavigation {
  /** `available` when typed entry points were prepared. */
  entryPoints: string | null;
  /** The complete-only reasoner contract (often failed for large ontologies). */
  inference: { status: string; reason?: string | null } | null;
  /** The partial ELK taxonomy; never disabled by the complete-only status above. */
  partial: PartialInference | null;
}
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
  /** exact-description/2 + presentation/1 metadata, when the artifact declares it (23 C03). */
  preparationKind?: string | null;
  supportStatus?: string | null;
  supportReasons?: string[];
  reviewerKind?: string | null;
  renderRevision?: string | null;
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
  /** Per-ontology navigation (entry points, complete and partial inference). */
  navigationByOntology?: Record<string, OntologyNavigation>;
  graphExpansion: boolean;
  profiles: string;
  comparison: string;
  evidence: string;
  /** Recorded comparison metadata (exact-evidence-metadata/1) can be read for its pairs. */
  evidenceMetadata?: boolean;
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
  /** Paginated typed entry points of one ontology for a basis (absent when not prepared). */
  entryPoints?: (ontology: string, kind: EntityKind, basis: Basis, cursor: string | null, signal?: AbortSignal) => Promise<EntryPointPage>;
  /** Members of an entity's equivalence group in an inferred basis. */
  equivalents?: (entity: EntityRef, basis: Basis, cursor: string | null, signal?: AbortSignal) => Promise<Page<EntityRef>>;
  /** Recorded pair-evidence completeness and matcher comparisons (exact-evidence-metadata/1). */
  evidenceMetadata?: (pair: PairScope, signal?: AbortSignal) => Promise<PairEvidenceMetadata | null>;
  search: (ontology: string, term: string, cursor: string | null, signal?: AbortSignal) => Promise<Page<SearchItem>>;
  /** Later pages of one fact category, continuing an entity context's own cursor. */
  facts?: (entity: EntityRef, category: string, cursor: string, signal?: AbortSignal) => Promise<Page<Fact>>;
  evidence: (pair: PairScope, signal?: AbortSignal) => Promise<EvidenceBundle>;
  remoteLabels?: import("../labels").RemoteLabelSource;
  /** Local label lookup; null means the shared remote label cache applies. */
  labels: ((ontology: string) => (iri: string) => { status: "loading" | "available" | "absent" | "failed" | "not_included"; value: string | null } | undefined) | null;
}

/** What a selected graph line is. Only a matcher `feature` line rests on a selected original record. */
export type GraphEdgeKind = "feature" | "asserted" | "structural" | "inferred" | "bridge";

/**
 * Typed actions the workspace reports; the shell maps them to telemetry or lesson evidence.
 * An interaction is not evidence by itself: actions carry the exact identities a lesson needs
 * (evidence ID, the inspected record, the typed entity) so a shell can decide whether it
 * satisfies a requirement, and nothing else is inferred (21 F01).
 */
export type WorkspaceAction =
  | { type: "tab_open"; tab: string }
  /** A citation under prepared generated text was opened. */
  | { type: "citation_open"; factId: string }
  /** An original record or axiom was shown (a card's "Show original axiom" or an inspector opened on a record). */
  | { type: "axiom_open"; factId: string }
  | { type: "hierarchy_expand"; side: Side; iri: string }
  | { type: "hierarchy_collapse"; side: Side; iri: string }
  /** `returned`: focus reached the compared entity again after being elsewhere (any route, not page load). */
  | { type: "hierarchy_focus"; side: Side; iri: string; kind: EntityKind; via: "parent" | "child" | "search" | "card" | "return"; returned?: boolean }
  | { type: "search"; side: Side }
  /** A selected matcher feature, with the original records shown for it (first is the one inspected). */
  | { type: "evidence_select"; evidenceId: string; from: "graph" | "list"; factIds?: string[] }
  | { type: "graph_edge"; edgeKind: GraphEdgeKind; evidenceId: string | null; factId?: string | null }
  | { type: "graph_zoom" }
  | { type: "graph_fit" }
  | { type: "graph_reset" }
  | { type: "locate_fact"; factId: string; inList: boolean }
  | { type: "copy_iri"; side: Side }
  | { type: "context_open"; side: Side; iri: string };
