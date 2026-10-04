// Normalizes frozen, policy-filtered study explanation resources (exact-study/1.0
// `study_explanation`) into the shared workspace shapes. Nothing is fetched or inferred:
// interpretation, availability, capabilities and limitations are carried through, and a
// category the resource does not prepare is reported as not prepared, never as absent.
// Pure module (type-only imports) so it runs under Node's test runner.

import type { ExplanationResource, GroundedClaim, OriginalFact, StudyEvidenceLink, StudyHierarchyEdge } from "../../study/types";
import type { Availability, Axiom, Claim, EntityContext, EntityKind, EntityRef, Fact, GeneratedExplanation, HierarchyEdge, HierarchyPage, Page, Scope, SearchItem, SelectedEvidence, Term } from "../types";
import type { Basis, EvidenceBundle, ExplanationResult, FactRef, ResolvedFact } from "./types";

/** Categories the current study builder prepares for every case entity (first page each). */
export const BUILDER_CATEGORIES = ["labels", "definitions", "synonyms", "hierarchy", "restrictions"];

export interface ResourceIndex {
  entities: Map<string, EntityRef>;
  /** Every copy of an original record, one per typed subject it is stored under. */
  facts: Map<string, OriginalFact[]>;
  bySubject: Map<string, OriginalFact[]>;
  labels: Map<string, string>;
  profiles: GroundedClaim[];
  comparisons: GroundedClaim[];
  hierarchy: StudyHierarchyEdge[];
  evidence: StudyEvidenceLink[];
  limitations: string[];
  capabilities: Record<string, string>;
  /** `${iri}|${category}` pairs the builder reported as bounded to a prepared page. */
  truncated: Set<string>;
  labelsTruncated: boolean;
  /** Categories prepared for every entity; anything else is "not prepared", not absent. */
  prepared: Set<string> | "all";
}

export function entityKey(entity: { ontology_version_id: string; iri: string; kind: string }): string {
  return `${entity.ontology_version_id}|${entity.kind}|${entity.iri}`;
}

function labelKey(ontology: string, iri: string) {
  return `${ontology}|${iri}`;
}

const RANK: Record<string, number> = { available: 0, partial: 1, not_requested: 2, not_exported: 3, unsupported: 4 };

/** Merge several resources; capabilities keep the least complete status reported. */
export function indexResources(resources: ExplanationResource[], options: { prepared?: Set<string> | "all"; extraLabels?: Record<string, string> } = {}): ResourceIndex {
  const entities = new Map<string, EntityRef>();
  const facts = new Map<string, OriginalFact[]>();
  const bySubject = new Map<string, OriginalFact[]>();
  const labels = new Map<string, string>(Object.entries(options.extraLabels ?? {}));
  const capabilities: Record<string, string> = {};
  const truncated = new Set<string>();
  let labelsTruncated = false;
  const addLabel = (fact: OriginalFact) => {
    if (fact.category === "labels" && fact.value.term_type === "literal" && fact.value.lexical_form) {
      const key = labelKey(fact.subject.ontology_version_id, fact.subject.iri);
      if (!labels.has(key)) labels.set(key, fact.value.lexical_form);
    }
  };
  for (const resource of resources) {
    resource.entities.forEach((entity) => entities.set(entityKey(entity), entity));
    for (const fact of resource.facts) {
      const copies = facts.get(fact.fact_id) ?? [];
      if (!copies.some((copy) => entityKey(copy.subject) === entityKey(fact.subject))) {
        facts.set(fact.fact_id, [...copies, fact]);
        const subject = entityKey(fact.subject);
        bySubject.set(subject, [...(bySubject.get(subject) ?? []), fact]);
      }
      addLabel(fact);
    }
    // Referenced terms provide labels only; they never expand the focal case's facts.
    (resource.referenced_labels ?? []).forEach(addLabel);
    labelsTruncated = labelsTruncated || Boolean(resource.referenced_labels_truncated);
    for (const [name, status] of Object.entries(resource.capabilities ?? {})) {
      if (!(name in capabilities) || (RANK[status] ?? 9) > (RANK[capabilities[name]] ?? 9)) capabilities[name] = status;
    }
    for (const limit of resource.limitations) {
      const original = /^Original (\w+) for (\S+) are bounded to the prepared page\.$/.exec(limit);
      if (original) truncated.add(`${original[2]}|${original[1]}`);
      const parents = /^Named parents for (\S+) are bounded to the prepared page\.$/.exec(limit);
      if (parents) truncated.add(`${parents[1]}|parents`);
    }
  }
  const unique = <T,>(items: T[], id: (item: T) => string) => Array.from(new Map(items.map((item) => [id(item), item])).values());
  return {
    entities,
    facts,
    bySubject,
    labels,
    profiles: unique(resources.flatMap((resource) => resource.entity_profiles), (claim) => claim.claim_id),
    comparisons: unique(resources.flatMap((resource) => resource.pair_comparison), (claim) => claim.claim_id),
    hierarchy: unique(resources.flatMap((resource) => resource.hierarchy), (edge) => `${entityKey(edge.child)}>${entityKey(edge.parent)}>${edge.basis}`),
    evidence: unique(resources.flatMap((resource) => resource.evidence), (item) => `${item.candidate_id}|${item.evidence_id}`),
    limitations: Array.from(new Set(resources.flatMap((resource) => resource.limitations))),
    capabilities,
    truncated,
    labelsTruncated,
    prepared: options.prepared ?? new Set(BUILDER_CATEGORIES),
  };
}

export function labelOf(index: ResourceIndex, ontology: string, iri: string): string | null {
  return index.labels.get(labelKey(ontology, iri)) ?? null;
}

function term(fact: OriginalFact): Term {
  const value = fact.value;
  if (value.term_type === "literal") return { term_type: "literal", lexical_form: value.lexical_form ?? "", datatype: value.datatype ?? null, language: value.language ?? null };
  if (value.term_type === "iri") return { term_type: "iri", iri: value.iri ?? "" };
  return { term_type: "expression_ref", expression_id: value.expression_id ?? fact.axiom_ref };
}

export function toFact(fact: OriginalFact): Fact {
  return {
    id: fact.fact_id,
    fact_id: fact.fact_id,
    subject: fact.subject,
    predicate_iri: fact.predicate_iri,
    value: term(fact),
    axiom_ref: fact.axiom_ref,
    axiom_id: fact.fact_id,
    interpretation: fact.interpretation,
    origins: (fact.origins ?? []).map((origin) => ({ document_sha256: origin.document_sha256, axiom_id: origin.axiom_id, source_span: origin.source_span, provenance_status: origin.provenance_status })),
    premise_ids: fact.premises ?? [],
    derivation_id: fact.derivation ?? null,
    category: fact.category,
    qualifiers: fact.qualifiers ?? [],
  };
}

/** The original typed AST, or an exact AnnotationAssertion for a literal/IRI annotation fact. */
export function factAst(fact: OriginalFact): { ast: Record<string, unknown> | null; reconstructed: boolean } {
  if (fact.value.ast) return { ast: fact.value.ast as Record<string, unknown>, reconstructed: false };
  if (!fact.predicate_iri || fact.value.term_type === "expression_ref") return { ast: null, reconstructed: false };
  const qualifiers = fact.qualifiers ?? [];
  const value = fact.value.term_type === "literal"
    ? { type: "Literal", lexical_form: fact.value.lexical_form ?? "", language: fact.value.language ?? null, ...(fact.value.datatype && !fact.value.language ? { datatype: { type: "IRI", value: fact.value.datatype } } : {}) }
    : { type: "IRI", value: fact.value.iri };
  return {
    ast: {
      type: "AnnotationAssertion",
      annotations: qualifiers,
      property: { type: "AnnotationProperty", iri: { type: "IRI", value: fact.predicate_iri } },
      subject: { type: "IRI", value: fact.subject.iri },
      value,
    },
    reconstructed: true,
  };
}

export function toAxiom(fact: OriginalFact): Axiom | null {
  const { ast } = factAst(fact);
  if (!ast) return null;
  return {
    id: fact.fact_id,
    fact_id: fact.fact_id,
    axiom_id: fact.fact_id,
    ontology_version_id: fact.subject.ontology_version_id,
    category: fact.category,
    interpretation: { kind: fact.interpretation, scope: "study_resource" },
    availability: "available",
    ast: ast as Axiom["ast"],
    original_availability: "not_exported",
    original_format: "study-resource",
    rendering: { status: "unsupported", text: "" },
  };
}

function scope(ontology: string): Scope {
  return { ontology_version_id: ontology, context_revision: "study-resource", basis: "prepared", visibility_policy_hash: "", filter_id: "" };
}

function page<T>(items: T[], ontology: string, status: Availability, reason: string | null, truncated = false): Page<T> {
  return { items, returned_count: items.length, total_count: truncated ? null : items.length, next_cursor: null, truncated, scope: scope(ontology), status, reason };
}

function categoryStatus(index: ResourceIndex, entity: EntityRef, category: string, count: number): { status: Availability; reason: string | null; truncated: boolean } {
  const truncated = index.truncated.has(`${entity.iri}|${category}`);
  if (count) return { status: truncated ? "partial" : "available", reason: truncated ? "Only the first prepared page of these facts is included in this study case." : null, truncated };
  const prepared = index.prepared === "all" || index.prepared.has(category);
  if (prepared && index.entities.has(entityKey(entity))) return { status: "absent_in_scope", reason: null, truncated: false };
  return { status: "not_exported", reason: "Not prepared for this study case.", truncated: false };
}

const EXTRA_CATEGORIES = ["alternate_definitions", "types", "assertions", "domains", "ranges", "characteristics", "comments", "definition_citations", "term_metadata", "annotations", "xrefs", "mappings", "axioms", "labels"];

const NODE_KINDS: Record<string, EntityKind> = { Class: "class", ObjectProperty: "object_property", DataProperty: "data_property", NamedIndividual: "individual" };
const RECORDED_KINDS: Record<string, EntityKind> = { class: "class", object_property: "object_property", data_property: "data_property", individual: "individual", named_individual: "individual" };

/** The type an AST entity node records (its `kind`, else its OWL node type); null when it records none or contradicts itself. */
function recordedKind(node: { kind?: unknown; type?: unknown }): EntityKind | null {
  const kind = RECORDED_KINDS[String(node.kind)] ?? null;
  const typed = NODE_KINDS[String(node.type)] ?? null;
  if (kind && typed && kind !== typed) return null;
  return kind ?? typed;
}

/**
 * The typed parent of a stored subclass axiom (19 F21): the recorded hierarchy edge for this
 * fact when there is one, else the type the axiom's own AST records. Never a guessed kind; a
 * parent with no recorded type stays unprojected and is shown, not opened.
 */
function parentProjection(index: ResourceIndex, fact: OriginalFact, entity: EntityRef, parent: { iri: string; kind?: unknown; type?: unknown }): Fact["hierarchy_projection"] {
  const edge = index.hierarchy.find((item) => item.basis === "literal_asserted" && item.fact_ids.includes(fact.fact_id) && entityKey(item.child) === entityKey(entity) && item.parent.iri === parent.iri);
  const kind = edge ? null : recordedKind(parent);
  const typed = edge?.parent ?? (kind ? { ontology_version_id: entity.ontology_version_id, iri: parent.iri, kind } : null);
  return typed ? { id: `${fact.fact_id}|${entityKey(entity)}`, child: entity, parent: typed, basis: "literal_asserted" } : undefined;
}

export function entityContextFrom(index: ResourceIndex, entity: EntityRef, scopeNote: string): EntityContext {
  const facts = index.bySubject.get(entityKey(entity)) ?? [];
  const ontology = entity.ontology_version_id;
  const of = (category: string) => facts.filter((fact) => fact.category === category).map(toFact);
  const pageOf = (category: string, items = of(category)) => {
    const status = categoryStatus(index, entity, category, items.length);
    return page(items, ontology, status.status, status.reason, status.truncated);
  };
  // Named parents are the hierarchy facts whose stored axiom has this entity as the subclass.
  // Each keeps its fact identity and provenance and carries the parent's recorded type.
  const parentFacts = facts
    .filter((fact) => fact.category === "hierarchy")
    .flatMap((fact): Fact[] => {
      const ast = fact.value.ast as Record<string, unknown> | null | undefined;
      const sub = (ast?.sub_class as { iri?: { value?: string } } | undefined)?.iri?.value;
      const sup = (ast?.super_class as { iri?: { value?: string }; kind?: unknown; type?: unknown } | undefined);
      if (ast?.type !== "SubClassOf" || sub !== entity.iri || !sup?.iri?.value) return [];
      const projection = parentProjection(index, fact, entity, { iri: sup.iri.value, kind: sup.kind, type: sup.type });
      return [{ ...toFact(fact), value: { term_type: "iri", iri: sup.iri.value } as Term, ...(projection ? { hierarchy_projection: projection } : {}) }];
    });
  const parentStatus = categoryStatus(index, entity, "hierarchy", parentFacts.length);
  const label = labelOf(index, ontology, entity.iri);
  const labelFact = facts.find((fact) => fact.category === "labels");
  const categories: Record<string, Page<Fact>> = { restrictions: pageOf("restrictions") };
  for (const category of EXTRA_CATEGORIES) categories[category] = pageOf(category);
  return {
    artifact_type: "entity_context",
    entity,
    ontology_version_id: ontology,
    preferred_label: label ? { status: "available", value: label, fact_id: labelFact?.fact_id } : { status: index.entities.has(entityKey(entity)) ? "absent_in_scope" : "not_exported", value: null },
    labels: facts.filter((fact) => fact.category === "labels" && fact.value.term_type === "literal").map((fact) => ({ text: fact.value.lexical_form ?? "", language: fact.value.language ?? null, predicate_iri: fact.predicate_iri ?? "" })),
    definitions: pageOf("definitions"),
    synonyms: pageOf("synonyms"),
    parents: page(parentFacts, ontology, parentStatus.status, parentStatus.reason, index.truncated.has(`${entity.iri}|parents`) || parentStatus.truncated),
    categories,
    alignment_eligible: null,
    alignment_eligibility_status: "not_recorded",
    capabilities: index.capabilities,
    completeness: { scope: "study_resource", extraction: scopeNote, imports_complete: true, domain_completeness: index.capabilities.context ?? "partial" },
    context_scope: scope(ontology),
  };
}

/** Resolve one cited/displayed record by identity and, when given, typed subject. */
export function resolveFact(index: ResourceIndex, ref: FactRef): ResolvedFact {
  const copies = index.facts.get(ref.factId) ?? [];
  const chosen = (ref.subject ? copies.find((copy) => entityKey(copy.subject) === entityKey(ref.subject!)) : null) ?? copies[0];
  if (!chosen) {
    return {
      ref, status: "not_exported", reason: "This record is not part of the information prepared for this view.", ontology: null, category: null,
      interpretation: null, predicateIri: null, subjects: ref.subject ? [{ entity: ref.subject, side: null }] : [], literal: null, axiom: null,
      reconstructed: false, origins: [], qualifiers: [], derivation: null,
    };
  }
  const { reconstructed } = factAst(chosen);
  return {
    ref,
    status: "available",
    reason: null,
    ontology: chosen.subject.ontology_version_id,
    category: chosen.category,
    interpretation: chosen.interpretation,
    predicateIri: chosen.predicate_iri,
    subjects: copies.map((copy) => ({ entity: copy.subject, side: null })),
    literal: chosen.value.term_type === "literal" ? { text: chosen.value.lexical_form ?? "", language: chosen.value.language ?? null, datatype: chosen.value.datatype ?? null } : null,
    axiom: toAxiom(chosen),
    reconstructed,
    origins: (chosen.origins ?? []).map((origin) => ({ document: origin.document_sha256, span: origin.source_span ? `${origin.source_span.start}–${origin.source_span.end} ${origin.source_span.unit}` : null, status: origin.provenance_status })),
    qualifiers: chosen.qualifiers ?? [],
    derivation: chosen.derivation ?? null,
  };
}

function claimSubjects(index: ResourceIndex, claim: GroundedClaim): Set<string> {
  if (claim.scoped_entities?.length) return new Set(claim.scoped_entities.map(entityKey));
  return new Set(claim.fact_ids.flatMap((id) => (index.facts.get(id) ?? []).map((fact) => entityKey(fact.subject))));
}

function explanationOf(index: ResourceIndex, task: "entity_profile" | "pair_comparison", entities: EntityRef[], claims: GroundedClaim[], synthetic: boolean): ExplanationResult {
  const generated: GeneratedExplanation = {
    artifact_type: "generated_explanation",
    explanation_id: `study:${task}:${entities.map(entityKey).join(">")}`,
    task,
    entities,
    claims: claims.map((claim): Claim => ({ claim_id: claim.claim_id, text: claim.text, fact_ids: claim.fact_ids, category: claim.category })),
    grounding_status: "validated",
    limitations: [],
    manifest: { requested_model: "", returned_model: null, provider: null, status: "validated", language: "" },
  };
  return {
    status: "available",
    explanation: generated,
    factOntologies: Array.from(new Set(entities.map((entity) => entity.ontology_version_id))),
    provenance: {
      kind: "study",
      grounding: "validated",
      manifestHashes: Array.from(new Set(claims.map((claim) => claim.generation_manifest_sha256).filter((value): value is string => Boolean(value)))),
      claimGrounding: Object.fromEntries(claims.map((claim) => [claim.claim_id, claim.grounding])),
      synthetic,
    },
  };
}

export function profileFor(index: ResourceIndex, entity: EntityRef, synthetic = false): ExplanationResult {
  const claims = index.profiles.filter((claim) => {
    const subjects = claimSubjects(index, claim);
    return subjects.size === 1 && subjects.has(entityKey(entity));
  });
  if (!claims.length) {
    const status = index.capabilities.profiles;
    return status === "not_exported" ? { status: "not_exported", reason: "Generated descriptions were not included for this study." } : { status: "not_requested" };
  }
  return explanationOf(index, "entity_profile", [entity], claims, synthetic);
}

export function comparisonFor(index: ResourceIndex, source: EntityRef, target: EntityRef, synthetic = false): ExplanationResult {
  const pair = new Set([entityKey(source), entityKey(target)]);
  const claims = index.comparisons.filter((claim) => {
    if (claim.scoped_entities?.length === 2) return claim.scoped_entities.every((entity) => pair.has(entityKey(entity)));
    const subjects = claimSubjects(index, claim);
    return subjects.size > 0 && [...subjects].every((key) => pair.has(key)) && subjects.has(entityKey(target));
  });
  if (!claims.length) {
    const status = index.capabilities.comparison;
    return status === "not_exported" ? { status: "not_exported", reason: "Generated comparisons were not included for this study." } : { status: "not_requested" };
  }
  return explanationOf(index, "pair_comparison", [source, target], claims, synthetic);
}

/** Prepared parent/child edges only; navigation beyond them reports that it was not prepared. */
export function hierarchyFor(index: ResourceIndex, entity: EntityRef, direction: "parents" | "children", basis: Basis, navigation: "complete" | "prepared"): HierarchyPage {
  const ontology = entity.ontology_version_id;
  const matches = index.hierarchy.filter((edge) => edge.basis === basis && entityKey(direction === "parents" ? edge.child : edge.parent) === entityKey(entity));
  const items: HierarchyEdge[] = matches.map((edge) => ({
    id: `${edge.fact_ids[0]}|${entityKey(edge.child)}`,
    axiom_id: edge.fact_ids[0],
    basis: edge.basis,
    category: "hierarchy",
    child: edge.child,
    parent: edge.parent,
    relation: "subclass_of",
    interpretation: { kind: edge.basis === "literal_asserted" ? "asserted" : edge.basis === "reasoner_inferred" ? "reasoner_inferred" : "structurally_derived", rule: edge.derivation ?? null },
  }));
  const focal = index.entities.has(entityKey(entity));
  let status: Availability = items.length ? "available" : "absent_in_scope";
  let reason: string | null = null;
  if (basis !== "literal_asserted" && !index.hierarchy.some((edge) => edge.basis === basis)) {
    status = basis === "reasoner_inferred" ? "not_run" : "not_exported";
    reason = basis === "reasoner_inferred" ? "No reasoner results were prepared." : "This hierarchy basis was not prepared for this view.";
  } else if (navigation === "prepared") {
    const bounded = direction === "children" || !focal || index.truncated.has(`${entity.iri}|parents`);
    if (direction === "children" && !items.length) {
      status = "not_exported";
      reason = "Children were not prepared for this study case.";
    } else if (!focal && !items.length) {
      status = "not_exported";
      reason = "Parents of this entity were not prepared for this study case.";
    } else if (bounded && items.length) status = "partial";
  }
  return { ...page(items, ontology, status, reason, status === "partial"), nodes: [], basis, node_budget_reached: false };
}

/** Prefix search over the labels this view holds; the scope is reported, not hidden. */
export function searchIn(index: ResourceIndex, ontology: string, rawTerm: string, partial: boolean): Page<SearchItem> {
  const needle = rawTerm.trim().toLowerCase();
  const seen = new Map<string, SearchItem>();
  const kinds = new Map<string, EntityKind>();
  index.entities.forEach((entity) => { if (entity.ontology_version_id === ontology) kinds.set(entity.iri, entity.kind); });
  for (const [key, label] of index.labels) {
    const [labelOntology, iri] = [key.slice(0, key.indexOf("|")), key.slice(key.indexOf("|") + 1)];
    if (labelOntology !== ontology) continue;
    const words = label.toLowerCase().split(/[\s,;()/-]+/);
    if (!(label.toLowerCase().startsWith(needle) || words.some((word) => word.startsWith(needle)) || iri.toLowerCase() === needle)) continue;
    seen.set(iri, { entity: { ontology_version_id: ontology, iri, kind: kinds.get(iri) ?? "class" }, preferred_label: { status: "available", value: label } });
  }
  const items = Array.from(seen.values()).sort((a, b) => (a.preferred_label.value ?? "").localeCompare(b.preferred_label.value ?? ""));
  return page(items, ontology, partial ? "partial" : "available", partial ? "Search covers only the entities included in this case's prepared information." : null, false);
}

export function evidenceFor(index: ResourceIndex, source: EntityRef, candidate: EntityRef, candidateId: string | null): EvidenceBundle {
  const items = index.evidence.filter((item) => {
    if (candidateId && item.candidate_id === candidateId) return true;
    return item.fact_ids.some((id) => (index.facts.get(id) ?? []).some((fact) => entityKey(fact.subject) === entityKey(candidate)));
  });
  const axioms: Record<string, Axiom | null> = {};
  const selected: SelectedEvidence[] = items.map((item) => {
    const copies = item.fact_ids.flatMap((id) => index.facts.get(id) ?? []);
    const sideEntity = item.role === "source" ? source : candidate;
    const own = copies.filter((fact) => entityKey(fact.subject) === entityKey(sideEntity));
    const facts = own.length ? own : copies;
    facts.forEach((fact) => { axioms[fact.fact_id] = toAxiom(fact); });
    const subject = facts[0]?.subject ?? sideEntity;
    const values: Record<string, number> = {};
    if (item.saved_value != null) values[item.value_meaning ?? "saved value"] = item.saved_value;
    (item.scores ?? []).forEach((score) => { values[score.name] = score.value; });
    return {
      evidence_id: item.evidence_id,
      channel: item.channel,
      side: entityKey(subject) === entityKey(source) ? "source" : "target",
      role: item.role,
      entity: subject,
      feature_id: null,
      fact_ids: Array.from(new Set(facts.map((fact) => fact.fact_id))),
      source_axiom_refs: [],
      semantic_terms: {},
      historical_item_alias: null,
      display: {},
      values,
      interpretation: item.interpretation,
      provenance_status: "recorded",
      status: (item.status === "unresolved" ? "partial" : item.status) as Availability,
      reason: item.status === "available" ? null : item.status === "unresolved" ? "The original facts for this feature could not be resolved when the study was prepared." : "The original facts for this feature were not included.",
    };
  });
  const status = index.capabilities.evidence ?? (selected.length ? "available" : "not_exported");
  return { items: selected, total: selected.length, status: selected.length ? "available" : status, reason: selected.length ? null : "No matcher evidence was prepared for this candidate.", axioms };
}
