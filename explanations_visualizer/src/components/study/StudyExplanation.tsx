"use client";

// Explanation-condition panels built only from the frozen, policy-filtered study resource.
// They reuse the exploration renderers but never call exploration routes; components not in
// the frozen study definition are not shown at all.

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";

import { SideMarker } from "@/components/common/Icons";
import { channelName, type EvidenceBundle } from "@/components/explore/evidenceData";
import { claimCategoryName } from "@/components/explore/GeneratedBlock";
import { Expression, hasReading } from "@/components/owl/Expression";
import { curie, entityKey, predicateName } from "@/lib/iri";
import { LabelSourceContext, type LabelLookup } from "@/lib/labelSource";
import { axiomRelation, functionalSyntax } from "@/lib/owl";
import type { Axiom, EntityRef, OwlNode, SelectedEvidence } from "@/lib/types";
import type { ExplanationResource, GroundedClaim, OriginalFact, StudyCandidate, StudyCase } from "@/study/types";

const EvidenceGraph = dynamic(() => import("@/components/explore/EvidenceGraph").then((module) => module.EvidenceGraph), {
  ssr: false,
  loading: () => <div className="graph-canvas skeleton" aria-busy="true" />,
});

export type Component = "original_context" | "entity_description" | "hierarchy" | "evidence_table" | "evidence_graph" | "pair_comparison";
type Emit = (type: "definition_open" | "axiom_open" | "hierarchy_expand" | "hierarchy_collapse" | "evidence_open" | "table_open" | "graph_open" | "comparison_open" | "graph_zoom" | "graph_fit", element?: string) => void;

export interface ResourceIndex {
  facts: Map<string, OriginalFact>;
  bySubject: Map<string, OriginalFact[]>;
  labels: Map<string, string>;
  profiles: GroundedClaim[];
  comparisons: GroundedClaim[];
  hierarchy: ExplanationResource["hierarchy"];
  evidence: ExplanationResource["evidence"];
  limitations: string[];
}

export function indexResources(resources: ExplanationResource[]): ResourceIndex {
  const facts = new Map<string, OriginalFact>();
  const bySubject = new Map<string, OriginalFact[]>();
  const labels = new Map<string, string>();
  for (const resource of resources) {
    for (const fact of resource.facts) {
      // One original axiom (e.g. SubClassOf(child, parent)) is a fact about several
      // entities; resources may record it under different subjects. Keep each subject's copy.
      if (!facts.has(fact.fact_id)) facts.set(fact.fact_id, fact);
      const key = entityKey(fact.subject);
      const existing = bySubject.get(key) ?? [];
      if (existing.some((item) => item.fact_id === fact.fact_id)) continue;
      bySubject.set(key, [...existing, fact]);
      if (fact.category === "labels" && fact.value.term_type === "literal" && fact.value.lexical_form) {
        const labelKey = `${fact.subject.ontology_version_id}|${fact.subject.iri}`;
        if (!labels.has(labelKey)) labels.set(labelKey, fact.value.lexical_form);
      }
    }
  }
  // Per-pair resources repeat the source's own facts, profile and edges; keep one of each.
  const unique = <T,>(items: T[], id: (item: T) => string) => Array.from(new Map(items.map((item) => [id(item), item])).values());
  return {
    facts,
    bySubject,
    labels,
    profiles: unique(resources.flatMap((resource) => resource.entity_profiles), (claim) => claim.claim_id),
    comparisons: unique(resources.flatMap((resource) => resource.pair_comparison), (claim) => claim.claim_id),
    hierarchy: unique(resources.flatMap((resource) => resource.hierarchy), (edge) => `${entityKey(edge.child)}>${entityKey(edge.parent)}>${edge.basis}`),
    evidence: unique(resources.flatMap((resource) => resource.evidence), (item) => `${item.candidate_id}|${item.evidence_id}`),
    limitations: Array.from(new Set(resources.flatMap((resource) => resource.limitations))),
  };
}

export function useStudyLabelSource(index: ResourceIndex | null, extra: Record<string, string>) {
  return useMemo(
    () =>
      (ontology: string): LabelLookup =>
      (iri: string) => {
        const value = extra[`${ontology}|${iri}`] ?? index?.labels.get(`${ontology}|${iri}`);
        return value ? { status: "available", value } : { status: "not_included", value: null };
      },
    [index, extra],
  );
}

function literal(fact: OriginalFact): string | null {
  return fact.value.term_type === "literal" ? fact.value.lexical_form ?? null : null;
}

function factAst(fact: OriginalFact): OwlNode | null {
  if (fact.value.ast) return fact.value.ast as OwlNode;
  if (fact.category === "hierarchy" && fact.value.term_type === "iri" && fact.value.iri) {
    return {
      type: "SubClassOf",
      sub_class: { type: "Class", iri: { type: "IRI", value: fact.subject.iri } },
      super_class: { type: "Class", iri: { type: "IRI", value: fact.value.iri } },
    };
  }
  if (fact.value.term_type === "literal" && fact.predicate_iri) {
    return {
      type: "AnnotationAssertion",
      property: { type: "AnnotationProperty", iri: { type: "IRI", value: fact.predicate_iri } },
      subject: { type: "IRI", value: fact.subject.iri },
      value: { type: "Literal", lexical_form: fact.value.lexical_form ?? "", language: fact.value.language ?? null },
    };
  }
  return null;
}

function claimSubjects(claim: GroundedClaim, index: ResourceIndex): Set<string> {
  return new Set(claim.fact_ids.map((id) => index.facts.get(id)).filter(Boolean).map((fact) => entityKey(fact!.subject)));
}

function FactAxiom({ fact, emit }: { fact: OriginalFact; emit: Emit }) {
  const [open, setOpen] = useState(false);
  const ast = factAst(fact);
  if (!ast) return null;
  const relation = axiomRelation(ast, fact.subject.iri);
  const readable = relation && hasReading(relation.other);
  return (
    <div className="fact-block">
      {readable && relation ? (
        <p className="fact-reading">
          <span className="fact-relation">{relation.relation}</span> <Expression node={relation.other} ontology={fact.subject.ontology_version_id} />
        </p>
      ) : (
        <p className="meta">No plain-language template exists; the original form is shown.</p>
      )}
      {readable && (
        <button
          type="button"
          className="btn btn-quiet btn-sm"
          aria-expanded={open}
          onClick={() => {
            if (!open) emit("axiom_open", fact.fact_id);
            setOpen((value) => !value);
          }}
        >
          {open ? "Hide original axiom" : "Show original axiom"}
        </button>
      )}
      {(open || !readable) && (
        <div className="original-axiom">
          <span className="meta">Original · OWL Functional Syntax</span>
          <code className="fs-code">{functionalSyntax(ast, { abbreviate: true })}</code>
        </div>
      )}
    </div>
  );
}

export function StudyEntityCard({
  side,
  title,
  entity,
  label,
  index,
  components,
  emit,
}: {
  side: "source" | "target";
  title: string;
  entity: EntityRef;
  label: string;
  index: ResourceIndex | null;
  components: Set<Component>;
  emit: Emit;
}) {
  const facts = index?.bySubject.get(entityKey(entity)) ?? [];
  const definitions = facts.filter((fact) => fact.category === "definitions" || fact.category === "alternate_definitions");
  const synonyms = facts.filter((fact) => fact.category === "synonyms" && literal(fact) && literal(fact) !== label);
  const restrictions = facts.filter((fact) => ["restrictions", "equivalences", "types", "assertions"].includes(fact.category));
  const parents = (index?.hierarchy ?? []).filter((edge) => entityKey(edge.child) === entityKey(entity));
  const profile = (index?.profiles ?? []).filter((claim) => {
    const subjects = claimSubjects(claim, index!);
    return subjects.size === 1 && subjects.has(entityKey(entity));
  });
  const lookupParent = (iri: string) => index?.labels.get(`${entity.ontology_version_id}|${iri}`);
  const showOriginal = components.has("original_context");
  return (
    <article className={`entity-card entity-card-${side} compact`} aria-label={title}>
      <div className="entity-card-head">
        <SideMarker side={side} />
        <span className={`eyebrow text-${side}`}>{title}</span>
      </div>
      <div className="entity-title-block">
        <h2 className="entity-title">{label}</h2>
        <span className="iri">{entity.iri}</span>
      </div>
      {!index && <div className="skeleton-row" aria-busy="true" />}
      {index && showOriginal && (
        <>
          <section className="card-section">
            <div className="section-head">
              <h3>Definition</h3>
              {definitions[0] && <span className="meta">Original · {predicateName(definitions[0].predicate_iri)}</span>}
            </div>
            {definitions.length ? (
              definitions.map((fact) => (
                <p key={fact.fact_id} className="definition-text">
                  {literal(fact)}
                </p>
              ))
            ) : (
              <p className="note">No definition is stated for this concept in the prepared ontology scope. This does not mean one exists nowhere.</p>
            )}
          </section>
          {synonyms.length > 0 && (
            <section className="card-section">
              <div className="section-head">
                <h3>Also called</h3>
              </div>
              <ul className="chip-list">
                {synonyms.map((fact) => (
                  <li key={fact.fact_id} className="chip">
                    {literal(fact)}
                    <span className="chip-meta">{predicateName(fact.predicate_iri).replace(/ synonym$/, "")}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
          {restrictions.length > 0 && (
            <section className="card-section">
              <div className="section-head">
                <h3>Defining facts</h3>
                <span className="meta">{restrictions.length} asserted</span>
              </div>
              <div className="fact-stack">
                {restrictions.map((fact) => (
                  <FactAxiom key={fact.fact_id} fact={fact} emit={emit} />
                ))}
              </div>
            </section>
          )}
          {components.has("hierarchy") && (
            <section className="card-section">
              <div className="section-head">
                <h3>Parents</h3>
                {parents.length > 1 && <span className="meta">{parents.length} asserted · multiple inheritance</span>}
              </div>
              {parents.length ? (
                <ul className="chip-list">
                  {parents.map((edge) => {
                    const name = lookupParent(edge.parent.iri);
                    return (
                      <li key={edge.parent.iri} className="chip" title={edge.parent.iri}>
                        {name ?? <span className="mono">{curie(edge.parent.iri)}</span>}
                        {name && <span className="chip-id">{curie(edge.parent.iri)}</span>}
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="meta">No named parent in the prepared scope.</p>
              )}
            </section>
          )}
        </>
      )}
      {index && components.has("entity_description") && profile.length > 0 && (
        <section className="generated" aria-label="Generated description">
          <div className="generated-head">
            <h3 className="generated-title">Generated description</h3>
            <span className="meta">Prepared before the study · cites original facts</span>
          </div>
          <ul className="claim-list">
            {profile.map((claim) => (
              <li key={claim.claim_id} className="claim">
                <span className="claim-category">{claimCategoryName(claim.category)}</span>
                <span className="claim-text">{claim.text}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}

function StudyComparison({ source, candidate, index }: { source: EntityRef; candidate: EntityRef; index: ResourceIndex }) {
  const pair = new Set([entityKey(source), entityKey(candidate)]);
  const claims = index.comparisons.filter((claim) => {
    if (claim.scoped_entities.length === 2) return claim.scoped_entities.every((entity) => pair.has(entityKey(entity)));
    const subjects = claimSubjects(claim, index);
    return subjects.size > 0 && [...subjects].every((key) => pair.has(key)) && subjects.has(entityKey(candidate));
  });
  const groups: [string, string[]][] = [
    ["Shared", ["agreement"]],
    ["Differences and one-sided information", ["difference", "scope"]],
    ["Explicit incompatibilities", ["explicit_incompatibility"]],
    ["Open questions", ["unknown", "review_question"]],
  ];
  return (
    <section className="comparison study-comparison" aria-labelledby="study-cmp-h">
      <div className="comparison-head">
        <h3 id="study-cmp-h">Comparison with the source</h3>
        <span className="origin-tag origin-generated">Generated before the study</span>
      </div>
      {!claims.length && <p className="note">No comparison was prepared for this candidate.</p>}
      {claims.length > 0 && (
        <div className="comparison-grid">
          {groups.map(([title, categories]) => {
            const items = claims.filter((claim) => categories.includes(claim.category));
            if (!items.length && title !== "Explicit incompatibilities") return null;
            return (
              <div key={title} className="comparison-group">
                <h4>{title}</h4>
                {items.length ? (
                  <ul>
                    {items.map((claim) => (
                      <li key={claim.claim_id} className="claim">
                        {claim.text}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="meta">No opposing statement is recorded in the cited facts.</p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function StudyEvidence({ items, index, source, candidate }: { items: ExplanationResource["evidence"]; index: ResourceIndex; source: EntityRef; candidate: EntityRef }) {
  if (!items.length) return <p className="note">No matcher evidence was prepared for this candidate.</p>;
  return (
    <ul className="evidence-items">
      {items.map((item) => {
        const facts = item.fact_ids.map((id) => index.facts.get(id)).filter(Boolean) as OriginalFact[];
        const side = facts[0] ? (entityKey(facts[0].subject) === entityKey(source) ? "source" : entityKey(facts[0].subject) === entityKey(candidate) ? "target" : item.role === "source" ? "source" : "target") : item.role === "source" ? "source" : "target";
        return (
          <li key={item.evidence_id} className="evidence-item">
            <div className="evidence-head">
              <SideMarker side={side} size="0.875rem" />
              <span className="evidence-kind">
                {channelName(item.channel)} feature · {side === "source" ? "source side" : "candidate side"}
              </span>
              <span className="meta">Projected by Exact from an asserted fact</span>
            </div>
            {facts.map((fact) => {
              const ast = factAst(fact);
              const relation = ast ? axiomRelation(ast, fact.subject.iri) : null;
              return relation && hasReading(relation.other) ? (
                <p key={fact.fact_id} className="fact-reading">
                  <span className="fact-relation">{relation.relation}</span> <Expression node={relation.other} ontology={fact.subject.ontology_version_id} />
                </p>
              ) : literal(fact) ? (
                <p key={fact.fact_id} className="fact-reading">
                  <span className="fact-relation">{predicateName(fact.predicate_iri)}</span> “{literal(fact)}”
                </p>
              ) : null;
            })}
          </li>
        );
      })}
    </ul>
  );
}

function toBundle(items: ExplanationResource["evidence"], index: ResourceIndex, source: EntityRef, candidate: EntityRef): EvidenceBundle {
  const axioms: Record<string, Axiom | null> = {};
  const evidence: SelectedEvidence[] = items.map((item) => {
    const facts = item.fact_ids.map((id) => index.facts.get(id)).filter(Boolean) as OriginalFact[];
    facts.forEach((fact) => {
      const ast = factAst(fact);
      axioms[fact.fact_id] = ast
        ? ({ id: fact.fact_id, fact_id: fact.fact_id, axiom_id: fact.fact_id, ontology_version_id: fact.subject.ontology_version_id, category: fact.category, interpretation: { kind: "asserted", scope: "prepared" }, availability: "available", ast, original_availability: "not_exported", original_format: "study-resource", rendering: { status: "unsupported", text: "" } } as Axiom)
        : null;
    });
    const subject = facts[0]?.subject ?? (item.role === "source" ? source : candidate);
    return {
      evidence_id: item.evidence_id,
      channel: item.channel,
      side: entityKey(subject) === entityKey(source) ? "source" : "target",
      role: item.role,
      entity: subject,
      feature_id: null,
      fact_ids: facts.map((fact) => fact.fact_id),
      source_axiom_refs: [],
      semantic_terms: {},
      historical_item_alias: null,
      display: {},
      values: {},
      interpretation: item.interpretation,
      provenance_status: "recorded",
      status: facts.length ? "available" : "not_exported",
      reason: null,
    } as SelectedEvidence;
  });
  return { items: evidence, total: evidence.length, status: evidence.length ? "available" : "not_exported", reason: null, axioms };
}

type PanelTab = "meaning" | "hierarchy" | "evidence" | "graph";

export function ExplanationPanels({
  studyCase,
  candidate,
  index,
  error,
  components,
  emit,
  labelSource,
}: {
  studyCase: StudyCase;
  candidate: StudyCandidate;
  index: ResourceIndex | null;
  error: string | null;
  components: Set<Component>;
  emit: Emit;
  labelSource: (ontology: string) => LabelLookup;
}) {
  const [tab, setTab] = useState<PanelTab>("meaning");
  const tabs: { key: PanelTab; label: string; show: boolean }[] = [
    { key: "meaning", label: components.has("pair_comparison") ? "Meaning and comparison" : "Meaning", show: true },
    { key: "hierarchy", label: "Hierarchy", show: components.has("hierarchy") },
    { key: "evidence", label: "Evidence", show: components.has("evidence_table") },
    { key: "graph", label: "Graph", show: components.has("evidence_graph") },
  ];
  const evidence = (index?.evidence ?? []).filter((item) => {
    if (item.candidate_id === candidate.candidate_id) return true;
    return item.fact_ids.some((id) => {
      const fact = index?.facts.get(id);
      return fact && entityKey(fact.subject) === entityKey(candidate.entity);
    });
  });
  const open = (next: PanelTab) => {
    setTab(next);
    if (next === "hierarchy") emit("hierarchy_expand", candidate.candidate_id);
    if (next === "evidence") emit("table_open", candidate.candidate_id);
    if (next === "graph") emit("graph_open", candidate.candidate_id);
    if (next === "meaning") emit("comparison_open", candidate.candidate_id);
  };
  return (
    <LabelSourceContext.Provider value={labelSource}>
      <section className="card explanation-panels" aria-label="Explanation for the inspected candidate">
        <div className="explanation-head">
          <span className="meta">
            Inspecting initial position {candidate.display_position} of {studyCase.candidates.length}
          </span>
          <div className="tab-list" role="tablist" aria-label="Explanation views">
            {tabs
              .filter((item) => item.show)
              .map((item) => (
                <button key={item.key} type="button" role="tab" aria-selected={tab === item.key} className={tab === item.key ? "tab on" : "tab"} onClick={() => open(item.key)}>
                  {item.label}
                </button>
              ))}
          </div>
        </div>
        <div className="details-panel" role="tabpanel">
          {error && <p className="note note-bad">{error}</p>}
          {tab === "meaning" && (
            <>
              <StudyEntityCard side="target" title="Candidate" entity={candidate.entity} label={candidate.label} index={index} components={components} emit={emit} />
              {index && components.has("pair_comparison") && <StudyComparison source={studyCase.source} candidate={candidate.entity} index={index} />}
            </>
          )}
          {tab === "hierarchy" && index && (
            <div className="mini-hierarchy">
              {[
                ["source", studyCase.source, studyCase.source_label] as const,
                ["target", candidate.entity, candidate.label] as const,
              ].map(([side, entity, name]) => {
                const parents = index.hierarchy.filter((edge) => entityKey(edge.child) === entityKey(entity));
                return (
                  <div key={side} className="mini-parents">
                    <h3 className="eyebrow">{side === "source" ? "Source" : "Candidate"} · asserted parents</h3>
                    {parents.length ? (
                      <ul>
                        {parents.map((edge) => (
                          <li key={edge.parent.iri}>
                            <span className="edge-solid" aria-hidden="true" />
                            {index.labels.get(`${entity.ontology_version_id}|${edge.parent.iri}`) ?? curie(edge.parent.iri)} <span className="iri">{curie(edge.parent.iri)}</span>
                          </li>
                        ))}
                        <li className="mini-focus">{name}</li>
                      </ul>
                    ) : (
                      <p className="meta">No named parent in the prepared scope.</p>
                    )}
                  </div>
                );
              })}
              <p className="meta">Only direct asserted parents were prepared. Explore further in Protégé if you need to.</p>
            </div>
          )}
          {tab === "evidence" && index && <StudyEvidence items={evidence} index={index} source={studyCase.source} candidate={candidate.entity} />}
          {tab === "graph" &&
            index &&
            (evidence.length ? (
              <EvidenceGraph
                source={studyCase.source}
                target={candidate.entity}
                bundle={toBundle(evidence, index, studyCase.source, candidate.entity)}
                selected={null}
                onSelect={() => undefined}
                canExpand={false}
                onViewChange={(action) => emit(action === "zoom" ? "graph_zoom" : "graph_fit")}
              />
            ) : (
              <p className="note">No matcher evidence was prepared for this candidate, so there is nothing to draw.</p>
            ))}
          {index && index.limitations.length > 0 && tab === "meaning" && (
            <details className="limits">
              <summary>Limits of the prepared information</summary>
              <ul>
                {index.limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      </section>
    </LabelSourceContext.Provider>
  );
}
