"use client";

// A meaning card for one entity. Source and target cards are the same component with equal
// prominence; every section states its own availability instead of silently disappearing.

import { useEffect, useMemo, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconCopy, SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { GeneratedProfile, useExplanation, type CitedFact } from "@/components/explore/GeneratedBlock";
import { AxiomBlock } from "@/components/owl/AxiomBlock";
import { curie, predicateName, sideTitle } from "@/lib/iri";
import { seedLabel, useLabels } from "@/lib/labels";
import type { EntityContext, EntityRef, Fact } from "@/lib/types";
import type { AsyncState } from "@/lib/useAsync";

const DEFINING = ["restrictions", "types", "assertions", "domains", "ranges", "characteristics"];
const MORE = ["comments", "definition_citations", "term_metadata", "annotations", "xrefs", "mappings", "axioms"];
const MORE_NAMES: Record<string, string> = {
  comments: "Comments",
  definition_citations: "Definition citations",
  term_metadata: "Term metadata",
  annotations: "Other annotations",
  xrefs: "Cross-references",
  mappings: "Mappings",
  axioms: "Other axioms",
};

function literalText(fact: Fact): string | null {
  return fact.value?.term_type === "literal" ? fact.value.lexical_form : null;
}

function factAnchor(fact: Fact): string {
  return `fact-${fact.fact_id.slice(7, 19)}`;
}

export function CopyIri({ iri }: { iri: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="btn btn-sm"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(iri);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1600);
        } catch {
          setCopied(false);
        }
      }}
    >
      <IconCopy />
      <span aria-live="polite">{copied ? "Copied" : "Copy IRI"}</span>
    </button>
  );
}

function Section({ title, meta, children }: { title: string; meta?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="card-section">
      <div className="section-head">
        <h3>{title}</h3>
        {meta ? <span className="meta">{meta}</span> : null}
      </div>
      {children}
    </section>
  );
}

export function EntityCard({
  side,
  entity,
  context,
  ontologyLabel,
  cite,
  onOpenEntity,
  compact = false,
}: {
  side: "source" | "target";
  entity: EntityRef;
  context: AsyncState<EntityContext>;
  ontologyLabel: string;
  cite: (factId: string) => CitedFact | undefined;
  onOpenEntity?: (iri: string) => void;
  compact?: boolean;
}) {
  const [allSynonyms, setAllSynonyms] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const ctx = context.data;
  const profile = useExplanation(`profile|${entity.ontology_version_id}|${entity.kind}|${entity.iri}`, {
    ontology_version_id: entity.ontology_version_id,
    iri: entity.iri,
    kind: entity.kind,
    task: "entity_profile",
  });

  const parentIris = useMemo(
    () => (ctx?.parents.items ?? []).map((fact) => (fact.value?.term_type === "iri" ? fact.value.iri : null)).filter((iri): iri is string => Boolean(iri)),
    [ctx],
  );
  const parentLabel = useLabels(entity.ontology_version_id, parentIris);
  useEffect(() => {
    if (ctx) seedLabel(entity.ontology_version_id, entity.iri, ctx.preferred_label.value);
  }, [ctx, entity]);

  const title = sideTitle(side, entity.kind);
  const label = ctx?.preferred_label.value;
  const synonyms = (ctx?.synonyms.items ?? []).filter((fact) => literalText(fact) && literalText(fact) !== label);
  const hiddenDuplicates = (ctx?.synonyms.items.length ?? 0) - synonyms.length;
  const alternate = ctx?.categories.alternate_definitions;
  const defining = DEFINING.flatMap((category) => ctx?.categories[category]?.items ?? []);
  const definingStatus = DEFINING.map((category) => ctx?.categories[category]?.status).find((status) => status && status !== "absent_in_scope") ?? "absent_in_scope";
  const scopeNote = ctx ? (ctx.completeness.scope === "root" ? "root document; imports not loaded" : "resolved import closure") : undefined;
  const moreCategories = MORE.filter((category) => (ctx?.categories[category]?.items.length ?? 0) > 0);

  return (
    <article className={`entity-card entity-card-${side}${compact ? " compact" : ""}`} aria-label={title}>
      <div className="entity-card-head">
        <SideMarker side={side} />
        <span className={`eyebrow text-${side}`}>{title}</span>
        <span className="meta">{ontologyLabel}</span>
        <span className="entity-card-spacer" />
        <CopyIri iri={entity.iri} />
      </div>
      <div className="entity-title-block">
        <h2 className="entity-title">
          {label ?? (context.loading ? <span className="skeleton skeleton-title" /> : <span className="muted">No label in scope</span>)}
        </h2>
        <span className="iri">{entity.iri}</span>
      </div>

      {context.error ? <ErrorNote error={context.error} onRetry={context.reload} what="Entity context" /> : null}
      {!ctx && context.loading ? <Skeleton lines={5} /> : null}

      {ctx && (
        <>
          <Section title="Definition" meta={ctx.definitions.items.length ? `Original · ${predicateName(ctx.definitions.items[0].predicate_iri)}` : undefined}>
            {ctx.definitions.items.length ? (
              <div className="definitions">
                {ctx.definitions.items.map((fact) => (
                  <p key={fact.fact_id} id={factAnchor(fact)} className="definition-text" lang={fact.value?.term_type === "literal" ? fact.value.language ?? undefined : undefined}>
                    {literalText(fact) ?? "Structured value: open the original axiom."}
                  </p>
                ))}
                {ctx.definitions.items.length > 1 && <p className="meta">{ctx.definitions.items.length} definitions are asserted; all are shown.</p>}
              </div>
            ) : (
              <EmptyReason status={ctx.definitions.status} what="definition" scopeNote={scopeNote} reason={ctx.definitions.reason} />
            )}
            {alternate && alternate.items.length > 0 && (
              <div className="alternate-definitions">
                <span className="meta">Alternate {alternate.items.length === 1 ? "definition" : "definitions"} · {predicateName(alternate.items[0].predicate_iri)}</span>
                {alternate.items.map((fact) => (
                  <p key={fact.fact_id} id={factAnchor(fact)} className="definition-text definition-alt">
                    {literalText(fact)}
                  </p>
                ))}
              </div>
            )}
          </Section>

          <Section title="Also called" meta={synonyms.length ? `${synonyms.length} ${synonyms.length === 1 ? "synonym" : "synonyms"}${ctx.synonyms.truncated ? " · more exist" : ""}` : undefined}>
            {synonyms.length ? (
              <>
                <ul className="chip-list">
                  {(allSynonyms ? synonyms : synonyms.slice(0, 6)).map((fact) => (
                    <li key={fact.fact_id} id={factAnchor(fact)} className="chip" title={predicateName(fact.predicate_iri)}>
                      {literalText(fact)}
                      <span className="chip-meta">{fact.synonym_scope ? `${fact.synonym_scope}` : predicateName(fact.predicate_iri).replace(/ synonym$/, "")}</span>
                    </li>
                  ))}
                </ul>
                {synonyms.length > 6 && (
                  <button type="button" className="btn btn-sm" aria-expanded={allSynonyms} onClick={() => setAllSynonyms((value) => !value)}>
                    {allSynonyms ? "Show fewer" : `Show ${synonyms.length - 6} more`}
                  </button>
                )}
              </>
            ) : hiddenDuplicates > 0 ? (
              <p className="meta">Only a synonym identical to the label is recorded.</p>
            ) : (
              <EmptyReason status={ctx.synonyms.status} what="synonyms" scopeNote={scopeNote} reason={ctx.synonyms.reason} />
            )}
            {synonyms.length > 0 && hiddenDuplicates > 0 && <p className="meta">{hiddenDuplicates} synonym identical to the label is not repeated.</p>}
          </Section>

          <Section title="Defining facts" meta={defining.length ? `${defining.length} asserted` : undefined}>
            {defining.length ? (
              <div className="fact-stack">
                {defining.map((fact) => (
                  <AxiomBlock key={fact.fact_id} ontology={entity.ontology_version_id} axiomId={fact.axiom_id ?? fact.fact_id} subjectIri={entity.iri} onOpen={onOpenEntity} />
                ))}
              </div>
            ) : (
              <EmptyReason status={definingStatus} what="restrictions or other defining axioms" scopeNote={scopeNote} />
            )}
          </Section>

          <Section title="Parents" meta={parentIris.length ? `${parentIris.length} asserted${parentIris.length > 1 ? " · multiple inheritance" : ""}${ctx.parents.truncated ? " · more exist" : ""}` : undefined}>
            {parentIris.length ? (
              <ul className="parent-list">
                {ctx.parents.items.map((fact) => {
                  const iri = fact.value?.term_type === "iri" ? fact.value.iri : null;
                  if (!iri) return null;
                  const text = parentLabel(iri);
                  return (
                    <li key={fact.fact_id} id={factAnchor(fact)}>
                      <button type="button" className={`parent-link parent-${side}`} onClick={() => onOpenEntity?.(iri)}>
                        <span>{text?.value ?? curie(iri)}</span>
                        <span className="iri">{curie(iri)}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <EmptyReason status={ctx.parents.status} what="named parent" scopeNote={scopeNote} reason={ctx.parents.reason} />
            )}
          </Section>

          {!compact && moreCategories.length > 0 && (
            <div className="card-section">
              <button type="button" className="btn btn-sm" aria-expanded={moreOpen} onClick={() => setMoreOpen((value) => !value)}>
                {moreOpen ? "Hide other recorded facts" : `Other recorded facts (${moreCategories.reduce((sum, c) => sum + (ctx.categories[c]?.items.length ?? 0), 0)})`}
              </button>
              {moreOpen && (
                <div className="fact-stack">
                  {moreCategories.map((category) => (
                    <div key={category} className="more-category">
                      <h4>{MORE_NAMES[category]}</h4>
                      {ctx.categories[category].items.map((fact) =>
                        literalText(fact) ? (
                          <p key={fact.fact_id} id={factAnchor(fact)} className="more-literal">
                            {literalText(fact)} <span className="meta">· {predicateName(fact.predicate_iri)}</span>
                          </p>
                        ) : (
                          <AxiomBlock key={fact.fact_id} ontology={entity.ontology_version_id} axiomId={fact.axiom_id ?? fact.fact_id} subjectIri={entity.iri} onOpen={onOpenEntity} />
                        ),
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {ctx.completeness.imports_complete === false && (
            <p className="note note-warn">This ontology declares imports that were not resolved; imported labels and axioms may be missing.</p>
          )}
        </>
      )}

      <GeneratedProfile state={profile} cite={cite} />
    </article>
  );
}
