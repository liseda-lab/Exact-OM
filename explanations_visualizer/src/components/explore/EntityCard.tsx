"use client";

// A meaning card for one entity. Source and target cards are the same component with equal
// prominence; every section states its own availability instead of silently disappearing.
// The card reads through the active workspace, so exploration, study and tutorial share it.

import { useEffect, useMemo, useRef, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconCopy, SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { GeneratedProfile, type CitedFact } from "@/components/explore/GeneratedBlock";
import { AxiomBlock } from "@/components/owl/AxiomBlock";
import { curie, predicateName, sideTitle } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import { seedLabel } from "@/lib/labels";
import type { EntityContext, EntityRef, Fact, Page } from "@/lib/types";
import type { AsyncState } from "@/lib/useAsync";
import { useExplanation, useWorkspace, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

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

/** Copies an IRI; when the browser refuses clipboard access, the IRI is shown selected instead. */
export function CopyIri({ iri, label, onCopied }: { iri: string; label?: string; onCopied?: (method: "clipboard" | "selected") => void }) {
  const [copied, setCopied] = useState(false);
  const [fallback, setFallback] = useState(false);
  return (
    <span className="copy-iri">
      <button
        type="button"
        className="btn btn-sm"
        aria-label={label}
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(iri);
            setCopied(true);
            setFallback(false);
            onCopied?.("clipboard");
            window.setTimeout(() => setCopied(false), 1600);
          } catch {
            setCopied(false);
            setFallback(true);
            onCopied?.("selected");
          }
        }}
      >
        <IconCopy />
        <span aria-live="polite">{copied ? "Copied" : "Copy IRI"}</span>
      </button>
      {fallback && (
        <span className="copy-fallback">
          <input
            className="input mono"
            readOnly
            value={iri}
            aria-label={`IRI${label ? ` (${label.replace(/^Copy IRI of /, "")})` : ""}`}
            ref={(node) => node?.select()}
            onFocus={(event) => event.target.select()}
          />
          <span className="meta" role="status">This browser blocked copying. The IRI is selected: press Ctrl+C or ⌘C.</span>
        </span>
      )}
    </span>
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

function scopeText(ctx: EntityContext): string {
  if (ctx.completeness.scope === "study_resource") return "the information prepared for this view";
  return ctx.completeness.scope === "root" ? "root document; imports not loaded" : "resolved import closure";
}

/**
 * One recorded fact category, continuing the context's own cursor page by page. Later
 * pages are read only on request; a failure stays local and never reads as absence.
 */
function MoreCategory({ category, page, entity, onOpenEntity }: { category: string; page: Page<Fact>; entity: EntityRef; onOpenEntity?: (iri: string) => void }) {
  const source = useWorkspace();
  const [extra, setExtra] = useState<{ base: Page<Fact>; items: Fact[]; cursor: string | null } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const current = extra?.base === page ? extra : null;
  const items = current ? current.items : page.items;
  const cursor = current ? current.cursor : page.next_cursor;
  const loadMore = async () => {
    if (!cursor || !source.facts || loading) return;
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    setLoading(true);
    setError(null);
    try {
      const next = await source.facts(entity, category, cursor, abort.signal);
      const seen = new Set(items.map((fact) => fact.fact_id));
      setExtra({ base: page, items: [...items, ...next.items.filter((fact) => !seen.has(fact.fact_id))], cursor: next.next_cursor });
    } catch (failure) {
      if (!abort.signal.aborted) setError(failure);
    } finally {
      if (!abort.signal.aborted) setLoading(false);
    }
  };
  const total = page.total_count;
  return (
    <div className="more-category">
      <h4>
        {MORE_NAMES[category]} <span className="meta">{total != null ? `${items.length} of ${total}` : `${items.length} shown${cursor ? " · more exist" : ""}`}</span>
      </h4>
      {items.map((fact) =>
        literalText(fact) ? (
          <p key={fact.fact_id} data-fact-id={fact.fact_id} className="more-literal">
            {literalText(fact)} <span className="meta">· {predicateName(fact.predicate_iri)}</span>
          </p>
        ) : (
          <AxiomBlock key={fact.fact_id} ontology={entity.ontology_version_id} factId={fact.fact_id} subject={entity} onOpen={onOpenEntity} />
        ),
      )}
      {error ? <ErrorNote error={error} onRetry={loadMore} what={MORE_NAMES[category]} /> : null}
      {cursor &&
        (source.facts ? (
          <button type="button" className="btn btn-sm" disabled={loading} onClick={loadMore}>
            {loading ? "Loading…" : `Load more ${MORE_NAMES[category].toLowerCase()}`}
          </button>
        ) : (
          <p className="meta">More are recorded but are not available in this view.</p>
        ))}
    </div>
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
  titleOverride,
  showOriginal = true,
  showProfile = true,
}: {
  side: "source" | "target";
  entity: EntityRef;
  context: AsyncState<EntityContext>;
  ontologyLabel: string;
  cite: (factId: string) => CitedFact | undefined;
  onOpenEntity?: (iri: string) => void;
  compact?: boolean;
  titleOverride?: string;
  /** Study publications admit components individually; absent ones are not rendered at all. */
  showOriginal?: boolean;
  showProfile?: boolean;
}) {
  const [allSynonyms, setAllSynonyms] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const source = useWorkspace();
  const report = useWorkspaceAction();
  const ctx = context.data;
  const entities = useMemo(() => [entity], [entity]);
  const profile = useExplanation("entity_profile", entities);

  const parentIris = useMemo(
    () => (ctx?.parents.items ?? []).map((fact) => (fact.hierarchy_projection?.parent.iri ?? (fact.value?.term_type === "iri" ? fact.value.iri : null))).filter((iri): iri is string => Boolean(iri)),
    [ctx],
  );
  const parentLabel = useLabelLookup(entity.ontology_version_id, parentIris);
  useEffect(() => {
    if (ctx && source.kind === "exploration") seedLabel(entity.ontology_version_id, entity.iri, ctx.preferred_label.value);
  }, [ctx, entity, source.kind]);

  const title = titleOverride ?? sideTitle(side, entity.kind);
  const label = ctx?.preferred_label.value;
  const synonyms = (ctx?.synonyms.items ?? []).filter((fact) => literalText(fact) && literalText(fact) !== label);
  const hiddenDuplicates = (ctx?.synonyms.items ?? []).filter((fact) => literalText(fact) === label);
  const alternate = ctx?.categories.alternate_definitions;
  const defining = DEFINING.flatMap((category) => ctx?.categories[category]?.items ?? []);
  const definingStatuses = DEFINING.map((category) => ctx?.categories[category]?.status).filter(Boolean) as string[];
  const definingStatus = definingStatuses.includes("absent_in_scope") ? "absent_in_scope" : definingStatuses.find((status) => status !== "available") ?? "absent_in_scope";
  const definingUnprepared = definingStatuses.some((status) => status === "not_exported");
  const definingPartial = DEFINING.some((category) => ctx?.categories[category]?.truncated);
  const scopeNote = ctx ? scopeText(ctx) : undefined;
  const moreCategories = MORE.filter((category) => (ctx?.categories[category]?.items.length ?? 0) > 0);

  return (
    <article className={`entity-card entity-card-${side}${compact ? " compact" : ""}`} aria-label={`${title}${label ? `: ${label}` : ""}`}>
      <div className="entity-card-head">
        <SideMarker side={side} />
        <span className={`eyebrow text-${side}`}>{title}</span>
        <span className="meta">{ontologyLabel}</span>
        <span className="entity-card-spacer" />
        <CopyIri iri={entity.iri} label={`Copy IRI of ${label ?? curie(entity.iri)}`} onCopied={() => report({ type: "copy_iri", side })} />
      </div>
      <div className="entity-title-block">
        <h2 className="entity-title">
          {label ?? (context.loading ? <span className="skeleton skeleton-title" /> : <span className="muted">No label in scope</span>)}
        </h2>
        <span className="iri">{entity.iri}</span>
      </div>

      {context.error ? <ErrorNote error={context.error} onRetry={context.reload} what="Entity context" /> : null}
      {!ctx && context.loading ? <Skeleton lines={5} /> : null}

      {ctx && showOriginal && (
        <>
          <Section title="Definition" meta={ctx.definitions.items.length ? `Original · ${predicateName(ctx.definitions.items[0].predicate_iri)}` : undefined}>
            {ctx.definitions.items.length ? (
              <div className="definitions">
                {ctx.definitions.items.map((fact) => (
                  <p key={fact.fact_id} data-fact-id={fact.fact_id} className="definition-text" lang={fact.value?.term_type === "literal" ? fact.value.language ?? undefined : undefined}>
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
                  <p key={fact.fact_id} data-fact-id={fact.fact_id} className="definition-text definition-alt">
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
                    <li key={fact.fact_id} data-fact-id={fact.fact_id} className="chip" title={predicateName(fact.predicate_iri)}>
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
            ) : hiddenDuplicates.length > 0 ? (
              <p className="meta">Only a synonym identical to the label is recorded.</p>
            ) : (
              <EmptyReason status={ctx.synonyms.status} what="synonyms" scopeNote={scopeNote} reason={ctx.synonyms.reason} />
            )}
            {synonyms.length > 0 && hiddenDuplicates.length > 0 && (
              <p className="meta">
                {hiddenDuplicates.length} {hiddenDuplicates.length === 1 ? "synonym identical to the label is" : "synonyms identical to the label are"} not repeated; citations to {hiddenDuplicates.length === 1 ? "it" : "them"} still open the record.
              </p>
            )}
          </Section>

          <Section title="Defining facts" meta={defining.length ? `${defining.length} ${definingPartial ? "shown · more exist" : "asserted"}` : undefined}>
            {defining.length ? (
              <div className="fact-stack">
                {defining.map((fact) => (
                  <AxiomBlock key={fact.fact_id} ontology={entity.ontology_version_id} factId={fact.fact_id} subject={entity} onOpen={onOpenEntity} />
                ))}
              </div>
            ) : (
              <EmptyReason status={definingStatus} what="restrictions or other defining axioms" scopeNote={scopeNote} />
            )}
            {definingUnprepared && !defining.length && definingStatus === "absent_in_scope" && <p className="meta">Some other kinds of defining axioms were not prepared for this view.</p>}
          </Section>

          <Section title="Parents" meta={parentIris.length ? `${parentIris.length} asserted${parentIris.length > 1 ? " · multiple inheritance" : ""}${ctx.parents.truncated ? " · more exist" : ""}` : undefined}>
            {parentIris.length ? (
              <ul className="parent-list">
                {ctx.parents.items.map((fact) => {
                  const iri = fact.hierarchy_projection?.parent.iri ?? (fact.value?.term_type === "iri" ? fact.value.iri : null);
                  if (!iri) return null;
                  const text = parentLabel(iri);
                  return (
                    <li key={fact.fact_id} data-fact-id={fact.fact_id}>
                      <button type="button" className={`parent-link parent-${side}`} onClick={() => onOpenEntity?.(iri)} disabled={!onOpenEntity}>
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

          {moreCategories.length > 0 && (
            <div className="card-section">
              <button type="button" className="btn btn-sm" aria-expanded={moreOpen} onClick={() => setMoreOpen((value) => !value)}>
                {moreOpen ? "Hide other recorded facts" : `Other recorded facts (${moreCategories.reduce((sum, c) => sum + (ctx.categories[c]?.total_count ?? ctx.categories[c]?.items.length ?? 0), 0)})`}
              </button>
              {moreOpen && (
                <div className="fact-stack">
                  {moreCategories.map((category) => (
                    <MoreCategory key={category} category={category} page={ctx.categories[category]} entity={entity} onOpenEntity={onOpenEntity} />
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

      {showProfile && <GeneratedProfile state={profile} subject={entity} cite={cite} />}
    </article>
  );
}
