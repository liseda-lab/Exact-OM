"use client";

// A meaning card for one entity. Source and target cards are the same component with equal
// prominence; every section states its own availability instead of silently disappearing.
// The card reads through the active workspace, so exploration, study and tutorial share it.
// Every displayed category continues from its own cursor (19 F16): definitions, alternate
// definitions, synonyms, defining facts, parents and other recorded facts say how many are
// shown, never claim completeness while more exist, and load further pages on request.

import { useCallback, useEffect, useMemo, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconCopy, SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { GeneratedProfile, type CitedFact } from "@/components/explore/GeneratedBlock";
import { AxiomBlock } from "@/components/owl/AxiomBlock";
import { curie, predicateName, sideTitle } from "@/lib/iri";
import { shownText } from "@/lib/continuation";
import { useLabelLookup } from "@/lib/labelSource";
import { seedLabel } from "@/lib/labels";
import type { EntityContext, EntityRef, Fact, Page } from "@/lib/types";
import { useContinuation, type Continued } from "@/components/explore/useContinuation";
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

const factKey = (fact: Fact) => fact.fact_id;
const NOUNS: Record<string, { one: string; many: string }> = {
  definitions: { one: "definition", many: "definitions" },
  alternate_definitions: { one: "alternate definition", many: "alternate definitions" },
  synonyms: { one: "synonym", many: "synonyms" },
  parents: { one: "parent", many: "parents" },
  restrictions: { one: "restriction", many: "restrictions" },
  types: { one: "type", many: "types" },
  assertions: { one: "assertion", many: "assertions" },
  domains: { one: "domain", many: "domains" },
  ranges: { one: "range", many: "ranges" },
  characteristics: { one: "characteristic", many: "characteristics" },
};
const nounFor = (category: string) => NOUNS[category] ?? { one: (MORE_NAMES[category] ?? category).toLowerCase().replace(/s$/, ""), many: (MORE_NAMES[category] ?? category).toLowerCase() };

/** Reads further pages of one fact category from the active source, when it can. */
function useFactPages(entity: EntityRef, category: string, page: Page<Fact> | undefined): Continued<Fact> {
  const source = useWorkspace();
  const load = useCallback((cursor: string, signal: AbortSignal) => source.facts!(entity, category, cursor, signal), [source, entity, category]);
  return useContinuation(page, source.facts ? load : null, factKey);
}

/** Shown/total counts, then a continuation, a local failure with retry, or an honest limit. */
function Continuation<T>({ state, noun }: { state: Continued<T>; noun: { one: string; many: string } }) {
  if (state.complete && !state.error) return null;
  return (
    <div className="continuation">
      <span className="meta">{shownText(state.items.length, state.total, state.hasMore, noun)}</span>
      {state.error ? <ErrorNote error={state.error} onRetry={state.loadMore} what={`More ${noun.many}`} /> : null}
      {state.hasMore &&
        (state.available ? (
          <button type="button" className="btn btn-sm" disabled={state.loading} aria-busy={state.loading || undefined} onClick={state.loadMore}>
            {state.loading ? "Loading…" : `Load more ${noun.many}`}
          </button>
        ) : (
          <span className="meta">More are recorded but are not available in this view.</span>
        ))}
    </div>
  );
}

function FactLiteral({ fact, className }: { fact: Fact; className: string }) {
  return (
    <p data-fact-id={fact.fact_id} className={className} lang={fact.value?.term_type === "literal" ? fact.value.language ?? undefined : undefined}>
      {literalText(fact) ?? "Structured value: open the original axiom."}
    </p>
  );
}

/** One recorded fact category under "Other recorded facts". */
function MoreCategory({ category, page, entity, onOpenEntity }: { category: string; page: Page<Fact>; entity: EntityRef; onOpenEntity?: (iri: string) => void }) {
  const pages = useFactPages(entity, category, page);
  return (
    <div className="more-category">
      <h4>
        {MORE_NAMES[category]} <span className="meta">{pages.total != null ? `${pages.items.length} of ${pages.total}` : `${pages.items.length} shown${pages.hasMore ? " · more exist" : ""}`}</span>
      </h4>
      {pages.items.map((fact) =>
        literalText(fact) ? (
          <p key={fact.fact_id} data-fact-id={fact.fact_id} className="more-literal">
            {literalText(fact)} <span className="meta">· {predicateName(fact.predicate_iri)}</span>
          </p>
        ) : (
          <AxiomBlock key={fact.fact_id} ontology={entity.ontology_version_id} factId={fact.fact_id} subject={entity} onOpen={onOpenEntity} />
        ),
      )}
      <Continuation state={pages} noun={nounFor(category)} />
    </div>
  );
}

/** One defining-fact category (restrictions, types, …) with its own continuation. */
function DefiningCategory({ category, page, entity, onOpenEntity }: { category: string; page: Page<Fact>; entity: EntityRef; onOpenEntity?: (iri: string) => void }) {
  const pages = useFactPages(entity, category, page);
  return (
    <div className="defining-category" data-category={category}>
      {pages.items.map((fact) => (
        <AxiomBlock key={fact.fact_id} ontology={entity.ontology_version_id} factId={fact.fact_id} subject={entity} onOpen={onOpenEntity} />
      ))}
      <Continuation state={pages} noun={nounFor(category)} />
    </div>
  );
}

interface ParentItem {
  key: string;
  iri: string;
  factId: string | null;
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
  /** Also gates the request: an unshown description is never read (it may be restricted). */
  showProfile?: boolean;
}) {
  const [allSynonyms, setAllSynonyms] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const source = useWorkspace();
  const report = useWorkspaceAction();
  const ctx = context.data;
  const entities = useMemo(() => [entity], [entity]);
  const profile = useExplanation("entity_profile", showProfile ? entities : null);

  const definitions = useFactPages(entity, "definitions", ctx?.definitions);
  const alternate = useFactPages(entity, "alternate_definitions", ctx?.categories.alternate_definitions);
  const synonymPages = useFactPages(entity, "synonyms", ctx?.synonyms);
  const parentBase = useMemo(
    () =>
      ctx
        ? {
            items: ctx.parents.items.flatMap((fact): ParentItem[] => {
              const iri = fact.hierarchy_projection?.parent.iri ?? (fact.value?.term_type === "iri" ? fact.value.iri : null);
              return iri ? [{ key: fact.hierarchy_projection?.id ?? fact.fact_id, iri, factId: fact.fact_id }] : [];
            }),
            next_cursor: ctx.parents.next_cursor,
            total_count: ctx.parents.total_count,
          }
        : null,
    [ctx],
  );
  const loadParents = useCallback(
    (cursor: string, signal: AbortSignal) =>
      source.hierarchy(entity, "parents", "literal_asserted", cursor, signal).then((page) => ({
        items: page.items.map((edge): ParentItem => ({ key: edge.id, iri: edge.parent.iri, factId: null })),
        next_cursor: page.next_cursor,
      })),
    [source, entity],
  );
  const parents = useContinuation(parentBase, loadParents, (item) => item.key);

  const parentIris = useMemo(() => parents.items.map((item) => item.iri), [parents.items]);
  const parentLabel = useLabelLookup(entity.ontology_version_id, parentIris);
  useEffect(() => {
    if (ctx && source.kind === "exploration") seedLabel(entity.ontology_version_id, entity.iri, ctx.preferred_label.value);
  }, [ctx, entity, source.kind]);

  const title = titleOverride ?? sideTitle(side, entity.kind);
  const label = ctx?.preferred_label.value;
  const synonyms = synonymPages.items.filter((fact) => literalText(fact) && literalText(fact) !== label);
  const hiddenDuplicates = synonymPages.items.filter((fact) => literalText(fact) === label);
  const definingCategories = DEFINING.filter((category) => (ctx?.categories[category]?.items.length ?? 0) > 0);
  const definingStatuses = DEFINING.map((category) => ctx?.categories[category]?.status).filter(Boolean) as string[];
  const definingStatus = definingStatuses.includes("absent_in_scope") ? "absent_in_scope" : definingStatuses.find((status) => status !== "available") ?? "absent_in_scope";
  const definingUnprepared = definingStatuses.some((status) => status === "not_exported");
  const definingTotals = definingCategories.map((category) => ctx?.categories[category]?.total_count ?? null);
  const definingMore = definingCategories.some((category) => Boolean(ctx?.categories[category]?.next_cursor));
  const definingMeta = !definingCategories.length
    ? undefined
    : definingTotals.every((value) => value != null)
      ? `${definingTotals.reduce((sum, value) => sum! + value!, 0)} recorded`
      : `${definingCategories.reduce((sum, category) => sum + (ctx?.categories[category]?.items.length ?? 0), 0)} on the first page${definingMore ? " · more exist" : ""}`;
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
          <Section title="Definition" meta={definitions.items.length ? `Original · ${predicateName(definitions.items[0].predicate_iri)}` : undefined}>
            {definitions.items.length ? (
              <div className="definitions" data-category="definitions">
                {definitions.items.map((fact) => (
                  <FactLiteral key={fact.fact_id} fact={fact} className="definition-text" />
                ))}
                {definitions.complete && definitions.items.length > 1 && <p className="meta">{definitions.items.length} definitions are asserted; all are shown.</p>}
                <Continuation state={definitions} noun={NOUNS.definitions} />
              </div>
            ) : (
              <EmptyReason status={ctx.definitions.status} what="definition" scopeNote={scopeNote} reason={ctx.definitions.reason} />
            )}
            {alternate.items.length > 0 && (
              <div className="alternate-definitions" data-category="alternate_definitions">
                <span className="meta">Alternate {alternate.items.length === 1 ? "definition" : "definitions"} · {predicateName(alternate.items[0].predicate_iri)}</span>
                {alternate.items.map((fact) => (
                  <FactLiteral key={fact.fact_id} fact={fact} className="definition-text definition-alt" />
                ))}
                <Continuation state={alternate} noun={NOUNS.alternate_definitions} />
              </div>
            )}
          </Section>

          <Section title="Also called" meta={synonyms.length ? `${synonyms.length} ${synonyms.length === 1 ? "synonym" : "synonyms"}${synonymPages.complete ? "" : " loaded"}` : undefined}>
            <div data-category="synonyms">
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
              ) : hiddenDuplicates.length > 0 && synonymPages.complete ? (
                <p className="meta">Only a synonym identical to the label is recorded.</p>
              ) : synonymPages.items.length === 0 ? (
                <EmptyReason status={ctx.synonyms.status} what="synonyms" scopeNote={scopeNote} reason={ctx.synonyms.reason} />
              ) : null}
              {synonyms.length > 0 && hiddenDuplicates.length > 0 && (
                <p className="meta">
                  {hiddenDuplicates.length} {hiddenDuplicates.length === 1 ? "synonym identical to the label is" : "synonyms identical to the label are"} not repeated; citations to {hiddenDuplicates.length === 1 ? "it" : "them"} still open the record.
                </p>
              )}
              <Continuation state={synonymPages} noun={NOUNS.synonyms} />
            </div>
          </Section>

          <Section title="Defining facts" meta={definingMeta}>
            {definingCategories.length ? (
              <div className="fact-stack">
                {definingCategories.map((category) => (
                  <DefiningCategory key={category} category={category} page={ctx.categories[category]} entity={entity} onOpenEntity={onOpenEntity} />
                ))}
              </div>
            ) : (
              <EmptyReason status={definingStatus} what="restrictions or other defining axioms" scopeNote={scopeNote} />
            )}
            {definingUnprepared && !definingCategories.length && definingStatus === "absent_in_scope" && <p className="meta">Some other kinds of defining axioms were not prepared for this view.</p>}
          </Section>

          <Section title="Parents" meta={parents.items.length ? `${parents.items.length}${parents.complete ? " asserted" : parents.total != null ? ` of ${parents.total}` : " shown"}${parents.items.length > 1 || (parents.total ?? 0) > 1 ? " · multiple inheritance" : ""}` : undefined}>
            <div data-section="parents">
            {parents.items.length ? (
              <ul className="parent-list" data-category="parents">
                {parents.items.map((item) => {
                  const text = parentLabel(item.iri);
                  return (
                    <li key={item.key} data-fact-id={item.factId ?? undefined} data-edge-id={item.key}>
                      <button type="button" className={`parent-link parent-${side}`} onClick={() => onOpenEntity?.(item.iri)} disabled={!onOpenEntity}>
                        <span>{text?.value ?? curie(item.iri)}</span>
                        <span className="iri">{curie(item.iri)}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <EmptyReason status={ctx.parents.status} what="named parent" scopeNote={scopeNote} reason={ctx.parents.reason} />
            )}
            <Continuation state={parents} noun={NOUNS.parents} />
            </div>
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
