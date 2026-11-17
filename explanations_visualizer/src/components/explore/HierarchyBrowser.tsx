"use client";

// Independent hierarchy browser for one ontology. It never depends on the match graph:
// every step is a bounded, paginated query by typed entity through the active workspace, with
// the hierarchy basis stated explicitly. A workspace that holds only part of an ontology says
// where its information ends instead of showing an empty tree. The partial ELK mode (23 C02)
// shows positive relationships only; an empty partial result is unknown, never a negative.

import { useCallback, useEffect, useState } from "react";

import { Disclosure } from "@/components/common/Disclosure";
import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconChevronDown, IconChevronUp, SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { EntitySearch } from "@/components/explore/EntitySearch";
import { useContinuation, type Continued } from "@/components/explore/useContinuation";
import { shownText } from "@/lib/continuation";
import { curie, KIND_NAMES } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import type { EntityKind, EntityRef, EntryPointItem, HierarchyEdge, HierarchyPage } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { BASIS_OPTIONS, basisAvailability, entryStart, offeredBases, unsupportedConstructs, type EntryStart } from "@/lib/workspace/navigation";
import type { Basis, PartialInference } from "@/lib/workspace/types";
import { useWorkspace, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

export type { Basis } from "@/lib/workspace/types";
export type FocusVia = "parent" | "child" | "search" | "card" | "return";

// The chosen mode per workspace scope and side survives tab switches and deliberate navigation.
const chosenModes = new Map<string, Basis>();
// Declared curated starting points per workspace, ontology, kind and basis, as served (HR4).
const curatedStarts = new Map<string, string>();

const isPartial = (basis: Basis) => basis === "elk_inferred_partial";

function emptyText(basis: Basis, what: "parents" | "children"): string {
  if (isPartial(basis)) return `ELK returned no ${what} here. Its result is partial, so ${what} may still exist.`;
  return what === "parents" ? "No parent shown." : "No children shown.";
}

/** An empty page: a partial ELK result is unknown, stated plainly; anything else keeps its status wording. */
function Empty({ page, basis, what }: { page: HierarchyPage; basis: Basis; what: "parents" | "children" }) {
  if (isPartial(basis) && page.status === "partial") return <p className="empty-reason">{emptyText(basis, what)}</p>;
  return <EmptyReason status={page.status} what={what} reason={page.reason} absent={emptyText(basis, what)} />;
}

/** Typed identity of a node in an ancestry path (punned IRIs of different kinds are distinct). */
export const typedKey = (entity: { ontology_version_id: string; kind: string; iri: string }) => `${entity.ontology_version_id}|${entity.kind}|${entity.iri}`;

/** Levels of ancestors shown under one focus; deeper levels continue by refocusing (R6). */
export const MAX_ANCESTOR_DEPTH = 12;

/**
 * Move keyboard focus to the earlier occurrence of a class in this ancestry path: `levelsUp`
 * tree items above this row, or the focused entity's card when `levelsUp` is null.
 */
function focusEarlier(from: HTMLElement, levelsUp: number | null) {
  let item: HTMLElement | null = levelsUp === null ? null : from.closest("li.tree-item");
  for (let step = 0; levelsUp !== null && step < levelsUp && item; step += 1) item = item.parentElement?.closest<HTMLElement>("li.tree-item") ?? null;
  const target = item?.querySelector<HTMLElement>(":scope > .tree-row .tree-label") ?? from.closest(".browser")?.querySelector<HTMLElement>(".focus-card") ?? null;
  if (!target) return;
  target.scrollIntoView({ block: "center" });
  if (target.tabIndex < 0 && !target.matches("button")) target.setAttribute("tabindex", "-1");
  target.focus({ preventScroll: true });
}

function ParentBranch({
  ontology,
  iri,
  kind,
  basis,
  depth,
  path,
  onFocus,
  side,
}: {
  ontology: string;
  iri: string;
  kind: EntityKind;
  basis: Basis;
  depth: number;
  /** Typed keys from the focused entity down to this node's child: the ancestry this node extends. */
  path: string[];
  onFocus: (iri: string, kind: EntityKind, via: FocusVia) => void;
  side: "source" | "target";
}) {
  const workspace = useWorkspace();
  const report = useWorkspaceAction();
  const [open, setOpen] = useState(false);
  const key = typedKey({ ontology_version_id: ontology, kind, iri });
  // A class already on this path is a cycle (possible in partial or asserted data): no further
  // reads, and an explicit way back to where it first appears (21 F06).
  const earlier = path.indexOf(key);
  const atLimit = depth >= MAX_ANCESTOR_DEPTH;
  const state = useAsync(open && earlier < 0 && !atLimit ? `${workspace.key}|parents|${ontology}|${kind}|${iri}|${basis}` : null, (signal) =>
    workspace.hierarchy({ ontology_version_id: ontology, iri, kind }, "parents", basis, null, signal),
  );
  const loadMore = useCallback((cursor: string, signal: AbortSignal) => workspace.hierarchy({ ontology_version_id: ontology, iri, kind }, "parents", basis, cursor, signal), [workspace, ontology, iri, kind, basis]);
  const pages = useContinuation(state.data, loadMore, edgeKey);
  const parents = pages.items.filter((edge) => edge.child.iri === iri && edge.child.kind === kind);
  const label = useLabelLookup(ontology, [iri, ...parents.map((edge) => edge.parent.iri)]);
  const name = label(iri)?.value ?? curie(iri);
  if (earlier >= 0) {
    return (
      <li className="tree-item tree-cycle">
        <div className="tree-row">
          <span className="tree-toggle-spacer" aria-hidden="true" />
          <span className={`tree-label-static text-${side}`}>{name}</span>
          <span className="iri">{curie(iri)}</span>
          <span className="meta">· already in this path</span>
          {/* path[0] is the focused entity (its card); path[i] is the ancestor at depth i - 1. */}
          <button type="button" className="link-btn" onClick={(event) => focusEarlier(event.currentTarget, earlier === 0 ? null : depth - (earlier - 1))}>
            Show where it appears
          </button>
        </div>
      </li>
    );
  }
  return (
    <li className="tree-item">
      <div className="tree-row">
        {atLimit ? (
          <span className="tree-toggle-spacer" aria-hidden="true" />
        ) : (
          <button
            type="button"
            className="tree-toggle"
            aria-expanded={open}
            aria-label={`${open ? "Hide" : "Show"} parents of ${name}`}
            onClick={() => {
              report({ type: open ? "hierarchy_collapse" : "hierarchy_expand", side, iri });
              setOpen((value) => !value);
            }}
          >
            {open ? <IconChevronDown /> : <IconChevronUp />}
          </button>
        )}
        <button type="button" className={`tree-label tree-${side}`} data-tour={depth === 0 ? `parent-${side}` : undefined} onClick={() => onFocus(iri, kind, "parent")}>
          {label(iri)?.value ?? <span className="muted">{curie(iri)}</span>}
        </button>
        <span className="iri">{curie(iri)}</span>
      </div>
      {atLimit && (
        <p className="meta tree-limit">
          {MAX_ANCESTOR_DEPTH} levels are shown under one focus.{" "}
          <button type="button" className="link-btn" onClick={() => onFocus(iri, kind, "parent")}>
            Continue from {name}
          </button>
        </p>
      )}
      {open && (
        <div className="tree-children">
          {state.error ? <ErrorNote error={state.error} onRetry={state.reload} /> : null}
          {!state.data && !state.error ? <Skeleton lines={1} title={false} /> : null}
          {state.data && !parents.length && <Empty page={state.data} basis={basis} what="parents" />}
          {parents.length > 0 && (
            <ul>
              {parents.map((edge) => (
                <ParentBranch key={edge.id} ontology={ontology} iri={edge.parent.iri} kind={edge.parent.kind} basis={basis} depth={depth + 1} path={[...path, key]} onFocus={onFocus} side={side} />
              ))}
            </ul>
          )}
          <EdgeContinuation state={pages} noun={PARENT_NOUN} />
        </div>
      )}
    </li>
  );
}

const edgeKey = (edge: HierarchyEdge) => edge.id;
const PARENT_NOUN = { one: "parent", many: "parents" };
const CHILD_NOUN = { one: "child", many: "children" };
const ENTRY_NOUN = { one: "top-level entry", many: "top-level entries" };
const EQUIVALENT_NOUN = { one: "equivalent class", many: "equivalent classes" };

/** A page that continues on request; failures stay local and keep what is shown. */
function EdgeContinuation<T>({ state, noun }: { state: Continued<T>; noun: { one: string; many: string } }) {
  if (state.complete && !state.error) return null;
  return (
    <div className="continuation">
      <span className="meta">{shownText(state.items.length, state.total, state.hasMore, noun)}</span>
      {state.error ? <ErrorNote error={state.error} onRetry={state.loadMore} what={`More ${noun.many}`} /> : null}
      {state.hasMore && (
        <button type="button" className="btn btn-sm" disabled={state.loading} onClick={state.loadMore}>
          {state.loading ? "Loading…" : `Load more ${noun.many}`}
        </button>
      )}
    </div>
  );
}

function edgeNote(edge: HierarchyEdge): string | null {
  if (edge.basis === "structural_navigation") return edge.interpretation.rule ? `derived: ${edge.interpretation.rule}` : "derived by structural rule";
  if (edge.basis === "reasoner_inferred") return "inferred";
  if (edge.basis === "elk_inferred_partial") return "ELK";
  return null;
}

/** The partial ELK result in one sentence, with the details people may need on request. */
function PartialDetails({ partial }: { partial: PartialInference }) {
  const unsupported = unsupportedConstructs(partial);
  const version = partial.versions ? Object.entries(partial.versions).map(([name, value]) => `${name} ${value}`).join(", ") : partial.reasoner;
  return (
    <Disclosure label="About ELK partial results" openLabel="Hide ELK details" className="basis-details">
      <ul className="plain-list">
        <li>Each link was returned by the ELK reasoner{version ? ` (${version})` : ""} for the constructs it supports. It is not a proof: no justification was exported for individual links.</li>
        {unsupported.length > 0 && <li>Not covered: {unsupported.join(", ")}. Relationships that depend on them may be missing.</li>}
        <li>“Direct” means direct in the returned result; it may skip a level of the complete hierarchy. A missing link, an empty list or a missing equivalence is unknown, not a negative.</li>
        <li>Classes only{partial.unrepresented_class_count ? `; ${partial.unrepresented_class_count.toLocaleString()} classes are not represented in the result` : ""}.</li>
        {partial.consistency?.value === true && partial.consistency.complete === false && <li>The reasoner reported no inconsistency, but its check did not cover the unsupported constructs.</li>}
        {partial.source_imports_complete === false && <li>Imported ontologies were not complete in the reasoner&apos;s input.</li>}
      </ul>
    </Disclosure>
  );
}

/** How the partial ELK parents differ from the asserted ones for the focused entity. */
function BasisDifference({ focus, partialParents, complete }: { focus: EntityRef; partialParents: HierarchyEdge[]; complete: boolean }) {
  const workspace = useWorkspace();
  const asserted = useAsync(`${workspace.key}|p|${focus.ontology_version_id}|${focus.kind}|${focus.iri}|literal_asserted`, (signal) => workspace.hierarchy(focus, "parents", "literal_asserted", null, signal));
  if (!asserted.data || asserted.data.next_cursor || !complete) return null;
  const assertedParents = new Set(asserted.data.items.filter((edge) => edge.child.iri === focus.iri).map((edge) => `${edge.parent.kind}|${edge.parent.iri}`));
  const partial = new Set(partialParents.map((edge) => `${edge.parent.kind}|${edge.parent.iri}`));
  const added = [...partial].filter((key) => !assertedParents.has(key)).length;
  const notReturned = [...assertedParents].filter((key) => !partial.has(key)).length;
  return (
    <p className="meta basis-difference" role="note">
      {added === 0 && notReturned === 0
        ? "Same parents as Asserted."
        : [added ? `${added} ${added === 1 ? "parent" : "parents"} not asserted` : null, notReturned ? `${notReturned} asserted ${notReturned === 1 ? "parent" : "parents"} not returned by ELK (unknown, not removed)` : null].filter(Boolean).join(" · ")}
    </p>
  );
}

/**
 * Paginated typed entry points for the current mode; choosing one focuses it. Without a focus,
 * the page decides the start (`entryStart`): a declared curated default or the single root is
 * opened; several ordinary entries stay a list (21 F06, HR4).
 */
function EntryPoints({ ontology, kind, basis, side, onFocus, onStart }: { ontology: string; kind: EntityKind; basis: Basis; side: "source" | "target"; onFocus: (iri: string, kind: EntityKind, via: FocusVia) => void; onStart?: (start: EntryStart) => void }) {
  const workspace = useWorkspace();
  const first = useAsync(`${workspace.key}|entry|${ontology}|${kind}|${basis}`, (signal) => workspace.entryPoints!(ontology, kind, basis, null, signal));
  const start = entryStart(first.data, ontology, kind);
  const suggested = start.kind === "curated" ? start.entity : null;
  useEffect(() => {
    if (!first.data) return;
    const key = `${workspace.key}|${ontology}|${kind}|${basis}`;
    if (suggested) curatedStarts.set(key, suggested.iri);
    else curatedStarts.delete(key);
    // Only the page for this exact ontology, kind and basis reaches here (useAsync never shows
    // another key's data); a focus chosen meanwhile removes onStart, so it is never overridden.
    if (onStart) onStart(start);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [first.data]);
  const loadMore = useCallback((cursor: string, signal: AbortSignal) => workspace.entryPoints!(ontology, kind, basis, cursor, signal), [workspace, ontology, kind, basis]);
  const pages = useContinuation(first.data, loadMore, (item: EntryPointItem) => `${item.entity.kind}|${item.entity.iri}`);
  const label = useLabelLookup(ontology, pages.items.filter((item) => !item.preferred_label?.value).map((item) => item.entity.iri));
  return (
    <div className="entry-points">
      {first.error ? <ErrorNote error={first.error} onRetry={first.reload} what="Top-level entries" /> : null}
      {!first.data && !first.error ? <Skeleton lines={2} title={false} /> : null}
      {first.data && !pages.items.length && <EmptyReason status={first.data.status} what="top-level entries" reason={first.data.reason} absent={isPartial(basis) ? "ELK returned no top-level entries here." : "No top-level entries shown."} />}
      {pages.items.length > 0 && (
        <ul className="child-list">
          {pages.items.map((item) => (
            <li key={`${item.entity.kind}|${item.entity.iri}`}>
              <button type="button" className={`tree-label tree-${side}`} onClick={() => onFocus(item.entity.iri, item.entity.kind, "search")}>
                {item.preferred_label?.value ?? label(item.entity.iri)?.value ?? <span className="muted">{curie(item.entity.iri)}</span>}
              </button>
              <span className="iri">{curie(item.entity.iri)}</span>
              {suggested && item.entity.iri === suggested.iri && item.entity.kind === suggested.kind && <span className="pill pill-suggested">Suggested starting point</span>}
              {item.entry_kind === "component_entry" && <span className="meta">· entry to a group without a single root</span>}
            </li>
          ))}
        </ul>
      )}
      <EdgeContinuation state={pages} noun={ENTRY_NOUN} />
      {isPartial(basis) && first.data && pages.items.length > 0 && <p className="meta">Top-level in the ELK result: some may have parents that ELK did not return.</p>}
    </div>
  );
}

/** Members of the focused class's equivalence group in the partial result, when there are several. */
/**
 * Members of the focused class's equivalence group in the partial result (R4): bounded pages
 * with the service's cursor, a shown/total count, typed links that refocus this side, and
 * local loading, failure and Retry. A new focus, scope or basis starts from the first page; an
 * unlisted class is unknown, not "not equivalent".
 */
function Equivalents({ focus, count, basis, side, onFocus, Sub }: { focus: EntityRef; count: number; basis: Basis; side: "source" | "target"; onFocus: (iri: string, kind: EntityKind, via: FocusVia) => void; Sub: "h3" | "h4" }) {
  const workspace = useWorkspace();
  const first = useAsync(workspace.equivalents ? `${workspace.key}|eq|${focus.ontology_version_id}|${focus.kind}|${focus.iri}|${basis}` : null, (signal) => workspace.equivalents!(focus, basis, null, signal));
  const loadMore = useCallback((cursor: string, signal: AbortSignal) => workspace.equivalents!(focus, basis, cursor, signal), [workspace, focus, basis]);
  const pages = useContinuation(first.data, workspace.equivalents ? loadMore : null, (item: EntityRef) => `${item.ontology_version_id}|${item.kind}|${item.iri}`);
  const others = pages.items.filter((item) => !(item.iri === focus.iri && item.kind === focus.kind && item.ontology_version_id === focus.ontology_version_id));
  const label = useLabelLookup(focus.ontology_version_id, others.map((item) => item.iri));
  // The group's size counts the focused class itself.
  const total = Math.max(0, (first.data?.total_count ?? count) - 1);
  const shown = others.length;
  return (
    <div className="browser-section equivalents" aria-busy={first.loading || pages.loading || undefined}>
      <div className="section-head">
        <Sub className="eyebrow">Equivalent in the ELK result</Sub>
        <span className="meta">{first.data ? (pages.hasMore || shown < total ? `${shown} of ${total} shown` : `${total} other ${total === 1 ? "class" : "classes"}`) : `${total} other ${total === 1 ? "class" : "classes"}`}</span>
      </div>
      {first.error ? <ErrorNote error={first.error} onRetry={first.reload} what="Equivalent classes" /> : null}
      {!first.data && !first.error ? <Skeleton lines={1} title={false} /> : null}
      {shown > 0 && (
        <ul className="child-list">
          {others.map((item) => (
            <li key={`${item.kind}|${item.iri}`}>
              <button type="button" className={`tree-label tree-${side}`} onClick={() => onFocus(item.iri, item.kind, "search")}>
                {label(item.iri)?.value ?? <span className="muted">{curie(item.iri)}</span>}
              </button>
              <span className="iri">{curie(item.iri)}</span>
              {item.kind !== "class" && <span className="meta">· {KIND_NAMES[item.kind]}</span>}
            </li>
          ))}
        </ul>
      )}
      <EdgeContinuation state={pages} noun={EQUIVALENT_NOUN} />
      <p className="meta">Equivalence returned by ELK for the parts it supports; a class not listed may still be equivalent.</p>
    </div>
  );
}

export function HierarchyBrowser({
  side,
  ontology,
  ontologyLabel,
  focusIri: explicitIri,
  kind = "class",
  pinnedIri,
  pinnedKind = "class",
  onFocus,
  entityCount,
  scopeNote,
  onOpenContext,
  headingLevel = 2,
}: {
  side: "source" | "target";
  ontology: string;
  ontologyLabel: string;
  focusIri: string | null;
  kind?: EntityKind;
  pinnedIri?: string | null;
  pinnedKind?: EntityKind;
  onFocus: (iri: string, kind: EntityKind, via: FocusVia) => void;
  entityCount?: number;
  scopeNote?: React.ReactNode;
  onOpenContext?: (iri: string, kind: EntityKind) => void;
  headingLevel?: 2 | 3;
}) {
  const workspace = useWorkspace();
  const report = useWorkspaceAction();
  const modeKey = `${workspace.key}|${side}|${ontology}`;
  const [basis, setBasisState] = useState<Basis>(() => chosenModes.get(modeKey) ?? "literal_asserted");
  const setBasis = (value: Basis) => {
    chosenModes.set(modeKey, value);
    setBasisState(value);
  };
  const [modeNote, setModeNote] = useState<string | null>(null);
  const [entriesOpen, setEntriesOpen] = useState(false);
  // A declared curated start applies only while no explicit, remembered or pinned focus exists;
  // it is local navigation (no URL, history or remembered choice), so a reload starts there again
  // and any deliberate focus always wins (HR4).
  const startKey = `${workspace.key}|${side}|${ontology}|${kind}`;
  const [suggested, setSuggested] = useState<{ key: string; entity: EntityRef; basis: Basis } | null>(null);
  const suggestedIri = !explicitIri && suggested?.key === startKey ? suggested.entity.iri : null;
  const focusIri = explicitIri ?? suggestedIri;
  const focus = focusIri ? { ontology_version_id: ontology, iri: focusIri, kind } : null;
  const available = (value: Basis) => basisAvailability(workspace.capabilities, ontology, kind, value);
  const offered = offeredBases(workspace.capabilities, ontology);
  const partialInfo = workspace.capabilities.navigationByOntology?.[ontology]?.partial ?? null;
  // A mode chosen for a class may not apply to a property: fall back to Asserted, saying why.
  const effective: Basis = available(basis).available ? basis : "literal_asserted";
  useEffect(() => {
    if (effective !== basis && focusIri) setModeNote(available(basis).reason);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effective, basis, focusIri]);

  const parents = useAsync(focus ? `${workspace.key}|p|${ontology}|${kind}|${focusIri}|${effective}` : null, (signal) => workspace.hierarchy(focus!, "parents", effective, null, signal));
  const children = useAsync(focus ? `${workspace.key}|c|${ontology}|${kind}|${focusIri}|${effective}` : null, (signal) => workspace.hierarchy(focus!, "children", effective, null, signal));
  // Cursors are bound to their query and basis; each base page starts its own continuation.
  const loadMoreParents = useCallback((cursor: string, signal: AbortSignal) => workspace.hierarchy({ ontology_version_id: ontology, iri: focusIri!, kind }, "parents", effective, cursor, signal), [workspace, ontology, focusIri, kind, effective]);
  const loadMoreChildren = useCallback((cursor: string, signal: AbortSignal) => workspace.hierarchy({ ontology_version_id: ontology, iri: focusIri!, kind }, "children", effective, cursor, signal), [workspace, ontology, focusIri, kind, effective]);
  const parentPages = useContinuation(parents.data, focusIri ? loadMoreParents : null, edgeKey);
  const childPages = useContinuation(children.data, focusIri ? loadMoreChildren : null, edgeKey);
  const parentEdges = parentPages.items.filter((edge) => edge.child.iri === focusIri && edge.child.kind === kind);
  const childEdges = childPages.items.filter((edge) => edge.parent.iri === focusIri && edge.parent.kind === kind);
  const label = useLabelLookup(ontology, [focusIri ?? "", pinnedIri ?? "", ...parentEdges.map((edge) => edge.parent.iri), ...childEdges.map((edge) => edge.child.iri)].filter(Boolean));
  const Heading = headingLevel === 2 ? "h2" : "h3";
  const Sub = headingLevel === 2 ? "h3" : "h4";
  const sideName = side === "source" ? "Source" : "Target";
  const partialScope = workspace.capabilities.navigation === "partial";
  const away = Boolean(pinnedIri && focusIri && (pinnedIri !== focusIri || pinnedKind !== kind));
  const entryPoints = Boolean(workspace.entryPoints) && workspace.capabilities.navigationByOntology?.[ontology]?.entryPoints === "available";
  const semantics = (parents.data as HierarchyPage | null)?.entity_semantics ?? null;
  const optionNote = BASIS_OPTIONS.find((option) => option.value === effective)?.note;
  // Labelled only under a basis whose entry page declares it (partial ELK declares none).
  const isSuggestedStart = Boolean(focusIri && ((suggestedIri === focusIri && suggested?.basis === effective) || curatedStarts.get(`${workspace.key}|${ontology}|${kind}|${effective}`) === focusIri));

  return (
    <section className={`card browser browser-${side}`} aria-label={`${sideName} ontology browser`}>
      <div className="browser-head">
        <div className="browser-title">
          <SideMarker side={side} />
          <Heading>{/^(Source|Target) ontology$/.test(ontologyLabel) ? ontologyLabel : `${sideName} ontology · ${ontologyLabel}`}</Heading>
          {entityCount != null && <span className="meta">{entityCount.toLocaleString()} entities</span>}
        </div>
        {pinnedIri && (
          // The compared entity stays identifiable; the way back sits with the search and mode controls (21 F02).
          <div className="compared-row">
            <span className="meta">Compared</span>
            <span className={`compared-name text-${side}`}>{label(pinnedIri)?.value ?? curie(pinnedIri)}</span>
            {away ? (
              <button type="button" className="btn btn-sm" data-tour={`return-${side}`} onClick={() => onFocus(pinnedIri, pinnedKind, "return")}>
                Back to compared concept
              </button>
            ) : (
              <span className="compared-state">Viewing compared concept</span>
            )}
          </div>
        )}
        <EntitySearch
          ontology={ontology}
          label={`Search the ${side} ontology`}
          placeholder="Search by label, synonym or IRI"
          tourId={`search-${side}`}
          onChoose={(item) => {
            report({ type: "search", side });
            onFocus(item.entity.iri, item.entity.kind, "search");
          }}
        />
        <div
          className="basis-group"
          role="radiogroup"
          aria-label="Relationships shown"
          onKeyDown={(event) => {
            // The ARIA radio pattern: arrows move between modes and choose an available one.
            const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
            if (!step) return;
            event.preventDefault();
            const buttons = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>("[role=radio]"));
            const at = buttons.indexOf(document.activeElement as HTMLButtonElement);
            const next = buttons[(Math.max(0, at) + step + buttons.length) % buttons.length];
            next?.focus();
            next?.click();
          }}
        >
          {offered.map((value) => {
            const option = BASIS_OPTIONS.find((item) => item.value === value)!;
            const state = available(value);
            return (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={effective === value}
                aria-disabled={!state.available}
                tabIndex={effective === value || (!offered.includes(effective) && value === offered[0]) ? 0 : -1}
                className={effective === value ? "basis-option on" : !state.available ? "basis-option unavailable" : "basis-option"}
                onClick={() => {
                  if (!state.available) {
                    setModeNote(state.reason);
                    return;
                  }
                  setModeNote(null);
                  setBasis(value);
                }}
              >
                {option.label}
                {!state.available ? <span className="sr-only"> (not available: {state.reason})</span> : null}
              </button>
            );
          })}
        </div>
        <p className="meta" aria-live="polite">
          {modeNote ?? optionNote}
        </p>
        {isPartial(effective) && partialInfo && <PartialDetails partial={partialInfo} />}
        {entryPoints && (
          <div className="entry-toggle">
            <button type="button" className="btn btn-quiet btn-sm" aria-expanded={entriesOpen} onClick={() => setEntriesOpen((value) => !value)}>
              {entriesOpen ? "Hide top-level classes" : kind === "class" ? "Top-level classes" : `Top-level ${KIND_NAMES[kind]}s`}
            </button>
          </div>
        )}
      </div>
      {scopeNote}
      {partialScope && (
        <p className="meta browser-note" role="note">
          {workspace.capabilities.scopeNote ?? "Only part of this ontology is available here."}
        </p>
      )}
      {entryPoints && (entriesOpen || !focusIri) && (
        <div className="browser-section">
          <Sub className="eyebrow">Top-level {kind === "class" ? "classes" : `${KIND_NAMES[kind]}s`}</Sub>
          <EntryPoints
            ontology={ontology}
            kind={kind}
            basis={effective}
            side={side}
            onFocus={(iri, entityKind, via) => {
              setEntriesOpen(false);
              onFocus(iri, entityKind, via);
            }}
            onStart={
              !focusIri
                ? (start) => {
                    if (start.kind === "curated") setSuggested({ key: startKey, entity: start.entity, basis: effective });
                    // One root (legacy and single-root ontologies): focus it as before (21 F06).
                    else if (start.kind === "single") onFocus(start.entity.iri, start.entity.kind, "search");
                  }
                : undefined
            }
          />
        </div>
      )}
      {!focusIri ? (
        !entryPoints && (
          <div className="browser-body">
            <p className="note">Search for an entity to start browsing this ontology.</p>
          </div>
        )
      ) : (
        <div className="browser-body">
          <div className="browser-section">
            <div className="section-head">
              <Sub className="eyebrow">Parents</Sub>
              {parents.data && (
                <span className="meta">
                  {parentEdges.length
                    ? `${parentEdges.length}${parentPages.complete ? "" : parentPages.total != null ? ` of ${parentPages.total}` : " shown"}${parentEdges.length > 1 || (parentPages.total ?? 0) > 1 ? " · multiple inheritance" : ""}${parents.data.status === "partial" && !isPartial(effective) ? " · more may exist" : ""}`
                    : ""}
                </span>
              )}
            </div>
            {parents.error ? <ErrorNote error={parents.error} onRetry={parents.reload} what="Parents" /> : null}
            {!parents.data && !parents.error ? <Skeleton lines={2} title={false} /> : null}
            {parents.data && !parentEdges.length && <Empty page={parents.data} basis={effective} what="parents" />}
            {parentEdges.length > 0 && (
              <ul className="tree">
                {parentEdges.map((edge) => (
                  <ParentBranch key={edge.id} ontology={ontology} iri={edge.parent.iri} kind={edge.parent.kind} basis={effective} depth={0} path={[typedKey({ ontology_version_id: ontology, kind, iri: focusIri })]} onFocus={onFocus} side={side} />
                ))}
              </ul>
            )}
            <EdgeContinuation state={parentPages} noun={PARENT_NOUN} />
            {isPartial(effective) && focus && parents.data && <BasisDifference focus={focus} partialParents={parentEdges} complete={parentPages.complete} />}
          </div>

          <div className={`focus-card focus-${side}`} tabIndex={-1} aria-label={`Focused ${KIND_NAMES[kind]}: ${label(focusIri)?.value ?? curie(focusIri)}`}>
            <span className={`eyebrow text-${side}`}>Focused {KIND_NAMES[kind]}{pinnedIri && !away ? " · compared" : ""}</span>
            <span className="focus-label">{label(focusIri)?.value ?? <span className="muted">{label(focusIri)?.status === "loading" ? "Loading label…" : "No label in scope"}</span>}</span>
            <span className="iri">{focusIri}</span>
            {isSuggestedStart && (
              // A navigation choice for this review, not an ontology statement (HR4).
              <p className="suggested-start">
                <span className="pill pill-suggested">Suggested starting point</span>
                {!entriesOpen && entryPoints && (
                  <button type="button" className="link-btn" onClick={() => setEntriesOpen(true)}>
                    Show all top-level {kind === "class" ? "classes" : `${KIND_NAMES[kind]}s`}
                  </button>
                )}
              </p>
            )}
            {onOpenContext && (
              <div className="focus-actions">
                <button type="button" className="btn btn-sm" onClick={() => onOpenContext(focusIri, kind)}>
                  Open full context
                </button>
              </div>
            )}
          </div>
          {isPartial(effective) && focus && semantics && semantics.equivalent_entity_count > 1 && <Equivalents focus={focus} count={semantics.equivalent_entity_count} basis={effective} side={side} onFocus={onFocus} Sub={Sub} />}

          <div className="browser-section">
            <div className="section-head">
              <Sub className="eyebrow">Children</Sub>
              {children.data && childEdges.length > 0 && <span className="meta">{shownText(childEdges.length, childPages.total, childPages.hasMore, CHILD_NOUN)}</span>}
            </div>
            {children.error ? <ErrorNote error={children.error} onRetry={children.reload} what="Children" /> : null}
            {!children.data && !children.error ? <Skeleton lines={2} title={false} /> : null}
            {children.data && !childEdges.length && <Empty page={children.data} basis={effective} what="children" />}
            {childEdges.length > 0 && (
              <ul className="child-list">
                {childEdges.map((edge) => (
                  <li key={edge.id}>
                    <button type="button" className={`tree-label tree-${side}`} data-tour={`child-${side}`} onClick={() => onFocus(edge.child.iri, edge.child.kind, "child")}>
                      {label(edge.child.iri)?.value ?? <span className="muted">{curie(edge.child.iri)}</span>}
                    </button>
                    <span className="iri">{curie(edge.child.iri)}</span>
                    {edgeNote(edge) && <span className="meta">· {edgeNote(edge)}</span>}
                    {pinnedIri === edge.child.iri && pinnedKind === edge.child.kind && <span className="pill">compared entity</span>}
                  </li>
                ))}
              </ul>
            )}
            <EdgeContinuation state={childPages} noun={CHILD_NOUN} />
          </div>
        </div>
      )}
    </section>
  );
}
