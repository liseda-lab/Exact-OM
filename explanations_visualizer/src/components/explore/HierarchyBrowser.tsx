"use client";

// Independent hierarchy browser for one ontology. It never depends on the match graph:
// every step is a bounded query by entity through the active workspace, with the hierarchy
// basis stated explicitly. A workspace that holds only part of an ontology says where its
// information ends instead of showing an empty tree.

import { useCallback, useEffect, useRef, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconChevronDown, IconChevronUp, SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { EntitySearch } from "@/components/explore/EntitySearch";
import { useContinuation, type Continued } from "@/components/explore/useContinuation";
import { appendPage, shownText } from "@/lib/continuation";
import { curie, KIND_NAMES } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import type { EntityKind, HierarchyEdge } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import type { Basis } from "@/lib/workspace/types";
import { useWorkspace, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

export type { Basis } from "@/lib/workspace/types";
export type FocusVia = "parent" | "child" | "search" | "card" | "return";

const BASES: { value: Basis; label: string; note: string }[] = [
  { value: "literal_asserted", label: "Asserted", note: "Named parents exactly as asserted in the ontology file." },
  { value: "structural_navigation", label: "Structural navigation", note: "Also named parts of equivalence and intersection axioms, by a documented rule." },
  { value: "reasoner_inferred", label: "Reasoner inferred", note: "Only when a reasoner was run for this bundle." },
];

function ParentBranch({
  ontology,
  iri,
  kind,
  basis,
  depth,
  onFocus,
  side,
}: {
  ontology: string;
  iri: string;
  kind: EntityKind;
  basis: Basis;
  depth: number;
  onFocus: (iri: string, kind: EntityKind, via: FocusVia) => void;
  side: "source" | "target";
}) {
  const workspace = useWorkspace();
  const report = useWorkspaceAction();
  const [open, setOpen] = useState(false);
  const state = useAsync(open ? `${workspace.key}|parents|${ontology}|${kind}|${iri}|${basis}` : null, (signal) =>
    workspace.hierarchy({ ontology_version_id: ontology, iri, kind }, "parents", basis, null, signal),
  );
  const loadMore = useCallback((cursor: string, signal: AbortSignal) => workspace.hierarchy({ ontology_version_id: ontology, iri, kind }, "parents", basis, cursor, signal), [workspace, ontology, iri, kind, basis]);
  const pages = useContinuation(state.data, loadMore, edgeKey);
  const parents = pages.items.filter((edge) => edge.child.iri === iri);
  const label = useLabelLookup(ontology, [iri, ...parents.map((edge) => edge.parent.iri)]);
  return (
    <li className="tree-item">
      <div className="tree-row">
        <button
          type="button"
          className="tree-toggle"
          aria-expanded={open}
          aria-label={`${open ? "Hide" : "Show"} parents of ${label(iri)?.value ?? curie(iri)}`}
          onClick={() => {
            report({ type: open ? "hierarchy_collapse" : "hierarchy_expand", side, iri });
            setOpen((value) => !value);
          }}
          disabled={depth > 12}
        >
          {open ? <IconChevronDown /> : <IconChevronUp />}
        </button>
        <button type="button" className={`tree-label tree-${side}`} onClick={() => onFocus(iri, kind, "parent")}>
          {label(iri)?.value ?? <span className="muted">{curie(iri)}</span>}
        </button>
        <span className="iri">{curie(iri)}</span>
      </div>
      {open && (
        <div className="tree-children">
          {state.error ? <ErrorNote error={state.error} onRetry={state.reload} /> : null}
          {!state.data && !state.error ? <Skeleton lines={1} title={false} /> : null}
          {state.data && !parents.length && <EmptyReason status={state.data.status} what="named parent" reason={state.data.reason} />}
          {parents.length > 0 && (
            <ul>
              {parents.map((edge) => (
                <ParentBranch key={edge.id} ontology={ontology} iri={edge.parent.iri} kind={edge.parent.kind} basis={basis} depth={depth + 1} onFocus={onFocus} side={side} />
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

/** A hierarchy page that continues on request; failures stay local and keep what is shown. */
function EdgeContinuation({ state, noun }: { state: Continued<HierarchyEdge>; noun: { one: string; many: string } }) {
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
  return null;
}

export function HierarchyBrowser({
  side,
  ontology,
  ontologyLabel,
  focusIri,
  kind = "class",
  pinnedIri,
  pinnedKind = "class",
  onFocus,
  reasonerStatus,
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
  reasonerStatus?: string;
  entityCount?: number;
  scopeNote?: React.ReactNode;
  onOpenContext?: (iri: string, kind: EntityKind) => void;
  headingLevel?: 2 | 3;
}) {
  const workspace = useWorkspace();
  const report = useWorkspaceAction();
  const [basis, setBasis] = useState<Basis>("literal_asserted");
  const [childPages, setChildPages] = useState<HierarchyEdge[]>([]);
  const [childCursor, setChildCursor] = useState<string | null>(null);
  const [childMoreError, setChildMoreError] = useState<unknown>(null);
  const focus = focusIri ? { ontology_version_id: ontology, iri: focusIri, kind } : null;

  const parents = useAsync(focus ? `${workspace.key}|p|${ontology}|${kind}|${focusIri}|${basis}` : null, (signal) => workspace.hierarchy(focus!, "parents", basis, null, signal));
  const children = useAsync(focus ? `${workspace.key}|c|${ontology}|${kind}|${focusIri}|${basis}` : null, (signal) => workspace.hierarchy(focus!, "children", basis, null, signal));
  useEffect(() => {
    setChildPages(children.data?.items ?? []);
    setChildCursor(children.data?.next_cursor ?? null);
    setChildMoreError(null);
  }, [children.data]);

  const moreController = useRef<AbortController | null>(null);
  useEffect(() => () => moreController.current?.abort(), [ontology, focusIri, kind, basis]);
  const loadMoreChildren = useCallback(async () => {
    if (!focusIri || !childCursor) return;
    moreController.current?.abort();
    const controller = new AbortController();
    moreController.current = controller;
    try {
      const page = await workspace.hierarchy({ ontology_version_id: ontology, iri: focusIri, kind }, "children", basis, childCursor, controller.signal);
      if (controller.signal.aborted) return;
      setChildPages((list) => appendPage(list, page.items, edgeKey));
      setChildCursor(page.next_cursor);
    } catch (error) {
      if (!controller.signal.aborted) setChildMoreError(error);
    }
  }, [basis, childCursor, focusIri, kind, ontology, workspace]);

  const loadMoreParents = useCallback(
    (cursor: string, signal: AbortSignal) => workspace.hierarchy({ ontology_version_id: ontology, iri: focusIri!, kind }, "parents", basis, cursor, signal),
    [workspace, ontology, focusIri, kind, basis],
  );
  const parentPages = useContinuation(parents.data, focusIri ? loadMoreParents : null, edgeKey);
  const parentEdges = parentPages.items.filter((edge) => edge.child.iri === focusIri);
  const label = useLabelLookup(ontology, [focusIri ?? "", pinnedIri ?? "", ...parentEdges.map((edge) => edge.parent.iri), ...childPages.map((edge) => edge.child.iri)].filter(Boolean));
  const prepared = (value: Basis) =>
    workspace.kind === "exploration" ? value !== "reasoner_inferred" || reasonerStatus === "available" : workspace.capabilities.bases.includes(value);
  const Heading = headingLevel === 2 ? "h2" : "h3";
  const Sub = headingLevel === 2 ? "h3" : "h4";
  const sideName = side === "source" ? "Source" : "Target";
  const partialScope = workspace.capabilities.navigation === "partial";

  return (
    <section className={`card browser browser-${side}`} aria-label={`${sideName} ontology browser`}>
      <div className="browser-head">
        <div className="browser-title">
          <SideMarker side={side} />
          <Heading>{/^(Source|Target) ontology$/.test(ontologyLabel) ? ontologyLabel : `${sideName} ontology · ${ontologyLabel}`}</Heading>
          {entityCount != null && <span className="meta">{entityCount.toLocaleString()} entities</span>}
        </div>
        <EntitySearch
          ontology={ontology}
          label={`Search the ${side} ontology`}
          placeholder="Search by label, synonym prefix or full IRI"
          onChoose={(item) => {
            report({ type: "search", side });
            onFocus(item.entity.iri, item.entity.kind, "search");
          }}
        />
        <div className="basis-group" role="radiogroup" aria-label="Hierarchy basis">
          {BASES.map((option) => {
            const disabled = !prepared(option.value);
            return (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={basis === option.value}
                aria-disabled={disabled}
                title={option.note}
                className={basis === option.value ? "basis-option on" : disabled ? "basis-option unavailable" : "basis-option"}
                onClick={() => !disabled && setBasis(option.value)}
              >
                {option.label}
                {disabled ? " · not prepared" : ""}
              </button>
            );
          })}
        </div>
        <p className="meta">{BASES.find((option) => option.value === basis)?.note}</p>
      </div>
      {scopeNote}
      {partialScope && (
        <p className="note browser-note" role="note">
          {workspace.capabilities.scopeNote ?? "Only part of this ontology is available here."} Where the prepared information ends, the browser says so.
        </p>
      )}
      {!focusIri ? (
        <div className="browser-body">
          <p className="note">Search for an entity to start browsing this ontology.</p>
        </div>
      ) : (
        <div className="browser-body">
          <div className="browser-section">
            <div className="section-head">
              <Sub className="eyebrow">Parents</Sub>
              {parents.data && (
                <span className="meta">
                  {parentEdges.length
                    ? `${parentEdges.length}${parentPages.complete ? "" : parentPages.total != null ? ` of ${parentPages.total}` : " shown"}${parentEdges.length > 1 || (parentPages.total ?? 0) > 1 ? " · multiple inheritance" : ""}${parents.data.status === "partial" ? " · more may exist" : ""}`
                    : ""}
                </span>
              )}
            </div>
            {parents.error ? <ErrorNote error={parents.error} onRetry={parents.reload} what="Parents" /> : null}
            {!parents.data && !parents.error ? <Skeleton lines={2} title={false} /> : null}
            {parents.data && !parentEdges.length && <EmptyReason status={parents.data.status} what="named parent" reason={parents.data.reason} />}
            {parentEdges.length > 0 && (
              <ul className="tree">
                {parentEdges.map((edge) => (
                  <ParentBranch key={edge.id} ontology={ontology} iri={edge.parent.iri} kind={edge.parent.kind} basis={basis} depth={0} onFocus={onFocus} side={side} />
                ))}
              </ul>
            )}
            <EdgeContinuation state={parentPages} noun={PARENT_NOUN} />
          </div>

          <div className={`focus-card focus-${side}`}>
            <span className={`eyebrow text-${side}`}>Focused {KIND_NAMES[kind]}</span>
            <span className="focus-label">{label(focusIri)?.value ?? <span className="muted">No label in scope</span>}</span>
            <span className="iri">{focusIri}</span>
            <div className="focus-actions">
              {onOpenContext && (
                <button type="button" className="btn btn-sm" onClick={() => onOpenContext(focusIri, kind)}>
                  Open full context
                </button>
              )}
              {pinnedIri && (pinnedIri !== focusIri || pinnedKind !== kind) && (
                <button type="button" className="btn btn-sm" onClick={() => onFocus(pinnedIri, pinnedKind, "return")}>
                  Return to {label(pinnedIri)?.value ?? curie(pinnedIri)}
                </button>
              )}
            </div>
          </div>

          <div className="browser-section">
            <div className="section-head">
              <Sub className="eyebrow">Children</Sub>
              {children.data && (
                <span className="meta">
                  {childPages.length ? (children.data.total_count != null ? `${childPages.length} of ${children.data.total_count}` : `${childPages.length} shown${children.data.status === "partial" ? " · more may exist" : ""}`) : ""}
                </span>
              )}
            </div>
            {children.error ? <ErrorNote error={children.error} onRetry={children.reload} what="Children" /> : null}
            {!children.data && !children.error ? <Skeleton lines={2} title={false} /> : null}
            {children.data && !childPages.length && <EmptyReason status={children.data.status} what="named child" reason={children.data.reason} />}
            {childPages.length > 0 && (
              <ul className="child-list">
                {childPages.map((edge) => (
                  <li key={edge.id}>
                    <button type="button" className={`tree-label tree-${side}`} onClick={() => onFocus(edge.child.iri, edge.child.kind, "child")}>
                      {label(edge.child.iri)?.value ?? <span className="muted">{curie(edge.child.iri)}</span>}
                    </button>
                    <span className="iri">{curie(edge.child.iri)}</span>
                    {edgeNote(edge) && <span className="meta">· {edgeNote(edge)}</span>}
                    {pinnedIri === edge.child.iri && pinnedKind === edge.child.kind && <span className="pill">compared entity</span>}
                  </li>
                ))}
              </ul>
            )}
            {childMoreError ? <ErrorNote error={childMoreError} onRetry={loadMoreChildren} /> : null}
            {childCursor && (
              <button type="button" className="btn btn-sm" onClick={loadMoreChildren}>
                Load next 50 children
              </button>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
