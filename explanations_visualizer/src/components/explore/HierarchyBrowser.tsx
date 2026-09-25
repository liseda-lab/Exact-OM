"use client";

// Independent hierarchy browser for one ontology. It never depends on the match graph:
// every step is a bounded query by entity, with the hierarchy basis stated explicitly.

import { useCallback, useEffect, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconChevronDown, IconChevronUp, SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { EntitySearch } from "@/components/explore/EntitySearch";
import { getJson } from "@/lib/api";
import { curie, KIND_NAMES } from "@/lib/iri";
import { useLabels } from "@/lib/labels";
import type { EntityKind, HierarchyEdge, HierarchyPage } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";

export type Basis = "literal_asserted" | "structural_navigation" | "reasoner_inferred";

const BASES: { value: Basis; label: string; note: string }[] = [
  { value: "literal_asserted", label: "Asserted", note: "Named parents exactly as asserted in the ontology file." },
  { value: "structural_navigation", label: "Structural navigation", note: "Also named parts of equivalence and intersection axioms, by a documented rule." },
  { value: "reasoner_inferred", label: "Reasoner inferred", note: "Only when a reasoner was run for this bundle." },
];

function fetchHierarchy(ontology: string, iri: string, kind: EntityKind, direction: "parents" | "children", basis: Basis, cursor?: string | null, signal?: AbortSignal) {
  return getJson<HierarchyPage>("/api/v1/hierarchy", { ontology_version_id: ontology, iri, kind, direction, basis, limit: 50, cursor }, signal);
}

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
  onFocus: (iri: string) => void;
  side: "source" | "target";
}) {
  const [open, setOpen] = useState(false);
  const state = useAsync(open ? `parents|${ontology}|${iri}|${basis}` : null, (signal) => fetchHierarchy(ontology, iri, kind, "parents", basis, null, signal));
  const parents = (state.data?.items ?? []).filter((edge) => edge.child.iri === iri);
  const label = useLabels(ontology, [iri, ...parents.map((edge) => edge.parent.iri)]);
  return (
    <li className="tree-item">
      <div className="tree-row">
        <button type="button" className="tree-toggle" aria-expanded={open} aria-label={`${open ? "Hide" : "Show"} parents of ${label(iri)?.value ?? curie(iri)}`} onClick={() => setOpen((value) => !value)} disabled={depth > 12}>
          {open ? <IconChevronDown /> : <IconChevronUp />}
        </button>
        <button type="button" className={`tree-label tree-${side}`} onClick={() => onFocus(iri)}>
          {label(iri)?.value ?? <span className="muted">{curie(iri)}</span>}
        </button>
        <span className="iri">{curie(iri)}</span>
      </div>
      {open && (
        <div className="tree-children">
          {state.error ? <ErrorNote error={state.error} onRetry={state.reload} /> : null}
          {!state.data && !state.error ? <Skeleton lines={1} title={false} /> : null}
          {state.data && !parents.length && <p className="meta">No named parent in this basis and scope.</p>}
          {parents.length > 0 && (
            <ul>
              {parents.map((edge) => (
                <ParentBranch key={edge.id} ontology={ontology} iri={edge.parent.iri} kind={kind} basis={basis} depth={depth + 1} onFocus={onFocus} side={side} />
              ))}
            </ul>
          )}
        </div>
      )}
    </li>
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
  onFocus,
  reasonerStatus,
  entityCount,
  scopeNote,
  onOpenContext,
}: {
  side: "source" | "target";
  ontology: string;
  ontologyLabel: string;
  focusIri: string | null;
  kind?: EntityKind;
  pinnedIri?: string | null;
  onFocus: (iri: string) => void;
  reasonerStatus?: string;
  entityCount?: number;
  scopeNote?: React.ReactNode;
  onOpenContext?: (iri: string) => void;
}) {
  const [basis, setBasis] = useState<Basis>("literal_asserted");
  const [childPages, setChildPages] = useState<HierarchyEdge[]>([]);
  const [childCursor, setChildCursor] = useState<string | null>(null);
  const [childMoreError, setChildMoreError] = useState<unknown>(null);

  const parents = useAsync(focusIri ? `p|${ontology}|${focusIri}|${basis}` : null, (signal) => fetchHierarchy(ontology, focusIri!, kind, "parents", basis, null, signal));
  const children = useAsync(focusIri ? `c|${ontology}|${focusIri}|${basis}` : null, (signal) => fetchHierarchy(ontology, focusIri!, kind, "children", basis, null, signal));
  useEffect(() => {
    setChildPages(children.data?.items ?? []);
    setChildCursor(children.data?.next_cursor ?? null);
    setChildMoreError(null);
  }, [children.data]);

  const loadMoreChildren = useCallback(async () => {
    if (!focusIri || !childCursor) return;
    try {
      const page = await fetchHierarchy(ontology, focusIri, kind, "children", basis, childCursor);
      setChildPages((list) => [...list, ...page.items]);
      setChildCursor(page.next_cursor);
    } catch (error) {
      setChildMoreError(error);
    }
  }, [basis, childCursor, focusIri, kind, ontology]);

  const parentEdges = (parents.data?.items ?? []).filter((edge) => edge.child.iri === focusIri);
  const label = useLabels(ontology, [focusIri ?? "", pinnedIri ?? "", ...parentEdges.map((edge) => edge.parent.iri), ...childPages.map((edge) => edge.child.iri)].filter(Boolean));
  const reasonerAvailable = reasonerStatus === "available";

  return (
    <section className={`card browser browser-${side}`} aria-label={`${side === "source" ? "Source" : "Target"} ontology browser`}>
      <div className="browser-head">
        <div className="browser-title">
          <SideMarker side={side} />
          <h2>{/^(Source|Target) ontology$/.test(ontologyLabel) ? ontologyLabel : `${side === "source" ? "Source" : "Target"} ontology · ${ontologyLabel}`}</h2>
          {entityCount != null && <span className="meta">{entityCount.toLocaleString()} entities</span>}
        </div>
        <EntitySearch ontology={ontology} label={`Search the ${side} ontology`} placeholder="Search by label, synonym prefix or full IRI" onChoose={(item) => onFocus(item.entity.iri)} />
        <div className="basis-group" role="radiogroup" aria-label="Hierarchy basis">
          {BASES.map((option) => {
            const disabled = option.value === "reasoner_inferred" && !reasonerAvailable;
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
      {!focusIri ? (
        <div className="browser-body">
          <p className="note">Search for an entity to start browsing this ontology.</p>
        </div>
      ) : (
        <div className="browser-body">
          <div className="browser-section">
            <div className="section-head">
              <h3 className="eyebrow">Parents</h3>
              {parents.data && <span className="meta">{parentEdges.length ? `${parentEdges.length}${parentEdges.length > 1 ? " · multiple inheritance" : ""}` : ""}</span>}
            </div>
            {parents.error ? <ErrorNote error={parents.error} onRetry={parents.reload} what="Parents" /> : null}
            {!parents.data && !parents.error ? <Skeleton lines={2} title={false} /> : null}
            {parents.data && !parentEdges.length && <EmptyReason status={parents.data.status} what="named parent" reason={parents.data.reason} />}
            {parentEdges.length > 0 && (
              <ul className="tree">
                {parentEdges.map((edge) => (
                  <ParentBranch key={edge.id} ontology={ontology} iri={edge.parent.iri} kind={kind} basis={basis} depth={0} onFocus={onFocus} side={side} />
                ))}
              </ul>
            )}
          </div>

          <div className={`focus-card focus-${side}`}>
            <span className={`eyebrow text-${side}`}>Focused {KIND_NAMES[kind]}</span>
            <span className="focus-label">{label(focusIri)?.value ?? <span className="muted">No label in scope</span>}</span>
            <span className="iri">{focusIri}</span>
            <div className="focus-actions">
              {onOpenContext && (
                <button type="button" className="btn btn-sm" onClick={() => onOpenContext(focusIri)}>
                  Open full context
                </button>
              )}
              {pinnedIri && pinnedIri !== focusIri && (
                <button type="button" className="btn btn-sm" onClick={() => onFocus(pinnedIri)}>
                  Return to {label(pinnedIri)?.value ?? curie(pinnedIri)}
                </button>
              )}
            </div>
          </div>

          <div className="browser-section">
            <div className="section-head">
              <h3 className="eyebrow">Children</h3>
              {children.data && (
                <span className="meta">
                  {childPages.length ? `${childPages.length} of ${children.data.total_count ?? "?"}` : ""}
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
                    <button type="button" className={`tree-label tree-${side}`} onClick={() => onFocus(edge.child.iri)}>
                      {label(edge.child.iri)?.value ?? <span className="muted">{curie(edge.child.iri)}</span>}
                    </button>
                    <span className="iri">{curie(edge.child.iri)}</span>
                    {edgeNote(edge) && <span className="meta">· {edgeNote(edge)}</span>}
                    {pinnedIri === edge.child.iri && <span className="pill">compared entity</span>}
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
