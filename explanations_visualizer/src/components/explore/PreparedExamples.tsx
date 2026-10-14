"use client";

// Prepared examples in the exploration rail (spec 24 HR2): a compact, keyboard-reachable switch
// between the development review's prepared examples and every source in the run. Examples come
// from the bound catalogue only; the full source pool, its paging and the matcher's candidate
// order are unchanged. Exploration only: the study and its shared workspace never mount this.

import { useMemo, useState } from "react";

import { Skeleton } from "@/components/common/ErrorNote";
import { SideMarker } from "@/components/common/Icons";
import { getJson } from "@/lib/api";
import { curie } from "@/lib/iri";
import { useLabels } from "@/lib/labels";
import { catalogueStateFromError, parseCatalogue, type CatalogueState, type ReviewCase } from "@/lib/reviewCatalogue";
import { useAsync } from "@/lib/useAsync";
import { useNarrow } from "@/lib/useMedia";

export type SourceMode = "examples" | "all";

/** The bound catalogue for the open package; one cancellable read, retried on request. */
export function useReviewCatalogue(packageId: string | null): { state: CatalogueState; reload: () => void } {
  const read = useAsync(packageId ? `review-bundle|${packageId}` : null, async (signal) => parseCatalogue(await getJson<unknown>("/api/v1/review-bundle", undefined, signal)));
  const state = useMemo<CatalogueState>(() => (read.data ? { status: "available", catalogue: read.data } : read.error ? catalogueStateFromError(read.error) : { status: "loading" }), [read.data, read.error]);
  return { state, reload: read.reload };
}

const FAILURE_TEXT: Record<string, string> = {
  incompatible: "Prepared examples are not shown: this service's review catalogue has a format this page does not read.",
  invalid: "Prepared examples are not shown: the service reports that its review catalogue failed validation.",
  denied: "Prepared examples are not available to this session.",
  unreachable: "Prepared examples could not be loaded: the service is not responding.",
  error: "Prepared examples could not be loaded.",
};

/** Visible reasons the examples are not offered; never an empty list presented as complete. */
export function CatalogueNotice({ state, otherRun, onRetry, requested = false }: { state: CatalogueState; otherRun: boolean; onRetry: () => void; requested?: boolean }) {
  if (state.status === "failed")
    return (
      <div className="note note-warn catalogue-note" role="status">
        <span>{FAILURE_TEXT[state.reason]}</span>{" "}
        <button type="button" className="link-btn" onClick={onRetry}>
          Try again
        </button>
      </div>
    );
  const unavailable = state.status === "available" && otherRun
    ? "The prepared examples belong to a saved run that is not in this bundle."
    : state.status === "available" && !state.catalogue.cases.length
      ? "No prepared examples are listed for this bundle. All sources remain available."
      : state.status === "absent" && requested
        ? "Prepared examples are not available in this bundle. All sources remain available."
        : null;
  if (unavailable) return <p className="note catalogue-note">{unavailable}{" "}<button type="button" className="link-btn" onClick={onRetry}>Try again</button></p>;
  return null;
}

/** Two choices in the ARIA radio pattern: arrows move and choose; Tab leaves the group. */
export function SourceModeSwitch({ mode, onMode, examples, sources }: { mode: SourceMode; onMode: (mode: SourceMode) => void; examples: number | null; sources: number | null }) {
  const options: { value: SourceMode; label: string; count: number | null }[] = [
    { value: "examples", label: "Prepared examples", count: examples },
    { value: "all", label: "All sources", count: sources },
  ];
  return (
    <div
      className="basis-group source-mode"
      role="radiogroup"
      aria-label="Sources shown"
      onKeyDown={(event) => {
        const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
        if (!step) return;
        event.preventDefault();
        const at = options.findIndex((option) => option.value === mode);
        const next = options[(at + step + options.length) % options.length];
        onMode(next.value);
        event.currentTarget.querySelector<HTMLButtonElement>(`[data-mode="${next.value}"]`)?.focus();
      }}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          data-mode={option.value}
          aria-checked={mode === option.value}
          tabIndex={mode === option.value ? 0 : -1}
          className={mode === option.value ? "basis-option on" : "basis-option"}
          onClick={() => onMode(option.value)}
        >
          {option.label}
          {option.count !== null && <span className="meta"> · {option.count.toLocaleString()}</span>}
        </button>
      ))}
    </div>
  );
}

/** The catalogue's examples in catalogue order; choosing one opens its source and a prepared pair. */
export function ExampleList({ state, currentCaseId, onChoose, otherRun }: { state: CatalogueState; currentCaseId: string | null; onChoose: (item: ReviewCase) => void; otherRun: boolean }) {
  const narrow = useNarrow();
  const [open, setOpen] = useState(false);
  const cases = state.status === "available" && !otherRun ? state.catalogue.cases : [];
  const label = useLabels(cases[0]?.source.ontology_version_id, cases.map((item) => item.source.iri));
  const collapsed = narrow && !open && Boolean(currentCaseId);
  if (state.status === "loading")
    return (
      <div aria-busy="true">
        <p className="meta">Loading prepared examples…</p>
        <Skeleton lines={3} title={false} />
      </div>
    );
  if (!cases.length) return null;
  const current = cases.find((item) => item.case_id === currentCaseId) ?? null;
  return (
    <>
      <p className="meta rail-sub">Pairs prepared for this development review. Every source and candidate stays available under All sources.</p>
      {collapsed && current ? (
        <div className="collapsed-choice">
          <span className="source-item current">
            <SideMarker side="source" size="0.875rem" />
            <span className="source-item-text">
              <span className="source-item-label">{label(current.source.iri)?.value ?? curie(current.source.iri)}</span>
              <span className="iri">{curie(current.source.iri)}</span>
            </span>
          </span>
          <button type="button" className="btn btn-sm" aria-expanded={false} onClick={() => setOpen(true)}>
            Change example ({cases.length})
          </button>
        </div>
      ) : (
        <ul className="source-list example-list" aria-label="Prepared examples">
          {cases.map((item) => {
            const isCurrent = item.case_id === currentCaseId;
            const text = label(item.source.iri);
            return (
              <li key={item.case_id}>
                <button
                  type="button"
                  className={isCurrent ? "source-item current" : "source-item"}
                  aria-current={isCurrent ? "true" : undefined}
                  onClick={() => {
                    onChoose(item);
                    setOpen(false);
                  }}
                >
                  <SideMarker side="source" size="0.875rem" />
                  <span className="source-item-text">
                    <span className="source-item-label">{text?.value ?? (text?.status === "loading" ? "Loading label…" : curie(item.source.iri))}</span>
                    <span className="iri">{curie(item.source.iri)}</span>
                    <span className="meta">
                      {item.targets.length} {item.targets.length === 1 ? "candidate" : "candidates"} prepared for review
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </>
  );
}
