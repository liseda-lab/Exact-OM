"use client";

import { useMemo, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconSearch, SideMarker } from "@/components/common/Icons";
import { formatScore, orderDescription, primaryScore, type OrderBasis } from "@/components/explore/candidates";
import { curie, KIND_NAMES } from "@/lib/iri";
import { useLabels } from "@/lib/labels";
import type { Candidate, SourceSummary } from "@/lib/types";
import { useNarrow } from "@/lib/useMedia";

export function SourcePicker({
  sources,
  total,
  loading,
  error,
  onRetry,
  onMore,
  canLoadMore,
  selectedIri,
  onSelect,
}: {
  sources: SourceSummary[];
  total: number | null;
  loading: boolean;
  error: unknown;
  onRetry: () => void;
  onMore: () => void;
  canLoadMore: boolean;
  selectedIri: string | null;
  onSelect: (source: SourceSummary) => void;
}) {
  const [filter, setFilter] = useState("");
  // On narrow screens the list collapses to the current source, so the comparison is reachable.
  const narrow = useNarrow();
  const [open, setOpen] = useState(false);
  const collapsed = narrow && !open && Boolean(selectedIri);
  const ontology = sources[0]?.entity.ontology_version_id;
  const label = useLabels(ontology, sources.map((source) => source.entity.iri));
  const index = sources.findIndex((source) => source.entity.iri === selectedIri);
  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return sources;
    return sources.filter((source) => {
      const text = `${label(source.entity.iri)?.value ?? ""} ${source.entity.iri} ${curie(source.entity.iri)}`.toLowerCase();
      return text.includes(needle);
    });
  }, [filter, sources, label]);
  const kind = sources.find((source) => source.entity.iri === selectedIri)?.entity.kind ?? sources[0]?.entity.kind ?? "class";

  return (
    <section className="card rail-section" aria-labelledby="source-picker-h">
      <h2 id="source-picker-h" className="eyebrow">
        Source {KIND_NAMES[kind] ?? kind}
      </h2>
      {collapsed ? (
        <div className="collapsed-choice">
          <span className="source-item current">
            <SideMarker side="source" size="0.875rem" />
            <span className="source-item-text">
              <span className="source-item-label">{label(selectedIri!)?.value ?? curie(selectedIri!)}</span>
              <span className="iri">{curie(selectedIri!)}</span>
            </span>
          </span>
          <button type="button" className="btn btn-sm" aria-expanded={false} onClick={() => setOpen(true)}>
            Change source ({total ?? sources.length})
          </button>
        </div>
      ) : (
        <>
      <div className="search-field">
        <IconSearch />
        <label htmlFor="source-filter" className="sr-only">
          Filter sources in this run
        </label>
        <input id="source-filter" type="search" className="search-input" placeholder="Filter sources in this run" value={filter} onChange={(event) => setFilter(event.target.value)} />
      </div>
      {error ? <ErrorNote error={error} onRetry={onRetry} what="Sources" /> : null}
      {loading && !sources.length ? <Skeleton lines={3} title={false} /> : null}
      <ul className="source-list" aria-label="Sources in this run">
        {visible.map((source) => {
          const current = source.entity.iri === selectedIri;
          const text = label(source.entity.iri);
          return (
            <li key={source.entity_id}>
              <button
                type="button"
                className={current ? "source-item current" : "source-item"}
                aria-current={current ? "true" : undefined}
                onClick={() => {
                  onSelect(source);
                  setOpen(false);
                }}
              >
                <SideMarker side="source" size="0.875rem" />
                <span className="source-item-text">
                  <span className="source-item-label">{text?.value ?? (text?.status === "loading" ? "Loading label…" : curie(source.entity.iri))}</span>
                  <span className="iri">{curie(source.entity.iri)}</span>
                </span>
              </button>
            </li>
          );
        })}
        {!loading && !error && visible.length === 0 && <li className="meta">No loaded source matches this filter.</li>}
      </ul>
      <div className="rail-footer">
        <span className="meta">
          {index >= 0 ? `Source ${index + 1} of ${total ?? sources.length}` : `${total ?? sources.length} sources in this run`}
        </span>
        {canLoadMore && (
          <button type="button" className="btn btn-sm" onClick={onMore} disabled={loading}>
            Load more sources
          </button>
        )}
      </div>
        </>
      )}
    </section>
  );
}

export function CandidateList({
  candidates,
  basis,
  loading,
  error,
  onRetry,
  selectedPair,
  onSelect,
  truncated,
}: {
  candidates: Candidate[];
  basis: OrderBasis;
  loading: boolean;
  error: unknown;
  onRetry: () => void;
  selectedPair: string | null;
  onSelect: (candidate: Candidate) => void;
  truncated: boolean;
}) {
  const ontology = candidates[0]?.target.ontology_version_id;
  const label = useLabels(ontology, candidates.map((candidate) => candidate.target.iri));
  const narrow = useNarrow();
  const [open, setOpen] = useState(false);
  const collapsed = narrow && !open && Boolean(selectedPair);
  return (
    <section className="card rail-section rail-candidates" aria-labelledby="candidates-h">
      <div className="rail-heading">
        <h2 id="candidates-h" className="eyebrow">
          Target candidates
        </h2>
        <span className="meta">{candidates.length ? `${candidates.length}${truncated ? "+" : ""}` : ""}</span>
      </div>
      {candidates.length > 0 && <p className="meta rail-sub">{orderDescription(basis)}</p>}
      {error ? <ErrorNote error={error} onRetry={onRetry} what="Candidates" /> : null}
      {loading && !candidates.length ? <Skeleton lines={4} title={false} /> : null}
      {!loading && !error && candidates.length === 0 && <p className="note">Exact saved no candidates for this source.</p>}
      <ol className="candidate-list">
        {candidates.map((candidate, index) => {
          const current = candidate.pair_id === selectedPair;
          if (collapsed && !current) return null;
          const text = label(candidate.target.iri);
          const score = primaryScore(candidate);
          return (
            <li key={candidate.pair_id}>
              <button
                type="button"
                className={current ? "candidate-item current" : "candidate-item"}
                aria-current={current ? "true" : undefined}
                onClick={() => {
                  onSelect(candidate);
                  setOpen(false);
                }}
              >
                <span className="position-badge" aria-label={`Position ${index + 1}`}>
                  {index + 1}
                </span>
                <span className="candidate-text">
                  <span className="candidate-label">{text?.value ?? (text?.status === "loading" ? "Loading label…" : "No label in scope")}</span>
                  <span className="iri">{curie(candidate.target.iri)}</span>
                  <span className="candidate-meta">
                    {score ? <span>Matching score {formatScore(score.value)}</span> : <span>No score recorded</span>}
                    {candidate.saved_alignment_member === true && <span className="member-flag">In saved alignment</span>}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      {narrow && candidates.length > 1 && (
        <button type="button" className="btn btn-sm" aria-expanded={!collapsed} onClick={() => setOpen((value) => !value)}>
          {collapsed ? `Show all ${candidates.length} candidates` : "Show only the selected candidate"}
        </button>
      )}
    </section>
  );
}
