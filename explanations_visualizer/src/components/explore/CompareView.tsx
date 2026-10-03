"use client";

// Main exploration flow. Reading and keyboard order: choose a candidate, read the two
// meaning cards and the comparison, then open optional details (hierarchy, evidence,
// decision trace, graph, scores). The pair workspace is the same composition the study's
// explanation condition uses. All navigation state is in the URL.

import { useCallback, useEffect, useMemo, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconChevronLeft, IconChevronRight } from "@/components/common/Icons";
import { formatScore, orderCandidates, primaryScore } from "@/components/explore/candidates";
import { DecisionTrace, ScoresPanel } from "@/components/explore/DecisionTrace";
import { ontologyName, useExplore } from "@/components/explore/ExploreContext";
import { ReviewPanel } from "@/components/explore/ReviewPanel";
import { CandidateList, SourcePicker } from "@/components/explore/SourceRail";
import { PairWorkspace, type Focus, type PairNavigation } from "@/components/workspace/PairWorkspace";
import { getJson } from "@/lib/api";
import { curie } from "@/lib/iri";
import type { Candidate, EntityKind, EntityRef, Page, PairTrace, SourceSummary } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { useNarrow } from "@/lib/useMedia";
import { useUrlState } from "@/lib/urlState";
import { useEntityContext as useWorkspaceEntityContext } from "@/lib/workspace/WorkspaceContext";

/** Kept for the browse page's full-context dialog. */
export function useEntityContext(entity: EntityRef | null) {
  return useWorkspaceEntityContext(entity);
}

const FIRST_LOAD_PAGES = 5;
const PAGE = 100;

function kindOf(value: string | null): EntityKind {
  return value === "object_property" || value === "data_property" || value === "individual" ? value : "class";
}

export function CompareView() {
  const state = useExplore();
  const [params, setParams] = useUrlState();
  const runId = params.get("run") ?? state.runs[0]?.run_id ?? null;
  const run = state.runs.find((item) => item.run_id === runId) ?? null;
  const sourceIri = params.get("source");
  const pairId = params.get("pair");
  const tab = params.get("details");

  // Sources: first page, then explicit "load more".
  const [extraSources, setExtraSources] = useState<SourceSummary[]>([]);
  const [sourceCursor, setSourceCursor] = useState<string | null>(null);
  const sources = useAsync<Page<SourceSummary>>(runId ? `sources|${runId}` : null, (signal) =>
    getJson<Page<SourceSummary>>(`/api/v1/runs/${encodeURIComponent(runId!)}/sources`, { limit: 100 }, signal),
  );
  useEffect(() => {
    setExtraSources([]);
    setSourceCursor(sources.data?.next_cursor ?? null);
  }, [sources.data]);
  const allSources = useMemo(() => [...(sources.data?.items ?? []), ...extraSources], [sources.data, extraSources]);
  const [sourceMoreError, setSourceMoreError] = useState<unknown>(null);
  const loadMoreSources = useCallback(async () => {
    if (!runId || !sourceCursor) return;
    try {
      const page = await getJson<Page<SourceSummary>>(`/api/v1/runs/${encodeURIComponent(runId)}/sources`, { limit: 100, cursor: sourceCursor });
      setExtraSources((list) => [...list, ...page.items]);
      setSourceCursor(page.next_cursor);
    } catch (error) {
      setSourceMoreError(error);
    }
  }, [runId, sourceCursor]);

  const source = allSources.find((item) => item.entity.iri === sourceIri) ?? null;
  useEffect(() => {
    if (!sourceIri && allSources.length) setParams({ source: allSources[0].entity.iri, pair: null }, "replace");
  }, [sourceIri, allSources, setParams]);

  // Candidates: the backend pages by pair identity, so a first bounded load is sorted by the
  // recorded rank and the remainder stays one explicit "load more" away.
  const sourceKind = source?.entity.kind ?? "class";
  const candidateKey = runId && sourceIri ? `cands|${runId}|${sourceKind}|${sourceIri}` : null;
  const first = useAsync<{ items: Candidate[]; cursor: string | null; total: number | null }>(candidateKey, async (signal) => {
    const items: Candidate[] = [];
    let cursor: string | null = null;
    let total: number | null = null;
    for (let pageIndex = 0; pageIndex < FIRST_LOAD_PAGES; pageIndex += 1) {
      const page: Page<Candidate> = await getJson<Page<Candidate>>(`/api/v1/runs/${encodeURIComponent(runId!)}/candidates`, { source: sourceIri!, source_kind: sourceKind, limit: PAGE, cursor }, signal);
      items.push(...page.items);
      total = page.total_count;
      cursor = page.next_cursor;
      if (!cursor) break;
    }
    return { items, cursor, total };
  });
  const [more, setMore] = useState<{ key: string | null; items: Candidate[]; cursor: string | null; loading: boolean; error: unknown }>({ key: null, items: [], cursor: null, loading: false, error: null });
  const continued = more.key === candidateKey ? more : null;
  const cursor = continued ? continued.cursor : first.data?.cursor ?? null;
  const loadMoreCandidates = useCallback(async () => {
    if (!runId || !sourceIri || !cursor) return;
    const base = continued ?? { key: candidateKey, items: [], cursor, loading: false, error: null };
    setMore({ ...base, loading: true, error: null });
    try {
      const page = await getJson<Page<Candidate>>(`/api/v1/runs/${encodeURIComponent(runId)}/candidates`, { source: sourceIri, source_kind: sourceKind, limit: PAGE, cursor });
      setMore({ key: candidateKey, items: [...base.items, ...page.items], cursor: page.next_cursor, loading: false, error: null });
    } catch (error) {
      setMore({ ...base, loading: false, error });
    }
  }, [runId, sourceIri, sourceKind, cursor, continued, candidateKey]);
  const loaded = useMemo(() => [...(first.data?.items ?? []), ...(continued?.items ?? [])], [first.data, continued]);
  const { ordered, basis } = useMemo(() => orderCandidates(loaded), [loaded]);
  const candidate = ordered.find((item) => item.pair_id === pairId) ?? null;
  useEffect(() => {
    if (ordered.length && !candidate) setParams({ pair: ordered[0].pair_id }, "replace");
  }, [ordered, candidate, setParams]);

  const sourceEntity = candidate?.source ?? source?.entity ?? null;
  const targetEntity = candidate?.target ?? null;
  const trace = useAsync<PairTrace>(runId && candidate ? `trace|${runId}|${candidate.pair_id}` : null, (signal) =>
    getJson<PairTrace>(`/api/v1/runs/${encodeURIComponent(runId!)}/pair`, { pair_id: candidate!.pair_id }, signal),
  );

  const pair = useMemo(
    () => (candidate && sourceEntity && targetEntity ? { source: sourceEntity, target: targetEntity, runId, pairId: candidate.pair_id } : null),
    [candidate, sourceEntity, targetEntity, runId],
  );

  // Hierarchy focus per side lives in the URL; a new candidate keeps the source side's focus.
  const navigation: PairNavigation = useMemo(() => {
    const focus = (iriKey: string, kindKey: string): Focus | null => {
      const iri = params.get(iriKey);
      return iri ? { iri, kind: kindOf(params.get(kindKey)) } : null;
    };
    return {
      source: focus("hs", "hsk"),
      target: focus("ht", "htk"),
      set: (side, value) => setParams(side === "source" ? { hs: value?.iri ?? null, hsk: value?.kind ?? null } : { ht: value?.iri ?? null, htk: value?.kind ?? null }, "replace"),
    };
  }, [params, setParams]);

  const index = candidate ? ordered.indexOf(candidate) : -1;
  // Phones read top to bottom: the review belongs after the comparison, not before it.
  const narrow = useNarrow();
  const openBrowse = useCallback(() => {
    const next = new URLSearchParams();
    if (runId) next.set("run", runId);
    if (sourceEntity) {
      next.set("s", navigation.source?.iri ?? sourceEntity.iri);
      next.set("pin_s", sourceEntity.iri);
    }
    if (targetEntity) {
      next.set("t", navigation.target?.iri ?? targetEntity.iri);
      next.set("pin_t", targetEntity.iri);
    }
    if (sourceIri) next.set("from_source", sourceIri);
    if (pairId) next.set("from_pair", pairId);
    window.location.assign(`/browse/?${next.toString()}`);
  }, [runId, sourceEntity, targetEntity, sourceIri, pairId, navigation]);

  if (!state.runs.length) {
    return (
      <div className="page-message">
        <h1>This bundle has no saved matching run</h1>
        <p className="muted">It contains ontology context only, so there are no candidates to compare. You can still browse both ontologies and their original axioms.</p>
        <a className="btn btn-primary" href="/browse/">
          Browse ontologies
        </a>
      </div>
    );
  }

  const score = candidate ? primaryScore(candidate) : undefined;
  const selection = trace.data?.events.find((event) => event.stage === "selection");
  const total = first.data?.total ?? null;
  const complete = !cursor;

  return (
    <div className="compare-layout">
      <aside className="compare-rail" aria-label="Choose a source and candidate">
        <SourcePicker
          sources={allSources}
          total={sources.data?.total_count ?? null}
          loading={sources.loading}
          error={sources.error ?? sourceMoreError}
          onRetry={sources.reload}
          onMore={loadMoreSources}
          canLoadMore={Boolean(sourceCursor)}
          selectedIri={sourceIri}
          onSelect={(item) => setParams({ source: item.entity.iri, pair: null, hs: null, hsk: null, ht: null, htk: null })}
        />
        <CandidateList
          candidates={ordered}
          basis={basis}
          loading={first.loading}
          error={first.error ?? continued?.error}
          onRetry={continued?.error ? loadMoreCandidates : first.reload}
          selectedPair={candidate?.pair_id ?? null}
          onSelect={(item) => setParams({ pair: item.pair_id, ht: null, htk: null })}
          complete={complete}
          total={total}
          onLoadMore={loadMoreCandidates}
          loadingMore={Boolean(continued?.loading)}
        />
        {!narrow && candidate && state.health?.package_id && <ReviewPanel packageId={state.health.package_id} pairId={candidate.pair_id} />}
      </aside>

      <div className="compare-main" id="main" tabIndex={-1}>
        {!candidate && (first.loading || sources.loading) && <Skeleton lines={6} />}
        {!candidate && !first.loading && first.data && !ordered.length && <p className="note">No candidate pair is available for this source.</p>}
        {candidate && pair && sourceEntity && targetEntity && (
          <>
            <PairWorkspace
              pair={pair}
              viewKey={`${runId}|${candidate.pair_id}`}
              ontologyLabel={(id) => ontologyName(state, id, run)}
              tab={tab}
              onTab={(next) => setParams({ details: next }, "replace")}
              order={["hierarchy", "evidence", "trace", "graph", "scores"]}
              extraTabs={[
                {
                  key: "trace",
                  label: "Decision trace",
                  render: () => (trace.data ? <DecisionTrace trace={trace.data} /> : trace.error ? <ErrorNote error={trace.error} onRetry={trace.reload} what="Decision trace" /> : <Skeleton lines={4} title={false} />),
                },
                { key: "scores", label: "Scores", render: () => <ScoresPanel candidate={candidate} /> },
              ]}
              navigation={navigation}
              hierarchyFooter={
                <button type="button" className="btn" onClick={openBrowse}>
                  Open both in the full ontology browser
                </button>
              }
              header={(names) => (
                <>
                  <div className="pair-header">
                    <div className="pair-heading">
                      <span className="meta">
                        Candidate {index + 1} of {ordered.length}
                        {complete ? "" : " loaded"}
                      </span>
                      <h1 className="pair-question">
                        Does <span className="text-source">{names.source ?? curie(sourceEntity.iri)}</span> mean the same as <span className="text-target">{names.target ?? curie(targetEntity.iri)}</span>?
                      </h1>
                    </div>
                    <div className="pair-nav">
                      <button type="button" className="icon-btn" aria-label="Previous candidate" disabled={index <= 0} onClick={() => setParams({ pair: ordered[index - 1].pair_id, ht: null, htk: null })}>
                        <IconChevronLeft />
                      </button>
                      <button type="button" className="icon-btn" aria-label="Next candidate" disabled={index >= ordered.length - 1} onClick={() => setParams({ pair: ordered[index + 1].pair_id, ht: null, htk: null })}>
                        <IconChevronRight />
                      </button>
                    </div>
                  </div>
                  <div className="matcher-bar">
                    <span className="origin-tag origin-matcher">Matcher record</span>
                    <span>
                      {selection?.status === "completed" ? (selection.outcome === "selected" ? "Exact selected this candidate" : selection.outcome === "not_selected" ? "Exact did not select this candidate" : `Selection: ${selection.outcome.replace(/_/g, " ")}`) : "Selection not recorded"}
                      {" · "}
                      {candidate.saved_alignment_member === true ? <strong>in saved alignment</strong> : candidate.saved_alignment_member === false ? "not in saved alignment" : "saved alignment unknown"}
                      {score ? ` · matching score ${formatScore(score.value)}` : ""}
                    </span>
                    <span className="meta">{score?.calibration_status === "validated" ? "Calibrated score." : "Not a probability of being correct."}</span>
                    <span className="matcher-bar-spacer" />
                    <button type="button" className="btn btn-quiet btn-sm" onClick={() => setParams({ details: "trace" }, "replace")}>
                      Open decision trace
                    </button>
                  </div>
                </>
              )}
            />
            {narrow && state.health?.package_id && <ReviewPanel packageId={state.health.package_id} pairId={candidate.pair_id} />}
          </>
        )}
      </div>
    </div>
  );
}
