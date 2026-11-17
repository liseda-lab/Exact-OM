"use client";

// Compare saved candidates, inspect their ontology context, evidence and decision trace.
// Explicit URLs take precedence over the last choice saved for this package and run.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconChevronLeft, IconChevronRight } from "@/components/common/Icons";
import { formatScore, orderCandidates, primaryScore, scoreHelpText, scoreLabel } from "@/components/explore/candidates";
import { DecisionTrace, ScoresPanel } from "@/components/explore/DecisionTrace";
import { ontologyName, useExplore } from "@/components/explore/ExploreContext";
import { ReviewPanel } from "@/components/explore/ReviewPanel";
import { CandidateList, SourcePicker } from "@/components/explore/SourceRail";
import { PairWorkspace, type Focus, type PairNavigation, type RevealRequest } from "@/components/workspace/PairWorkspace";
import { getJson } from "@/lib/api";
import { choiceParams, comparisonBrowseParams, needsLinkedPair, resolveComparison, sourceParams } from "@/lib/compareNavigation";
import { curie } from "@/lib/iri";
import { savedChoiceFor, type SavedChoice } from "@/lib/sourceSelection";
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
/** Bound automatic linked-pair lookup; visitors may continue explicitly. */
const PAIR_SEARCH_PAGES = 20;
const CHOICE_KEY = "exact.compare.choice";

function readChoice(): string | null {
  try {
    return window.localStorage.getItem(CHOICE_KEY);
  } catch {
    return null;
  }
}

function writeChoice(choice: SavedChoice) {
  try {
    window.localStorage.setItem(CHOICE_KEY, JSON.stringify(choice));
  } catch {
    /* a convenience only */
  }
}

function kindOf(value: string | null): EntityKind {
  return value === "object_property" || value === "data_property" || value === "individual" ? value : "class";
}

export function CompareView() {
  const state = useExplore();
  const [params, setParams] = useUrlState();
  const runId = params.get("run") ?? state.runs[0]?.run_id ?? null;
  const run = state.runs.find((item) => item.run_id === runId) ?? null;
  const pairId = params.get("pair");
  const tab = params.get("details");
  const packageId = state.health?.package_id ?? null;

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

  // A pair identifies its query even when the URL also names a repeated source.
  // Wait for that identity before requesting candidates from any query.
  const linked = useAsync<PairTrace>(needsLinkedPair(params) && runId ? `linked|${runId}|${pairId}` : null, (signal) =>
    getJson<PairTrace>(`/api/v1/runs/${encodeURIComponent(runId!)}/pair`, { pair_id: pairId! }, signal),
  );
  const { sourceIri, sourceKind, queryId, source, waitingForPair, invalidQuery, updates } = resolveComparison(params, allSources, runId, linked.data);
  useEffect(() => {
    if (updates) setParams(updates, "replace");
  }, [updates, setParams]);
  useEffect(() => {
    if (state.phase !== "ready" || !runId || sourceIri || pairId) return;
    const saved = savedChoiceFor(readChoice(), packageId, runId);
    if (saved) setParams(choiceParams(saved), "replace");
    else if (allSources[0]) setParams(sourceParams(allSources[0]), "replace");
  }, [state.phase, runId, sourceIri, pairId, packageId, allSources, setParams]);

  // Candidates: the backend pages by pair identity, so a first bounded load is sorted by the
  // recorded rank and the remainder stays one explicit "load more" away.
  const candidateKey = runId && sourceIri && sources.data && !waitingForPair && !invalidQuery ? `cands|${runId}|${sourceKind}|${sourceIri}|${queryId}` : null;
  const first = useAsync<{ items: Candidate[]; cursor: string | null; total: number | null }>(candidateKey, async (signal) => {
    const items: Candidate[] = [];
    let cursor: string | null = null;
    let total: number | null = null;
    for (let pageIndex = 0; pageIndex < FIRST_LOAD_PAGES; pageIndex += 1) {
      const page: Page<Candidate> = await getJson<Page<Candidate>>(`/api/v1/runs/${encodeURIComponent(runId!)}/candidates`, { source: sourceIri!, source_kind: sourceKind, query_id: queryId, limit: PAGE, cursor }, signal);
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
  const candidateRead = useRef<AbortController | null>(null);
  const cancelCandidateRead = useCallback(() => {
    candidateRead.current?.abort();
    candidateRead.current = null;
    setMore((previous) => previous.loading ? { ...previous, loading: false } : previous);
  }, []);
  // Back/Forward and deliberate choices both cancel obsolete continuation reads. The identity
  // guard also discards a response that completed just as the visitor made another choice.
  useEffect(() => cancelCandidateRead, [candidateKey, pairId, cancelCandidateRead]);
  const loadMoreCandidates = useCallback(async () => {
    if (!runId || !sourceIri || !cursor || candidateRead.current) return;
    const controller = new AbortController();
    candidateRead.current = controller;
    const base = continued ?? { key: candidateKey, items: [], cursor, loading: false, error: null };
    setMore({ ...base, loading: true, error: null });
    try {
      const page = await getJson<Page<Candidate>>(`/api/v1/runs/${encodeURIComponent(runId)}/candidates`, { source: sourceIri, source_kind: sourceKind, query_id: queryId, limit: PAGE, cursor }, controller.signal);
      if (!controller.signal.aborted && candidateRead.current === controller)
        setMore({ key: candidateKey, items: [...base.items, ...page.items], cursor: page.next_cursor, loading: false, error: null });
    } catch (error) {
      if (!controller.signal.aborted && candidateRead.current === controller) setMore({ ...base, loading: false, error });
    } finally {
      if (candidateRead.current === controller) candidateRead.current = null;
    }
  }, [runId, sourceIri, sourceKind, queryId, cursor, continued, candidateKey]);
  const loaded = useMemo(() => [...(first.data?.items ?? []), ...(continued?.items ?? [])], [first.data, continued]);
  const { ordered, basis } = useMemo(() => orderCandidates(loaded), [loaded]);
  const candidate = ordered.find((item) => item.pair_id === pairId) ?? null;
  useEffect(() => {
    if (ordered.length && !candidate && !pairId) setParams({ pair: ordered[0].pair_id }, "replace");
  }, [ordered, candidate, pairId, setParams]);
  const searchKey = `${candidateKey}|${pairId ?? ""}`;
  const [searched, setSearched] = useState<{ key: string; pages: number }>({ key: "", pages: 0 });
  const searchedPages = searched.key === searchKey ? searched.pages : 0;
  const pairMissing = Boolean(pairId && !candidate && first.data && !first.loading);
  useEffect(() => {
    if (!pairMissing || !cursor || continued?.loading || continued?.error || searchedPages >= PAIR_SEARCH_PAGES) return;
    setSearched({ key: searchKey, pages: searchedPages + 1 });
    void loadMoreCandidates();
  }, [pairMissing, cursor, continued?.loading, continued?.error, searchedPages, searchKey, loadMoreCandidates]);
  const searchStopped = !continued?.loading && (!cursor || searchedPages >= PAIR_SEARCH_PAGES || Boolean(continued?.error));
  const pairNotFound = pairMissing && searchStopped;
  const continueSearch = () => {
    setSearched({ key: searchKey, pages: 0 });
    if (continued?.error) void loadMoreCandidates();
  };

  // Deliberate choices are kept for this exact package and run only.
  const remember = useCallback(
    (iri: string, kind: EntityKind, pair: string | null, query: number | null = queryId) => {
      if (packageId && runId) writeChoice({ package_id: packageId, run_id: runId, source: iri, source_kind: kind, query_id: query, pair });
    },
    [packageId, runId, queryId],
  );
  const choosePair = useCallback(
    (item: Candidate) => {
      cancelCandidateRead();
      remember(item.source.iri, item.source.kind, item.pair_id);
      setParams({ pair: item.pair_id, q: queryId?.toString() ?? null, ht: null, htk: null });
    },
    [remember, setParams, cancelCandidateRead, queryId],
  );

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
  // "Open decision trace" opens the tab and brings it into view every time (21 F03).
  const [reveal, setReveal] = useState<RevealRequest | null>(null);
  const [scoreHelp, setScoreHelp] = useState(false);
  const openTrace = () => {
    setParams({ details: "trace" }, "replace");
    setReveal((current) => ({ tab: "trace", target: ".trace-heading", nonce: (current?.nonce ?? 0) + 1 }));
  };
  // Phones read top to bottom: the review belongs after the comparison, not before it.
  const narrow = useNarrow();
  const openBrowse = useCallback(() => {
    const next = comparisonBrowseParams(params, { runId, source: sourceEntity, target: targetEntity, queryId, sourceFocus: navigation.source, targetFocus: navigation.target });
    window.location.assign(`/browse/?${next.toString()}`);
  }, [runId, sourceEntity, targetEntity, navigation, params, queryId]);

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
          selectedKind={sourceKind}
          selectedQueryId={queryId}
          onSelect={(item) => {
            cancelCandidateRead();
            remember(item.entity.iri, item.entity.kind, null, item.query_id ?? null);
            setParams(sourceParams(item));
          }}
        >
        </SourcePicker>
        <CandidateList
          candidates={ordered}
          basis={basis}
          loading={first.loading || (waitingForPair && !linked.error)}
          error={first.error ?? continued?.error}
          onRetry={continued?.error ? loadMoreCandidates : first.reload}
          selectedPair={candidate?.pair_id ?? null}
          onSelect={choosePair}
          complete={complete}
          total={total}
          onLoadMore={loadMoreCandidates}
          loadingMore={Boolean(continued?.loading)}
        />
        {!narrow && candidate && state.health?.package_id && <ReviewPanel packageId={state.health.package_id} pairId={candidate.pair_id} />}
      </aside>

      <div className="compare-main" id="main" tabIndex={-1}>
        {!candidate && !pairNotFound && (first.loading || sources.loading || (waitingForPair && !linked.error) || (!sourceIri && !linked.error && !(sources.data && !allSources.length)) || pairMissing) && (
          <div aria-busy="true">
            {pairMissing && <p className="meta">Finding the linked pair among this source&apos;s candidates…</p>}
            <Skeleton lines={6} />
          </div>
        )}
        {waitingForPair && linked.error ? (
          <div className="note note-warn" role="alert">
            <p>The linked pair was not found in this run.</p>
            <div className="note-actions">
              <button type="button" className="btn btn-sm" onClick={() => setParams({ source: null, sk: null, q: null, pair: null })}>
                Browse all sources
              </button>
            </div>
          </div>
        ) : null}
        {invalidQuery && <p className="note note-warn" role="alert">The query in this link is invalid. Choose a source query from the list.</p>}
        {pairNotFound && (
          <div className="note note-warn" role="alert">
            <p>The linked pair is not among this source&apos;s {ordered.length} loaded candidates in this run.</p>
            <div className="note-actions">
              {cursor && <button type="button" className="btn btn-sm" onClick={continueSearch}>Continue pair search</button>}
              {ordered.length > 0 && (
                <button type="button" className="btn btn-sm" onClick={() => choosePair(ordered[0])}>
                  Open the first candidate
                </button>
              )}
            </div>
          </div>
        )}
        {!candidate && !first.loading && first.data && !ordered.length && (
          <div className="note">
            <p>{queryId === null ? "No candidate pair is available for this source." : `No candidate pair is available for query ${queryId}.`}</p>
          </div>
        )}
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
                { key: "scores", label: "Scores", render: () => <ScoresPanel candidate={candidate} ranking={trace.data?.ranking} /> },
              ]}
              navigation={navigation}
              reveal={reveal}
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
                        {queryId !== null && <>Query {queryId} · </>}Candidate {index + 1} of {ordered.length}
                        {complete ? "" : " loaded"}
                      </span>
                      <h1 className="pair-question">
                        Does <span className="text-source">{names.source ?? curie(sourceEntity.iri)}</span> mean the same as <span className="text-target">{names.target ?? curie(targetEntity.iri)}</span>?
                      </h1>
                    </div>
                    <div className="pair-nav">
                      <button type="button" className="icon-btn" aria-label="Previous candidate" disabled={index <= 0} onClick={() => choosePair(ordered[index - 1])}>
                        <IconChevronLeft />
                      </button>
                      <button type="button" className="icon-btn" aria-label="Next candidate" disabled={index >= ordered.length - 1} onClick={() => choosePair(ordered[index + 1])}>
                        <IconChevronRight />
                      </button>
                    </div>
                  </div>
                  <div className="matcher-bar">
                    <span className="origin-tag origin-matcher">Matcher</span>
                    <span>
                      {trace.data?.ranking ? <>Ranked {trace.data.ranking.rank} of {trace.data.ranking.candidate_count} in the original run</> : <>
                        {selection?.status === "completed" ? (selection.outcome === "selected" ? "Selected" : selection.outcome === "not_selected" ? "Not selected" : `Selection: ${selection.outcome.replace(/_/g, " ")}`) : "Selection not recorded"}
                        {" · "}
                        {candidate.saved_alignment_member === true ? <strong>in saved alignment</strong> : candidate.saved_alignment_member === false ? "not in saved alignment" : "saved alignment unknown"}
                      </>}
                      {score ? (
                        <>
                          {" · "}
                          <button type="button" className="link-btn score-help" aria-expanded={scoreHelp} onClick={() => setScoreHelp((value) => !value)}>
                            {scoreLabel(score)} {formatScore(score.value)}
                          </button>
                        </>
                      ) : null}
                    </span>
                    <span className="matcher-bar-spacer" />
                    <button type="button" className="btn btn-sm" onClick={openTrace}>
                      Open decision trace
                    </button>
                    {scoreHelp && score && (
                      <p className="meta matcher-bar-note">
                        {scoreHelpText(score)}
                      </p>
                    )}
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
