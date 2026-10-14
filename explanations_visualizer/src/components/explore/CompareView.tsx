"use client";

// Main exploration flow. Reading and keyboard order: choose a candidate, read the two
// meaning cards and the comparison, then open optional details (hierarchy, evidence,
// decision trace, graph, scores). The pair workspace is the same composition the study's
// explanation condition uses. All navigation state is in the URL.
//
// First entry (spec 24 HR2): an explicit URL wins, then a deliberate choice saved for this exact
// package and run, then the development review's first prepared example (when the bound
// catalogue offers one), then the run's first source. Prepared examples sit beside All sources;
// neither filters nor re-ranks the run's sources and candidates.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconChevronLeft, IconChevronRight } from "@/components/common/Icons";
import { formatScore, orderCandidates, primaryScore } from "@/components/explore/candidates";
import { DecisionTrace, ScoresPanel } from "@/components/explore/DecisionTrace";
import { ontologyName, useExplore } from "@/components/explore/ExploreContext";
import { CatalogueNotice, ExampleList, SourceModeSwitch, useReviewCatalogue, type SourceMode } from "@/components/explore/PreparedExamples";
import { ReviewPanel } from "@/components/explore/ReviewPanel";
import { CandidateList, SourcePicker } from "@/components/explore/SourceRail";
import { PairWorkspace, type Focus, type PairNavigation, type RevealRequest } from "@/components/workspace/PairWorkspace";
import { getJson } from "@/lib/api";
import { curie } from "@/lib/iri";
import { caseForSource, examplePair, landingChoice, savedChoiceFor, sourceListMode, type ReviewCase, type SavedChoice } from "@/lib/reviewCatalogue";
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
/** A slow catalogue read never blocks ordinary browsing for longer than this. */
const CATALOGUE_WAIT_MS = 8_000;
/** Each automatic linked-pair or prepared-target search reads at most this many further pages. */
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
  const sourceIri = params.get("source");
  const pairId = params.get("pair");
  const tab = params.get("details");
  const listParam = params.get("list");
  const listMode: SourceMode | null = listParam === "examples" || listParam === "all" ? listParam : null;
  const packageId = state.health?.package_id ?? null;

  // The bound development-review catalogue (exploration only); absent on generic bundles.
  const catalogue = useReviewCatalogue(state.phase === "ready" ? packageId : null);
  const runIds = useMemo(() => state.runs.map((item) => item.run_id), [state.runs]);
  const otherRun = catalogue.state.status === "available" && !runIds.includes(catalogue.state.catalogue.source_run_id);
  const examples = catalogue.state.status === "available" && !otherRun && catalogue.state.catalogue.cases.length ? catalogue.state.catalogue : null;

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
  // A source beyond the loaded pages keeps its typed kind from the URL (example or link).
  const sourceKind = source?.entity.kind ?? kindOf(params.get("sk"));
  const sourceRef = useMemo<EntityRef | null>(() => (sourceIri && run ? { ontology_version_id: source?.entity.ontology_version_id ?? run.source_ontology_version_id, iri: sourceIri, kind: sourceKind } : null), [sourceIri, run, source, sourceKind]);
  const activeCase = examples && runId === examples.source_run_id ? caseForSource(examples, sourceRef) : null;

  // First entry; a late catalogue never overrides a source already chosen or All sources.
  const [waited, setWaited] = useState(false);
  const mode = sourceListMode({ requested: listMode, catalogue: catalogue.state, hasExamples: Boolean(examples), hasSource: Boolean(sourceIri), sourceIsExample: Boolean(activeCase), waited });
  useEffect(() => {
    const timer = window.setTimeout(() => setWaited(true), CATALOGUE_WAIT_MS);
    return () => window.clearTimeout(timer);
  }, []);
  useEffect(() => {
    if (state.phase !== "ready" || !runId) return;
    const decision = landingChoice({
      explicitSource: Boolean(sourceIri),
      explicitPair: Boolean(pairId),
      saved: savedChoiceFor(readChoice(), packageId, runId),
      catalogue: catalogue.state,
      runIds,
      listMode,
      waited,
      firstSource: allSources[0]?.entity ?? null,
    });
    if (decision.kind === "saved") setParams({ source: decision.choice.source, sk: decision.choice.source_kind, pair: decision.choice.pair }, "replace");
    else if (decision.kind === "example") setParams({ ...(decision.run_id !== runId ? { run: decision.run_id } : {}), source: decision.item.source.iri, sk: decision.item.source.kind, pair: null }, "replace");
    else if (decision.kind === "first_source") setParams({ source: decision.source.iri, pair: null }, "replace");
  }, [state.phase, runId, sourceIri, pairId, packageId, catalogue.state, runIds, listMode, waited, allSources, setParams]);

  // A link that names a pair without its source: the run's own record says which source it is.
  const linked = useAsync<PairTrace>(!sourceIri && pairId && runId ? `linked|${runId}|${pairId}` : null, (signal) =>
    getJson<PairTrace>(`/api/v1/runs/${encodeURIComponent(runId!)}/pair`, { pair_id: pairId! }, signal),
  );
  useEffect(() => {
    if (!sourceIri && linked.data && linked.data.pair_id === pairId) setParams({ source: linked.data.source.iri, sk: linked.data.source.kind }, "replace");
  }, [sourceIri, pairId, linked.data, setParams]);

  // Candidates: the backend pages by pair identity, so a first bounded load is sorted by the
  // recorded rank and the remainder stays one explicit "load more" away.
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
  const candidateRead = useRef<AbortController | null>(null);
  const cancelCandidateRead = useCallback(() => {
    candidateRead.current?.abort();
    candidateRead.current = null;
    setMore((previous) => previous.loading ? { ...previous, loading: false } : previous);
  }, []);
  // Back/Forward and deliberate choices both cancel obsolete continuation reads. The identity
  // guard also discards a response that completed just as the visitor made another choice.
  useEffect(() => cancelCandidateRead, [candidateKey, pairId, mode, cancelCandidateRead]);
  const loadMoreCandidates = useCallback(async () => {
    if (!runId || !sourceIri || !cursor || candidateRead.current) return;
    const controller = new AbortController();
    candidateRead.current = controller;
    const base = continued ?? { key: candidateKey, items: [], cursor, loading: false, error: null };
    setMore({ ...base, loading: true, error: null });
    try {
      const page = await getJson<Page<Candidate>>(`/api/v1/runs/${encodeURIComponent(runId)}/candidates`, { source: sourceIri, source_kind: sourceKind, limit: PAGE, cursor }, controller.signal);
      if (!controller.signal.aborted && candidateRead.current === controller)
        setMore({ key: candidateKey, items: [...base.items, ...page.items], cursor: page.next_cursor, loading: false, error: null });
    } catch (error) {
      if (!controller.signal.aborted && candidateRead.current === controller) setMore({ ...base, loading: false, error });
    } finally {
      if (candidateRead.current === controller) candidateRead.current = null;
    }
  }, [runId, sourceIri, sourceKind, cursor, continued, candidateKey]);
  const loaded = useMemo(() => [...(first.data?.items ?? []), ...(continued?.items ?? [])], [first.data, continued]);
  const { ordered, basis } = useMemo(() => orderCandidates(loaded), [loaded]);
  const candidate = ordered.find((item) => item.pair_id === pairId) ?? null;
  const preparedCandidate = activeCase ? examplePair(ordered, activeCase) : null;
  // Choosing an example never silently opens an unrelated target. While the catalogue or its
  // candidate pages resolve, the visitor can still choose any loaded candidate deliberately.
  useEffect(() => {
    if (!ordered.length || candidate || pairId) return;
    if (mode === "examples" && catalogue.state.status === "loading") return;
    if (mode === "examples" && activeCase) {
      if (preparedCandidate) setParams({ pair: preparedCandidate.pair_id }, "replace");
    } else setParams({ pair: ordered[0].pair_id }, "replace");
  }, [ordered, candidate, pairId, mode, activeCase, preparedCandidate, catalogue.state.status, setParams]);
  // A linked pair and a catalogue's typed target use the same bounded continuation. The budget
  // pauses, rather than converts, a prepared-example request. Further reads require a choice.
  const searchKey = `${candidateKey}|${pairId ?? (mode === "examples" ? activeCase?.case_id : "")}`;
  const [searched, setSearched] = useState<{ key: string; pages: number }>({ key: "", pages: 0 });
  const searchedPages = searched.key === searchKey ? searched.pages : 0;
  const pairMissing = Boolean(pairId && !candidate && first.data && !first.loading);
  const exampleMissing = Boolean(!pairId && mode === "examples" && activeCase && !preparedCandidate && first.data && !first.loading);
  useEffect(() => {
    if (!(pairMissing || exampleMissing) || !cursor || continued?.loading || continued?.error || searchedPages >= PAIR_SEARCH_PAGES) return;
    setSearched({ key: searchKey, pages: searchedPages + 1 });
    void loadMoreCandidates();
  }, [pairMissing, exampleMissing, cursor, continued?.loading, continued?.error, searchedPages, searchKey, loadMoreCandidates]);
  const searchStopped = !continued?.loading && (!cursor || searchedPages >= PAIR_SEARCH_PAGES || Boolean(continued?.error));
  const pairNotFound = pairMissing && searchStopped;
  const exampleNotFound = exampleMissing && searchStopped;
  const continueSearch = () => {
    setSearched({ key: searchKey, pages: 0 });
    if (continued?.error) void loadMoreCandidates();
  };

  // Deliberate choices are kept for this exact package and run only (spec 24 HR2 §3).
  const remember = useCallback(
    (iri: string, kind: EntityKind, pair: string | null) => {
      if (packageId && runId) writeChoice({ package_id: packageId, run_id: runId, source: iri, source_kind: kind, pair });
    },
    [packageId, runId],
  );
  const chooseExample = useCallback(
    (item: ReviewCase) => {
      cancelCandidateRead();
      setSearched({ key: "", pages: 0 });
      remember(item.source.iri, item.source.kind, null);
      setParams({ ...(examples && examples.source_run_id !== runId ? { run: examples.source_run_id } : {}), source: item.source.iri, sk: item.source.kind, pair: null, list: "examples", hs: null, hsk: null, ht: null, htk: null });
    },
    [remember, setParams, examples, runId, cancelCandidateRead],
  );
  const choosePair = useCallback(
    (item: Candidate) => {
      cancelCandidateRead();
      remember(item.source.iri, item.source.kind, item.pair_id);
      setParams({ pair: item.pair_id, ht: null, htk: null });
    },
    [remember, setParams, cancelCandidateRead],
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
    // The comparison's own view comes back unchanged: source kind, list, tab and each side's
    // independent hierarchy focus (spec 24 HR2 §5).
    for (const key of ["sk", "list", "details", "hs", "hsk", "ht", "htk"]) {
      const value = params.get(key);
      if (value) next.set(`from_${key}`, value);
    }
    window.location.assign(`/browse/?${next.toString()}`);
  }, [runId, sourceEntity, targetEntity, sourceIri, pairId, navigation, params]);

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
        {/* One card in both modes, so the switch keeps keyboard focus when the list changes. */}
        <SourcePicker
          sources={allSources}
          total={sources.data?.total_count ?? null}
          loading={sources.loading}
          error={sources.error ?? sourceMoreError}
          onRetry={sources.reload}
          onMore={loadMoreSources}
          canLoadMore={Boolean(sourceCursor)}
          selectedIri={sourceIri}
          header={
            <>
              {(examples || catalogue.state.status === "loading") && <SourceModeSwitch mode={mode} onMode={(next) => { cancelCandidateRead(); setParams({ list: next }); }} examples={examples?.cases.length ?? null} sources={sources.data?.total_count ?? null} />}
              <CatalogueNotice state={catalogue.state} otherRun={otherRun} onRetry={catalogue.reload} requested={listMode === "examples"} />
            </>
          }
          onSelect={(item) => {
            cancelCandidateRead();
            remember(item.entity.iri, item.entity.kind, null);
            setParams({ source: item.entity.iri, sk: item.entity.kind, pair: null, list: "all", hs: null, hsk: null, ht: null, htk: null });
          }}
        >
          {mode === "examples" ? <ExampleList state={catalogue.state} otherRun={otherRun} currentCaseId={activeCase?.case_id ?? null} onChoose={chooseExample} /> : null}
        </SourcePicker>
        <CandidateList
          candidates={ordered}
          basis={basis}
          loading={first.loading}
          error={first.error ?? continued?.error}
          onRetry={continued?.error ? loadMoreCandidates : first.reload}
          selectedPair={candidate?.pair_id ?? null}
          onSelect={choosePair}
          prepared={activeCase?.targets ?? []}
          complete={complete}
          total={total}
          onLoadMore={loadMoreCandidates}
          loadingMore={Boolean(continued?.loading)}
        />
        {!narrow && candidate && state.health?.package_id && <ReviewPanel packageId={state.health.package_id} pairId={candidate.pair_id} />}
      </aside>

      <div className="compare-main" id="main" tabIndex={-1}>
        {!candidate && !pairNotFound && !exampleNotFound && (first.loading || sources.loading || (!sourceIri && !linked.error && !(sources.data && !allSources.length)) || pairMissing || exampleMissing) && (
          <div aria-busy="true">
            {!sourceIri && catalogue.state.status === "loading" && <p className="meta">Opening a prepared example…</p>}
            {pairMissing && <p className="meta">Finding the linked pair among this source&apos;s candidates…</p>}
            {exampleMissing && <p className="meta">Finding a prepared target among this source&apos;s candidates…</p>}
            <Skeleton lines={6} />
          </div>
        )}
        {!sourceIri && linked.error ? (
          <div className="note note-warn" role="alert">
            <p>The linked pair was not found in this run.</p>
            <div className="note-actions">
              {examples && (
                <button type="button" className="btn btn-sm" onClick={() => chooseExample(examples.cases[0])}>
                  Open a prepared example
                </button>
              )}
              <button type="button" className="btn btn-sm" onClick={() => setParams({ pair: null, list: "all" })}>
                Browse all sources
              </button>
            </div>
          </div>
        ) : null}
        {exampleNotFound && (
          <div className="note note-warn" role="alert">
            <p>{continued?.error ? "A prepared target could not be resolved because a candidate page failed to load." : cursor ? `The prepared-target search paused after ${PAIR_SEARCH_PAGES} further pages. No prepared target was found among the ${ordered.length} loaded candidates.` : "None of this example's prepared targets is present in this source's candidate list."}</p>
            <p>No unprepared candidate has been selected automatically. You can choose a loaded candidate or browse all sources.</p>
            <div className="note-actions">
              {cursor && <button type="button" className="btn btn-sm" onClick={continueSearch}>{continued?.error ? "Retry prepared-target search" : "Continue prepared-target search"}</button>}
              <button type="button" className="btn btn-sm" onClick={() => { cancelCandidateRead(); setParams({ list: "all" }); }}>Browse all sources</button>
            </div>
          </div>
        )}
        {pairNotFound && (
          <div className="note note-warn" role="alert">
            <p>The linked pair is not among this source&apos;s {ordered.length} loaded candidates in this run.</p>
            <div className="note-actions">
              {ordered.length > 0 && (
                <button type="button" className="btn btn-sm" onClick={() => choosePair(ordered[0])}>
                  Open the first candidate
                </button>
              )}
              {examples && (
                <button type="button" className="btn btn-sm" onClick={() => chooseExample(activeCase ?? examples.cases[0])}>
                  Open a prepared example
                </button>
              )}
            </div>
          </div>
        )}
        {!candidate && !exampleMissing && !first.loading && first.data && !ordered.length && (
          <div className="note">
            <p>No candidate pair is available for this source.</p>
            {examples && (
              <div className="note-actions">
                <button type="button" className="btn btn-sm" onClick={() => chooseExample(examples.cases[0])}>
                  Open a prepared example
                </button>
              </div>
            )}
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
                { key: "scores", label: "Scores", render: () => <ScoresPanel candidate={candidate} /> },
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
                        Candidate {index + 1} of {ordered.length}
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
                      {selection?.status === "completed" ? (selection.outcome === "selected" ? "Selected" : selection.outcome === "not_selected" ? "Not selected" : `Selection: ${selection.outcome.replace(/_/g, " ")}`) : "Selection not recorded"}
                      {" · "}
                      {candidate.saved_alignment_member === true ? <strong>in saved alignment</strong> : candidate.saved_alignment_member === false ? "not in saved alignment" : "saved alignment unknown"}
                      {score ? (
                        <>
                          {" · "}
                          <button type="button" className="link-btn score-help" aria-expanded={scoreHelp} onClick={() => setScoreHelp((value) => !value)}>
                            matching score {formatScore(score.value)}
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
                        {score.calibration_status === "validated" ? "A calibrated score." : "A matching score ranks candidates for this source. It is not a probability of being correct."}
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
