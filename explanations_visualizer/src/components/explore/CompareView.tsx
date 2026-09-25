"use client";

// Main exploration flow. Reading and keyboard order: choose a candidate, read the two
// meaning cards and the comparison, then open optional details (hierarchy, evidence,
// decision trace, graph, scores). All navigation state is in the URL.

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconChevronLeft, IconChevronRight } from "@/components/common/Icons";
import { formatScore, orderCandidates, primaryScore } from "@/components/explore/candidates";
import { ComparisonPanel } from "@/components/explore/ComparisonPanel";
import { DecisionTrace, ScoresPanel } from "@/components/explore/DecisionTrace";
import { EntityCard } from "@/components/explore/EntityCard";
import { usePairEvidence } from "@/components/explore/evidenceData";
import { EvidenceList } from "@/components/explore/EvidenceList";
import { ontologyName, useExplore } from "@/components/explore/ExploreContext";
import type { CitedFact } from "@/components/explore/GeneratedBlock";
import { ReviewPanel } from "@/components/explore/ReviewPanel";
import { CandidateList, SourcePicker } from "@/components/explore/SourceRail";
import { getJson } from "@/lib/api";
import { curie } from "@/lib/iri";
import { useLabels } from "@/lib/labels";
import type { Candidate, EntityContext, EntityRef, Page, PairTrace, SourceSummary } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { useNarrow } from "@/lib/useMedia";
import { useUrlState } from "@/lib/urlState";

const EvidenceGraph = dynamic(() => import("@/components/explore/EvidenceGraph").then((module) => module.EvidenceGraph), {
  ssr: false,
  loading: () => <div className="graph-canvas skeleton" aria-busy="true" />,
});

type Tab = "hierarchy" | "evidence" | "trace" | "graph" | "scores";
const TABS: { key: Tab; label: string }[] = [
  { key: "hierarchy", label: "Hierarchy" },
  { key: "evidence", label: "Evidence" },
  { key: "trace", label: "Decision trace" },
  { key: "graph", label: "Evidence graph" },
  { key: "scores", label: "Scores" },
];

export function useEntityContext(entity: EntityRef | null) {
  return useAsync<EntityContext>(entity ? `ctx|${entity.ontology_version_id}|${entity.kind}|${entity.iri}` : null, (signal) =>
    getJson<EntityContext>("/api/v1/entity-context", { ontology_version_id: entity!.ontology_version_id, iri: entity!.iri, kind: entity!.kind }, signal),
  );
}

/** Map every fact visible in the two contexts to a short citation label. */
function citationIndex(source: EntityContext | undefined, target: EntityContext | undefined) {
  const index = new Map<string, CitedFact>();
  for (const [side, ctx] of [
    ["source", source],
    ["target", target],
  ] as const) {
    if (!ctx) continue;
    const prefix = side === "source" ? "Source" : "Target";
    const add = (items: { fact_id: string; predicate_iri: string | null; category?: string }[], name: string) => items.forEach((fact) => index.set(fact.fact_id, { label: `${prefix} ${name}`, side }));
    add(ctx.definitions.items, "definition");
    add(ctx.synonyms.items, "synonym");
    add(ctx.parents.items, "parent");
    for (const [category, page] of Object.entries(ctx.categories)) {
      page.items.forEach((fact) => {
        if (!index.has(fact.fact_id)) index.set(fact.fact_id, { label: `${prefix} ${category === "labels" ? "label" : category.replace(/_/g, " ").replace(/s$/, "")}`, side });
      });
    }
  }
  return index;
}

export function CompareView() {
  const state = useExplore();
  const [params, setParams] = useUrlState();
  const runId = params.get("run") ?? state.runs[0]?.run_id ?? null;
  const run = state.runs.find((item) => item.run_id === runId) ?? null;
  const sourceIri = params.get("source");
  const pairId = params.get("pair");
  const tab = (params.get("details") as Tab | null) ?? null;

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

  const sourceKind = source?.entity.kind ?? "class";
  const candidates = useAsync<{ items: Candidate[]; truncated: boolean }>(runId && sourceIri ? `cands|${runId}|${sourceKind}|${sourceIri}` : null, async (signal) => {
    const items: Candidate[] = [];
    let cursor: string | null = null;
    let truncated = false;
    for (let guard = 0; guard < 5; guard += 1) {
      const page: Page<Candidate> = await getJson<Page<Candidate>>(`/api/v1/runs/${encodeURIComponent(runId!)}/candidates`, { source: sourceIri!, source_kind: sourceKind, limit: 100, cursor }, signal);
      items.push(...page.items);
      cursor = page.next_cursor;
      truncated = Boolean(page.next_cursor);
      if (!cursor) break;
    }
    return { items, truncated };
  });
  const { ordered, basis } = useMemo(() => orderCandidates(candidates.data?.items ?? []), [candidates.data]);
  const candidate = ordered.find((item) => item.pair_id === pairId) ?? null;
  useEffect(() => {
    if (ordered.length && !candidate) setParams({ pair: ordered[0].pair_id }, "replace");
  }, [ordered, candidate, setParams]);

  const sourceEntity = candidate?.source ?? source?.entity ?? null;
  const targetEntity = candidate?.target ?? null;
  const sourceCtx = useEntityContext(sourceEntity);
  const targetCtx = useEntityContext(targetEntity);
  const trace = useAsync<PairTrace>(runId && candidate ? `trace|${runId}|${candidate.pair_id}` : null, (signal) =>
    getJson<PairTrace>(`/api/v1/runs/${encodeURIComponent(runId!)}/pair`, { pair_id: candidate!.pair_id }, signal),
  );
  const evidence = usePairEvidence(tab === "evidence" || tab === "graph" ? runId : null, tab === "evidence" || tab === "graph" ? candidate?.pair_id ?? null : null);
  const [selectedEvidence, setSelectedEvidence] = useState<string | null>(null);

  const cites = useMemo(() => citationIndex(sourceCtx.data, targetCtx.data), [sourceCtx.data, targetCtx.data]);
  const cite = useCallback((factId: string) => cites.get(factId), [cites]);
  const labels = useLabels(targetEntity?.ontology_version_id, targetEntity ? [targetEntity.iri] : []);

  const index = candidate ? ordered.indexOf(candidate) : -1;
  // Phones read top to bottom: the review belongs after the comparison, not before it.
  const narrow = useNarrow();
  const openBrowse = useCallback(
    (side: "source" | "target", iri: string) => {
      const next = new URLSearchParams();
      if (runId) next.set("run", runId);
      if (sourceEntity) {
        next.set("s", side === "source" ? iri : sourceEntity.iri);
        next.set("pin_s", sourceEntity.iri);
      }
      if (targetEntity) {
        next.set("t", side === "target" ? iri : targetEntity.iri);
        next.set("pin_t", targetEntity.iri);
      }
      if (sourceIri) next.set("from_source", sourceIri);
      if (pairId) next.set("from_pair", pairId);
      window.location.assign(`/browse/?${next.toString()}`);
    },
    [runId, sourceEntity, targetEntity, sourceIri, pairId],
  );

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
  const targetName = targetCtx.data?.preferred_label.value ?? (targetEntity ? labels(targetEntity.iri)?.value : null);
  const sourceName = sourceCtx.data?.preferred_label.value;

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
          onSelect={(item) => setParams({ source: item.entity.iri, pair: null })}
        />
        <CandidateList
          candidates={ordered}
          basis={basis}
          loading={candidates.loading}
          error={candidates.error}
          onRetry={candidates.reload}
          selectedPair={candidate?.pair_id ?? null}
          onSelect={(item) => setParams({ pair: item.pair_id })}
          truncated={Boolean(candidates.data?.truncated)}
        />
        {!narrow && candidate && state.health?.package_id && <ReviewPanel packageId={state.health.package_id} pairId={candidate.pair_id} />}
      </aside>

      <div className="compare-main" id="main" tabIndex={-1}>
        {!candidate && (candidates.loading || sources.loading) && <Skeleton lines={6} />}
        {!candidate && !candidates.loading && candidates.data && !ordered.length && <p className="note">No candidate pair is available for this source.</p>}
        {candidate && sourceEntity && targetEntity && (
          <>
            <div className="pair-header">
              <div className="pair-heading">
                <span className="meta">
                  Candidate {index + 1} of {ordered.length}
                </span>
                <h1 className="pair-question">
                  Does <span className="text-source">{sourceName ?? curie(sourceEntity.iri)}</span> mean the same as <span className="text-target">{targetName ?? curie(targetEntity.iri)}</span>?
                </h1>
              </div>
              <div className="pair-nav">
                <button type="button" className="icon-btn" aria-label="Previous candidate" disabled={index <= 0} onClick={() => setParams({ pair: ordered[index - 1].pair_id })}>
                  <IconChevronLeft />
                </button>
                <button type="button" className="icon-btn" aria-label="Next candidate" disabled={index >= ordered.length - 1} onClick={() => setParams({ pair: ordered[index + 1].pair_id })}>
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

            <div className="card-pair">
              <EntityCard
                side="source"
                entity={sourceEntity}
                context={sourceCtx}
                ontologyLabel={ontologyName(state, sourceEntity.ontology_version_id, run)}
                cite={cite}
                onOpenEntity={(iri) => openBrowse("source", iri)}
              />
              <EntityCard
                side="target"
                entity={targetEntity}
                context={targetCtx}
                ontologyLabel={ontologyName(state, targetEntity.ontology_version_id, run)}
                cite={cite}
                onOpenEntity={(iri) => openBrowse("target", iri)}
              />
            </div>

            <ComparisonPanel
              pairKey={`${sourceEntity.iri}|${targetEntity.iri}`}
              params={{
                ontology_version_id: sourceEntity.ontology_version_id,
                iri: sourceEntity.iri,
                kind: sourceEntity.kind,
                counterpart_ontology_version_id: targetEntity.ontology_version_id,
                counterpart_iri: targetEntity.iri,
                counterpart_kind: targetEntity.kind,
              }}
              cite={cite}
            />

            <section className="card details" aria-label="Optional details">
              <div className="tab-list" role="tablist" aria-label="Optional details">
                {TABS.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    role="tab"
                    id={`tab-${item.key}`}
                    aria-selected={tab === item.key}
                    aria-controls="details-panel"
                    className={tab === item.key ? "tab on" : "tab"}
                    onClick={() => setParams({ details: tab === item.key ? null : item.key }, "replace")}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              <div id="details-panel" role="tabpanel" aria-labelledby={tab ? `tab-${tab}` : undefined} className="details-panel">
                {!tab && <p className="meta">Optional details stay closed until you open one. They never change the comparison above.</p>}
                {tab === "hierarchy" && (
                  <div className="mini-hierarchy">
                    {[
                      ["source", sourceCtx.data] as const,
                      ["target", targetCtx.data] as const,
                    ].map(([side, ctx]) => (
                      <MiniParents key={side} side={side} ctx={ctx} onOpen={(iri) => openBrowse(side, iri)} />
                    ))}
                    <button type="button" className="btn" onClick={() => openBrowse("source", sourceEntity.iri)}>
                      Open both in the ontology browser
                    </button>
                  </div>
                )}
                {tab === "evidence" && <EvidenceList state={evidence} selected={selectedEvidence} onOpenEntity={openBrowse} />}
                {tab === "graph" &&
                  (evidence.data ? (
                    evidence.data.items.length ? (
                      <EvidenceGraph source={sourceEntity} target={targetEntity} bundle={evidence.data} selected={selectedEvidence} onSelect={setSelectedEvidence} />
                    ) : (
                      <EvidenceList state={evidence} />
                    )
                  ) : evidence.error ? (
                    <ErrorNote error={evidence.error} onRetry={evidence.reload} what="Evidence" />
                  ) : (
                    <div className="graph-canvas skeleton" aria-busy="true" />
                  ))}
                {tab === "trace" && (trace.data ? <DecisionTrace trace={trace.data} /> : trace.error ? <ErrorNote error={trace.error} onRetry={trace.reload} what="Decision trace" /> : <Skeleton lines={4} title={false} />)}
                {tab === "scores" && <ScoresPanel candidate={candidate} />}
              </div>
            </section>
            {narrow && state.health?.package_id && <ReviewPanel packageId={state.health.package_id} pairId={candidate.pair_id} />}
          </>
        )}
      </div>
    </div>
  );
}

function MiniParents({ side, ctx, onOpen }: { side: "source" | "target"; ctx: EntityContext | undefined; onOpen: (iri: string) => void }) {
  const iris = (ctx?.parents.items ?? []).map((fact) => (fact.value?.term_type === "iri" ? fact.value.iri : "")).filter(Boolean);
  const label = useLabels(ctx?.entity.ontology_version_id, iris);
  if (!ctx) return <Skeleton lines={2} title={false} />;
  return (
    <div className="mini-parents">
      <h3 className="eyebrow">
        {side === "source" ? "Source" : "Target"} · asserted parents
      </h3>
      {iris.length ? (
        <ul>
          {iris.map((iri) => (
            <li key={iri}>
              <span className="edge-solid" aria-hidden="true" />
              <button type="button" className={`tree-label tree-${side}`} onClick={() => onOpen(iri)}>
                {label(iri)?.value ?? curie(iri)}
              </button>
            </li>
          ))}
          <li className="mini-focus">{ctx.preferred_label.value ?? curie(ctx.entity.iri)}</li>
        </ul>
      ) : (
        <p className="meta">No named parent in the loaded scope.</p>
      )}
    </div>
  );
}
