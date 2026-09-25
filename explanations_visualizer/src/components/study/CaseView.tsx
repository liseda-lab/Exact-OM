"use client";

// One scored case. The server decides which case and condition this is and which
// resources exist; the baseline never receives explanation data. Drafts save as you work,
// submitting is explicit and final, and the case timer starts only once content is usable.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconCopy, SideMarker } from "@/components/common/Icons";
import { RankingPanel, type RankingCandidate, type RankingValue } from "@/components/study/RankingPanel";
import { ResourceDownloads } from "@/components/study/Stages";
import { ExplanationPanels, indexResources, StudyEntityCard, useStudyLabelSource, type Component, type ResourceIndex } from "@/components/study/StudyExplanation";
import { describeError, getJson } from "@/lib/api";
import { curie } from "@/lib/iri";
import { LabelSourceContext } from "@/lib/labelSource";
import type { StudySession } from "@/study/session";
import type { Telemetry } from "@/study/telemetry";
import type { ExplanationResource, StudyCase, StudyState } from "@/study/types";

const ALL_COMPONENTS: Component[] = ["original_context", "entity_description", "hierarchy", "evidence_table", "evidence_graph", "pair_comparison"];

function frozenComponents(state: StudyState): Set<Component> {
  const matrix = state.forms.final.find((question) => question.id === "component_usefulness")?.matrix;
  return new Set((matrix ? Object.keys(matrix) : ALL_COMPONENTS) as Component[]);
}

function CopyButton({ text, label, onCopied }: { text: string; label: string; onCopied?: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="btn btn-sm"
      aria-label={label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          onCopied?.();
          window.setTimeout(() => setCopied(false), 1500);
        } catch {
          setCopied(false);
        }
      }}
    >
      <IconCopy /> {copied ? "Copied" : "Copy IRI"}
    </button>
  );
}

export function CaseView({ state, session, telemetry }: { state: StudyState; session: StudySession; telemetry: Telemetry }) {
  const caseKey = `${state.current_case_id}|${state.current_presentation_id}`;
  const [studyCase, setStudyCase] = useState<StudyCase | null>(null);
  const [caseError, setCaseError] = useState<unknown>(null);
  const [resources, setResources] = useState<ResourceIndex | null>(null);
  const [resourceError, setResourceError] = useState<string | null>(null);
  const [value, setValue] = useState<RankingValue>({ responseType: null, ranked: [] });
  const [inspecting, setInspecting] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);
  const [phoneAnyway, setPhoneAnyway] = useState(false);
  const loadStarted = useRef(performance.now());
  const draftTimer = useRef<number | null>(null);

  // Load the server-assigned case, then any condition-permitted explanation resources.
  useEffect(() => {
    let cancelled = false;
    loadStarted.current = performance.now();
    setStudyCase(null);
    setResources(null);
    setCaseError(null);
    setResourceError(null);
    (async () => {
      try {
        const current = await getJson<StudyCase>("/api/v1/study/cases/current");
        if (cancelled) return;
        setStudyCase(current);
        const saved = state.ranking && state.ranking.presentation_id === current.presentation_id ? state.ranking : null;
        setValue(saved ? { responseType: saved.response_type, ranked: saved.ranked_candidate_ids } : { responseType: null, ranked: [] });
        setInspecting(current.candidates.find((candidate) => candidate.display_position === 1)?.candidate_id ?? null);
        if (current.condition === "explanation" && current.explanation_refs.length) {
          try {
            const loaded = await Promise.all(current.explanation_refs.map((ref) => getJson<ExplanationResource>(`/api/v1/study/resources/${encodeURIComponent(ref)}`)));
            if (!cancelled) setResources(indexResources(loaded));
          } catch (error) {
            if (!cancelled) {
              setResources(indexResources([]));
              setResourceError(`The prepared explanations could not be loaded: ${describeError(error)} You can still rank the candidates.`);
            }
          }
        }
        if (!cancelled) telemetry.markCaseReady(performance.now() - loadStarted.current);
      } catch (error) {
        if (!cancelled) setCaseError(error);
      }
    })();
    return () => {
      cancelled = true;
    };
    // The case identity is the only trigger; restoring from state happens once per case.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseKey]);

  // After a conflict the server's saved draft replaces what was on screen.
  const handledConflict = useRef(0);
  useEffect(() => {
    if (session.save.kind !== "conflict" || session.save.at === handledConflict.current || !studyCase) return;
    handledConflict.current = session.save.at;
    const saved = state.ranking && state.ranking.presentation_id === studyCase.presentation_id ? state.ranking : null;
    setValue(saved ? { responseType: saved.response_type, ranked: saved.ranked_candidate_ids } : { responseType: null, ranked: [] });
  }, [session.save, state.ranking, studyCase]);

  const locked = state.ranking?.workflow_state === "submitted" && state.ranking.presentation_id === studyCase?.presentation_id;

  const saveDraft = useCallback(
    (next: RankingValue) => {
      if (!studyCase || !state.current_case_id) return;
      if (draftTimer.current) window.clearTimeout(draftTimer.current);
      draftTimer.current = window.setTimeout(() => {
        void session
          .mutate({
            method: "PUT",
            path: `/api/v1/study/cases/${encodeURIComponent(state.current_case_id!)}/draft`,
            body: { presentation_id: studyCase.presentation_id, response_type: next.responseType, ranked_candidate_ids: next.responseType === "ranked_candidates" ? next.ranked : [] },
            coalesce: `draft:${state.current_case_id}`,
            persist: true,
          })
          .catch(() => undefined);
      }, 500);
    },
    [session, state.current_case_id, studyCase],
  );

  const submit = async () => {
    if (!studyCase || !state.current_case_id) return;
    if (draftTimer.current) window.clearTimeout(draftTimer.current);
    setSubmitting(true);
    telemetry.emit("submit");
    await telemetry.flushTiming();
    try {
      await session.mutate({
        method: "POST",
        path: `/api/v1/study/cases/${encodeURIComponent(state.current_case_id)}/submit`,
        body: { presentation_id: studyCase.presentation_id, response_type: value.responseType, ranked_candidate_ids: value.responseType === "ranked_candidates" ? value.ranked : [] },
      });
    } catch {
      /* the header explains; the answer stays on screen for another try */
    } finally {
      setSubmitting(false);
    }
  };

  const components = useMemo(() => frozenComponents(state), [state]);
  const extraLabels = useMemo(() => {
    const labels: Record<string, string> = {};
    if (studyCase) {
      labels[`${studyCase.source.ontology_version_id}|${studyCase.source.iri}`] = studyCase.source_label;
      studyCase.candidates.forEach((candidate) => {
        labels[`${candidate.entity.ontology_version_id}|${candidate.entity.iri}`] = candidate.label;
      });
    }
    return labels;
  }, [studyCase]);
  const labelSource = useStudyLabelSource(resources, extraLabels);
  const emitPanel = useCallback((type: Parameters<Telemetry["emit"]>[0], element?: string) => telemetry.emit(type, { component: "explanation", element }), [telemetry]);

  if (caseError) return <ErrorNote error={caseError} what="This case could not be loaded" />;
  if (!studyCase) return <Skeleton lines={6} />;

  const rankingCandidates: RankingCandidate[] = studyCase.candidates.map((candidate) => ({
    id: candidate.candidate_id,
    position: candidate.display_position,
    label: candidate.label,
    identifier: curie(candidate.entity.iri),
    score: candidate.score.toFixed(2),
    scoreMeaning: candidate.score_meaning,
  }));
  const explanation = studyCase.condition === "explanation";
  const inspected = studyCase.candidates.find((candidate) => candidate.candidate_id === inspecting) ?? studyCase.candidates[0];

  return (
    <LabelSourceContext.Provider value={labelSource}>
      <div className="case-page">
        {!phoneAnyway && (
          <div className="phone-notice note note-info" role="note">
            <span>
              Scored cases are designed for a desktop computer with Protégé open. Your progress is saved: open your private link on that computer to continue exactly here.
            </span>
            <button type="button" className="btn btn-sm" onClick={() => setPhoneAnyway(true)}>
              Show the case here anyway
            </button>
          </div>
        )}
        <div className={phoneAnyway ? "case-content" : "case-content phone-hidden"}>
          <div className="instruction-strip">
            {showInstructions ? (
              <p>
                {state.instructions}{" "}
                <button type="button" className="btn btn-quiet btn-sm" onClick={() => setShowInstructions(false)}>
                  Hide
                </button>
              </p>
            ) : (
              <button type="button" className="btn btn-quiet btn-sm" onClick={() => setShowInstructions(true)}>
                Show instructions
              </button>
            )}
          </div>
          <div className="case-grid">
            <div className="case-left">
              {explanation ? (
                <StudyEntityCard side="source" title="Source concept" entity={studyCase.source} label={studyCase.source_label} index={resources} components={components} emit={emitPanel} />
              ) : (
                <article className="entity-card entity-card-source compact" aria-label="Source concept">
                  <div className="entity-card-head">
                    <SideMarker side="source" />
                    <span className="eyebrow text-source">Source concept</span>
                  </div>
                  <h2 className="entity-title">{studyCase.source_label}</h2>
                  <div className="iri-row">
                    <span className="iri">{studyCase.source.iri}</span>
                    <CopyButton text={studyCase.source.iri} label={`Copy IRI of ${studyCase.source_label}`} />
                  </div>
                </article>
              )}
              <RankingPanel
                candidates={rankingCandidates}
                value={value}
                locked={locked || submitting}
                inspecting={explanation ? inspected?.candidate_id : null}
                onInspect={
                  explanation
                    ? (id) => {
                        setInspecting(id);
                        telemetry.emit("candidate_inspected", { component: "ranking", element: id });
                      }
                    : undefined
                }
                rowExtra={
                  explanation
                    ? undefined
                    : (candidate) => {
                        const entity = studyCase.candidates.find((item) => item.candidate_id === candidate.id)!.entity;
                        return <CopyButton text={entity.iri} label={`Copy IRI of ${candidate.label}`} />;
                      }
                }
                onChange={(next, event, element) => {
                  setValue(next);
                  telemetry.emit(event, { component: "ranking", element });
                  saveDraft(next);
                }}
                onSubmit={submit}
                submitting={submitting}
              />
            </div>
            <div className="case-right">
              {explanation && inspected ? (
                <ExplanationPanels studyCase={studyCase} candidate={inspected} index={resources} error={resourceError} components={components} emit={emitPanel} labelSource={labelSource} />
              ) : (
                <section className="study-card baseline-tools">
                  <h2>Inspect with your own tools</h2>
                  <p>In this block, look candidates up in Protégé or in the ontology files. Copy an IRI and paste it into Protégé&apos;s search to find the class.</p>
                  <div onClickCapture={(event) => (event.target as HTMLElement).closest("a") && telemetry.emit("external_resource_link", { component: "downloads" })}>
                    <ResourceDownloads state={state} compact />
                  </div>
                  <p className="meta">Time spent in Protégé before you submit counts as part of this case. Switching windows does not pause anything.</p>
                </section>
              )}
            </div>
          </div>
        </div>
      </div>
    </LabelSourceContext.Provider>
  );
}
