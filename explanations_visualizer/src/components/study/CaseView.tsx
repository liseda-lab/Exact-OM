"use client";

// One scored case. The server decides which case and condition this is and which resources
// exist; the baseline never receives explanation data. The explanation condition renders
// the same shared workspace as the exploration app over participant-safe data. Drafts save
// as you work, submitting is explicit and final, and the case timer starts only once the
// required content is usable.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { CaseLayout, answerSummaryOf } from "@/components/study/CaseLayout";
import { IdentityCard } from "@/components/study/IdentityCard";
import { AnswerPanel, CandidateRows, useRanking, type RankingCandidate, type RankingValue } from "@/components/study/RankingPanel";
import { ResourceAccessButton, ResourceList } from "@/components/study/Resources";
import { TutorialHelpButton } from "@/components/study/Tutorial";
import { ALL_COMPONENTS, PairWorkspace, type Component, type Focus } from "@/components/workspace/PairWorkspace";
import { describeError, getJson } from "@/lib/api";
import { curie } from "@/lib/iri";
import type { EntityRef } from "@/lib/types";
import { createApiSource, loadWorkspaceCapabilities, scopedRead } from "@/lib/workspace/apiSource";
import { indexResources } from "@/lib/workspace/resourceIndex";
import { createResourceSource } from "@/lib/workspace/resourceSource";
import type { WorkspaceAction, WorkspaceSource, WorkspaceCapabilities } from "@/lib/workspace/types";
import { WorkspaceProvider } from "@/lib/workspace/WorkspaceContext";
import { orderedKeys } from "@/study/formOrder";
import type { StudySession } from "@/study/session";
import type { Telemetry } from "@/study/telemetry";
import type { ExplanationResource, PublicAsset, StudyCase, StudyState } from "@/study/types";
import { workspaceEvent } from "@/study/workspaceEvents";

/** Components frozen in the final questionnaire's rating matrix are the ones shown. */
export function frozenComponents(state: StudyState): Set<Component> {
  const question = state.forms.final.find((item) => item.id === "component_usefulness");
  if (!question?.matrix) return new Set(ALL_COMPONENTS);
  try {
    return new Set(orderedKeys(question, "rows", state.forms.version) as Component[]);
  } catch {
    return new Set(Object.keys(question.matrix) as Component[]);
  }
}

export function ontologyLabelFor(resources: PublicAsset[], studyCase: { source: EntityRef; candidates: { entity: EntityRef }[] }) {
  return (ontology: string) => {
    const asset = resources.find((item) => item.ontology_version_id === ontology);
    if (asset?.title) return asset.title;
    if (ontology === studyCase.source.ontology_version_id) return "Source ontology";
    if (studyCase.candidates.some((candidate) => candidate.entity.ontology_version_id === ontology)) return "Target ontology";
    return "Ontology";
  };
}

function viewedKey(presentationId: string) {
  return `exact.study.viewed.${presentationId}`;
}

function readViewed(presentationId: string): Set<string> {
  try {
    return new Set(JSON.parse(window.sessionStorage.getItem(viewedKey(presentationId)) ?? "[]") as string[]);
  } catch {
    return new Set();
  }
}

export function CaseView({ state, session, telemetry, timingEnabled = true }: { state: StudyState; session: StudySession; telemetry: Telemetry; timingEnabled?: boolean }) {
  const caseKey = `${state.session_id}|${state.study_revision}|${state.current_case_id}|${state.current_presentation_id}`;
  const [studyCase, setStudyCase] = useState<StudyCase | null>(null);
  const [caseError, setCaseError] = useState<unknown>(null);
  const [resources, setResources] = useState<ExplanationResource[] | null>(null);
  const [workspaceCapabilities, setWorkspaceCapabilities] = useState<WorkspaceCapabilities | null>(null);
  const [resourceError, setResourceError] = useState<string | null>(null);
  const [value, setValue] = useState<RankingValue>({ responseType: null, ranked: [] });
  const [inspecting, setInspecting] = useState<string | null>(null);
  const [viewed, setViewed] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);
  const [reload, setReload] = useState(0);
  const [contentReady, setContentReady] = useState(false);
  const [tab, setTab] = useState<string | null>(null);
  const [sourceFocus, setSourceFocus] = useState<Focus | null>(null);
  const [targetFocus, setTargetFocus] = useState<Record<string, Focus | null>>({});
  const edited = useRef(false);
  const loadStarted = useRef(performance.now());

  // Load the server-assigned case, then any condition-permitted explanation resources.
  useEffect(() => {
    let cancelled = false;
    loadStarted.current = performance.now();
    setStudyCase(null);
    setResources(null);
    setWorkspaceCapabilities(null);
    setCaseError(null);
    setResourceError(null);
    setContentReady(false);
    (async () => {
      try {
        const current = await scopedRead<StudyCase>("/api/v1/study", "/cases/current", undefined, undefined, state.session_id);
        if (cancelled) return;
        setStudyCase(current);
        const saved = state.ranking && state.ranking.presentation_id === current.presentation_id ? state.ranking : null;
        setValue(saved ? { responseType: saved.response_type, ranked: saved.ranked_candidate_ids } : { responseType: null, ranked: [] });
        const first = current.candidates.find((candidate) => candidate.display_position === 1)?.candidate_id ?? null;
        setInspecting(first);
        const seen = readViewed(current.presentation_id);
        if (first) seen.add(first);
        setViewed(seen);
        if (current.condition === "explanation" && current.workspace?.scope_id) {
          // A scoped workspace (16 B1) is ready only once the source's context actually loads.
          try {
            const base = `/api/v1/study/workspace/${encodeURIComponent(current.workspace.scope_id)}`;
            const [capabilities] = await Promise.all([
              loadWorkspaceCapabilities(base, state.session_id),
              scopedRead(base, "/entity-context", { ontology_version_id: current.source.ontology_version_id, iri: current.source.iri, kind: current.source.kind }, undefined, state.session_id),
            ]);
            if (!cancelled) setWorkspaceCapabilities(capabilities);
          } catch (error) {
            if (!cancelled) setResourceError(`The case information could not be loaded: ${describeError(error)} Reconnect and retry before answering this case.`);
            return;
          }
        } else if (current.condition === "explanation" && current.explanation_refs.length) {
          try {
            const loaded = await Promise.all(current.explanation_refs.map((ref) => getJson<ExplanationResource>(`/api/v1/study/resources/${encodeURIComponent(ref)}`)));
            if (!cancelled) setResources(loaded);
          } catch (error) {
            if (!cancelled) setResourceError(`The prepared explanations could not be loaded: ${describeError(error)} Reconnect and retry before answering this case.`);
            return;
          }
        }
        if (!cancelled) setContentReady(true);
      } catch (error) {
        if (!cancelled) setCaseError(error);
      }
    })();
    return () => {
      cancelled = true;
    };
    // The case identity is the only trigger; restoring from state happens once per case.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseKey, reload]);

  useEffect(() => {
    if (contentReady && !submitting && timingEnabled) telemetry.markCaseReady(performance.now() - loadStarted.current);
  }, [contentReady, submitting, timingEnabled, telemetry.markCaseReady]);
  useEffect(() => {
    if (edited.current || !studyCase) return;
    const saved = state.ranking;
    if (saved?.presentation_id === studyCase.presentation_id) setValue({ responseType: saved.response_type, ranked: saved.ranked_candidate_ids });
  }, [state.ranking, studyCase]);
  useEffect(() => {
    if (!studyCase) return;
    try {
      window.sessionStorage.setItem(viewedKey(studyCase.presentation_id), JSON.stringify(Array.from(viewed)));
    } catch {
      /* inspection markers are a reading aid only */
    }
  }, [viewed, studyCase]);

  // After a conflict the server's saved draft replaces what was on screen.
  const handledConflict = useRef(0);
  useEffect(() => {
    if (session.save.kind !== "conflict" || session.save.at === handledConflict.current || !studyCase) return;
    handledConflict.current = session.save.at;
    edited.current = false;
    const saved = state.ranking && state.ranking.presentation_id === studyCase.presentation_id ? state.ranking : null;
    setValue(saved ? { responseType: saved.response_type, ranked: saved.ranked_candidate_ids } : { responseType: null, ranked: [] });
  }, [session.save, state.ranking, studyCase]);

  const locked = state.ranking?.workflow_state === "submitted" && state.ranking.presentation_id === studyCase?.presentation_id;

  const saveDraft = useCallback(
    (next: RankingValue) => {
      if (!studyCase || !state.current_case_id) return;
      void session
        .mutate({
          method: "PUT",
          path: `/api/v1/study/cases/${encodeURIComponent(state.current_case_id)}/draft`,
          body: { presentation_id: studyCase.presentation_id, response_type: next.responseType, ranked_candidate_ids: next.responseType === "ranked_candidates" ? next.ranked : [] },
          coalesce: `draft:${state.current_case_id}`,
          debounceMs: 500,
        })
        .catch(() => undefined);
    },
    [session, state.current_case_id, studyCase],
  );

  const submit = async () => {
    if (!studyCase || !state.current_case_id || submitting) return;
    setSubmitting(true);
    telemetry.emit("submit");
    try {
      await session.mutate({
        method: "POST",
        path: `/api/v1/study/cases/${encodeURIComponent(state.current_case_id)}/submit`,
        body: { presentation_id: studyCase.presentation_id, response_type: value.responseType, ranked_candidate_ids: value.responseType === "ranked_candidates" ? value.ranked : [] },
        transition: true,
      });
    } catch {
      /* the header explains; the answer stays on screen for another try */
    } finally {
      setSubmitting(false);
    }
  };

  const components = useMemo(() => frozenComponents(state), [state]);
  const caseLabels = useMemo(() => {
    const labels: Record<string, string> = {};
    if (studyCase) {
      labels[`${studyCase.source.ontology_version_id}|${studyCase.source.iri}`] = studyCase.source_label;
      studyCase.candidates.forEach((candidate) => {
        labels[`${candidate.entity.ontology_version_id}|${candidate.entity.iri}`] = candidate.label;
      });
    }
    return labels;
  }, [studyCase]);
  // Stable across autosaves: every state update brings a new resources array.
  const resourcesRef = useRef(state.ontology_resources);
  resourcesRef.current = state.ontology_resources;
  const ontologyLabel = useCallback((ontology: string) => (studyCase ? ontologyLabelFor(resourcesRef.current, studyCase)(ontology) : "Ontology"), [studyCase]);

  // One workspace source per presentation: nothing from another case or session can appear.
  const source: WorkspaceSource | null = useMemo(() => {
    if (!studyCase || studyCase.condition !== "explanation") return null;
    const key = `study|${state.session_id}|${state.study_revision}|${studyCase.presentation_id}`;
    if (studyCase.workspace?.scope_id) {
      if (!workspaceCapabilities) return null;
      return createApiSource({ key, sessionId: state.session_id, kind: "study_resource", capabilities: workspaceCapabilities, base: `/api/v1/study/workspace/${encodeURIComponent(studyCase.workspace.scope_id)}`, ontologyName: (id) => ontologyLabel(id) });
    }
    if (!resources) return null;
    return createResourceSource({
      key,
      kind: "study_resource",
      index: indexResources(resources, { extraLabels: caseLabels }),
      navigation: "prepared",
      scopeNote: "Only the information prepared for this case is available here.",
      ontologyName: (id) => ontologyLabel(id),
    });
  }, [studyCase, resources, caseLabels, ontologyLabel, state.session_id, workspaceCapabilities]);

  const onAction = useCallback(
    (action: WorkspaceAction) => {
      const mapped = workspaceEvent(action);
      if (mapped) telemetry.emit(mapped.type, { component: mapped.component, element: mapped.element });
    },
    [telemetry],
  );

  const rankingCandidates: RankingCandidate[] = useMemo(
    () =>
      (studyCase?.candidates ?? []).map((candidate) => ({
        id: candidate.candidate_id,
        position: candidate.display_position,
        label: candidate.label,
        identifier: curie(candidate.entity.iri),
        score: candidate.score.toFixed(2),
        scoreMeaning: candidate.score_meaning,
      })),
    [studyCase],
  );
  const controller = useRanking({
    candidates: rankingCandidates,
    value,
    locked,
    disabled: !contentReady,
    submitting,
    resetKey: handledConflict.current,
    onChange: (next, event, element) => {
      edited.current = true;
      setValue(next);
      telemetry.emit(event, { component: "ranking", element });
      saveDraft(next);
    },
  });

  if (caseError) return <ErrorNote error={caseError} what="This case could not be loaded" />;
  if (!studyCase) return <Skeleton lines={6} />;

  const explanation = studyCase.condition === "explanation";
  const inspected = studyCase.candidates.find((candidate) => candidate.candidate_id === inspecting) ?? studyCase.candidates[0];
  const inspect = (id: string) => {
    setInspecting(id);
    setViewed((current) => new Set([...current, id]));
    telemetry.emit("candidate_inspected", { component: "ranking", element: id });
  };
  const pendingNote = submitting && session.save.kind === "offline" ? "Your answer is waiting in this tab and will be sent when the connection returns. It is not submitted until the study server confirms it." : null;

  const question = (names: { source: string | null; target: string | null }) => (
    <div className="pair-header">
      <div className="pair-heading">
        <span className="meta">
          Inspecting initial position {inspected.display_position} of {studyCase.candidates.length} · matching score {inspected.score.toFixed(2)}
        </span>
        <h1 className="pair-question">
          Does <span className="text-source">{names.source ?? studyCase.source_label}</span> mean the same as <span className="text-target">{names.target ?? inspected.label}</span>?
        </h1>
      </div>
    </div>
  );

  const workspace = explanation ? (
    resourceError ? (
      <p className="note note-bad" role="alert">
        {resourceError}{" "}
        <button type="button" className="btn btn-sm" onClick={() => setReload((count) => count + 1)}>
          Retry explanations
        </button>
      </p>
    ) : source ? (
      <WorkspaceProvider source={source} onAction={onAction}>
        <PairWorkspace
          pair={{ source: studyCase.source, target: inspected.entity, candidateId: inspected.candidate_id }}
          viewKey={`${studyCase.presentation_id}|${inspected.candidate_id}`}
          ontologyLabel={ontologyLabel}
          components={components}
          tab={tab}
          onTab={setTab}
          navigation={{
            source: sourceFocus,
            target: targetFocus[inspected.candidate_id] ?? null,
            set: (side, focus) => (side === "source" ? setSourceFocus(focus) : setTargetFocus((current) => ({ ...current, [inspected.candidate_id]: focus }))),
          }}
          header={question}
          cardTitles={{ source: "Source concept", target: `Candidate · initial position ${inspected.display_position}` }}
          compactCards
        />
      </WorkspaceProvider>
    ) : (
      <Skeleton lines={6} />
    )
  ) : (
    <div className="pair-workspace baseline-workspace">
      {question({ source: studyCase.source_label, target: inspected.label })}
      <div className="card-pair">
        <IdentityCard side="source" entity={studyCase.source} label={studyCase.source_label} ontology={ontologyLabel(studyCase.source.ontology_version_id)} title="Source concept" onCopied={() => onAction({ type: "copy_iri", side: "source" })} />
        <IdentityCard side="target" entity={inspected.entity} label={inspected.label} ontology={ontologyLabel(inspected.entity.ontology_version_id)} title={`Candidate · initial position ${inspected.display_position}`} onCopied={() => onAction({ type: "copy_iri", side: "target" })} />
      </div>
      <section className="study-card baseline-tools">
        <h2>Inspect with your own methods</h2>
        <p>
          This block shows no explanations. Inspect the source and candidates in any way you choose: an ontology editor such as Protégé, a viewer, the files directly, queries, or none of these. Copy an IRI to find an entity in the files or in your tool.
        </p>
        <ResourceList resources={state.ontology_resources} compact onDownload={(asset) => telemetry.emit("external_resource_link", { component: "downloads", element: asset })} />
        <p className="meta">Time you spend inspecting before you submit counts as part of this case. Switching windows does not pause anything.</p>
      </section>
    </div>
  );

  return (
    <div className="case-page">
      <CaseLayout
        answerId="case-answer"
        answerSummary={answerSummaryOf(value, locked)}
        toolbar={
          <div className="case-toolbar">
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
            <ResourceAccessButton resources={state.ontology_resources} onOpen={() => telemetry.emit("resources_open", { component: "downloads" })} onDownload={(asset) => telemetry.emit("external_resource_link", { component: "downloads", element: asset })} />
            {state.tutorial && <TutorialHelpButton tutorial={state.tutorial} onOpen={() => telemetry.emit("help_open", { component: "tutorial_help", scope: "help" })} />}
          </div>
        }
        rows={<CandidateRows controller={controller} inspecting={inspected.candidate_id} viewed={viewed} onInspect={inspect} compact />}
        answer={<AnswerPanel id="case-answer" controller={controller} onSubmit={submit} submitting={submitting} pendingNote={pendingNote} compact />}
        workspace={workspace}
      />
    </div>
  );
}
