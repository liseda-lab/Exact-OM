"use client";

// One scored case. The server decides which case and condition this is and which resources
// exist; the baseline never receives explanation data. The explanation condition renders
// the same shared workspace as the exploration app over its authorized scope. Ranking and
// submission are enabled, and the case timer starts, only once this presentation's required
// content is validated and rendered (19 F12); a failure blocks the case with a retry and
// keeps the answer. Drafts save as you work; submitting is explicit and final.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Skeleton } from "@/components/common/ErrorNote";
import { CaseLayout, answerSummaryOf } from "@/components/study/CaseLayout";
import { RenderProbe, useCaseContent, WorkspaceBoundary, type CaseReadiness } from "@/components/study/caseContent";
import { IdentityCard } from "@/components/study/IdentityCard";
import { AnswerPanel, CandidateRows, useRanking, type RankingCandidate, type RankingValue } from "@/components/study/RankingPanel";
import { ResourceAccessButton, ResourceList } from "@/components/study/Resources";
import { TutorialHelpButton } from "@/components/study/Tutorial";
import { ALL_COMPONENTS, PairWorkspace, type Component, type Focus } from "@/components/workspace/PairWorkspace";
import { curie } from "@/lib/iri";
import type { EntityRef } from "@/lib/types";
import type { WorkspaceAction } from "@/lib/workspace/types";
import { WorkspaceProvider } from "@/lib/workspace/WorkspaceContext";
import { orderedKeys } from "@/study/formOrder";
import type { StudySession } from "@/study/session";
import type { Telemetry } from "@/study/telemetry";
import type { PublicAsset, StudyCase, StudyState } from "@/study/types";
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

/** Loading progress or a blocking failure with its causes and a retry. */
function ReadinessNote({ readiness, onRetry }: { readiness: CaseReadiness; onRetry: () => void }) {
  if (readiness.kind === "usable") return null;
  if (readiness.kind !== "blocked")
    return (
      <p className="note case-readiness" role="status">
        {readiness.kind === "loading" && readiness.total > 1 ? `Loading this case's information (${readiness.done} of ${readiness.total})…` : "Loading this case's information…"} You can rank once it has loaded.
      </p>
    );
  return (
    <div className="note note-bad case-readiness" role="alert">
      <p>
        <strong>{readiness.title}</strong> Your answer so far is kept, and nothing is submitted. You cannot submit this case until it loads.
      </p>
      <ul>
        {readiness.failures.map((failure, index) => (
          <li key={`${failure.label}-${index}`}>
            {failure.label}: {failure.detail}
          </li>
        ))}
      </ul>
      <p className="meta">
        {readiness.cause === "session"
          ? "If you opened the study in another tab or device, continue there, or reload this page."
          : "Retry now, or use Pause at the top and come back later with your private link. If it keeps failing, contact the study team."}
      </p>
      <button type="button" className="btn btn-sm" onClick={onRetry}>
        Retry loading this case
      </button>
    </div>
  );
}

/** Baseline cases need no workspace; they are usable once their identity cards render. */
function BaselineRendered({ attempt, onRendered }: { attempt: number; onRendered: (attempt: number) => void }) {
  useEffect(() => {
    const frame = requestAnimationFrame(() => onRendered(attempt));
    return () => cancelAnimationFrame(frame);
  }, [attempt, onRendered]);
  return null;
}

export function CaseView({ state, session, telemetry, timingEnabled = true }: { state: StudyState; session: StudySession; telemetry: Telemetry; timingEnabled?: boolean }) {
  const [value, setValue] = useState<RankingValue>({ responseType: null, ranked: [] });
  const [inspecting, setInspecting] = useState<string | null>(null);
  const [viewed, setViewed] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);
  const [tab, setTab] = useState<string | null>(null);
  const [sourceFocus, setSourceFocus] = useState<Focus | null>(null);
  const [targetFocus, setTargetFocus] = useState<Record<string, Focus | null>>({});
  const edited = useRef(false);
  const loadStarted = useRef(performance.now());

  const components = useMemo(() => frozenComponents(state), [state]);
  // Stable across autosaves: every state update brings a new resources array.
  const resourcesRef = useRef(state.ontology_resources);
  resourcesRef.current = state.ontology_resources;
  const caseRef = useRef<StudyCase | null>(null);
  const ontologyLabel = useCallback((ontology: string) => (caseRef.current ? ontologyLabelFor(resourcesRef.current, caseRef.current)(ontology) : "Ontology"), []);
  const caseLabels = useCallback((current: StudyCase) => {
    const labels: Record<string, string> = { [`${current.source.ontology_version_id}|${current.source.iri}`]: current.source_label };
    current.candidates.forEach((candidate) => (labels[`${candidate.entity.ontology_version_id}|${candidate.entity.iri}`] = candidate.label));
    return labels;
  }, []);
  const content = useCaseContent({ state, components, ontologyName: ontologyLabel, caseLabels });
  const { studyCase, source, readiness } = content;
  caseRef.current = studyCase;
  const usable = readiness.kind === "usable";

  // Restore the saved draft and reading aids once per presentation.
  const restoredFor = useRef<string | null>(null);
  useEffect(() => {
    if (!studyCase || restoredFor.current === studyCase.presentation_id) return;
    restoredFor.current = studyCase.presentation_id;
    const saved = state.ranking && state.ranking.presentation_id === studyCase.presentation_id ? state.ranking : null;
    if (!edited.current) setValue(saved ? { responseType: saved.response_type, ranked: saved.ranked_candidate_ids } : { responseType: null, ranked: [] });
    const first = [...studyCase.candidates].sort((a, b) => a.display_position - b.display_position)[0]?.candidate_id ?? null;
    setInspecting(first);
    const seen = readViewed(studyCase.presentation_id);
    if (first) seen.add(first);
    setViewed(seen);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studyCase]);

  // The case timer starts only when this page's required content is usable and rendered.
  const presentationId = studyCase?.presentation_id ?? null;
  useEffect(() => {
    telemetry.setCaseUsable(usable ? presentationId : null);
    if (usable && !submitting && timingEnabled) telemetry.markCaseReady(performance.now() - loadStarted.current);
  }, [usable, submitting, timingEnabled, presentationId, telemetry.markCaseReady, telemetry.setCaseUsable]);
  useEffect(() => () => telemetry.setCaseUsable(null), [telemetry.setCaseUsable]);

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
    // Submit only an answer given against usable content for this exact presentation.
    if (!usable || studyCase.presentation_id !== state.current_presentation_id || studyCase.study_revision !== state.study_revision) return;
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

  const onAction = useCallback(
    (action: WorkspaceAction) => {
      const mapped = workspaceEvent(action);
      if (mapped) telemetry.emit(mapped.type, { component: mapped.component, element: mapped.element });
    },
    [telemetry],
  );

  const rankingCandidates: RankingCandidate[] = useMemo(
    () =>
      [...(studyCase?.candidates ?? [])]
        .sort((a, b) => a.display_position - b.display_position)
        .map((candidate) => ({
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
    disabled: !usable,
    submitting,
    resetKey: handledConflict.current,
    onChange: (next, event, element) => {
      edited.current = true;
      setValue(next);
      telemetry.emit(event, { component: "ranking", element });
      saveDraft(next);
    },
  });

  if (!studyCase) return readiness.kind === "blocked" ? <ReadinessNote readiness={readiness} onRetry={content.retry} /> : <Skeleton lines={6} />;

  const explanation = studyCase.condition === "explanation";
  // Generated descriptions exist only for the source and its five candidates; never request others.
  const focal = new Set([studyCase.source, ...studyCase.candidates.map((candidate) => candidate.entity)].map((entity) => `${entity.ontology_version_id}|${entity.kind}|${entity.iri}`));
  const isFocal = (entity: EntityRef) => focal.has(`${entity.ontology_version_id}|${entity.kind}|${entity.iri}`);
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

  // While blocked the workspace is not shown: nothing re-reads a failed dependency behind the
  // participant's back, and Retry is the one path that refetches it (19 F18). The answer,
  // inspected candidate, focus, tab and cached usable content are all kept.
  const workspace = explanation ? (
    source && readiness.kind !== "blocked" ? (
      <WorkspaceProvider source={source} onAction={onAction}>
        <WorkspaceBoundary key={content.attempt} onError={content.renderFailed}>
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
            profileAllowed={isFocal}
          />
          {readiness.kind === "rendering" && (
            <RenderProbe source={studyCase.source} target={inspected.entity} components={components} attempt={readiness.attempt} onRendered={content.rendered} onStalled={() => content.renderFailed(new Error("The case display did not finish."))} />
          )}
        </WorkspaceBoundary>
      </WorkspaceProvider>
    ) : (
      <div className="pair-workspace">
        {question({ source: studyCase.source_label, target: inspected.label })}
        {readiness.kind !== "blocked" && <Skeleton lines={6} />}
      </div>
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
      {readiness.kind === "rendering" && <BaselineRendered attempt={readiness.attempt} onRendered={content.rendered} />}
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
        status={usable ? null : <ReadinessNote readiness={readiness} onRetry={content.retry} />}
        rows={<CandidateRows controller={controller} inspecting={inspected.candidate_id} viewed={viewed} onInspect={inspect} compact />}
        answer={<AnswerPanel id="case-answer" controller={controller} onSubmit={submit} submitting={submitting} pendingNote={pendingNote} compact />}
        workspace={workspace}
      />
    </div>
  );
}
