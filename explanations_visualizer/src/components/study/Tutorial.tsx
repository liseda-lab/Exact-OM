"use client";

// exact-study/2.0 interactive tutorial and comprehension check (14, 16 B3). Lessons run in
// the production shared workspace over the publication's synthetic resources; the actions a
// lesson asks for are reported as typed requirement evidence, saved through the outbox and
// shown as saved only once the server acknowledges them. The five items are graded by the
// server, which returns specific feedback; retries are unlimited. Allocation happens only
// when the server accepts completion. Nothing here is scored or touches a scored case.
// The screen is a durable `position` (18 B12): navigation saves its destination, a reload or
// another device restores the server's acknowledged position, and focus follows deliberate
// navigation only (19 F13).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconCheck, IconWarning } from "@/components/common/Icons";
import { CaseLayout, answerSummaryOf } from "@/components/study/CaseLayout";
import { IdentityCard } from "@/components/study/IdentityCard";
import { AnswerPanel, CandidateRows, useRanking, type RankingEvent, type RankingValue } from "@/components/study/RankingPanel";
import { ResourceAccessButton, ResourceList } from "@/components/study/Resources";
import { METHOD_LABELS } from "@/components/study/SetupV2";
import { LessonCoach, TOUR_TARGETS, findTarget, highlight, type RequirementState } from "@/components/study/LessonCoach";
import { MessagePage } from "@/components/study/Stages";
import { Paragraphs } from "@/components/study/StudyChrome";
import { PairWorkspace, type Focus } from "@/components/workspace/PairWorkspace";
import { ApiError, describeError, getJson } from "@/lib/api";
import { curie } from "@/lib/iri";
import { indexResources } from "@/lib/workspace/resourceIndex";
import { createResourceSource } from "@/lib/workspace/resourceSource";
import type { WorkspaceAction } from "@/lib/workspace/types";
import { WorkspaceProvider } from "@/lib/workspace/WorkspaceContext";
import type { Telemetry } from "@/study/telemetry";
import { workspaceEvent } from "@/study/workspaceEvents";
import { uuid, type StudySession } from "@/study/session";
import { assessmentPosition, followAcknowledged, lessonPosition, restoredPosition } from "@/study/tutorialPosition";
import { AssessmentDrafts, canonical, type KeptConflict } from "@/study/assessmentDrafts";
import { DiscardedMutation, ReplacedMutation, type QueueState } from "@/study/mutationQueue";
import { LessonEvidence, evidenceFor, type LessonSignal } from "@/study/tutorialProgress";
import { isReviewProtocol, type AssessmentItem, type AssessmentResponse, type ExplanationResource, type LessonRequirement, type RequirementAction, type StudyState, type TutorialLesson, type TutorialPosition, type TutorialPublic } from "@/study/types";
import { orderedEntries, orderedKeys } from "@/study/formOrder";

const PRACTICE_KEY = "practice-case";

function ontologyNames(tutorial: TutorialPublic) {
  return (id: string) => tutorial.case.ontology_resources.find((asset) => asset.ontology_version_id === id)?.title ?? (id === tutorial.case.source.ontology_version_id ? "Practice source ontology" : "Practice target ontology");
}

export function TutorialStage({ state, session, onPause, telemetry }: { state: StudyState; session: StudySession; onPause: () => void; telemetry: Telemetry }) {
  const tutorial = state.tutorial ?? null;
  if (!tutorial) {
    return (
      <MessagePage title="The tutorial for this study is missing">
        <p>This study version requires an interactive tutorial, but none was included in its publication. The study cannot continue to scored cases without it, and this page will not substitute a different tutorial.</p>
        <p className="muted">Your progress is saved. Please tell the study team.</p>
      </MessagePage>
    );
  }
  // An incomplete or mismatched tutorial is a publication problem, never replaced by a fallback.
  const progressVersion = state.tutorial_progress?.tutorial_version;
  if (!tutorial.lessons.length || !tutorial.assessment.length || !tutorial.case?.candidates?.length || (progressVersion && progressVersion !== tutorial.version)) {
    return (
      <MessagePage title="The tutorial for this study cannot be shown">
        <p>The tutorial included with this study version is incomplete or does not match your saved progress, so this page will not continue with it or substitute another one. Your progress is saved.</p>
        <p className="muted">Please tell the study team. Tutorial version {tutorial.version}.</p>
      </MessagePage>
    );
  }
  return <Tutorial state={state} session={session} tutorial={tutorial} onPause={onPause} telemetry={telemetry} />;
}

function Tutorial({ state, session, tutorial, onPause, telemetry }: { state: StudyState; session: StudySession; tutorial: TutorialPublic; onPause: () => void; telemetry: Telemetry }) {
  const progress = state.tutorial_progress ?? null;
  const [resources, setResources] = useState<ExplanationResource[] | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [reload, setReload] = useState(0);
  const [position, setPosition] = useState<TutorialPosition>(() => restoredPosition(tutorial, progress?.position));
  const view = position.view === "lesson" ? position.lesson_id : "assessment";
  // Focus moves only after deliberate navigation, never on restore or an autosave.
  const focusAfterNavigation = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const [restoredPending] = useState(() => session.pendingCount() > 0);
  const [helpOpen, setHelpOpen] = useState(false);
  const acknowledged = useMemo(() => new Set(progress?.completed_requirements ?? []), [progress?.completed_requirements]);
  const acknowledgedRef = useRef(acknowledged);
  acknowledgedRef.current = acknowledged;

  // Visible assessment drafts above the keyed question cards (S0-R2, S0-FU-R1, S0-FU-R2): each
  // write is followed to its settlement, including writes restored after a reload; an older
  // acknowledgement never replaces a newer edit; and after a failure an answer is saved again only
  // when the saved answer still equals the one it started from. Otherwise the saved answer is
  // shown and this tab's answer is kept for the participant to choose (see assessmentDrafts.ts).
  const drafts = useRef<AssessmentDrafts | null>(null);
  const [, setDraftVersion] = useState(0);
  const conflictKey = `exact.study.assessment-conflicts.${state.session_id}`;
  /** The saved answers in the latest accepted state: the revision the next write is made against. */
  const observed = useCallback(() => session.current()?.tutorial_progress?.assessment_drafts, [session]);
  const keepConflicts = useCallback(() => {
    const kept = drafts.current!.conflicts();
    try {
      if (kept.length) window.sessionStorage.setItem(conflictKey, JSON.stringify(kept));
      else window.sessionStorage.removeItem(conflictKey);
    } catch {
      /* the recovery archive still holds the unsaved answer */
    }
  }, [conflictKey]);
  const sendDraftRef = useRef<(questionId: string, response: AssessmentResponse) => void>(() => undefined);
  /** After any settlement or new state: finish saved work, re-save safe failures, keep conflicts. */
  const settle = useCallback(() => {
    const resaves = drafts.current!.observe(observed());
    setDraftVersion((value) => value + 1);
    keepConflicts();
    for (const resave of resaves) sendDraftRef.current(resave.questionId, resave.response);
  }, [observed, keepConflicts]);
  const settleRef = useRef(settle);
  settleRef.current = settle;
  const track = useCallback((questionId: string, settled: Promise<QueueState>) => {
    settled.then(
      (next) => {
        drafts.current!.acknowledged(questionId, (next as StudyState).tutorial_progress?.assessment_drafts);
        settleRef.current();
      },
      (error) => {
        if (error instanceof ReplacedMutation) drafts.current!.replaced(questionId);
        else drafts.current!.failed(questionId);
        settleRef.current();
      },
    );
  }, []);
  if (!drafts.current) {
    const store = new AssessmentDrafts();
    store.restoreConflicts(readConflicts(conflictKey, tutorial));
    // Writes restored after a reload are followed to their settlement like new ones (S0-FU-R2).
    const restored = session.queued().flatMap((entry) => {
      const draft = entry.path === PROGRESS_PATH ? (entry.body.assessment_draft as { question_id?: string; response?: AssessmentResponse } | undefined) : undefined;
      if (!draft?.question_id || !draft.response) return [];
      const base = entry.meta?.assessment_base;
      return [{ questionId: draft.question_id, response: draft.response, base: typeof base === "string" ? base : null, settled: entry.settled }];
    });
    store.seed(restored);
    drafts.current = store;
    for (const write of restored) track(write.questionId, write.settled);
  }
  const serverDrafts = progress?.assessment_drafts;
  useEffect(() => settleRef.current(), [serverDrafts]);
  const sendDraft = useCallback(
    (questionId: string, response: AssessmentResponse) => {
      const store = drafts.current!;
      store.sent(questionId);
      // The base travels with the queued write (never to the server), so a reload keeps the proof.
      track(questionId, session.mutate({ method: "PUT", path: PROGRESS_PATH, body: { tutorial_version: tutorial.version, assessment_draft: { question_id: questionId, response } }, coalesce: `tutorial:item:${questionId}`, debounceMs: 500, meta: { assessment_base: store.baseOf(questionId) } }));
    },
    [session, tutorial.version, track],
  );
  sendDraftRef.current = sendDraft;
  const editDraft = useCallback(
    (questionId: string, response: AssessmentResponse) => {
      if (drafts.current!.edit(questionId, response, observed()?.[questionId])) sendDraft(questionId, response);
      setDraftVersion((value) => value + 1);
      keepConflicts();
    },
    [sendDraft, observed, keepConflicts],
  );
  const resolveDraft = useCallback(
    (questionId: string, choice: "mine" | "saved") => {
      const mine = drafts.current!.resolve(questionId, choice, observed()?.[questionId]);
      if (mine) sendDraft(questionId, mine);
      setDraftVersion((value) => value + 1);
      keepConflicts();
    },
    [sendDraft, observed, keepConflicts],
  );

  // Valid lesson evidence that the server has not acknowledged yet (21 F01.4–5). "Saving…" is
  // shown for it; a tick only for acknowledged requirements.
  const evidence = useRef(new LessonEvidence());
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState<{ tone: "info" | "warn"; text: string } | null>(null);
  useEffect(() => {
    evidence.current.acknowledge(acknowledged);
    setPending(evidence.current.pending());
  }, [acknowledged]);

  useEffect(() => {
    let cancelled = false;
    setLoadError(null);
    Promise.all(tutorial.case.explanation_refs.map((id) => getJson<ExplanationResource>(`/api/v1/study/resources/${encodeURIComponent(id)}`)))
      .then((loaded) => !cancelled && setResources(loaded))
      .catch((error) => !cancelled && setLoadError(error));
    return () => {
      cancelled = true;
    };
  }, [tutorial, reload]);

  // The source keeps its identity across state updates (autosaves create new state objects).
  const tutorialRef = useRef(tutorial);
  tutorialRef.current = tutorial;
  const ontologyName = useCallback((id: string) => ontologyNames(tutorialRef.current)(id), []);
  const source = useMemo(() => {
    if (!resources) return null;
    const labels: Record<string, string> = { [`${tutorial.case.source.ontology_version_id}|${tutorial.case.source.iri}`]: tutorial.case.source_label };
    tutorial.case.candidates.forEach((candidate) => (labels[`${candidate.entity.ontology_version_id}|${candidate.entity.iri}`] = candidate.label));
    return createResourceSource({
      key: `tutorial|${tutorial.version}|${state.session_id}`,
      kind: "tutorial",
      index: indexResources(resources, { prepared: "all", extraLabels: labels }),
      navigation: "complete",
      scopeNote: "This practice workspace holds the whole invented practice ontologies.",
      ontologyName,
      synthetic: true,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resources, tutorial.version, state.session_id, ontologyName]);

  const lesson = tutorial.lessons.find((item) => item.lesson_id === view) ?? null;
  const labelOf = useCallback((id: string) => tutorial.lessons.flatMap((item) => item.requirements).find((requirement) => requirement.requirement_id === id)?.label ?? id, [tutorial]);
  const practiceRef = useRef<Record<string, unknown> | null>(null);

  // Every lesson save carries only valid unacknowledged evidence; acknowledged work is never
  // resent, and evidence a terminal validation failure rejected is dropped, so a later save
  // (help, a practice change, the next lesson) cannot carry it again (U02).
  const saveRef = useRef<(body: Record<string, unknown>, coalesce: string) => void>(() => undefined);
  const save = useCallback(
    (body: Record<string, unknown>, coalesce: string) => {
      evidence.current.acknowledge(acknowledgedRef.current);
      const sent = evidence.current.snapshot();
      const completed = Array.from(new Set([...acknowledgedRef.current, ...sent.map((item) => item.requirement_id)]));
      session
        .mutate({ method: "PUT", path: "/api/v1/study/tutorial/progress", body: { tutorial_version: tutorial.version, actions: sent, ...body, completed_requirements: completed }, coalesce, debounceMs: 400 })
        .catch((error) => {
          if (error instanceof DiscardedMutation) {
            // Never sent: keep the evidence and send it again after the failure is handled.
            if (evidence.current.size) window.setTimeout(() => saveRef.current({}, "tutorial:resync"), 0);
            return;
          }
          if (!(error instanceof ApiError)) return; // replaced by a newer draft, or the session ended
          if (error.status === 409) {
            // The session refreshed to the other tab's state; resend only still-missing evidence.
            if (evidence.current.size) window.setTimeout(() => saveRef.current({}, "tutorial:resync"), 0);
            return;
          }
          if (error.status < 400 || error.status >= 500 || error.status === 401) return;
          // The server names the refused requirements; valid evidence sent with them is kept and resent.
          const refused = new Set(error.validation?.requirement_ids ?? []);
          const named = sent.filter((item) => refused.has(item.requirement_id));
          const removed = evidence.current.reject(named.length ? named : sent);
          setPending(evidence.current.pending());
          session.clearNotice();
          setNotice({
            tone: "warn",
            text: removed.length
              ? `Not saved: ${removed.map((id) => `“${labelOf(id)}”`).join(", ")}. The study server did not accept that action as lesson evidence. Try the step again; your other progress is kept.`
              : "A tutorial change was not saved. Try the step again; your other progress is kept.",
          });
          if (evidence.current.size) window.setTimeout(() => saveRef.current({}, "tutorial:resync"), 0);
        });
    },
    [session, tutorial.version, labelOf],
  );
  saveRef.current = save;

  // Follow the server's acknowledged position when it changes underneath an unchanged screen
  // (an outbox entry replayed after reload), and adopt it after a conflict.
  const serverPosition = progress?.position ?? null;
  const lastServer = useRef(serverPosition);
  useEffect(() => {
    const previous = lastServer.current;
    lastServer.current = serverPosition;
    setPosition((current) => followAcknowledged(current, previous, serverPosition, tutorial));
  }, [serverPosition, tutorial]);
  const handledConflict = useRef(0);
  useEffect(() => {
    if (session.save.kind !== "conflict" || session.save.at === handledConflict.current) return;
    handledConflict.current = session.save.at;
    setPosition(restoredPosition(tutorial, progress?.position));
  }, [session.save, progress?.position, tutorial]);

  const record = useCallback(
    (signal: LessonSignal | null, practice?: Record<string, unknown>) => {
      if (!lesson) return;
      const observation = signal?.kind === "workspace" ? workspaceEvent(signal.action) : signal?.kind === "rank" ? { type: signal.event, component: "ranking" } : signal?.kind === "inspect" ? { type: "candidate_inspected" as const } : signal?.kind === "downloads" ? { type: "resources_open" as const, component: "downloads" } : null;
      if (observation) telemetry.emit(observation.type, { component: observation.component, scope: "tutorial", lessonId: lesson.lesson_id });
      if (practice) practiceRef.current = practice;
      const produced = signal ? evidenceFor(lesson, signal, tutorial, resources ?? [], practiceRef.current) : [];
      const added = evidence.current.observe(produced, acknowledgedRef.current);
      if (added.length) {
        setPending(evidence.current.pending());
        setNotice(null);
      }
      if (!added.length && !practice) return;
      save({ lesson_id: lesson.lesson_id, ...(practiceRef.current ? { practice: practiceRef.current } : {}) }, `tutorial:lesson:${lesson.lesson_id}`);
    },
    [lesson, save, tutorial, resources, telemetry],
  );

  // Build the destination first, then save exactly that object; a newer destination may
  // replace an unsent one, but never another lesson's evidence, drafts or attempts.
  const navigate = (destination: TutorialPosition) => {
    focusAfterNavigation.current = true;
    setNotice(null);
    setPosition(destination);
    void session.mutate({ method: "PUT", path: "/api/v1/study/tutorial/progress", body: { tutorial_version: tutorial.version, position: destination }, coalesce: "tutorial:position", debounceMs: 300 }).catch(() => undefined);
  };
  const go = (next: string) => navigate(next === "assessment" ? assessmentPosition() : lessonPosition(next));
  const positionKey = `${position.view}|${position.lesson_id ?? ""}|${position.question_id ?? ""}`;
  useEffect(() => {
    if (!focusAfterNavigation.current) return;
    focusAfterNavigation.current = false;
    const target = position.view === "assessment" && position.question_id ? document.getElementById(`assessment-${position.question_id}-h`) : heading.current;
    if (!target) return;
    target.focus({ preventScroll: true });
    target.scrollIntoView({ block: "start" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [positionKey]);

  if (loadError) {
    return (
      <div className="study-page">
        <ErrorNote error={loadError} onRetry={() => setReload((value) => value + 1)} what="The practice material" />
      </div>
    );
  }
  if (!source) return <Skeleton lines={6} />;

  const index = tutorial.lessons.findIndex((item) => item.lesson_id === view);
  const outstandingLessons = tutorial.lessons.filter((item) => !item.optional && item.requirements.some((requirement) => !acknowledged.has(requirement.requirement_id)));
  const states: Record<string, RequirementState> = {};
  for (const item of tutorial.lessons) for (const requirement of item.requirements) states[requirement.requirement_id] = acknowledged.has(requirement.requirement_id) ? "saved" : pending.has(requirement.requirement_id) ? "saving" : "todo";
  const openHelp = () => {
    setHelpOpen(true);
    telemetry.emit("help_open", { component: "tutorial_help", scope: "help", lessonId: lesson?.lesson_id });
    save({ help_opened: true }, "tutorial:help");
  };
  return (
    <div className="study-page study-page-wide tutorial-page">
      <div className="tutorial-head">
        <div className="tutorial-title">
          <span className="pill">Practice · not scored · invented examples</span>
          <h1 ref={heading} id="tutorial-heading" tabIndex={-1}>
            {lesson ? `Lesson ${index + 1} of ${tutorial.lessons.length}: ${lesson.title}` : "Check your understanding"}
          </h1>
        </div>
        <nav className="lesson-nav" aria-label="Tutorial lessons">
          <ol>
            {tutorial.lessons.map((item, number) => {
              const finished = item.requirements.every((requirement) => acknowledged.has(requirement.requirement_id));
              return (
                <li key={item.lesson_id}>
                  <button type="button" className={item.lesson_id === view ? "lesson-link current" : "lesson-link"} aria-current={item.lesson_id === view ? "step" : undefined} title={item.title} onClick={() => go(item.lesson_id)}>
                    <span className="lesson-mark" aria-hidden="true">{finished ? <IconCheck /> : number + 1}</span>
                    <span className="lesson-link-text">
                      {item.title}
                      {item.optional ? <span className="meta"> · optional</span> : null}
                      <span className="sr-only">{finished ? " (done)" : " (not finished)"}</span>
                    </span>
                  </button>
                </li>
              );
            })}
            <li>
              <button type="button" className={view === "assessment" ? "lesson-link current" : "lesson-link"} aria-current={view === "assessment" ? "step" : undefined} title="Five short questions" onClick={() => go("assessment")}>
                <span className="lesson-mark" aria-hidden="true">{(progress?.passed_items.length ?? 0) === tutorial.assessment.length ? <IconCheck /> : "?"}</span>
                <span className="lesson-link-text">Five short questions</span>
              </button>
            </li>
          </ol>
        </nav>
      </div>
      {index === 0 && view !== "assessment" && <p className="tutorial-intro">{tutorial.intro}</p>}
      {restoredPending && session.pendingCount() > 0 && (
        <p className="note" role="status">
          Changes made before this page was reloaded are still being saved. Until the study server confirms them, this page shows your last saved position and steps.
        </p>
      )}

      {lesson ? (
        <>
          <LessonWorkspace
            key={lesson.lesson_id}
            tutorial={tutorial}
            lesson={lesson}
            state={state}
            source={source}
            record={record}
            coach={(showMe, tools) => (
              <LessonCoach
                tools={tools}
                lessonNumber={index + 1}
                lessonCount={tutorial.lessons.length}
                lesson={lesson}
                states={states}
                notice={notice}
                onShowMe={(requirement) => {
                  const missing = showMe(requirement);
                  setNotice(missing ? { tone: "info", text: missing } : null);
                }}
                onBack={() => go(tutorial.lessons[index - 1].lesson_id)}
                canGoBack={index > 0}
                onNext={() => go(tutorial.lessons[index + 1]?.lesson_id ?? "assessment")}
                nextLabel={index + 1 < tutorial.lessons.length ? "Next lesson" : "Go to the five questions"}
                extra={
                  <p className="meta">
                    Practice answers are never scored. You can move on and come back to any step.{" "}
                    <button type="button" className="link-btn" onClick={openHelp}>
                      Open help
                    </button>
                  </p>
                }
              />
            )}
          />
          <div className="study-actions lesson-actions">
            <button type="button" className="btn" onClick={openHelp}>
              Help
            </button>
            <button type="button" className="btn btn-quiet" onClick={onPause}>
              Save and return later
            </button>
          </div>
        </>
      ) : (
        <Assessment
          tutorial={tutorial}
          state={state}
          session={session}
          onRevisit={go}
          onQuestion={(questionId) => navigate(assessmentPosition(questionId))}
          focusedQuestion={position.question_id}
          outstandingLessons={outstandingLessons}
          draftOf={(item) => drafts.current!.value(item.question_id, serverDrafts?.[item.question_id], emptyResponse(item))}
          conflictOf={(item) => drafts.current!.conflict(item.question_id)}
          onDraft={editDraft}
          onResolve={resolveDraft}
        />
      )}
      {helpOpen && lesson && (
        <Dialog title={`Help · ${lesson.title}`} onClose={() => setHelpOpen(false)} wide>
          <Paragraphs text={tutorial.intro} />
          <ol className="lesson-steps">
            {lesson.steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <p className="meta">Your practice answer and saved progress stay exactly as they are. Close this help to continue.</p>
        </Dialog>
      )}
    </div>
  );
}

function LessonWorkspace({
  tutorial,
  lesson,
  state,
  source,
  record,
  coach,
}: {
  tutorial: TutorialPublic;
  lesson: TutorialLesson;
  state: StudyState;
  source: ReturnType<typeof createResourceSource>;
  record: (signal: LessonSignal | null, practice?: Record<string, unknown>) => void;
  /** The guided task bar, given "Show me" and the lesson's secondary tools. */
  coach: (showMe: (requirement: LessonRequirement) => string | null, tools: React.ReactNode) => React.ReactNode;
}) {
  const saved = state.tutorial_progress?.practice?.[PRACTICE_KEY];
  const [value, setValue] = useState<RankingValue>(() => ({ responseType: saved?.response_type ?? null, ranked: saved?.ranked_candidate_ids ?? [] }));
  const [inspecting, setInspecting] = useState(tutorial.case.candidates[0].candidate_id);
  const [viewed, setViewed] = useState<Set<string>>(() => new Set([tutorial.case.candidates[0].candidate_id]));
  const [tab, setTab] = useState<string | null>(null);
  const [sourceFocus, setSourceFocus] = useState<Focus | null>(null);
  const [targetFocus, setTargetFocus] = useState<Record<string, Focus | null>>({});
  const [feedback, setFeedback] = useState<string | null>(null);
  const tabRef = useRef(tab);
  tabRef.current = tab;
  const root = useRef<HTMLDivElement>(null);
  const candidates = tutorial.case.candidates.map((candidate) => ({
    id: candidate.candidate_id,
    position: candidate.display_position,
    label: candidate.label,
    identifier: curie(candidate.entity.iri),
    score: candidate.score.toFixed(2),
    scoreMeaning: candidate.score_meaning,
  }));
  const controller = useRanking({
    candidates,
    value,
    locked: false,
    onChange: (next, event: RankingEvent, element) => {
      setValue(next);
      setFeedback(null);
      record({ kind: "rank", event, element, detailsOpen: tabRef.current === "evidence" || tabRef.current === "graph" }, { key: PRACTICE_KEY, response_type: next.responseType, ranked_candidate_ids: next.ranked });
    },
  });
  const inspected = tutorial.case.candidates.find((candidate) => candidate.candidate_id === inspecting) ?? tutorial.case.candidates[0];
  const inspect = (id: string) => {
    const returning = viewed.has(id) && id !== inspecting;
    setInspecting(id);
    setViewed((current) => new Set([...current, id]));
    record({ kind: "inspect", position: tutorial.case.candidates.find((candidate) => candidate.candidate_id === id)?.display_position ?? 1, returning });
  };
  const onAction = useCallback(
    (action: WorkspaceAction) => record({ kind: "workspace", action, candidateId: inspected.candidate_id, ...(action.type === "copy_iri" ? { entity: action.side === "source" ? tutorial.case.source : inspected.entity } : {}) }),
    [record, tutorial.case.source, inspected.entity, inspected.candidate_id],
  );

  // "Show me": open the view the step needs, then reveal and focus its control once mounted.
  const [showing, setShowing] = useState<{ requirement: LessonRequirement; nonce: number } | null>(null);
  const showMe = useCallback(
    (requirement: LessonRequirement): string | null => {
      const target = TOUR_TARGETS[requirement.action];
      if (lesson.view === "baseline" && target.tab) return target.missing;
      if (target.tab && tabRef.current !== target.tab) setTab(target.tab);
      setShowing((current) => ({ requirement, nonce: (current?.nonce ?? 0) + 1 }));
      return null;
    },
    [lesson.view],
  );
  const [missing, setMissing] = useState<string | null>(null);
  useEffect(() => {
    if (!showing) return;
    const target = TOUR_TARGETS[showing.requirement.action];
    let frames = 0;
    let handle = 0;
    const attempt = () => {
      const element = findTarget(document, target);
      if (element) {
        setMissing(null);
        highlight(element);
        return;
      }
      if (frames++ < 60) handle = requestAnimationFrame(attempt);
      else setMissing(target.missing);
    };
    handle = requestAnimationFrame(attempt);
    return () => cancelAnimationFrame(handle);
  }, [showing]);

  const check = () => {
    const n = value.ranked.length;
    if (value.responseType === "ranked_candidates" && n >= 1 && n <= 4) {
      record({ kind: "practice_check", partial: true });
      setFeedback(`Checked: ${n} of 5 ranked, the rest left unranked. Practice answers are never scored.`);
    } else if (value.responseType === "ranked_candidates") setFeedback("All five are ranked. For this step, leave at least one unranked, then check again.");
    else if (value.responseType) setFeedback(`“${value.responseType === "none_of_these" ? "None of these" : "Insufficient information"}” is a valid answer. For this step, rank one to four candidates and check it.`);
    else setFeedback("Your practice answer is empty. Add one to four candidates for this step.");
  };
  const question = (names: { source: string | null; target: string | null }) => (
    <div className="pair-header">
      <div className="pair-heading">
        <span className="meta">
          Practice · inspecting initial position {inspected.display_position} of {tutorial.case.candidates.length} · illustrative score {inspected.score.toFixed(2)}
        </span>
        <h2 className="pair-question">
          Does <span className="text-source">{names.source ?? tutorial.case.source_label}</span> mean the same as <span className="text-target">{names.target ?? inspected.label}</span>?
        </h2>
      </div>
    </div>
  );
  const resources = tutorial.case.ontology_resources;
  const workspace =
    lesson.view === "baseline" ? (
      <div className="pair-workspace baseline-workspace">
        {question({ source: tutorial.case.source_label, target: inspected.label })}
        <div className="card-pair">
          <IdentityCard side="source" entity={tutorial.case.source} label={tutorial.case.source_label} ontology={ontologyNames(tutorial)(tutorial.case.source.ontology_version_id)} title="Source concept" onCopied={() => record({ kind: "workspace", action: { type: "copy_iri", side: "source" }, entity: tutorial.case.source })} />
          <IdentityCard side="target" entity={inspected.entity} label={inspected.label} ontology={ontologyNames(tutorial)(inspected.entity.ontology_version_id)} title={`Candidate · initial position ${inspected.display_position}`} onCopied={() => record({ kind: "workspace", action: { type: "copy_iri", side: "target" }, entity: inspected.entity })} />
        </div>
        <section className="study-card baseline-tools">
          <h3>Inspect with your own methods</h3>
          <p>This view shows no explanations. You can inspect the practice ontologies in any way you choose, or not at all.</p>
          <ResourceList resources={resources} compact onDownload={() => record({ kind: "downloads" })} />
        </section>
        <PracticeReports record={record} methods={practiceMethods(state)} />
      </div>
    ) : (
      <WorkspaceProvider source={source} onAction={onAction}>
        <PairWorkspace
          pair={{ source: tutorial.case.source, target: inspected.entity, candidateId: inspected.candidate_id }}
          viewKey={`tutorial|${inspected.candidate_id}`}
          ontologyLabel={ontologyNames(tutorial)}
          tab={tab}
          onTab={setTab}
          navigation={{
            source: sourceFocus,
            target: targetFocus[inspected.candidate_id] ?? null,
            set: (side, focus) => (side === "source" ? setSourceFocus(focus) : setTargetFocus((current) => ({ ...current, [inspected.candidate_id]: focus }))),
          }}
          header={question}
          cardTitles={{ source: "Practice source concept", target: `Practice candidate · initial position ${inspected.display_position}` }}
          compactCards
        />
      </WorkspaceProvider>
    );
  return (
    <div ref={root} className="lesson-workspace">
      <CaseLayout
        answerId="practice-answer"
        answerSummary={answerSummaryOf(value, false)}
        railLabel="Practice candidates and your practice answer"
        toolbar={
          <div className="coach-wrap" data-sticky>
            {coach(showMe, <ResourceAccessButton resources={resources} onOpen={() => record({ kind: "downloads" })} onDownload={() => record({ kind: "downloads" })} />)}
            {missing && (
              <p className="note coach-missing" role="status">
                {missing}
              </p>
            )}
          </div>
        }
        rows={<CandidateRows controller={controller} practice inspecting={inspected.candidate_id} viewed={viewed} onInspect={inspect} compact />}
        answer={
          <AnswerPanel
            id="practice-answer"
            controller={controller}
            practice
            onSubmit={check}
            submitting={false}
            submitLabel="Check practice answer"
            compact
            footer={
              feedback ? (
                <p className="note note-info" role="status">
                  {feedback}
                </p>
              ) : null
            }
          />
        }
        workspace={workspace}
      />
    </div>
  );
}

function PracticeReports({ record, methods }: { record: (signal: LessonSignal | null) => void; methods: [string, string][] }) {
  const tasks = [
    { id: "multi", title: "Practice report 1", situation: "Suppose that for a case you looked the source up in Protégé and also searched the ontology files in a text editor." },
    { id: "none", title: "Practice report 2", situation: "Suppose that for another case you used only the study pages and nothing outside them." },
  ] as const;
  const [answers, setAnswers] = useState<Record<string, { consulted: boolean | null; methods: string[]; other: string }>>({ multi: { consulted: null, methods: [], other: "" }, none: { consulted: null, methods: [], other: "" } });
  const [feedback, setFeedback] = useState<Record<string, string>>({});
  return (
    <section className="study-card practice-reports" aria-labelledby="practice-reports-h">
      <h3 id="practice-reports-h">Practise the question asked after each case</h3>
      <p className="muted">These practice reports are not study answers. After a real case you report what you actually did for that case; you can use different methods, or none, each time.</p>
      {tasks.map((task) => {
        const answer = answers[task.id];
        const set = (changes: Partial<typeof answer>) => {
          setAnswers((current) => ({ ...current, [task.id]: { ...current[task.id], ...changes } }));
          setFeedback((current) => ({ ...current, [task.id]: "" }));
        };
        return (
          <fieldset key={task.id} className="question" data-tour={task.id === "multi" ? "report-multi" : "report-none"}>
            <legend className="question-label">{task.title}</legend>
            <p>{task.situation} Did you inspect ontology information outside the study pages for that case?</p>
            <div className="option-grid">
              {(
                [
                  [true, "Yes"],
                  [false, "No"],
                ] as [boolean, string][]
              ).map(([value, label]) => (
                <label key={label} className={answer.consulted === value ? "option on" : "option"}>
                  <input type="radio" name={`practice-${task.id}`} checked={answer.consulted === value} onChange={() => set({ consulted: value, methods: value ? answer.methods : [] })} />
                  <span>{label}</span>
                </label>
              ))}
            </div>
            {answer.consulted && (
              <div className="option-grid option-grid-column" role="group" aria-label="Methods used">
                {methods.map(([code, label]) => (
                  <label key={code} className={answer.methods.includes(code) ? "option on" : "option"}>
                    <input type="checkbox" checked={answer.methods.includes(code)} onChange={() => set({ methods: answer.methods.includes(code) ? answer.methods.filter((item) => item !== code) : [...answer.methods, code] })} />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            )}
            {answer.consulted && answer.methods.includes("other_method") && (
              <div className="field">
                <label htmlFor={`practice-other-${task.id}`}>Which other methods? <span className="meta">Practice only · short names</span></label>
                <input id={`practice-other-${task.id}`} className="input" maxLength={160} value={answer.other} onChange={(event) => set({ other: event.target.value })} />
              </div>
            )}
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                if (task.id === "multi") {
                  if (answer.consulted && answer.methods.includes("other_method") && !answer.other.trim()) {
                    setFeedback((current) => ({ ...current, multi: "You selected Other method: name it, or clear Other method." }));
                  } else if (answer.consulted && answer.methods.length >= 2) {
                    record({ kind: "report", methods: answer.methods.length, methodCodes: answer.methods, consulted: true, otherMethod: answer.methods.includes("other_method") ? answer.other : null });
                    setFeedback((current) => ({ ...current, multi: "That is how you report two methods for one case: Yes, with every method you used selected." }));
                  } else setFeedback((current) => ({ ...current, multi: "In this situation you used two methods: answer Yes and select both of them." }));
                } else if (answer.consulted === false) {
                  record({ kind: "report", methods: 0, methodCodes: [], consulted: false });
                  setFeedback((current) => ({ ...current, none: "That is how you report a case where you used nothing outside the study pages: No." }));
                } else setFeedback((current) => ({ ...current, none: "In this situation you used nothing outside the study pages: answer No." }));
              }}
            >
              Check practice report
            </button>
            {feedback[task.id] && (
              <p className="note note-info" role="status">
                {feedback[task.id]}
              </p>
            )}
          </fieldset>
        );
      })}
    </section>
  );
}

function Assessment({
  tutorial,
  state,
  session,
  onRevisit,
  onQuestion,
  focusedQuestion,
  outstandingLessons,
  draftOf,
  conflictOf,
  onDraft,
  onResolve,
}: {
  tutorial: TutorialPublic;
  state: StudyState;
  session: StudySession;
  onRevisit: (lessonId: string) => void;
  onQuestion: (questionId: string) => void;
  focusedQuestion: string | null;
  outstandingLessons: TutorialLesson[];
  /** The visible answer for a question (this tab's unsaved answer, else the saved one). */
  draftOf: (item: AssessmentItem) => AssessmentResponse;
  /** This tab's answer kept when another tab or device saved a different one. */
  conflictOf: (item: AssessmentItem) => AssessmentResponse | null;
  onDraft: (questionId: string, response: AssessmentResponse) => void;
  onResolve: (questionId: string, choice: "mine" | "saved") => void;
}) {
  const progress = state.tutorial_progress ?? null;
  const passed = new Set(progress?.passed_items ?? []);
  const [busy, setBusy] = useState(false);
  const outstanding = progress?.outstanding ?? [...outstandingLessons.map((lesson) => lesson.lesson_id), ...tutorial.assessment.filter((item) => !passed.has(item.question_id)).map((item) => item.question_id)];
  // One question at a time (21 F02). The landing position (no question) shows the first one.
  const currentIndex = Math.max(0, tutorial.assessment.findIndex((item) => item.question_id === focusedQuestion));
  const current = tutorial.assessment[currentIndex];
  const finish = async () => {
    setBusy(true);
    try {
      await session.mutate({ method: "POST", path: "/api/v1/study/tutorial/complete", body: { tutorial_version: tutorial.version }, transition: true });
    } catch {
      /* the header explains; nothing is allocated unless the server accepts completion */
    } finally {
      setBusy(false);
    }
  };
  const labelOf = (id: string) => tutorial.lessons.find((lesson) => lesson.lesson_id === id)?.title ?? tutorial.assessment.find((item) => item.question_id === id)?.title ?? id;
  return (
    <>
      <section className="study-card assessment-intro">
        <p>
          These five questions check that the task is clear. They are not a test of medical or ontology knowledge, and you can try each one again as often as you like. Each answer is checked on the study server, which explains it straight away.
        </p>
        <nav aria-label="Questions">
          <ol className="question-nav">
            {tutorial.assessment.map((item, index) => (
              <li key={item.question_id}>
                <button type="button" className={index === currentIndex ? "btn btn-sm current" : "btn btn-sm"} aria-current={item.question_id === focusedQuestion ? "location" : index === currentIndex ? "step" : undefined} onClick={() => onQuestion(item.question_id)}>
                  {passed.has(item.question_id) ? <IconCheck /> : conflictOf(item) ? <IconWarning /> : null}
                  {index + 1} · {item.title}
                  {passed.has(item.question_id) ? <span className="sr-only"> (passed)</span> : conflictOf(item) ? <span className="sr-only"> (choose which answer to keep)</span> : null}
                </button>
              </li>
            ))}
          </ol>
        </nav>
      </section>
      <AssessmentCard key={current.question_id} item={current} number={currentIndex + 1} state={state} session={session} tutorial={tutorial} passed={passed.has(current.question_id)} onRevisit={onRevisit} response={draftOf(current)} conflict={conflictOf(current)} onChange={(next) => onDraft(current.question_id, next)} onResolve={(choice) => onResolve(current.question_id, choice)} />
      <div className="study-actions question-steps">
        <button type="button" className="btn" disabled={currentIndex === 0} onClick={() => onQuestion(tutorial.assessment[currentIndex - 1].question_id)}>
          Previous question
        </button>
        {currentIndex + 1 < tutorial.assessment.length && (
          <button type="button" className={passed.has(current.question_id) ? "btn btn-primary" : "btn"} onClick={() => onQuestion(tutorial.assessment[currentIndex + 1].question_id)}>
            Next question
          </button>
        )}
      </div>
      <section className="study-card" aria-labelledby="finish-h">
        <h2 id="finish-h">Finish the tutorial</h2>
        {outstanding.length ? (
          <>
            <p>Before the scored cases, the study server still needs:</p>
            <ul>
              {outstanding.map((id) => (
                <li key={id}>
                  {tutorial.lessons.some((lesson) => lesson.lesson_id === id) ? (
                    <button type="button" className="btn btn-quiet btn-sm" onClick={() => onRevisit(id)}>
                      Lesson: {labelOf(id)}
                    </button>
                  ) : (
                    <button type="button" className="btn btn-quiet btn-sm" onClick={() => onQuestion(id)}>
                      Question: {labelOf(id)}
                    </button>
                  )}
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p>Everything is done. Your cases are assigned when you start; every participant receives this same tutorial first.</p>
        )}
        <div className="study-actions">
          <button type="button" className="btn btn-primary btn-large" disabled={busy || outstanding.length > 0} onClick={finish}>
            {busy ? "Saving…" : "Finish the tutorial and start the scored cases"}
          </button>
        </div>
      </section>
    </>
  );
}

const PROGRESS_PATH = "/api/v1/study/tutorial/progress";

/** The per-case method list in the publication's own order (study-methods/3), else the legacy list. */
function practiceMethods(state: StudyState): [string, string][] {
  const question = state.forms.consultation.find((item) => item.id === "methods");
  if (isReviewProtocol(state) && question?.options) return orderedEntries(question.options, orderedKeys(question, "options", state.forms.version));
  return METHOD_LABELS;
}

/** Conflicts kept from before a reload in this tab, for this tutorial's questions only. */
function readConflicts(key: string, tutorial: TutorialPublic): KeptConflict[] {
  if (typeof window === "undefined") return [];
  try {
    const stored: unknown = JSON.parse(window.sessionStorage.getItem(key) ?? "[]");
    if (!Array.isArray(stored)) return [];
    const ids = new Set(tutorial.assessment.map((item) => item.question_id));
    return stored.filter(
      (item): item is KeptConflict =>
        Boolean(item) && typeof item.questionId === "string" && ids.has(item.questionId) && Boolean(item.response) && typeof item.response === "object" && !Array.isArray(item.response) && (item.base === null || typeof item.base === "string"),
    );
  } catch {
    return [];
  }
}

/** An answer in words, for the choice offered during a conflict. */
function describeResponse(item: AssessmentItem, response: AssessmentResponse): string {
  const label = (options: { code: string; label: string }[] | undefined, code: string) => options?.find((option) => option.code === code)?.label ?? code;
  if (item.kind === "single") return response.choice ? label(item.options, response.choice) : "No answer";
  if (item.kind === "multiple") return response.choices?.length ? response.choices.map((code) => label(item.options, code)).join(" · ") : "No answer";
  const parts = (item.rows ?? []).filter((row) => response.matches?.[row.row_id]).map((row) => `${row.label}: ${label(item.row_options, response.matches![row.row_id])}`);
  if (item.kind === "match_and_single" && item.part_b && response.part_b) parts.push(`${item.part_b.prompt} ${label(item.part_b.options, response.part_b)}`);
  return parts.length ? parts.join(" · ") : "No answer";
}

function emptyResponse(item: AssessmentItem): AssessmentResponse {
  if (item.kind === "multiple") return { choices: [] };
  if (item.kind === "match" || item.kind === "match_and_single") return { matches: {} };
  return {};
}

function complete(item: AssessmentItem, response: AssessmentResponse): boolean {
  if (item.kind === "single") return Boolean(response.choice);
  if (item.kind === "multiple") return Boolean(response.choices?.length);
  const rows = item.rows ?? [];
  const matched = rows.every((row) => response.matches?.[row.row_id]);
  return item.kind === "match" ? matched : matched && Boolean(response.part_b);
}

function AssessmentCard({
  item,
  number,
  state,
  session,
  tutorial,
  passed,
  onRevisit,
  response,
  conflict,
  onChange,
  onResolve,
}: {
  item: AssessmentItem;
  number: number;
  state: StudyState;
  session: StudySession;
  tutorial: TutorialPublic;
  passed: boolean;
  onRevisit: (lessonId: string) => void;
  /** The visible answer, held above this keyed card so navigation never loses it (S0-R2). */
  response: AssessmentResponse;
  /** This tab's unsaved answer when the saved answer changed elsewhere (S0-FU-R1). */
  conflict: AssessmentResponse | null;
  onChange: (next: AssessmentResponse) => void;
  onResolve: (choice: "mine" | "saved") => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const attempts = (state.tutorial_progress?.attempts ?? []).filter((attempt) => attempt.question_id === item.question_id);
  const latest = attempts.at(-1) ?? null;
  // Changed since the last check: compared with what was checked, so it survives remounts.
  const stale = latest ? canonical(latest.response) !== canonical(response) : false;
  const update = (next: AssessmentResponse) => {
    setError(null);
    onChange(next);
  };
  const errorRef = useRef<HTMLSpanElement>(null);
  const submit = async () => {
    if (!complete(item, response)) {
      setError("Answer every part before checking.");
      requestAnimationFrame(() => errorRef.current?.focus());
      return;
    }
    setBusy(true);
    try {
      // Each check is its own attempt: never coalesced, retried with the same identity.
      await session.mutate({ method: "POST", path: "/api/v1/study/tutorial/assessment", body: { tutorial_version: tutorial.version, question_id: item.question_id, attempt_id: uuid(), response } });
    } catch (failure) {
      setError(describeError(failure));
    } finally {
      setBusy(false);
    }
  };
  const name = `assessment-${item.question_id}`;
  const lesson = tutorial.lessons.find((value) => value.lesson_id === (latest?.revisit_lesson_id ?? item.lesson_id));
  return (
    <section id={name} className={passed ? "study-card assessment-item passed" : "study-card assessment-item"} aria-labelledby={`${name}-h`}>
      <div className="section-head">
        <h2 id={`${name}-h`} tabIndex={-1}>
          {number} · {item.title}
        </h2>
        {passed && (
          <span className="pill pill-ok">
            <IconCheck /> Passed
          </span>
        )}
      </div>
      {conflict && !passed && (
        <div className="note note-warn assessment-conflict" role="status">
          <IconWarning />
          <div>
            <p>
              <strong>Another tab or device saved a different answer to this question.</strong> The saved answer is shown below. This tab’s answer was not saved:
            </p>
            <p className="assessment-conflict-answer">{describeResponse(item, conflict)}</p>
            <div className="study-actions">
              <button type="button" className="btn btn-sm" onClick={() => onResolve("mine")}>
                Use this tab’s answer
              </button>
              <button type="button" className="btn btn-sm" onClick={() => onResolve("saved")}>
                Keep the saved answer
              </button>
            </div>
          </div>
        </div>
      )}
      <fieldset className="assessment-fieldset" disabled={passed || busy}>
        <legend className="question-label">{item.prompt}</legend>
        {item.kind === "single" && (
          <div className="option-grid option-grid-column">
            {item.options?.map((option) => (
              <label key={option.code} className={response.choice === option.code ? "option on" : "option"}>
                <input type="radio" name={name} checked={response.choice === option.code} onChange={() => update({ choice: option.code })} />
                <span>{option.label}</span>
              </label>
            ))}
          </div>
        )}
        {item.kind === "multiple" && (
          <div className="option-grid option-grid-column">
            {item.options?.map((option) => {
              const checked = response.choices?.includes(option.code) ?? false;
              return (
                <label key={option.code} className={checked ? "option on" : "option"}>
                  <input type="checkbox" checked={checked} onChange={() => update({ choices: checked ? (response.choices ?? []).filter((code) => code !== option.code) : [...(response.choices ?? []), option.code] })} />
                  <span>{option.label}</span>
                </label>
              );
            })}
          </div>
        )}
        {(item.kind === "match" || item.kind === "match_and_single") && (
          <div className="match-rows">
            {item.rows?.map((row) => (
              <fieldset key={row.row_id} className="match-row">
                <legend>{row.label}</legend>
                <div className="option-grid">
                  {item.row_options?.map((option) => (
                    <label key={option.code} className={response.matches?.[row.row_id] === option.code ? "option on" : "option"}>
                      <input type="radio" name={`${name}-${row.row_id}`} checked={response.matches?.[row.row_id] === option.code} onChange={() => update({ ...response, matches: { ...(response.matches ?? {}), [row.row_id]: option.code } })} />
                      <span>{option.label}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
          </div>
        )}
        {item.kind === "match_and_single" && item.part_b && (
          <fieldset className="match-row">
            <legend>{item.part_b.prompt}</legend>
            <div className="option-grid">
              {item.part_b.options.map((option) => (
                <label key={option.code} className={response.part_b === option.code ? "option on" : "option"}>
                  <input type="radio" name={`${name}-b`} checked={response.part_b === option.code} onChange={() => update({ ...response, part_b: option.code })} />
                  <span>{option.label}</span>
                </label>
              ))}
            </div>
          </fieldset>
        )}
      </fieldset>
      {!passed && (
        <div className="study-actions">
          <button type="button" className="btn btn-primary" disabled={busy || Boolean(conflict)} onClick={submit}>
            {busy ? "Checking…" : attempts.length ? "Check again" : "Check answer"}
          </button>
          {conflict && <span className="meta">Choose which answer to keep before checking.</span>}
          {error && (
            <span className="field-error" role="alert" tabIndex={-1} ref={errorRef}>
              {error}
            </span>
          )}
        </div>
      )}
      <div aria-live="polite">
        {latest && !stale && (
          <div className={latest.correct ? "note note-ok" : "note note-warn"}>
            <span>{latest.feedback}</span>
            {!latest.correct && lesson && (
              <button type="button" className="btn btn-sm" onClick={() => onRevisit(lesson.lesson_id)}>
                Revisit the lesson “{lesson.title}”
              </button>
            )}
          </div>
        )}
        {latest && stale && !passed && <p className="meta">Your answer changed since the last check. Check it again when you are ready.</p>}
      </div>
    </section>
  );
}

export type { RequirementAction };

/** Re-opens the tutorial's synthetic lesson text during a case. It reads nothing from the
 * server, so it cannot reach another case or unlock baseline explanations; the draft and
 * the case timer continue unchanged. */
export function TutorialHelpButton({ tutorial, onOpen }: { tutorial: TutorialPublic; onOpen?: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        className="btn btn-sm"
        aria-haspopup="dialog"
        onClick={() => {
          setOpen(true);
          onOpen?.();
        }}
      >
        Tutorial help
      </button>
      {open && (
        <Dialog title="Tutorial help" onClose={() => setOpen(false)} wide>
          <p className="muted">These are the practice lessons you completed, with their invented examples. Your answer to this case is unchanged, and the case continues while this is open.</p>
          {tutorial.lessons.map((lesson, index) => (
            <section key={lesson.lesson_id} className="help-lesson">
              <h3>
                {index + 1} · {lesson.title}
              </h3>
              <ol className="lesson-steps">
                {lesson.steps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            </section>
          ))}
        </Dialog>
      )}
    </>
  );
}
