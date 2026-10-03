"use client";

// exact-study/2.0 interactive tutorial and comprehension check (14, 16 B3). Lessons run in
// the production shared workspace over the publication's synthetic resources; the actions a
// lesson asks for are reported as typed requirement evidence, saved through the outbox and
// shown as saved only once the server acknowledges them. The five items are graded by the
// server, which returns specific feedback; retries are unlimited. Allocation happens only
// when the server accepts completion. Nothing here is scored or touches a scored case.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { IconCheck } from "@/components/common/Icons";
import { CaseLayout, answerSummaryOf } from "@/components/study/CaseLayout";
import { IdentityCard } from "@/components/study/IdentityCard";
import { AnswerPanel, CandidateRows, useRanking, type RankingEvent, type RankingValue } from "@/components/study/RankingPanel";
import { ResourceAccessButton, ResourceList } from "@/components/study/Resources";
import { METHOD_LABELS } from "@/components/study/SetupV2";
import { MessagePage } from "@/components/study/Stages";
import { Paragraphs } from "@/components/study/StudyChrome";
import { PairWorkspace, type Focus } from "@/components/workspace/PairWorkspace";
import { describeError, getJson } from "@/lib/api";
import { curie } from "@/lib/iri";
import { indexResources } from "@/lib/workspace/resourceIndex";
import { createResourceSource } from "@/lib/workspace/resourceSource";
import type { WorkspaceAction } from "@/lib/workspace/types";
import { WorkspaceProvider } from "@/lib/workspace/WorkspaceContext";
import { uuid, type StudySession } from "@/study/session";
import { requirementsFor, type LessonSignal } from "@/study/tutorialProgress";
import type { AssessmentItem, AssessmentResponse, ExplanationResource, RequirementAction, StudyState, TutorialLesson, TutorialPublic } from "@/study/types";

const PRACTICE_KEY = "practice-case";

function ontologyNames(tutorial: TutorialPublic) {
  return (id: string) => tutorial.case.ontology_resources.find((asset) => asset.ontology_version_id === id)?.title ?? (id === tutorial.case.source.ontology_version_id ? "Practice source ontology" : "Practice target ontology");
}

export function TutorialStage({ state, session, onPause }: { state: StudyState; session: StudySession; onPause: () => void }) {
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
  return <Tutorial state={state} session={session} tutorial={tutorial} onPause={onPause} />;
}

function Tutorial({ state, session, tutorial, onPause }: { state: StudyState; session: StudySession; tutorial: TutorialPublic; onPause: () => void }) {
  const progress = state.tutorial_progress ?? null;
  const [resources, setResources] = useState<ExplanationResource[] | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [reload, setReload] = useState(0);
  const [view, setView] = useState<string>(progress?.current_lesson_id ?? tutorial.lessons[0]?.lesson_id ?? "assessment");
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [helpOpen, setHelpOpen] = useState(false);
  const acknowledged = useMemo(() => new Set(progress?.completed_requirements ?? []), [progress?.completed_requirements]);
  const completed = useMemo(() => new Set([...acknowledged, ...pending]), [acknowledged, pending]);

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

  const save = useCallback(
    (body: Record<string, unknown>, coalesce: string) =>
      session.mutate({ method: "PUT", path: "/api/v1/study/tutorial/progress", body: { tutorial_version: tutorial.version, current_lesson_id: view === "assessment" ? null : view, ...body }, coalesce, debounceMs: 400 }).catch(() => undefined),
    [session, tutorial.version, view],
  );

  // Clear local pending marks once the server has acknowledged them.
  useEffect(() => {
    setPending((current) => {
      const next = new Set([...current].filter((id) => !acknowledged.has(id)));
      return next.size === current.size ? current : next;
    });
  }, [acknowledged]);

  const lesson = tutorial.lessons.find((item) => item.lesson_id === view) ?? null;
  // One cumulative draft per lesson (requirements so far plus the practice answer), so
  // coalescing replaces only an older snapshot of the same lesson and never drops evidence.
  const practiceRef = useRef<Record<string, unknown> | null>(null);
  const record = useCallback(
    (signal: LessonSignal | null, practice?: Record<string, unknown>) => {
      if (!lesson) return;
      if (practice) practiceRef.current = practice;
      const satisfied = signal ? requirementsFor(lesson, signal).filter((id) => !completed.has(id)) : [];
      if (!satisfied.length && !practice) return;
      const all = Array.from(new Set([...completed, ...satisfied]));
      if (satisfied.length) setPending((current) => new Set([...current, ...satisfied]));
      void save({ lesson_id: lesson.lesson_id, completed_requirements: all, ...(practiceRef.current ? { practice: practiceRef.current } : {}) }, `tutorial:lesson:${lesson.lesson_id}`);
    },
    [lesson, completed, save],
  );

  const go = (next: string) => {
    setView(next);
    window.scrollTo({ top: 0 });
    void save({ completed_requirements: Array.from(completed), lesson_id: next === "assessment" ? null : next }, "tutorial:position");
  };

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
  return (
    <div className="study-page study-page-wide tutorial-page">
      <div className="study-intro">
        <span className="pill pill-warn">Practice · not scored · synthetic</span>
        <h1>{lesson ? `Lesson ${index + 1} of ${tutorial.lessons.length}: ${lesson.title}` : "Check your understanding"}</h1>
        {index <= 0 && view !== "assessment" && <p className="lead">{tutorial.intro}</p>}
      </div>
      <nav className="lesson-nav" aria-label="Tutorial lessons">
        <ol>
          {tutorial.lessons.map((item, position) => {
            const finished = item.requirements.every((requirement) => acknowledged.has(requirement.requirement_id));
            return (
              <li key={item.lesson_id}>
                <button type="button" className={item.lesson_id === view ? "lesson-link current" : "lesson-link"} aria-current={item.lesson_id === view ? "step" : undefined} onClick={() => go(item.lesson_id)}>
                  <span className="lesson-mark" aria-hidden="true">{finished ? <IconCheck /> : position + 1}</span>
                  <span>
                    {item.title}
                    {item.optional ? <span className="meta"> · optional</span> : null}
                    <span className="sr-only">{finished ? " (done)" : " (not finished)"}</span>
                  </span>
                </button>
              </li>
            );
          })}
          <li>
            <button type="button" className={view === "assessment" ? "lesson-link current" : "lesson-link"} aria-current={view === "assessment" ? "step" : undefined} onClick={() => go("assessment")}>
              <span className="lesson-mark" aria-hidden="true">{(progress?.passed_items.length ?? 0) === tutorial.assessment.length ? <IconCheck /> : "?"}</span>
              <span>Five short questions</span>
            </button>
          </li>
        </ol>
      </nav>

      {lesson ? (
        <>
          <section className="study-card lesson-card" aria-labelledby="lesson-steps-h">
            <h2 id="lesson-steps-h" className="sr-only">
              What to do
            </h2>
            <ol className="lesson-steps">
              {lesson.steps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
            <ul className="lesson-checklist" aria-label="Steps in this lesson">
              {lesson.requirements.map((requirement) => {
                const saved = acknowledged.has(requirement.requirement_id);
                const local = pending.has(requirement.requirement_id);
                return (
                  <li key={requirement.requirement_id} className={saved ? "done" : local ? "saving" : ""}>
                    <span className="check-mark" aria-hidden="true">{saved || local ? <IconCheck /> : null}</span>
                    <span>{requirement.label}</span>
                    <span className="meta">{saved ? "Done · saved" : local ? "Done · saving" : "To do"}</span>
                  </li>
                );
              })}
            </ul>
          </section>
          <LessonWorkspace key={lesson.lesson_id} tutorial={tutorial} lesson={lesson} state={state} source={source} record={record} />
          <div className="study-actions lesson-actions">
            <button type="button" className="btn" disabled={index <= 0} onClick={() => go(tutorial.lessons[index - 1].lesson_id)}>
              Back
            </button>
            <button type="button" className="btn btn-primary" onClick={() => go(tutorial.lessons[index + 1]?.lesson_id ?? "assessment")}>
              {index + 1 < tutorial.lessons.length ? "Next lesson" : "Go to the five questions"}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                setHelpOpen(true);
                void save({ completed_requirements: Array.from(completed), help_opened: true }, "tutorial:help");
              }}
            >
              Help
            </button>
            <button type="button" className="btn btn-quiet" onClick={onPause}>
              Save and return later
            </button>
            {lesson.requirements.some((requirement) => !completed.has(requirement.requirement_id)) && <span className="meta">You can move on and come back to the remaining steps later.</span>}
          </div>
        </>
      ) : (
        <Assessment tutorial={tutorial} state={state} session={session} onRevisit={go} outstandingLessons={outstandingLessons} />
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
}: {
  tutorial: TutorialPublic;
  lesson: TutorialLesson;
  state: StudyState;
  source: ReturnType<typeof createResourceSource>;
  record: (signal: LessonSignal | null, practice?: Record<string, unknown>) => void;
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
  const onAction = useCallback((action: WorkspaceAction) => record({ kind: "workspace", action }), [record]);
  const check = () => {
    const n = value.ranked.length;
    if (value.responseType === "ranked_candidates" && n >= 1 && n <= 4) {
      record({ kind: "practice_check", partial: true });
      setFeedback(`Practice answer checked: ${n} of 5 ranked, the rest left unranked. This checks the action only; practice answers are never scored.`);
    } else if (value.responseType === "ranked_candidates") setFeedback("All five are ranked. For this step, leave at least one candidate unranked, then check again.");
    else if (value.responseType) setFeedback(`“${value.responseType === "none_of_these" ? "None of these" : "Insufficient information"}” is a valid answer. For this step, make a partial ranking of one to four candidates and check it.`);
    else setFeedback("Your practice answer is empty. An empty answer is never submitted; add one to four candidates for this step.");
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
          <IdentityCard side="source" entity={tutorial.case.source} label={tutorial.case.source_label} ontology={ontologyNames(tutorial)(tutorial.case.source.ontology_version_id)} title="Source concept" onCopied={() => record({ kind: "workspace", action: { type: "copy_iri", side: "source" } })} />
          <IdentityCard side="target" entity={inspected.entity} label={inspected.label} ontology={ontologyNames(tutorial)(inspected.entity.ontology_version_id)} title={`Candidate · initial position ${inspected.display_position}`} onCopied={() => record({ kind: "workspace", action: { type: "copy_iri", side: "target" } })} />
        </div>
        <section className="study-card baseline-tools">
          <h3>Inspect with your own methods</h3>
          <p>This view shows no explanations. You can inspect the practice ontologies in any way you choose, or not at all.</p>
          <ResourceList resources={resources} compact onDownload={() => record({ kind: "downloads" })} />
        </section>
        <PracticeReports record={record} />
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
    <CaseLayout
      answerId="practice-answer"
      answerSummary={answerSummaryOf(value, false)}
      railLabel="Practice candidates and your practice answer"
      toolbar={
        <div className="case-toolbar">
          <p className="note synthetic-banner">Synthetic practice: every name, fact, score and description here is invented. No matcher or model was run.</p>
          <ResourceAccessButton resources={resources} onOpen={() => record({ kind: "downloads" })} onDownload={() => record({ kind: "downloads" })} />
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
  );
}

function PracticeReports({ record }: { record: (signal: LessonSignal | null) => void }) {
  const tasks = [
    { id: "multi", title: "Practice report 1", situation: "Suppose that for a case you looked the source up in Protégé and also searched the ontology files in a text editor." },
    { id: "none", title: "Practice report 2", situation: "Suppose that for another case you used only the study pages and nothing outside them." },
  ] as const;
  const [answers, setAnswers] = useState<Record<string, { consulted: boolean | null; methods: string[] }>>({ multi: { consulted: null, methods: [] }, none: { consulted: null, methods: [] } });
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
          <fieldset key={task.id} className="question">
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
                {METHOD_LABELS.map(([code, label]) => (
                  <label key={code} className={answer.methods.includes(code) ? "option on" : "option"}>
                    <input type="checkbox" checked={answer.methods.includes(code)} onChange={() => set({ methods: answer.methods.includes(code) ? answer.methods.filter((item) => item !== code) : [...answer.methods, code] })} />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            )}
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                if (task.id === "multi") {
                  if (answer.consulted && answer.methods.length >= 2) {
                    record({ kind: "report", methods: answer.methods.length, consulted: true });
                    setFeedback((current) => ({ ...current, multi: "That is how you report two methods for one case: Yes, with every method you used selected." }));
                  } else setFeedback((current) => ({ ...current, multi: "In this situation you used two methods: answer Yes and select both of them." }));
                } else if (answer.consulted === false) {
                  record({ kind: "report", methods: 0, consulted: false });
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
  outstandingLessons,
}: {
  tutorial: TutorialPublic;
  state: StudyState;
  session: StudySession;
  onRevisit: (lessonId: string) => void;
  outstandingLessons: TutorialLesson[];
}) {
  const progress = state.tutorial_progress ?? null;
  const passed = new Set(progress?.passed_items ?? []);
  const [busy, setBusy] = useState(false);
  const outstanding = progress?.outstanding ?? [...outstandingLessons.map((lesson) => lesson.lesson_id), ...tutorial.assessment.filter((item) => !passed.has(item.question_id)).map((item) => item.question_id)];
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
      <section className="study-card">
        <p>
          These five questions check that the task is clear. They are not a test of medical or ontology knowledge, and you can try each one again as often as you like. Each answer is checked on the study server, which explains it straight away.
        </p>
      </section>
      {tutorial.assessment.map((item, index) => (
        <AssessmentCard key={item.question_id} item={item} number={index + 1} state={state} session={session} tutorial={tutorial} passed={passed.has(item.question_id)} onRevisit={onRevisit} />
      ))}
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
                    <span>Question: {labelOf(id)}</span>
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
}: {
  item: AssessmentItem;
  number: number;
  state: StudyState;
  session: StudySession;
  tutorial: TutorialPublic;
  passed: boolean;
  onRevisit: (lessonId: string) => void;
}) {
  const draft = state.tutorial_progress?.assessment_drafts?.[item.question_id];
  const [response, setResponse] = useState<AssessmentResponse>(() => draft ?? emptyResponse(item));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const attempts = (state.tutorial_progress?.attempts ?? []).filter((attempt) => attempt.question_id === item.question_id);
  const latest = attempts.at(-1) ?? null;
  const [stale, setStale] = useState(false);
  const update = (next: AssessmentResponse) => {
    setResponse(next);
    setStale(true);
    setError(null);
    void session
      .mutate({ method: "PUT", path: "/api/v1/study/tutorial/progress", body: { tutorial_version: tutorial.version, assessment_draft: { question_id: item.question_id, response: next } }, coalesce: `tutorial:item:${item.question_id}`, debounceMs: 500 })
      .catch(() => undefined);
  };
  const submit = async () => {
    if (!complete(item, response)) {
      setError("Answer every part before checking.");
      return;
    }
    setBusy(true);
    try {
      // Each check is its own attempt: never coalesced, retried with the same identity.
      await session.mutate({ method: "POST", path: "/api/v1/study/tutorial/assessment", body: { tutorial_version: tutorial.version, question_id: item.question_id, attempt_id: uuid(), response } });
      setStale(false);
    } catch (failure) {
      setError(describeError(failure));
    } finally {
      setBusy(false);
    }
  };
  const name = `assessment-${item.question_id}`;
  const lesson = tutorial.lessons.find((value) => value.lesson_id === (latest?.revisit_lesson_id ?? item.lesson_id));
  return (
    <section className={passed ? "study-card assessment-item passed" : "study-card assessment-item"} aria-labelledby={`${name}-h`}>
      <div className="section-head">
        <h2 id={`${name}-h`}>
          {number} · {item.title}
        </h2>
        {passed && (
          <span className="pill pill-ok">
            <IconCheck /> Passed
          </span>
        )}
      </div>
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
          <button type="button" className="btn btn-primary" disabled={busy} onClick={submit}>
            {busy ? "Checking…" : attempts.length ? "Check again" : "Check answer"}
          </button>
          {error && (
            <span className="field-error" role="alert">
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
