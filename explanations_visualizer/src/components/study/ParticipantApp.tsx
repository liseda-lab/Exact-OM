"use client";

// First-party participant application. The page GET does nothing server-side; the private
// link is exchanged by POST for an HttpOnly session cookie and removed from the address bar.
// The study contract version selects the flow: exact-study/1.0 sessions keep their frozen
// legacy steps, exact-study/2.0 uses tool-neutral setup, the interactive tutorial and
// durable per-case reports. Any other version fails visibly before collecting answers.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Skeleton } from "@/components/common/ErrorNote";
import { CaseView } from "@/components/study/CaseView";
import { ConsultationStageV2 } from "@/components/study/ConsultationV2";
import { LegacyConsultationStage, LegacyPracticeStage, LegacySetupStage } from "@/components/study/LegacyStages";
import { SetupStageV2 } from "@/components/study/SetupV2";
import { FormStage, GapDialog, MessagePage, PausedStage, WelcomeStage, type Protocol } from "@/components/study/Stages";
import { StudyHeader } from "@/components/study/StudyChrome";
import { TutorialStage } from "@/components/study/Tutorial";
import { getJson } from "@/lib/api";
import type { Mutation } from "@/study/mutationQueue";
import { useStudySession } from "@/study/session";
import { useTelemetry } from "@/study/telemetry";
import type { StudyCase, StudyState } from "@/study/types";

const PAUSABLE = new Set(["setup", "background", "practice", "tutorial", "case", "consultation", "final"]);
const SEEN_KEY = "exact.study.lastSeen";
const CONTRACTS: Record<string, Protocol> = { "exact-study/1.0": "v1", "exact-study/2.0": "v2" };

export function protocolOf(state: StudyState): Protocol | null {
  return CONTRACTS[state.contract_version ?? "exact-study/1.0"] ?? null;
}

function lastSeen(sessionId: string): number | null {
  try {
    const raw = window.localStorage.getItem(`${SEEN_KEY}.${sessionId}`);
    return raw ? Number(raw) : null;
  } catch {
    return null;
  }
}

function markSeen(sessionId: string) {
  try {
    window.localStorage.setItem(`${SEEN_KEY}.${sessionId}`, String(Date.now()));
  } catch {
    /* only used to ask a better question on return */
  }
}

export function ParticipantApp() {
  const rawSession = useStudySession();
  const state = rawSession.state;
  const telemetry = useTelemetry(rawSession.phase === "ready" ? state : null);
  const transitioning = useRef(false);
  // Only operations declared as transitions (consent, setup submit, tutorial completion,
  // questionnaire submit, ranking submit, final consultation save, pause, completion) close
  // timing first. Drafts, progress saves and individual assessment attempts never do.
  const session = useMemo(
    () => ({
      ...rawSession,
      mutate: async (mutation: Mutation) => {
        const transition = Boolean(mutation.transition);
        if (transition && transitioning.current) throw new Error("A step is already being saved.");
        if (transition) transitioning.current = true;
        try {
          if (transition) await telemetry.flushTiming();
          const next = await rawSession.mutate(mutation);
          if (transition && next.stage === rawSession.state?.stage) telemetry.resumeTiming();
          return next;
        } catch (error) {
          if (transition) telemetry.resumeTiming();
          throw error;
        } finally {
          if (transition) transitioning.current = false;
        }
      },
    }),
    [rawSession, telemetry],
  );
  const [gapAsked, setGapAsked] = useState<string | null>(null);
  const [needsGapAnswer, setNeedsGapAnswer] = useState(false);
  const [caseMeta, setCaseMeta] = useState<{ condition: string; sourceLabel: string } | null>(null);

  // On return to an in-progress case after the page was closed, ask what happened.
  useEffect(() => {
    if (!state || gapAsked === state.session_id) return;
    setGapAsked(state.session_id);
    setNeedsGapAnswer(false);
    if (state.stage === "case" || state.stage === "consultation") {
      const seen = lastSeen(state.session_id);
      if (seen === null || Date.now() - seen > 120_000) setNeedsGapAnswer(true);
    }
  }, [state, gapAsked]);

  useEffect(() => {
    if (!state) return;
    markSeen(state.session_id);
    const timer = window.setInterval(() => markSeen(state.session_id), 20_000);
    return () => window.clearInterval(timer);
  }, [state]);

  // Condition label for the header comes from the current case, never from client state.
  const caseId = state?.current_case_id;
  const stage = state?.stage;
  useEffect(() => {
    if (!caseId || !(stage === "case" || stage === "consultation")) {
      setCaseMeta(null);
      return;
    }
    let cancelled = false;
    getJson<StudyCase>("/api/v1/study/cases/current")
      .then((current) => {
        if (!cancelled)
          setCaseMeta({
            condition: current.condition === "explanation" ? "With explanation tools" : "With your own inspection methods",
            sourceLabel: current.source_label,
          });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [caseId, stage]);

  const resume = useCallback(
    async (activity: "external_work" | "break" | "unknown") => {
      telemetry.emit("resume");
      await session.mutate({ method: "POST", path: "/api/v1/study/resume", body: { gap_activity: activity } }).catch(() => undefined);
    },
    [session, telemetry],
  );

  const pause = useCallback(async () => {
    telemetry.emit("pause");
    await session.mutate({ method: "POST", path: "/api/v1/study/pause", body: {}, transition: true }).catch(() => undefined);
  }, [session, telemetry]);

  if (session.phase === "booting") {
    return (
      <div className="study-root">
        <main className="study-page" aria-busy="true">
          <Skeleton lines={4} />
        </main>
      </div>
    );
  }

  const protocol = state ? protocolOf(state) : null;
  const header = (
    <StudyHeader
      session={session}
      stage={state?.stage ?? null}
      condition={caseMeta?.condition ?? null}
      tutorialLabel={protocol === "v2" ? "Tutorial" : "Practice"}
      progress={
        state && (state.stage === "case" || state.stage === "consultation") && state.assigned_case_count
          ? `Case ${Math.min(state.completed_cases + 1, state.assigned_case_count)} of ${state.assigned_case_count}`
          : null
      }
      onPause={state && PAUSABLE.has(state.stage) ? pause : undefined}
    />
  );

  let body: React.ReactNode;
  if (session.phase === "no_link") {
    body = (
      <MessagePage title="Open your private link">
        <p>This page needs the private link you were given for the study. Open that link in this browser to start or continue where you left off.</p>
        <p className="muted">If you have lost the link and this browser no longer has your session, it cannot be recovered: we do not store names or email addresses.</p>
      </MessagePage>
    );
  } else if (session.phase === "link_unusable") {
    body = (
      <MessagePage title="This link cannot be used">
        <p>It may have been closed or replaced by the research team, or the study may have ended. No new session was started.</p>
        <p className="muted">If you think this is a mistake, contact the study team.</p>
      </MessagePage>
    );
  } else if (session.phase === "rate_limited") {
    body = (
      <MessagePage title="Please wait a moment">
        <p>Too many attempts were made from here. Wait a minute, then open your private link again.</p>
      </MessagePage>
    );
  } else if (session.phase === "unreachable" || !state) {
    body = (
      <MessagePage title="The study service is not responding">
        <p>Your saved answers are safe on the study server. Check your connection and reload this page.</p>
      </MessagePage>
    );
  } else if (!protocol) {
    body = (
      <MessagePage title="This study version is not supported here">
        <p>This page cannot show study version {state.contract_version}, so it does not collect answers. Nothing you saved earlier is lost.</p>
        <p className="muted">Please tell the study team; the study may need a matching version of this page.</p>
      </MessagePage>
    );
  } else {
    switch (state.stage) {
      case "welcome":
        body = <WelcomeStage state={state} session={session} protocol={protocol} />;
        break;
      case "setup":
        body = protocol === "v2" ? <SetupStageV2 key={state.session_id} state={state} session={session} /> : <LegacySetupStage key={state.session_id} state={state} session={session} />;
        break;
      case "background":
        body = (
          <FormStage
            state={state}
            session={session}
            formId="background"
            title="About your background"
            intro="Broad answers only. “Prefer not to say” is always available and is never read as a lack of experience."
            submitLabel={protocol === "v2" ? "Continue to the tutorial" : "Continue to practice"}
          />
        );
        break;
      case "practice":
      case "tutorial":
        body = protocol === "v2" ? <TutorialStage key={state.session_id} state={state} session={session} onPause={pause} /> : <LegacyPracticeStage state={state} session={session} />;
        break;
      case "case":
        body = <CaseView key={`${state.session_id}|${state.current_case_id}|${state.current_presentation_id}`} state={state} session={session} telemetry={telemetry} timingEnabled={!needsGapAnswer} />;
        break;
      case "consultation":
        body =
          protocol === "v2" ? (
            <ConsultationStageV2 key={`${state.session_id}|${state.current_presentation_id}`} state={state} session={session} sourceLabel={caseMeta?.sourceLabel ?? null} />
          ) : (
            <LegacyConsultationStage key={`${state.session_id}|${state.current_presentation_id}`} state={state} session={session} sourceLabel={caseMeta?.sourceLabel ?? null} />
          );
        break;
      case "final":
        body = (
          <FormStage
            state={state}
            session={session}
            formId="final"
            title="What helped, and what was hard?"
            intro="All cases are done. These answers describe your experience; they are not used to score your rankings."
            submitLabel="Finish the study"
            onSubmitted={async () => {
              await session.mutate({ method: "POST", path: "/api/v1/study/complete", body: {}, transition: true });
            }}
          />
        );
        break;
      case "paused":
        body = <PausedStage onResume={resume} />;
        break;
      case "completed":
        body = (
          <MessagePage title="Thank you. Your responses are recorded.">
            <p>All your answers were saved on the study server. Opening your private link again brings you back to this page; nothing more is needed from you.</p>
            <p className="muted">Correct answers are not shown while the study is running, so that every participant sees the cases the same way.</p>
          </MessagePage>
        );
        break;
      case "closed":
        body = (
          <MessagePage title="Participation is closed">
            <p>{state.consent && !state.consent.accepted ? "You chose not to take part. Nothing further is collected." : "This study session is closed."}</p>
          </MessagePage>
        );
        break;
      default:
        body = null;
    }
  }

  return (
    <div className="study-root">
      {header}
      <main id="main">{body}</main>
      {needsGapAnswer && state && (
        <GapDialog
          onAnswer={async (activity) => {
            setNeedsGapAnswer(false);
            await resume(activity);
          }}
        />
      )}
      {state?.synthetic && <p className="synthetic-note">Test session: responses are excluded from analysis.</p>}
    </div>
  );
}
