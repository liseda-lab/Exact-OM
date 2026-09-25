"use client";

// First-party participant application. The page GET does nothing server-side; the private
// link is exchanged by POST for an HttpOnly session cookie and removed from the address bar.

import { useCallback, useEffect, useMemo, useState } from "react";

import { Skeleton } from "@/components/common/ErrorNote";
import { CaseView } from "@/components/study/CaseView";
import { ConsultationStage, FormStage, GapDialog, MessagePage, PausedStage, PracticeStage, SetupStage, WelcomeStage } from "@/components/study/Stages";
import { StudyHeader } from "@/components/study/StudyChrome";
import { getJson } from "@/lib/api";
import { useStudySession } from "@/study/session";
import { useTelemetry } from "@/study/telemetry";
import type { StudyCase } from "@/study/types";

const PAUSABLE = new Set(["setup", "background", "practice", "case", "consultation", "final"]);
const SEEN_KEY = "exact.study.lastSeen";

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

/** Saves that can move the participant to another step; timing must be closed first. */
function changesStage(path: string, body: Record<string, unknown>): boolean {
  if (/\/(consent|setup|submit|consultation|pause|complete)$/.test(path)) return true;
  return path.includes("/questionnaires/") && body.submitted === true;
}

export function ParticipantApp() {
  const rawSession = useStudySession();
  const state = rawSession.state;
  const telemetry = useTelemetry(state);
  const session = useMemo(
    () => ({
      ...rawSession,
      mutate: async (mutation: Parameters<typeof rawSession.mutate>[0]) => {
        if (changesStage(mutation.path, mutation.body)) await telemetry.flushTiming();
        return rawSession.mutate(mutation);
      },
    }),
    [rawSession, telemetry],
  );
  const [gapAsked, setGapAsked] = useState(false);
  const [needsGapAnswer, setNeedsGapAnswer] = useState(false);
  const [caseMeta, setCaseMeta] = useState<{ condition: string; sourceLabel: string } | null>(null);

  // On return to an in-progress case after the page was closed, ask what happened.
  useEffect(() => {
    if (!state || gapAsked) return;
    setGapAsked(true);
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
            condition: current.condition === "explanation" ? "With explanation tools" : "With Protégé and the ontology files",
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
    await telemetry.flushTiming();
    await session.mutate({ method: "POST", path: "/api/v1/study/pause", body: {} }).catch(() => undefined);
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

  const header = (
    <StudyHeader
      session={session}
      stage={state?.stage ?? null}
      condition={caseMeta?.condition ?? null}
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
  } else {
    switch (state.stage) {
      case "welcome":
        body = <WelcomeStage state={state} session={session} />;
        break;
      case "setup":
        body = <SetupStage state={state} session={session} />;
        break;
      case "background":
        body = (
          <FormStage
            state={state}
            session={session}
            formId="background"
            title="About your background"
            intro="Broad answers only. “Prefer not to say” is always available and is never read as a lack of experience."
            submitLabel="Continue to practice"
          />
        );
        break;
      case "practice":
        body = <PracticeStage state={state} session={session} />;
        break;
      case "case":
        body = <CaseView key={`${state.current_case_id}|${state.current_presentation_id}`} state={state} session={session} telemetry={telemetry} />;
        break;
      case "consultation":
        body = <ConsultationStage state={state} session={session} sourceLabel={caseMeta?.sourceLabel ?? null} />;
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
              await session.mutate({ method: "POST", path: "/api/v1/study/complete", body: {} });
            }}
          />
        );
        break;
      case "paused":
        body = <PausedStage session={session} onResume={resume} />;
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

