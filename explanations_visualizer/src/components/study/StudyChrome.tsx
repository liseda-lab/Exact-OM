"use client";

import { IconCheck, IconOffline, IconSpinner, IconWarning } from "@/components/common/Icons";
import { TextSizeControl } from "@/components/shell/Preferences";
import type { StudySession } from "@/study/session";
import type { Stage } from "@/study/types";

const STEPS: { label: string; stages: Stage[] }[] = [
  { label: "Welcome", stages: ["welcome"] },
  { label: "Setup", stages: ["setup"] },
  { label: "About you", stages: ["background"] },
  { label: "Practice", stages: ["practice"] },
  { label: "Ranking", stages: ["case", "consultation"] },
  { label: "Feedback", stages: ["final", "completed"] },
];

export function stepIndex(stage: Stage): number {
  return STEPS.findIndex((step) => step.stages.includes(stage));
}

export function SaveIndicator({ session }: { session: StudySession }) {
  const save = session.save;
  if (save.kind === "saving")
    return (
      <span className="save-indicator" role="status">
        <IconSpinner className="icon spin" /> Saving…
      </span>
    );
  if (save.kind === "offline")
    return (
      <span className="save-indicator save-offline" role="status">
        <IconOffline /> Offline · {save.pending} {save.pending === 1 ? "change" : "changes"} waiting
        <button type="button" className="btn btn-sm" onClick={session.retryNow}>
          Retry now
        </button>
      </span>
    );
  if (save.kind === "saved")
    return (
      <span className="save-indicator save-ok" role="status">
        <IconCheck /> Saved
      </span>
    );
  if (!session.online)
    return (
      <span className="save-indicator save-offline" role="status">
        <IconOffline /> Offline
      </span>
    );
  return <span className="save-indicator" role="status" />;
}

export function StudyHeader({
  session,
  stage,
  progress,
  condition,
  onPause,
}: {
  session: StudySession;
  stage: Stage | null;
  progress?: string | null;
  condition?: string | null;
  onPause?: () => void;
}) {
  const index = stage ? stepIndex(stage) : -1;
  return (
    <header className="study-header">
      <div className="study-header-row">
        <span className="study-title">Ontology matching study</span>
        {condition && <span className="pill study-condition">{condition}</span>}
        {progress && <span className="study-progress">{progress}</span>}
        <span className="study-header-spacer" />
        <SaveIndicator session={session} />
        {onPause && (
          <button type="button" className="btn btn-sm" onClick={onPause}>
            Pause
          </button>
        )}
        <TextSizeControl compact />
      </div>
      {index >= 0 && (
        <nav aria-label="Study steps" className="study-steps">
          <ol>
            {STEPS.map((step, position) => (
              <li key={step.label} aria-current={position === index ? "step" : undefined} className={position < index ? "done" : position === index ? "current" : ""}>
                <span className="study-step-bar" aria-hidden="true" />
                <span className="study-step-label">
                  {step.label}
                  {position < index && <span className="sr-only"> (completed)</span>}
                </span>
              </li>
            ))}
          </ol>
        </nav>
      )}
      {session.save.kind === "conflict" && (
        <div className="study-banner banner-warn" role="alert">
          <IconWarning />
          <span>{session.save.message}</span>
          <button type="button" className="btn btn-sm" onClick={session.clearNotice}>
            Dismiss
          </button>
        </div>
      )}
      {session.save.kind === "error" && (
        <div className="study-banner banner-bad" role="alert">
          <IconWarning />
          <span>Your last change was not saved: {session.save.message}</span>
          <button type="button" className="btn btn-sm" onClick={session.clearNotice}>
            Dismiss
          </button>
        </div>
      )}
    </header>
  );
}

export function Paragraphs({ text }: { text: string }) {
  return (
    <>
      {text
        .split(/\n\s*\n/)
        .map((part) => part.trim())
        .filter(Boolean)
        .map((part, index) => (
          <p key={index}>{part}</p>
        ))}
    </>
  );
}
