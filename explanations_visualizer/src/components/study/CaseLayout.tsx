"use client";

// Case layout shared by scored cases and the tutorial. Wide: a bounded, sticky rail with the
// candidates and the answer beside one primary reading column. Narrow or enlarged text:
// candidates first, then the workspace, then the answer, with a sticky bar that always shows
// the answer status and jumps to it. The workspace keeps its tree position in both
// arrangements, so switching layout never resets inspection or the graph.

import { useCallback } from "react";

import { useNarrow } from "@/lib/useMedia";

export function CaseLayout({
  toolbar,
  status,
  rows,
  answer,
  workspace,
  answerId,
  answerSummary,
  railLabel = "Candidates and your answer",
}: {
  toolbar?: React.ReactNode;
  /** Readiness or service status for the whole case, shown above both columns. */
  status?: React.ReactNode;
  rows: React.ReactNode;
  answer: React.ReactNode;
  workspace: React.ReactNode;
  answerId: string;
  answerSummary: string;
  railLabel?: string;
}) {
  const narrow = useNarrow(64);
  const jump = useCallback(() => {
    const node = document.getElementById(answerId);
    node?.scrollIntoView({ block: "start" });
    node?.focus({ preventScroll: true });
  }, [answerId]);
  return (
    <div className={narrow ? "case-layout narrow" : "case-layout wide"}>
      {toolbar}
      {status}
      <div className="case-columns">
        <aside className="case-rail" aria-label={railLabel} hidden={narrow}>
          {!narrow && (
            <div className="answer-peek">
              <span aria-live="polite">
                Your answer: <strong>{answerSummary}</strong>
              </span>
              <button type="button" className="btn btn-quiet btn-sm" onClick={jump}>
                Go to your answer
              </button>
            </div>
          )}
          {!narrow && rows}
          {!narrow && answer}
        </aside>
        <div className="case-main">
          {narrow && rows}
          {workspace}
          {narrow && answer}
        </div>
      </div>
      {narrow && (
        <div className="answer-bar" role="region" aria-label="Answer status">
          <span className="answer-bar-text" aria-live="polite">
            Your answer: {answerSummary}
          </span>
          <button type="button" className="btn btn-sm" onClick={jump}>
            Go to your answer
          </button>
        </div>
      )}
    </div>
  );
}

export function answerSummaryOf(value: { responseType: string | null; ranked: string[] }, locked: boolean): string {
  if (locked) return "submitted";
  if (value.responseType === "ranked_candidates") return `${value.ranked.length} ranked`;
  if (value.responseType === "none_of_these") return "None of these";
  if (value.responseType === "insufficient_evidence") return "Insufficient information";
  return "empty";
}
