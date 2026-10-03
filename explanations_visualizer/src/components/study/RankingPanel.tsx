"use client";

// One ranking component for both study conditions, the tutorial and practice. The initial
// order stays visible and never changes; the answer starts empty. Inspecting a candidate is
// never ranking it. Partial rankings, explicit "None of these" and "Insufficient
// information" are first-class; nothing auto-submits. The controller is separate from the
// two regions so a layout can place the candidate list and the answer apart without losing
// the undo history.

import { useCallback, useEffect, useRef, useState } from "react";

import { IconArrowDown, IconArrowUp, IconCheck, IconUndo } from "@/components/common/Icons";
import type { EventType, ResponseType } from "@/study/types";

export interface RankingValue {
  responseType: ResponseType | null;
  ranked: string[];
}

export interface RankingCandidate {
  id: string;
  position: number;
  label: string;
  identifier: string;
  score: string;
  scoreMeaning: string;
}

export type RankingEvent = EventType;

export interface RankingController {
  candidates: RankingCandidate[];
  value: RankingValue;
  locked: boolean;
  controlsLocked: boolean;
  canUndo: boolean;
  add: (id: string) => void;
  remove: (index: number) => void;
  move: (from: number, to: number) => void;
  keepInitial: () => void;
  explicit: (type: "none_of_these" | "insufficient_evidence") => void;
  undo: () => void;
}

export function useRanking({
  candidates,
  value,
  onChange,
  locked,
  disabled = false,
  submitting = false,
  resetKey,
}: {
  candidates: RankingCandidate[];
  value: RankingValue;
  onChange: (next: RankingValue, event: RankingEvent, element?: string) => void;
  locked: boolean;
  disabled?: boolean;
  submitting?: boolean;
  resetKey?: string | number;
}): RankingController {
  const history = useRef<RankingValue[]>([]);
  const [, force] = useState(0);
  const controlsLocked = locked || disabled || submitting;
  useEffect(() => {
    history.current = [];
    force((n) => n + 1);
  }, [resetKey]);

  const apply = useCallback(
    (next: RankingValue, event: RankingEvent, element?: string) => {
      if (controlsLocked) return;
      history.current = [...history.current, value].slice(-50);
      force((n) => n + 1);
      onChange(next, event, element);
    },
    [controlsLocked, onChange, value],
  );
  return {
    candidates,
    value,
    locked,
    controlsLocked,
    canUndo: history.current.length > 0,
    add: (id) => apply({ responseType: "ranked_candidates", ranked: [...value.ranked.filter((item) => item !== id), id] }, "rank_add", id),
    remove: (index) => {
      const ranked = value.ranked.filter((_, position) => position !== index);
      apply({ responseType: ranked.length ? "ranked_candidates" : null, ranked }, "rank_remove", value.ranked[index]);
    },
    move: (from, to) => {
      if (to < 0 || to >= value.ranked.length || from === to) return;
      const ranked = [...value.ranked];
      const [item] = ranked.splice(from, 1);
      ranked.splice(to, 0, item);
      apply({ responseType: "ranked_candidates", ranked }, "rank_move", item);
    },
    keepInitial: () =>
      apply({ responseType: "ranked_candidates", ranked: [...candidates].sort((a, b) => a.position - b.position).map((candidate) => candidate.id) }, "keep_initial_order"),
    explicit: (type) => apply({ responseType: value.responseType === type ? null : type, ranked: [] }, "response_type_change", type),
    undo: () => {
      if (controlsLocked) return;
      const previous = history.current.pop();
      if (!previous) return;
      force((n) => n + 1);
      onChange(previous, "revision", "undo");
    },
  };
}

/** The frozen initial order, inspection state and the participant's rank per candidate. */
export function CandidateRows({
  controller,
  practice = false,
  inspecting,
  viewed,
  onInspect,
  rowExtra,
  compact = false,
}: {
  controller: RankingController;
  practice?: boolean;
  inspecting?: string | null;
  viewed?: Set<string>;
  onInspect?: (id: string) => void;
  rowExtra?: (candidate: RankingCandidate) => React.ReactNode;
  compact?: boolean;
}) {
  const { candidates, value, controlsLocked } = controller;
  const n = value.ranked.length;
  return (
    <section className={compact ? "ranking-initial ranking-section compact" : "card ranking-initial ranking-section"} aria-labelledby="initial-h">
      <div className="ranking-head">
        <h2 id="initial-h">Candidates</h2>
        <span className="meta">{practice ? "Illustrative practice order" : "Initial order: the system’s suggestion"}</span>
      </div>
      <ol className="initial-list">
        {[...candidates]
          .sort((a, b) => a.position - b.position)
          .map((candidate) => {
            const rank = value.ranked.indexOf(candidate.id);
            const isInspecting = inspecting === candidate.id;
            const seen = viewed?.has(candidate.id);
            return (
              <li key={candidate.id} className={isInspecting ? "initial-row inspecting" : "initial-row"}>
                <span className="position-badge" aria-label={`Initial position ${candidate.position}`}>
                  {candidate.position}
                </span>
                {onInspect ? (
                  <button type="button" className="initial-text inspect-target" aria-pressed={isInspecting} aria-label={`Inspect ${candidate.label}`} aria-describedby={`inspect-state-${candidate.id}`} onClick={() => onInspect(candidate.id)}>
                    <span className="initial-label">{candidate.label}</span>
                    <span className="meta">
                      <span className="iri">{candidate.identifier}</span> · {practice ? "illustrative score" : "matching score"} {candidate.score}
                    </span>
                    <span id={`inspect-state-${candidate.id}`} className={isInspecting ? "inspect-state on" : "inspect-state"}>
                      {isInspecting ? "Inspecting" : seen ? "Viewed" : "Not viewed yet"}, initial position {candidate.position}
                    </span>
                  </button>
                ) : (
                  <span className="initial-text">
                    <span className="initial-label">{candidate.label}</span>
                    <span className="meta">
                      <span className="iri">{candidate.identifier}</span> · {practice ? "illustrative score" : "matching score"} {candidate.score}
                    </span>
                  </span>
                )}
                <span className="initial-actions">
                  {rowExtra?.(candidate)}
                  {rank >= 0 ? (
                    <span className="rank-tag" aria-label={`Your rank ${rank + 1}`}>
                      Your #{rank + 1}
                    </span>
                  ) : (
                    <button type="button" className="btn btn-sm add-btn" disabled={controlsLocked} onClick={() => controller.add(candidate.id)} aria-label={`Add ${candidate.label} as rank ${n + 1}`}>
                      Add as #{n + 1}
                    </button>
                  )}
                </span>
              </li>
            );
          })}
      </ol>
      {candidates[0] && <p className="meta">{candidates[0].scoreMeaning}</p>}
    </section>
  );
}

export function AnswerPanel({
  controller,
  practice = false,
  onSubmit,
  submitting,
  submitLabel = "Submit answer",
  pendingNote,
  footer,
  compact = false,
  id,
}: {
  controller: RankingController;
  practice?: boolean;
  onSubmit: () => void;
  submitting: boolean;
  submitLabel?: string;
  /** Shown while a submission is queued locally and not yet acknowledged by the server. */
  pendingNote?: string | null;
  footer?: React.ReactNode;
  compact?: boolean;
  id?: string;
}) {
  const { candidates, value, locked, controlsLocked } = controller;
  const [drag, setDrag] = useState<number | null>(null);
  const byId = Object.fromEntries(candidates.map((candidate) => [candidate.id, candidate]));
  const n = value.ranked.length;
  const canSubmit = !controlsLocked && ((value.responseType === "ranked_candidates" && n > 0) || value.responseType === "none_of_these" || value.responseType === "insufficient_evidence");
  const answerNote =
    value.responseType === "ranked_candidates"
      ? `${n} of ${candidates.length} ranked · the rest stay unranked`
      : value.responseType === "none_of_these"
        ? "None of these"
        : value.responseType === "insufficient_evidence"
          ? "Insufficient information"
          : "Empty";
  const submitNote =
    value.responseType === "ranked_candidates"
      ? `Submits ${n} ranked ${n === 1 ? "candidate" : "candidates"}. You cannot change it afterwards.`
      : value.responseType === "none_of_these"
        ? "Submits “None of these”. You cannot change it afterwards."
        : value.responseType === "insufficient_evidence"
          ? "Submits “Insufficient information”. You cannot change it afterwards."
          : "Choose at least one candidate, or one of the two options.";

  return (
    <section id={id} tabIndex={id ? -1 : undefined} className={compact ? "ranking-answer ranking-section compact" : "card ranking-answer ranking-section"} aria-labelledby="answer-h">
      <div className="ranking-head">
        <h2 id="answer-h">Your answer</h2>
        <span className="meta" aria-live="polite">
          {answerNote}
        </span>
        <span className="ranking-head-spacer" />
        <button type="button" className="btn btn-sm" onClick={controller.undo} disabled={controlsLocked || !controller.canUndo}>
          <IconUndo /> Undo
        </button>
        <button type="button" className="btn btn-sm" onClick={controller.keepInitial} disabled={controlsLocked}>
          Keep initial order
        </button>
      </div>
      {n === 0 && !value.responseType && (
        <p className="answer-empty">Nothing ranked yet. Add the candidates you consider plausible equivalents, best first, or choose one of the two options below. An empty answer is not submitted.</p>
      )}
      {n > 0 && (
        <ol className="answer-list" aria-label="Your ranking">
          {value.ranked.map((rankedId, index) => {
            const candidate = byId[rankedId];
            return (
              <li
                key={rankedId}
                className={drag === index ? "answer-row dragging" : "answer-row"}
                draggable={!controlsLocked}
                onDragStart={(event) => {
                  setDrag(index);
                  event.dataTransfer.effectAllowed = "move";
                }}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  if (drag !== null) controller.move(drag, index);
                  setDrag(null);
                }}
                onDragEnd={() => setDrag(null)}
              >
                <span className="rank-number">{index + 1}</span>
                <span className="initial-text">
                  <span className="initial-label">{candidate?.label ?? rankedId}</span>
                  <span className="meta">
                    Initial position {candidate?.position} · <span className="iri">{candidate?.identifier}</span>
                  </span>
                </span>
                <span className="initial-actions">
                  <button type="button" className="icon-btn" aria-label={`Move ${candidate?.label} up`} disabled={controlsLocked || index === 0} onClick={() => controller.move(index, index - 1)}>
                    <IconArrowUp />
                  </button>
                  <button type="button" className="icon-btn" aria-label={`Move ${candidate?.label} down`} disabled={controlsLocked || index === n - 1} onClick={() => controller.move(index, index + 1)}>
                    <IconArrowDown />
                  </button>
                  <button type="button" className="btn btn-sm" aria-label={`Remove ${candidate?.label}`} disabled={controlsLocked} onClick={() => controller.remove(index)}>
                    Remove
                  </button>
                </span>
              </li>
            );
          })}
        </ol>
      )}
      <div className="explicit-group" role="radiogroup" aria-label="Or give an explicit answer">
        {(
          [
            ["none_of_these", "None of these", "No displayed candidate is equivalent"],
            ["insufficient_evidence", "Insufficient information", "I cannot judge from what is available"],
          ] as const
        ).map(([type, title, note]) => (
          <button key={type} type="button" role="radio" aria-checked={value.responseType === type} className={value.responseType === type ? "explicit-option on" : "explicit-option"} onClick={() => controller.explicit(type)} disabled={controlsLocked}>
            <span className="explicit-title">{title}</span>
            <span className="meta">{note}</span>
          </button>
        ))}
      </div>
      {!locked && (
        <div className="submit-row">
          <button type="button" className="btn btn-primary btn-large" disabled={!canSubmit || submitting} onClick={onSubmit}>
            {submitting ? "Submitting…" : submitLabel}
          </button>
          <span className="meta" role="status">
            {pendingNote ?? (practice ? "Practice only: this answer is not scored. You can change it and try again." : submitNote)}
          </span>
        </div>
      )}
      {locked && (
        <p className="note note-ok" role="status">
          <IconCheck /> Answer received and saved by the study server.
        </p>
      )}
      {footer}
    </section>
  );
}

/** Candidates and answer together, for stacked layouts (practice, narrow tutorial). */
export function RankingPanel({
  candidates,
  value,
  onChange,
  locked,
  disabled = false,
  resetKey,
  practice = false,
  inspecting,
  viewed,
  onInspect,
  rowExtra,
  onSubmit,
  submitting,
  submitLabel = "Submit answer",
  footer,
}: {
  candidates: RankingCandidate[];
  value: RankingValue;
  onChange: (next: RankingValue, event: RankingEvent, element?: string) => void;
  locked: boolean;
  disabled?: boolean;
  resetKey?: string | number;
  practice?: boolean;
  inspecting?: string | null;
  viewed?: Set<string>;
  onInspect?: (id: string) => void;
  rowExtra?: (candidate: RankingCandidate) => React.ReactNode;
  onSubmit: () => void;
  submitting: boolean;
  submitLabel?: string;
  footer?: React.ReactNode;
}) {
  const controller = useRanking({ candidates, value, onChange, locked, disabled, submitting, resetKey });
  return (
    <div className="ranking">
      <CandidateRows controller={controller} practice={practice} inspecting={inspecting} viewed={viewed} onInspect={onInspect} rowExtra={rowExtra} />
      <AnswerPanel controller={controller} practice={practice} onSubmit={onSubmit} submitting={submitting} submitLabel={submitLabel} footer={footer} />
    </div>
  );
}
