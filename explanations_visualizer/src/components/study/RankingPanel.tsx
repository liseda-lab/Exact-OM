"use client";

// One ranking component for both study conditions and the practice sandbox. The initial
// order stays visible and never changes; the answer starts empty. Partial rankings, explicit
// "None of these" and "Insufficient information" are first-class; nothing auto-submits.

import { useCallback, useRef, useState } from "react";

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

export function RankingPanel({
  candidates,
  value,
  onChange,
  locked,
  inspecting,
  onInspect,
  rowExtra,
  onSubmit,
  submitting,
  submitLabel = "Submit answer",
  footer,
}: {
  candidates: RankingCandidate[];
  value: RankingValue;
  onChange: (next: RankingValue, event: EventType, element?: string) => void;
  locked: boolean;
  inspecting?: string | null;
  onInspect?: (id: string) => void;
  rowExtra?: (candidate: RankingCandidate) => React.ReactNode;
  onSubmit: () => void;
  submitting: boolean;
  submitLabel?: string;
  footer?: React.ReactNode;
}) {
  const history = useRef<RankingValue[]>([]);
  const [, force] = useState(0);
  const [drag, setDrag] = useState<number | null>(null);
  const byId = Object.fromEntries(candidates.map((candidate) => [candidate.id, candidate]));

  const apply = useCallback(
    (next: RankingValue, event: EventType, element?: string) => {
      if (locked) return;
      history.current = [...history.current, value].slice(-50);
      force((n) => n + 1);
      onChange(next, event, element);
    },
    [locked, onChange, value],
  );

  const add = (id: string) => apply({ responseType: "ranked_candidates", ranked: [...value.ranked.filter((item) => item !== id), id] }, "rank_add", id);
  const remove = (index: number) => {
    const ranked = value.ranked.filter((_, position) => position !== index);
    apply({ responseType: ranked.length ? "ranked_candidates" : null, ranked }, "rank_remove", value.ranked[index]);
  };
  const move = (from: number, to: number) => {
    if (to < 0 || to >= value.ranked.length || from === to) return;
    const ranked = [...value.ranked];
    const [item] = ranked.splice(from, 1);
    ranked.splice(to, 0, item);
    apply({ responseType: "ranked_candidates", ranked }, "rank_move", item);
  };
  const keepInitial = () =>
    apply(
      {
        responseType: "ranked_candidates",
        ranked: [...candidates].sort((a, b) => a.position - b.position).map((candidate) => candidate.id),
      },
      "keep_initial_order",
    );
  const explicit = (type: "none_of_these" | "insufficient_evidence") => apply({ responseType: value.responseType === type ? null : type, ranked: [] }, "response_type_change", type);
  const undo = () => {
    const previous = history.current.pop();
    if (!previous) return;
    force((n) => n + 1);
    onChange(previous, "revision", "undo");
  };

  const n = value.ranked.length;
  const canSubmit = !locked && ((value.responseType === "ranked_candidates" && n > 0) || value.responseType === "none_of_these" || value.responseType === "insufficient_evidence");
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
    <div className="ranking">
      <section className="card ranking-initial" aria-labelledby="initial-h">
        <div className="ranking-head">
          <h2 id="initial-h">Initial order</h2>
          <span className="meta">The system&apos;s suggestion</span>
        </div>
        <ol className="initial-list">
          {[...candidates]
            .sort((a, b) => a.position - b.position)
            .map((candidate) => {
              const rank = value.ranked.indexOf(candidate.id);
              const isInspecting = inspecting === candidate.id;
              return (
                <li key={candidate.id} className={isInspecting ? "initial-row inspecting" : "initial-row"}>
                  <span className="position-badge" aria-label={`Initial position ${candidate.position}`}>
                    {candidate.position}
                  </span>
                  <span className="initial-text">
                    <span className="initial-label">{candidate.label}</span>
                    <span className="meta">
                      <span className="iri">{candidate.identifier}</span> · matching score {candidate.score}
                    </span>
                  </span>
                  <span className="initial-actions">
                    {rowExtra?.(candidate)}
                    {onInspect && (
                      <button type="button" className={isInspecting ? "btn btn-sm toggle on" : "btn btn-sm"} aria-pressed={isInspecting} onClick={() => onInspect(candidate.id)}>
                        Inspect
                      </button>
                    )}
                    {rank >= 0 ? (
                      <span className="rank-tag" aria-label={`Your rank ${rank + 1}`}>
                        Your #{rank + 1}
                      </span>
                    ) : (
                      <button type="button" className="btn btn-sm add-btn" disabled={locked} onClick={() => add(candidate.id)} aria-label={`Add ${candidate.label} as rank ${n + 1}`}>
                        Add as #{n + 1}
                      </button>
                    )}
                  </span>
                </li>
              );
            })}
        </ol>
        {candidates[0] && <p className="meta">Matching scores: {candidates[0].scoreMeaning}</p>}
      </section>

      <section className="card ranking-answer" aria-labelledby="answer-h">
        <div className="ranking-head">
          <h2 id="answer-h">Your answer</h2>
          <span className="meta" aria-live="polite">
            {answerNote}
          </span>
          <span className="ranking-head-spacer" />
          <button type="button" className="btn btn-sm" onClick={undo} disabled={locked || history.current.length === 0}>
            <IconUndo /> Undo
          </button>
          <button type="button" className="btn btn-sm" onClick={keepInitial} disabled={locked}>
            Keep initial order
          </button>
        </div>
        {n === 0 && !value.responseType && (
          <p className="answer-empty">Nothing ranked yet. Add the candidates you consider plausible equivalents, best first, or choose one of the two options below. An empty answer is not submitted.</p>
        )}
        {n > 0 && (
          <ol className="answer-list" aria-label="Your ranking">
            {value.ranked.map((id, index) => {
              const candidate = byId[id];
              return (
                <li
                  key={id}
                  className={drag === index ? "answer-row dragging" : "answer-row"}
                  draggable={!locked}
                  onDragStart={(event) => {
                    setDrag(index);
                    event.dataTransfer.effectAllowed = "move";
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();
                    if (drag !== null) move(drag, index);
                    setDrag(null);
                  }}
                  onDragEnd={() => setDrag(null)}
                >
                  <span className="rank-number">{index + 1}</span>
                  <span className="initial-text">
                    <span className="initial-label">{candidate?.label ?? id}</span>
                    <span className="meta">
                      Initial position {candidate?.position} · <span className="iri">{candidate?.identifier}</span>
                    </span>
                  </span>
                  <span className="initial-actions">
                    <button type="button" className="icon-btn" aria-label={`Move ${candidate?.label} up`} disabled={locked || index === 0} onClick={() => move(index, index - 1)}>
                      <IconArrowUp />
                    </button>
                    <button type="button" className="icon-btn" aria-label={`Move ${candidate?.label} down`} disabled={locked || index === n - 1} onClick={() => move(index, index + 1)}>
                      <IconArrowDown />
                    </button>
                    <button type="button" className="btn btn-sm" aria-label={`Remove ${candidate?.label}`} disabled={locked} onClick={() => remove(index)}>
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
            <button key={type} type="button" role="radio" aria-checked={value.responseType === type} className={value.responseType === type ? "explicit-option on" : "explicit-option"} onClick={() => explicit(type)} disabled={locked}>
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
            <span className="meta">{submitNote}</span>
          </div>
        )}
        {locked && (
          <p className="note note-ok" role="status">
            <IconCheck /> Answer submitted and saved.
          </p>
        )}
        {footer}
      </section>
    </div>
  );
}
