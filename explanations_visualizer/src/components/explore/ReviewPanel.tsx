"use client";

// Exploration-app review (ReviewDecision), kept in this browser. It is deliberately not a
// study response and never changes Exact's saved result.

import { useEffect, useState } from "react";

import { readStored, writeStored } from "@/lib/prefs";

type Relation = "" | "equivalent" | "source_narrower" | "source_broader" | "not_equivalent" | "unresolved";
type Action = "" | "accept" | "reject" | "defer";

interface Review {
  relation: Relation;
  action: Action;
  saved_at: string;
}

const RELATIONS: { value: Relation; label: string }[] = [
  { value: "", label: "Not judged yet" },
  { value: "equivalent", label: "Equivalent" },
  { value: "source_narrower", label: "Source is narrower" },
  { value: "source_broader", label: "Source is broader" },
  { value: "not_equivalent", label: "Not equivalent" },
  { value: "unresolved", label: "Unresolved" },
];

const ACTION_TEXT: Record<Exclude<Action, "">, string> = {
  accept: "Accepted",
  reject: "Rejected",
  defer: "Deferred",
};

export function ReviewPanel({ packageId, pairId }: { packageId: string; pairId: string }) {
  const storageKey = `exact.review.${packageId}.${pairId}`;
  const [review, setReview] = useState<Review | null>(null);

  useEffect(() => {
    setReview(readStored<Review | null>(storageKey, null));
  }, [storageKey]);

  const update = (changes: Partial<Review>) => {
    const next: Review = { relation: review?.relation ?? "", action: review?.action ?? "", ...changes, saved_at: new Date().toISOString() };
    setReview(next);
    writeStored(storageKey, next);
  };

  return (
    <section className="card rail-section" aria-labelledby="review-h">
      <div className="rail-heading">
        <h2 id="review-h" className="eyebrow">
          Your review
        </h2>
        <span className="meta">Saved in this browser</span>
      </div>
      <div className="field">
        <label htmlFor="review-relation" className="muted">
          Relation you see
        </label>
        <select id="review-relation" className="select" value={review?.relation ?? ""} onChange={(event) => update({ relation: event.target.value as Relation })}>
          {RELATIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
      <div className="review-actions" role="group" aria-label="Review action">
        {(["accept", "reject", "defer"] as const).map((action) => (
          <button key={action} type="button" className={review?.action === action ? "btn toggle on" : "btn toggle"} aria-pressed={review?.action === action} onClick={() => update({ action: review?.action === action ? "" : action })}>
            {action === "accept" ? "Accept" : action === "reject" ? "Reject" : "Defer"}
          </button>
        ))}
      </div>
      <p className="meta" aria-live="polite">
        {review?.action ? `${ACTION_TEXT[review.action]}. ` : ""}Your review does not change Exact&apos;s saved result.
      </p>
    </section>
  );
}
