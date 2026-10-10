// Visible assessment drafts that survive one-question-at-a-time navigation (S0-R2, 21 F02).
// A question card is keyed and unmounts when the participant moves on; its answer must not
// fall back to the last acknowledged server draft while a newer local edit is still queued,
// offline, or rejected by a conflict. This store sits above the cards for the whole tutorial
// session. Pure module (type-only imports) for Node's test runner.
//
// Rules:
// - An edit is shown immediately and stays "pending" until the server's draft equals it and no
//   save for that question is still outstanding (queued, in flight or being replaced).
// - With nothing pending locally, the server's draft wins (another tab, a replay, a refresh).
// - A pending edit is never overwritten by an older or competing server draft; the shell saves
//   pending drafts again after a conflict refresh or a discarded write.

import type { AssessmentResponse } from "./types";

/** Stable JSON with sorted keys, so equal answers compare equal whatever their key order. */
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).filter(([, item]) => item !== undefined).sort(([a], [b]) => a.localeCompare(b));
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
  }
  return JSON.stringify(value ?? null);
}

interface Entry {
  response: AssessmentResponse;
  pending: boolean;
  outstanding: number;
}

export class AssessmentDrafts {
  private entries = new Map<string, Entry>();

  /** What the question shows: a local edit, else the server's draft, else empty. */
  value(questionId: string, server: AssessmentResponse | undefined, empty: AssessmentResponse): AssessmentResponse {
    return this.entries.get(questionId)?.response ?? server ?? empty;
  }

  /** A participant edit: shown at once and pending until acknowledged. */
  edit(questionId: string, response: AssessmentResponse) {
    const entry = this.entries.get(questionId);
    this.entries.set(questionId, { response, pending: true, outstanding: entry?.outstanding ?? 0 });
  }

  /** A save for this question was handed to the outbox. */
  sent(questionId: string) {
    const entry = this.entries.get(questionId);
    if (entry) entry.outstanding += 1;
  }

  /** That save settled (acknowledged, replaced by a newer draft, discarded or rejected). */
  settled(questionId: string, server: Record<string, AssessmentResponse> | undefined) {
    const entry = this.entries.get(questionId);
    if (!entry) return;
    entry.outstanding = Math.max(0, entry.outstanding - 1);
    this.reconcile(server);
  }

  /** Queued drafts restored after a reload: shown until the server's draft matches them. */
  seed(drafts: { questionId: string; response: AssessmentResponse }[]) {
    for (const draft of drafts) this.entries.set(draft.questionId, { response: draft.response, pending: true, outstanding: 0 });
  }

  /** Align with the authoritative drafts without overwriting newer local edits. */
  reconcile(server: Record<string, AssessmentResponse> | undefined) {
    for (const [questionId, entry] of Array.from(this.entries)) {
      if (entry.outstanding > 0) continue;
      const saved = server?.[questionId];
      if (entry.pending && saved !== undefined && canonical(saved) === canonical(entry.response)) entry.pending = false;
      if (!entry.pending) this.entries.delete(questionId);
    }
  }

  /** Local edits the server does not hold yet. */
  pending(): { questionId: string; response: AssessmentResponse }[] {
    return Array.from(this.entries)
      .filter(([, entry]) => entry.pending)
      .map(([questionId, entry]) => ({ questionId, response: entry.response }));
  }

  isPending(questionId: string): boolean {
    return Boolean(this.entries.get(questionId)?.pending);
  }
}
