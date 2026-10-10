// Visible assessment drafts that survive one-question-at-a-time navigation, reloads and
// conflicts (S0-R2, S0-FU-R1, S0-FU-R2; 21 F02, 12 persistence). A question card is keyed and
// unmounts when the participant moves on, so its answer lives here, above the cards, for the
// whole tutorial session. Pure module (type-only imports) for Node's test runner.
//
// Rules:
// - An edit is shown at once. It stays this tab's unsaved work until the saved answer equals it
//   and none of this tab's writes for that question is still queued, in flight, restored after a
//   reload, or being replaced. An older write's acknowledgement never replaces a newer edit.
// - Each piece of unsaved work has a base: the saved answer it started from. Every write the
//   outbox sends carries the revision this tab last saw, so it can only succeed while the server
//   still holds the state this tab observed. A write for a question is therefore issued only
//   while the observed saved answer equals the base (or is this tab's own acknowledged answer,
//   which then becomes the base). That is what proves it cannot overwrite another tab's answer.
// - After a write fails (a conflict, or discarded behind another failure), the answer is saved
//   again automatically only when the saved answer still equals the base, at most
//   MAX_AUTOMATIC_RESAVES times without a success. Otherwise the saved answer is shown and this
//   tab's answer is kept as a conflict for the participant to resolve deliberately.

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

export const MAX_AUTOMATIC_RESAVES = 3;

type Saved = Record<string, AssessmentResponse> | undefined;

interface Entry {
  /** The latest local answer: shown, unless a conflict waits for the participant's choice. */
  response: AssessmentResponse;
  /** The saved answer (canonical) this work started from; null when unknown (an older outbox). */
  base: string | null;
  /** This tab's writes for the question that have not settled. */
  outstanding: number;
  /** A write failed: the server may not hold this answer, and nothing is sent until checked. */
  failed: boolean;
  /** The saved answer changed elsewhere: it is shown, and this answer kept for a choice. */
  conflict: boolean;
  /** Automatic re-saves since this tab's last acknowledged write for the question. */
  resends: number;
}

/** An answer that is safe to save again: the saved answer still equals its base. */
export interface Resave {
  questionId: string;
  response: AssessmentResponse;
}

/** This tab's answer kept during a conflict, persisted so a reload can still offer the choice. */
export interface KeptConflict {
  questionId: string;
  response: AssessmentResponse;
  base: string | null;
}

export class AssessmentDrafts {
  private entries = new Map<string, Entry>();

  /** What the question shows: this tab's answer, or the saved answer during a conflict or when nothing is unsaved. */
  value(questionId: string, saved: AssessmentResponse | undefined, empty: AssessmentResponse): AssessmentResponse {
    const entry = this.entries.get(questionId);
    return entry && !entry.conflict ? entry.response : saved ?? empty;
  }

  /** This tab's answer when another tab or device saved a different one: kept for a choice. */
  conflict(questionId: string): AssessmentResponse | null {
    const entry = this.entries.get(questionId);
    return entry?.conflict ? entry.response : null;
  }

  /** The saved answer this tab's unsaved work started from (sent with each write, never to the server). */
  baseOf(questionId: string): string | null {
    return this.entries.get(questionId)?.base ?? null;
  }

  /**
   * A participant edit, given the currently observed saved answer. Returns whether to write it.
   * Editing the saved answer shown during a conflict is a deliberate resolution: it starts new
   * work from that saved answer. An edit made after the saved answer changed elsewhere is kept
   * but not written: the question becomes a conflict.
   */
  edit(questionId: string, response: AssessmentResponse, saved: AssessmentResponse | undefined): boolean {
    const observed = canonical(saved);
    const entry = this.entries.get(questionId);
    if (!entry || entry.conflict) {
      this.entries.set(questionId, { response, base: observed, outstanding: entry?.outstanding ?? 0, failed: false, conflict: false, resends: 0 });
      return true;
    }
    entry.response = response;
    if (entry.base !== null && entry.base !== observed) {
      entry.conflict = true;
      return false;
    }
    return true;
  }

  /** A write for this question was handed to the outbox. */
  sent(questionId: string) {
    const entry = this.entries.get(questionId);
    if (!entry) return;
    entry.outstanding += 1;
    entry.failed = false;
  }

  /** This tab's write was acknowledged; `saved` is the state it produced (the new base). */
  acknowledged(questionId: string, saved: Saved) {
    const entry = this.entries.get(questionId);
    if (!entry) return;
    entry.outstanding = Math.max(0, entry.outstanding - 1);
    entry.base = canonical(saved?.[questionId]);
    entry.resends = 0;
  }

  /** The write was replaced by a newer unsent draft for the same question. */
  replaced(questionId: string) {
    const entry = this.entries.get(questionId);
    if (entry) entry.outstanding = Math.max(0, entry.outstanding - 1);
  }

  /** The write failed or was discarded: keep the answer, and check before any write. */
  failed(questionId: string) {
    const entry = this.entries.get(questionId);
    if (!entry) return;
    entry.outstanding = Math.max(0, entry.outstanding - 1);
    entry.failed = true;
  }

  /** Writes restored after a reload, oldest first: each counts until it settles. */
  seed(restored: { questionId: string; response: AssessmentResponse; base: string | null }[]) {
    for (const write of restored) {
      const entry = this.entries.get(write.questionId);
      if (entry) {
        entry.response = write.response;
        entry.outstanding += 1;
      } else this.entries.set(write.questionId, { response: write.response, base: write.base, outstanding: 1, failed: false, conflict: false, resends: 0 });
    }
  }

  /** Conflicts kept from before a reload. */
  restoreConflicts(kept: KeptConflict[]) {
    for (const item of kept) if (!this.entries.has(item.questionId)) this.entries.set(item.questionId, { response: item.response, base: item.base, outstanding: 0, failed: false, conflict: true, resends: 0 });
  }

  conflicts(): KeptConflict[] {
    return Array.from(this.entries)
      .filter(([, entry]) => entry.conflict)
      .map(([questionId, entry]) => ({ questionId, response: entry.response, base: entry.base }));
  }

  /**
   * Compare settled work with the observed saved answers (the state the next write is made
   * against). Work the server now holds is done; failed work is saved again only when the saved
   * answer still equals its base; anything else becomes a conflict. Returns the safe re-saves.
   */
  observe(saved: Saved): Resave[] {
    const resaves: Resave[] = [];
    for (const [questionId, entry] of Array.from(this.entries)) {
      if (entry.outstanding > 0) continue;
      const observed = canonical(saved?.[questionId]);
      if (observed === canonical(entry.response)) {
        this.entries.delete(questionId);
        continue;
      }
      if (entry.conflict || !entry.failed) continue;
      if (entry.base !== null && observed === entry.base && entry.resends < MAX_AUTOMATIC_RESAVES) {
        entry.resends += 1;
        entry.failed = false;
        resaves.push({ questionId, response: entry.response });
      } else entry.conflict = true;
    }
    return resaves;
  }

  /**
   * The participant's deliberate choice during a conflict. "mine" starts new work from the saved
   * answer they were shown and returns the answer to write; "saved" drops this tab's answer.
   */
  resolve(questionId: string, choice: "mine" | "saved", saved: AssessmentResponse | undefined): AssessmentResponse | null {
    const entry = this.entries.get(questionId);
    if (!entry?.conflict) return null;
    if (choice === "saved") {
      this.entries.delete(questionId);
      return null;
    }
    Object.assign(entry, { base: canonical(saved), failed: false, conflict: false, resends: 0 });
    return entry.response;
  }

  /** Whether the question has work the server does not hold yet. */
  isPending(questionId: string): boolean {
    return this.entries.has(questionId);
  }
}
