/** Monotonic observed intervals. Long event-loop gaps are left unknown, never filled in. */
export interface OpenSegment { stage: string; caseId: string | null; presentationId: string | null; start: number }
export interface Interval extends OpenSegment { end: number }
export function splitTiming(open: OpenSegment | null, end: number, close = false): { interval: Interval | null; next: OpenSegment | null } {
  if (!open) return { interval: null, next: null };
  const elapsed = end - open.start;
  const next = close ? null : { ...open, start: end };
  if (elapsed < 0 || elapsed > 299_000) return { interval: null, next };
  if (elapsed === 0) return { interval: null, next: close ? null : open };
  return { interval: { ...open, end }, next };
}

/** Parent stage effects may run after a child's ready effect; retain that same interval. */
export function retainCurrentTiming(open: OpenSegment | null, stage: string | undefined, presentationId: string | null): OpenSegment | null {
  if (!open || open.stage !== stage) return null;
  if ((stage === "case" || stage === "consultation") && open.presentationId !== presentationId) return null;
  return open;
}

/** A case-to-case transition changes timers even though both steps are named "case". */
export function sameTimingContext(before: { stage: string; current_presentation_id?: string | null } | null | undefined, after: { stage: string; current_presentation_id?: string | null }): boolean {
  return Boolean(before && before.stage === after.stage && (before.current_presentation_id ?? null) === (after.current_presentation_id ?? null));
}
