"use client";

// Optional telemetry never blocks an answer. Each retry retains its original event/segment ID.
import { useCallback, useEffect, useRef } from "react";
import { request } from "@/lib/api";
import { uuid } from "@/study/session";
import { retainCurrentTiming, splitTiming, type OpenSegment } from "@/study/timing";
import { V1_EVENT_TYPES, type EventType, type Stage, type StudyState } from "@/study/types";

export const BUILD_VERSION = "exact-explain-ui-1.2";
const TIMED: Stage[] = ["setup", "background", "practice", "tutorial", "case", "consultation", "final"];
interface StudyEvent {
  event_id: string; page_instance_id: string; sequence: number; case_id: string;
  presentation_id: string; type: EventType; component_id?: string; element_id?: string;
  client_monotonic_ms: number; build_version: string; visibility?: "visible" | "hidden"; loading_ms?: number;
}
function safeId(value: string | undefined): string | undefined {
  if (!value) return undefined;
  return value.replace(/[^A-Za-z0-9_.:/-]/g, "_").slice(0, 128).replace(/:\/\//g, "_");
}
export function useTelemetry(state: StudyState | null) {
  const pageInstance = useRef(uuid());
  const owner = useRef<string | null>(null);
  const sequence = useRef(0);
  const events = useRef<StudyEvent[]>([]);
  const segments = useRef<Record<string, unknown>[]>([]);
  const flushing = useRef<Promise<void> | null>(null);
  const segment = useRef<OpenSegment | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  // Only event types this service version accepts are sent; others are not invented as v1 types.
  const declared = state?.telemetry?.event_types;
  const supported = useRef<Set<string>>(new Set(V1_EVENT_TYPES));
  supported.current = new Set(declared?.length ? declared : V1_EVENT_TYPES);

  // A new invitation must never inherit another session's telemetry or ready marker.
  useEffect(() => {
    if (owner.current === (state?.session_id ?? null)) return;
    owner.current = state?.session_id ?? null;
    events.current = [];
    segments.current = [];
    segment.current = null;
    sequence.current = 0;
    pageInstance.current = uuid();
    flushing.current = null;
  }, [state?.session_id]);

  const flush = useCallback((): Promise<void> => {
    if (flushing.current) return flushing.current;
    const sessionId = owner.current;
    if (!sessionId) return Promise.resolve();
    const send = (path: string, body: unknown) => request<{ acknowledged_event_ids?: string[] }>(path, {
      method: "POST", headers: { "Content-Type": "application/json", "X-Study-Session": sessionId }, body: JSON.stringify(body),
    });
    const work = (async () => {
      try {
        // case_ready must be acknowledged before a case timing segment can be admitted.
        while (events.current.length && owner.current === sessionId) {
          const batch = events.current.slice(0, 100);
          const ack = await send("/api/v1/study/events", { events: batch });
          if (owner.current !== sessionId) return;
          const done = new Set(ack.acknowledged_event_ids);
          if (!batch.some((event) => done.has(event.event_id))) return;
          events.current = events.current.filter((event) => !done.has(event.event_id));
        }
        while (segments.current.length && owner.current === sessionId) {
          const pending = segments.current[0];
          await send("/api/v1/study/timing", pending);
          if (owner.current !== sessionId) return;
          segments.current = segments.current.filter((part) => part.segment_id !== pending.segment_id);
        }
      } catch {
        // Retry while this stage remains current. Bounded loss appears as sequence/time gaps.
        if (owner.current === sessionId) {
          events.current = events.current.slice(-500);
          segments.current = segments.current.slice(-20);
        }
      }
    })();
    flushing.current = work;
    void work.finally(() => { if (flushing.current === work) flushing.current = null; });
    return work;
  }, []);

  const emit = useCallback((type: EventType, extra: { component?: string; element?: string; visibility?: "visible" | "hidden"; loadingMs?: number } = {}) => {
    const current = stateRef.current;
    if (!supported.current.has(type)) return undefined;
    if (!current?.current_case_id || !current.current_presentation_id || !["case", "consultation", "paused"].includes(current.stage) || current.session_id !== owner.current) return undefined;
    const now = performance.now();
    events.current.push({ event_id: uuid(), page_instance_id: pageInstance.current, sequence: sequence.current++, case_id: current.current_case_id,
      presentation_id: current.current_presentation_id, type, component_id: safeId(extra.component), element_id: safeId(extra.element),
      client_monotonic_ms: now, build_version: BUILD_VERSION, visibility: extra.visibility, loading_ms: extra.loadingMs });
    if (type === "case_ready" || type === "submit" || events.current.length >= 20) void flush();
    return now;
  }, [flush]);

  const capture = useCallback((close = false) => {
    const result = splitTiming(segment.current, performance.now(), close);
    segment.current = result.next;
    if (!result.interval) return;
    const part = result.interval;
    segments.current.push({ segment_id: uuid(), page_instance_id: pageInstance.current, stage: part.stage, case_id: part.caseId,
      presentation_id: part.presentationId, monotonic_start_ms: part.start, monotonic_end_ms: part.end });
  }, []);

  const markCaseReady = useCallback((loadingMs: number) => {
    const current = stateRef.current;
    if (!current?.current_case_id || current.stage !== "case" || current.session_id !== owner.current) return;
    if (segment.current?.stage === "case" && segment.current.presentationId === current.current_presentation_id) return;
    const now = emit("case_ready", { loadingMs });
    if (now !== undefined) segment.current = { stage: "case", caseId: current.current_case_id, presentationId: current.current_presentation_id, start: now };
  }, [emit]);

  const resumeTiming = useCallback(() => {
    const current = stateRef.current;
    if (!current || segment.current || current.session_id !== owner.current || !TIMED.includes(current.stage)) return;
    if (current.stage === "case") { markCaseReady(0); return; }
    segment.current = { stage: current.stage, caseId: current.stage === "consultation" ? current.current_case_id : null,
      presentationId: current.stage === "consultation" ? current.current_presentation_id : null, start: performance.now() };
  }, [markCaseReady]);

  const stage = state?.stage;
  const presentationId = state?.current_presentation_id;
  useEffect(() => {
    segment.current = retainCurrentTiming(segment.current, stage, presentationId ?? null);
    events.current = events.current.filter((event) => event.type !== "case_ready" || (stage === "case" && event.presentation_id === presentationId));
    // The previous stage can no longer accept new timing; retain the gap honestly.
    segments.current = segments.current.filter((part) => part.stage === stage && (part.presentation_id === null || part.presentation_id === presentationId));
    if (stage !== "case") resumeTiming();
  }, [stage, presentationId, state?.session_id, resumeTiming]);

  const flushTiming = useCallback(async () => {
    capture(true); // Freeze at submit/pause intent, including while an acknowledgement is delayed.
    let timer: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([flush(), new Promise<void>((resolve) => { timer = setTimeout(resolve, 1500); })]);
    if (timer) clearTimeout(timer);
  }, [capture, flush]);

  useEffect(() => {
    const timer = setInterval(() => {
      if (segment.current && performance.now() - segment.current.start >= 60_000) capture();
      void flush();
    }, 15_000);
    const visibility = () => {
      emit("visibility", { visibility: document.visibilityState === "hidden" ? "hidden" : "visible" });
      capture(); // A hidden tab may be external work on the case; never pause automatically.
      void flush();
    };
    document.addEventListener("visibilitychange", visibility);
    return () => { clearInterval(timer); document.removeEventListener("visibilitychange", visibility); };
  }, [capture, emit, flush]);

  return { emit, flushTiming, markCaseReady, resumeTiming, pageInstanceId: pageInstance.current };
}
export type Telemetry = ReturnType<typeof useTelemetry>;
