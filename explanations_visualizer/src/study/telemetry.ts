"use client";

// Optional telemetry never blocks an answer. Each retry retains its original event/segment ID.
import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "@/lib/api";
import { uuid } from "@/study/session";
import { retainCurrentTiming, splitTiming, type OpenSegment } from "@/study/timing";
import { ObservationStore, TelemetryQueue, type DeliveryHealth, type Observation } from "@/study/telemetryQueue";
import { V1_EVENT_TYPES, type EventType, type Stage, type StudyState } from "@/study/types";

export const BUILD_VERSION = "exact-explain-ui-1.2";
const TIMED: Stage[] = ["setup", "background", "practice", "tutorial", "case", "consultation", "final"];
interface StudyEvent extends Observation {
  event_id: string; page_instance_id: string; sequence: number; case_id?: string;
  presentation_id?: string; scope?: "case" | "tutorial" | "help"; tutorial_version?: string; lesson_id?: string; type: EventType; component_id?: string; element_id?: string;
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
  const queue = useRef<TelemetryQueue | null>(null);
  const [delivery, setDelivery] = useState<DeliveryHealth>({ pending: 0, rejected: 0, delayed: false, storageUnavailable: false, lost: 0 });
  const segment = useRef<OpenSegment | null>(null);
  // The presentation whose required content this page has validated and rendered (19 F12).
  const usablePresentation = useRef<string | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  // Only event types this service version accepts are sent; others are not invented as v1 types.
  const declared = state?.telemetry?.event_types;
  const supported = useRef<Set<string>>(new Set(V1_EVENT_TYPES));
  supported.current = new Set(declared?.length ? declared : V1_EVENT_TYPES);

  // The durable outbox is isolated by session. Each document keeps its own monotonic clock.
  useEffect(() => {
    const sessionId = state?.session_id ?? null;
    owner.current = sessionId;
    segment.current = null;
    usablePresentation.current = null;
    sequence.current = 0;
    pageInstance.current = uuid();
    if (!sessionId) { queue.current = null; return; }
    let storage: Storage | null = null;
    try { storage = window.localStorage; } catch { /* The outbox will report unavailable storage. */ }
    const outbox = new ObservationStore(sessionId, storage);
    const deliveryQueue = new TelemetryQueue(outbox, async (path, body, keepalive) => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);
      try {
        return await request<Record<string, unknown>>(path, {
          method: "POST", headers: { "Content-Type": "application/json", "X-Study-Session": sessionId },
          body: JSON.stringify(body), signal: controller.signal, keepalive,
        });
      } finally { clearTimeout(timeout); }
    }, () => { if (queue.current === deliveryQueue) setDelivery(outbox.health()); });
    queue.current = deliveryQueue;
    setDelivery(outbox.health());
    void deliveryQueue.flush();
    return () => { deliveryQueue.close(); if (queue.current === deliveryQueue) queue.current = null; };
  }, [state?.session_id]);

  const flush = useCallback((keepalive = false): Promise<void> => queue.current?.flush(keepalive) ?? Promise.resolve(), []);
  const retryDelivery = useCallback(() => { void queue.current?.retry(); }, []);

  const emit = useCallback((type: EventType, extra: { component?: string; element?: string; visibility?: "visible" | "hidden"; loadingMs?: number; scope?: "case" | "tutorial" | "help"; lessonId?: string } = {}) => {
    const current = stateRef.current;
    if (!supported.current.has(type)) return undefined;
    if (!current || current.session_id !== owner.current) return undefined;
    const scope = extra.scope ?? (current.stage === "tutorial" ? "tutorial" : "case");
    if (scope !== "case" && (!current.telemetry?.scopes?.includes(scope) || !current.tutorial)) return undefined;
    if (scope === "case" && (!current.current_case_id || !current.current_presentation_id || !["case", "consultation", "paused"].includes(current.stage))) return undefined;
    const binding = scope === "case" ? { case_id: current.current_case_id!, presentation_id: current.current_presentation_id! } : { scope, tutorial_version: current.tutorial!.version, lesson_id: extra.lessonId };
    const now = performance.now();
    const event: StudyEvent = { event_id: uuid(), page_instance_id: pageInstance.current, sequence: sequence.current++, ...binding,
      type, component_id: safeId(extra.component), element_id: safeId(extra.element),
      client_monotonic_ms: now, build_version: BUILD_VERSION, visibility: extra.visibility, loading_ms: extra.loadingMs };
    if (!queue.current?.enqueue("event", event)) return undefined;
    if (type === "case_ready" || type === "submit" || (queue.current?.store.health().pending ?? 0) >= 20) void flush();
    return now;
  }, [flush]);

  const capture = useCallback((close = false) => {
    const result = splitTiming(segment.current, performance.now(), close);
    segment.current = result.next;
    if (!result.interval) return;
    const part = result.interval;
    queue.current?.enqueue("timing", { segment_id: uuid(), page_instance_id: pageInstance.current, stage: part.stage, case_id: part.caseId,
      presentation_id: part.presentationId, monotonic_start_ms: part.start, monotonic_end_ms: part.end });
  }, []);

  const setCaseUsable = useCallback((presentationId: string | null) => {
    usablePresentation.current = presentationId;
  }, []);

  /** One ready observation per page and presentation, only after its content is usable. */
  const markCaseReady = useCallback((loadingMs: number) => {
    const current = stateRef.current;
    if (!current?.current_case_id || current.stage !== "case" || current.session_id !== owner.current) return;
    if (!current.current_presentation_id || usablePresentation.current !== current.current_presentation_id) return;
    if (segment.current?.stage === "case" && segment.current.presentationId === current.current_presentation_id) return;
    const now = emit("case_ready", { loadingMs });
    if (now !== undefined) segment.current = { stage: "case", caseId: current.current_case_id, presentationId: current.current_presentation_id, start: now };
  }, [emit]);

  const resumeTiming = useCallback(() => {
    const current = stateRef.current;
    if (!current || segment.current || current.session_id !== owner.current || !TIMED.includes(current.stage)) return;
    // A case segment reopens only on a page whose content is already usable for it.
    if (current.stage === "case") { markCaseReady(0); return; }
    segment.current = { stage: current.stage, caseId: current.stage === "consultation" ? current.current_case_id : null,
      presentationId: current.stage === "consultation" ? current.current_presentation_id : null, start: performance.now() };
  }, [markCaseReady]);

  const stage = state?.stage;
  const presentationId = state?.current_presentation_id;
  useEffect(() => {
    segment.current = retainCurrentTiming(segment.current, stage, presentationId ?? null);
    // Historical records keep their original page/presentation identity. A stale ready
    // event is isolated by delivery; it must never be silently reassigned to a new case.
    if (stage !== "case") resumeTiming();
  }, [stage, presentationId, state?.session_id, resumeTiming]);

  const flushTiming = useCallback(async () => {
    capture(true); // Freeze at submit/pause intent, including while an acknowledgement is delayed.
    let timer: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([(async () => { await flush(); await flush(); })(), new Promise<void>((resolve) => { timer = setTimeout(resolve, 1500); })]);
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
      void flush(document.visibilityState === "hidden");
    };
    const leaving = () => { capture(true); void flush(true); };
    const online = () => { void flush(); };
    const restored = (event: PageTransitionEvent) => { if (event.persisted) resumeTiming(); void flush(); };
    document.addEventListener("visibilitychange", visibility);
    window.addEventListener("pagehide", leaving);
    window.addEventListener("pageshow", restored);
    window.addEventListener("online", online);
    return () => {
      clearInterval(timer); document.removeEventListener("visibilitychange", visibility);
      window.removeEventListener("pagehide", leaving); window.removeEventListener("pageshow", restored); window.removeEventListener("online", online);
    };
  }, [capture, emit, flush, resumeTiming]);

  return { emit, flushTiming, markCaseReady, setCaseUsable, resumeTiming, delivery, retryDelivery, pageInstanceId: pageInstance.current };
}
export type Telemetry = ReturnType<typeof useTelemetry>;
