"use client";

// Research telemetry: typed interaction events and observed timing segments. It records
// meaningful actions only (no pointer tracking, keystrokes or free text) and never blocks
// or delays saving an answer. Hidden tabs do not pause time: Protégé work happens there.

import { useCallback, useEffect, useRef } from "react";

import { sendJson } from "@/lib/api";
import { uuid } from "@/study/session";
import type { EventType, Stage, StudyState } from "@/study/types";

export const BUILD_VERSION = "exact-explain-ui-1.0";
const TIMED: Stage[] = ["setup", "background", "practice", "case", "consultation", "final"];
const SEGMENT_MS = 60_000;

interface StudyEvent {
  event_id: string;
  page_instance_id: string;
  sequence: number;
  case_id: string;
  presentation_id: string;
  type: EventType;
  component_id?: string;
  element_id?: string;
  client_monotonic_ms: number;
  build_version: string;
  visibility?: "visible" | "hidden";
  loading_ms?: number;
}

function safeId(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const cleaned = value.replace(/[^A-Za-z0-9_.:/-]/g, "_").slice(0, 128);
  return cleaned.includes("://") ? cleaned.replace(/:\/\//g, "_") : cleaned;
}

export function useTelemetry(state: StudyState | null) {
  const pageInstance = useRef(uuid());
  const sequence = useRef(0);
  const events = useRef<StudyEvent[]>([]);
  const flushing = useRef(false);
  const segment = useRef<{ stage: Stage; caseId: string | null; presentationId: string | null; start: number } | null>(null);
  const caseReadyAt = useRef<Record<string, number>>({});
  const stateRef = useRef(state);
  stateRef.current = state;

  const flushEvents = useCallback(async () => {
    if (flushing.current || !events.current.length) return;
    flushing.current = true;
    try {
      const batch = events.current.slice(0, 100);
      const ack = await sendJson<{ acknowledged_event_ids: string[] }>("POST", "/api/v1/study/events", { events: batch });
      const done = new Set(ack.acknowledged_event_ids);
      events.current = events.current.filter((event) => !done.has(event.event_id));
    } catch {
      // Keep a bounded backlog for retry; loss is visible server-side as sequence gaps.
      if (events.current.length > 500) events.current = events.current.slice(-500);
    } finally {
      flushing.current = false;
    }
  }, []);

  const emit = useCallback(
    (type: EventType, extra: { component?: string; element?: string; visibility?: "visible" | "hidden"; loadingMs?: number } = {}) => {
      const current = stateRef.current;
      if (!current?.current_case_id || !current.current_presentation_id) return;
      if (!["case", "consultation", "paused"].includes(current.stage)) return;
      events.current.push({
        event_id: uuid(),
        page_instance_id: pageInstance.current,
        sequence: sequence.current++,
        case_id: current.current_case_id,
        presentation_id: current.current_presentation_id,
        type,
        component_id: safeId(extra.component),
        element_id: safeId(extra.element),
        client_monotonic_ms: performance.now(),
        build_version: BUILD_VERSION,
        visibility: extra.visibility,
        loading_ms: extra.loadingMs,
      });
      if (type === "case_ready" || type === "submit" || events.current.length >= 20) void flushEvents();
    },
    [flushEvents],
  );

  const postSegment = useCallback(async (end: number) => {
    const open = segment.current;
    if (!open || end - open.start < 250) return;
    const body = {
      segment_id: uuid(),
      page_instance_id: pageInstance.current,
      stage: open.stage,
      case_id: open.caseId,
      presentation_id: open.presentationId,
      monotonic_start_ms: open.start,
      monotonic_end_ms: Math.min(end, open.start + 299_000),
    };
    segment.current = { ...open, start: body.monotonic_end_ms };
    try {
      await sendJson("POST", "/api/v1/study/timing", body);
    } catch {
      /* timing loss is recorded as a gap, never invented */
    }
  }, []);

  /** Close the running segment before a stage-changing save; never waits more than 1.5 s. */
  const flushTiming = useCallback(async () => {
    await Promise.race([Promise.all([postSegment(performance.now()), flushEvents()]), new Promise((resolve) => setTimeout(resolve, 1500))]);
  }, [flushEvents, postSegment]);

  const markCaseReady = useCallback(
    (loadingMs: number) => {
      const current = stateRef.current;
      if (!current?.current_case_id || current.stage !== "case") return;
      if (caseReadyAt.current[current.current_case_id] !== undefined) return;
      const now = performance.now();
      caseReadyAt.current[current.current_case_id] = now;
      emit("case_ready", { loadingMs });
      segment.current = { stage: "case", caseId: current.current_case_id, presentationId: current.current_presentation_id, start: now };
    },
    [emit],
  );

  // Start or switch segments when the server-owned stage changes.
  const stage = state?.stage;
  const caseId = state?.current_case_id ?? null;
  const presentationId = state?.current_presentation_id ?? null;
  useEffect(() => {
    if (!stage) return;
    const open = segment.current;
    if (open && (open.stage !== stage || open.caseId !== (stage === "case" || stage === "consultation" ? caseId : null))) {
      segment.current = null;
    }
    if (!segment.current && TIMED.includes(stage)) {
      // A case segment starts only when its content is usable (markCaseReady).
      if (stage === "case") return;
      segment.current = {
        stage,
        caseId: stage === "consultation" ? caseId : null,
        presentationId: stage === "consultation" ? presentationId : null,
        start: performance.now(),
      };
    }
  }, [stage, caseId, presentationId]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (segment.current && performance.now() - segment.current.start >= SEGMENT_MS) void postSegment(performance.now());
      void flushEvents();
    }, 15_000);
    const onVisibility = () => {
      const visibility = document.visibilityState === "hidden" ? "hidden" : "visible";
      emit("visibility", { visibility });
      if (visibility === "hidden") {
        void postSegment(performance.now());
        void flushEvents();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [emit, flushEvents, postSegment]);

  return { emit, flushTiming, markCaseReady, pageInstanceId: pageInstance.current };
}

export type Telemetry = ReturnType<typeof useTelemetry>;
