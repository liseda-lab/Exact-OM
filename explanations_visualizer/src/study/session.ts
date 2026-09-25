"use client";

// Participant session: the server is the only authority for stage, answers and revision.
// Mutations run one at a time, each with an idempotency key and the revision it expects.
// A lost acknowledgement is retried with the same key (the server replays it); a stale
// revision is a visible conflict, never a silent overwrite. Only the latest unsent draft
// is kept locally, so a reload can resend it.

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getJson, sendJson } from "@/lib/api";
import type { StudyState } from "@/study/types";

export type SaveStatus =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "saved"; at: number }
  | { kind: "offline"; pending: number }
  | { kind: "conflict"; message: string; at: number }
  | { kind: "error"; message: string };

export type Phase = "booting" | "no_link" | "link_unusable" | "rate_limited" | "ready" | "unreachable";

interface Mutation {
  method: "PUT" | "POST";
  path: string;
  body: Record<string, unknown>;
  /** Mutations with the same coalesce key replace each other while still unsent. */
  coalesce?: string;
  persist?: boolean;
}

interface Entry extends Mutation {
  key: string;
  expected: number | null;
  resolve: (state: StudyState) => void;
  reject: (error: unknown) => void;
}

const PENDING_KEY = "exact.study.pending";

export function uuid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

function storePending(sessionId: string, entry: Entry | null) {
  try {
    if (!entry) window.localStorage.removeItem(`${PENDING_KEY}.${sessionId}`);
    else
      window.localStorage.setItem(
        `${PENDING_KEY}.${sessionId}`,
        JSON.stringify({ method: entry.method, path: entry.path, body: entry.body, key: entry.key, expected: entry.expected, coalesce: entry.coalesce }),
      );
  } catch {
    /* browser storage is only a convenience; server-acknowledged data is the guarantee */
  }
}

function readPending(sessionId: string): (Mutation & { key: string; expected: number | null }) | null {
  try {
    const raw = window.localStorage.getItem(`${PENDING_KEY}.${sessionId}`);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function useStudySession() {
  const [phase, setPhase] = useState<Phase>("booting");
  const [state, setStateRaw] = useState<StudyState | null>(null);
  const [save, setSave] = useState<SaveStatus>({ kind: "idle" });
  const [online, setOnline] = useState(true);
  const stateRef = useRef<StudyState | null>(null);
  const queue = useRef<Entry[]>([]);
  const running = useRef(false);
  const wake = useRef<(() => void) | null>(null);
  const [bootNonce, setBootNonce] = useState(0);

  // A link pasted into a tab already on this page only changes the fragment: exchange it too.
  useEffect(() => {
    const onHash = () => {
      if (new URLSearchParams(window.location.hash.replace(/^#/, "")).get("invite")) {
        setPhase("booting");
        setBootNonce((value) => value + 1);
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const setState = useCallback((next: StudyState) => {
    const current = stateRef.current;
    if (current && current.session_id === next.session_id && next.revision < current.revision) return;
    stateRef.current = next;
    setStateRaw(next);
  }, []);

  const refresh = useCallback(async () => {
    const next = await getJson<StudyState>("/api/v1/study/state");
    stateRef.current = next;
    setStateRaw(next);
    return next;
  }, []);

  const pump = useCallback(async () => {
    if (running.current) return;
    running.current = true;
    try {
      while (queue.current.length) {
        const entry = queue.current[0];
        if (entry.expected === null) entry.expected = stateRef.current?.revision ?? 0;
        if (entry.persist && stateRef.current) storePending(stateRef.current.session_id, entry);
        setSave({ kind: "saving" });
        let delay = 1500;
        for (;;) {
          try {
            const next = await sendJson<StudyState>(entry.method, entry.path, { ...entry.body, idempotency_key: entry.key, expected_revision: entry.expected });
            queue.current.shift();
            if (entry.persist && stateRef.current) storePending(stateRef.current.session_id, null);
            setState(next);
            setOnline(true);
            setSave({ kind: "saved", at: Date.now() });
            entry.resolve(next);
            break;
          } catch (error) {
            const transient = error instanceof ApiError && (error.status === 0 || error.status >= 500 || error.status === 429);
            if (transient) {
              setOnline(false);
              setSave({ kind: "offline", pending: queue.current.length });
              await new Promise<void>((resolve) => {
                const timer = window.setTimeout(resolve, delay);
                wake.current = () => {
                  window.clearTimeout(timer);
                  resolve();
                };
              });
              wake.current = null;
              delay = Math.min(30000, delay * 2);
              continue;
            }
            queue.current.shift();
            if (entry.persist && stateRef.current) storePending(stateRef.current.session_id, null);
            if (error instanceof ApiError && error.status === 409) {
              // Load the authoritative state first so screens can re-sync to it.
              try {
                await refresh();
              } catch {
                /* the banner still explains what happened */
              }
              setSave({ kind: "conflict", message: "Your study changed in another tab or on another device. The latest saved version is shown.", at: Date.now() });
            } else if (error instanceof ApiError && error.status === 401) {
              setPhase("no_link");
            } else {
              setSave({ kind: "error", message: error instanceof ApiError ? error.message : "The change could not be saved." });
            }
            entry.reject(error);
            // Later entries were built on the rejected revision; let callers rebuild them.
            const dropped = queue.current.splice(0);
            dropped.forEach((item) => item.reject(error));
            break;
          }
        }
      }
    } finally {
      running.current = false;
    }
  }, [refresh, setState]);

  const mutate = useCallback(
    (mutation: Mutation): Promise<StudyState> =>
      new Promise((resolve, reject) => {
        if (mutation.coalesce) {
          // Replace an unsent (not in-flight) entry for the same thing, e.g. a draft ranking.
          const index = queue.current.findIndex((entry, position) => position > 0 && entry.coalesce === mutation.coalesce);
          if (index > 0) {
            const old = queue.current[index];
            queue.current[index] = {
              ...mutation,
              key: uuid(),
              expected: null,
              resolve: (next) => {
                old.resolve(next);
                resolve(next);
              },
              reject: (error) => {
                old.reject(error);
                reject(error);
              },
            };
            return;
          }
        }
        queue.current.push({ ...mutation, key: uuid(), expected: null, resolve, reject });
        void pump();
      }),
    [pump],
  );

  // Boot: exchange a fragment invitation for a session cookie, or resume with the cookie.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
      const secret = hash.get("invite");
      try {
        if (secret) {
          // Remove the bearer secret from the address bar and history before anything else.
          window.history.replaceState(null, "", window.location.pathname);
          const next = await sendJson<StudyState>("POST", "/api/v1/study/session", { secret });
          if (cancelled) return;
          stateRef.current = next;
          setStateRaw(next);
        } else {
          await refresh();
        }
        if (cancelled) return;
        setPhase("ready");
        // Resend a draft that never reached the server (same key: safe to replay).
        const pending = stateRef.current ? readPending(stateRef.current.session_id) : null;
        if (pending) {
          queue.current.push({
            ...pending,
            resolve: () => undefined,
            reject: () => undefined,
          });
          void pump();
        }
      } catch (error) {
        if (cancelled) return;
        if (secret && error instanceof ApiError && error.status === 401) setPhase("link_unusable");
        else if (error instanceof ApiError && error.status === 401) setPhase("no_link");
        else if (error instanceof ApiError && error.status === 429) setPhase("rate_limited");
        else if (error instanceof ApiError && error.status > 0) setPhase(secret ? "link_unusable" : "no_link");
        else setPhase("unreachable");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pump, refresh, bootNonce]);

  useEffect(() => {
    const up = () => {
      setOnline(true);
      wake.current?.();
    };
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    return () => {
      window.removeEventListener("online", up);
      window.removeEventListener("offline", down);
    };
  }, []);

  const retryNow = useCallback(() => wake.current?.(), []);
  const clearNotice = useCallback(() => setSave({ kind: "idle" }), []);

  return { phase, state, save, online, mutate, refresh, retryNow, clearNotice, pendingCount: () => queue.current.length };
}

export type StudySession = ReturnType<typeof useStudySession>;
