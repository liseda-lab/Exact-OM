"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getJson, request } from "@/lib/api";
import { MutationQueue, type Mutation } from "@/study/mutationQueue";
import type { StudyState } from "@/study/types";

export type SaveStatus =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "saved"; at: number }
  | { kind: "offline"; pending: number }
  | { kind: "conflict"; message: string; at: number }
  | { kind: "error"; message: string };
export type Phase = "booting" | "no_link" | "link_unusable" | "rate_limited" | "ready" | "unreachable";

export function uuid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}
function read(key: string) { try { return window.sessionStorage.getItem(key); } catch { return null; } }
function write(key: string, value: string | null) {
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
  } catch { /* Only server-acknowledged data survives browser/storage loss. */ }
}

export function useStudySession() {
  const [phase, setPhase] = useState<Phase>("booting");
  const [state, setStateRaw] = useState<StudyState | null>(null);
  const [save, setSave] = useState<SaveStatus>({ kind: "idle" });
  const [online, setOnline] = useState(true);
  const [recovery, setRecovery] = useState<string | null>(null);
  const stateRef = useRef<StudyState | null>(null);
  const queue = useRef<MutationQueue | null>(null);
  const pendingInvite = useRef<string | null>(null);
  const bootGeneration = useRef(0);
  const [bootNonce, setBootNonce] = useState(0);

  const setState = useCallback((next: StudyState) => {
    const current = stateRef.current;
    if (current && (current.session_id !== next.session_id || next.revision < current.revision)) return;
    stateRef.current = next;
    queue.current?.observe(next);
    setStateRaw(next);
  }, []);

  const refresh = useCallback(async () => {
    const next = await getJson<StudyState>("/api/v1/study/state");
    if (stateRef.current && stateRef.current.session_id !== next.session_id) throw new Error("The active private link changed. Reopen this session's original link.");
    setState(next);
    return next;
  }, [setState]);

  useEffect(() => {
    const onHash = () => {
      if (!new URLSearchParams(window.location.hash.replace(/^#/, "")).get("invite")) return;
      queue.current?.stop();
      setPhase("booting");
      setBootNonce((value) => value + 1);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const generation = ++bootGeneration.current;
    const controller = new AbortController();
    queue.current?.stop();
    queue.current = null;
    stateRef.current = null;
    setStateRaw(null);
    setRecovery(null);
    setSave({ kind: "idle" });
    const secret = new URLSearchParams(window.location.hash.replace(/^#/, "")).get("invite");
    if (secret) {
      pendingInvite.current = secret;
      window.history.replaceState(null, "", window.location.pathname);
    }
    const invite = pendingInvite.current;
    (async () => {
      try {
        const next = invite
          ? await request<StudyState>("/api/v1/study/session", { method: "POST", signal: controller.signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ secret: invite }) })
          : await getJson<StudyState>("/api/v1/study/state", undefined, controller.signal);
        if (controller.signal.aborted || generation !== bootGeneration.current) return;
        pendingInvite.current = null;
        setState(next);
        setPhase("ready");
        const key = `exact.study.outbox.${next.session_id}`;
        const recoveryKey = `${key}.recovery`;
        setRecovery(read(recoveryKey));
        const outbox = new MutationQueue({
          state: next,
          snapshot: read(key),
          key: uuid,
          save: (snapshot) => write(key, snapshot),
          recover: (snapshot) => {
            const previous = read(recoveryKey);
            let attempts: unknown[] = [];
            let discarded = 0;
            try {
              const saved = previous ? JSON.parse(previous) : null;
              attempts = saved?.attempts ?? (saved ? [saved] : []);
              discarded = saved?.discarded_attempts ?? 0;
            } catch { /* Ignore an unreadable older recovery archive. */ }
            try { attempts.push(JSON.parse(snapshot)); } catch { attempts.push({ unreadable_snapshot: true }); }
            while (attempts.length > 8 || (attempts.length > 1 && JSON.stringify(attempts).length > 1024 * 1024)) { attempts.shift(); discarded += 1; }
            const archive = JSON.stringify({ attempts, discarded_attempts: discarded });
            write(recoveryKey, archive);
            setRecovery(archive);
          },
          send: (entry, signal) => request<StudyState>(entry.path, {
            method: entry.method, signal,
            headers: { "Content-Type": "application/json", Accept: "application/json", "X-Study-Session": next.session_id },
            body: JSON.stringify({ ...entry.body, idempotency_key: entry.key, expected_revision: entry.expected }),
          }),
          update: (value) => setState(value as StudyState),
          status: (kind, pending) => {
            setOnline(kind !== "offline");
            setSave(kind === "offline" ? { kind, pending } : kind === "saved" ? { kind, at: Date.now() } : { kind });
          },
          retryable: (error) => error instanceof ApiError && (error.status === 0 || error.status >= 500 || error.status === 429),
          failed: async (error) => {
            if (error instanceof ApiError && error.status === 409) {
              let restored = false;
              try { await refresh(); restored = true; } catch { /* Recovery remains available. */ }
              setSave({ kind: "conflict", message: restored ? "Your study changed in another tab or device. The latest saved version is shown. Unsaved changes are available below for recovery." : "Your study changed in another tab or device. The latest version could not be loaded. Reconnect and reload; unsaved changes are available below for recovery.", at: Date.now() });
            } else if (error instanceof ApiError && error.status === 401) setPhase("no_link");
            else setSave({ kind: "error", message: error instanceof Error ? error.message : "The change could not be saved." });
          },
        });
        queue.current = outbox;
        void outbox.start();
      } catch (error) {
        if (controller.signal.aborted || generation !== bootGeneration.current) return;
        if (error instanceof ApiError && error.status === 401) setPhase(invite ? "link_unusable" : "no_link");
        else if (error instanceof ApiError && error.status === 429) setPhase("rate_limited");
        else if (error instanceof ApiError && error.status > 0 && error.status < 500) setPhase(invite ? "link_unusable" : "no_link");
        else setPhase("unreachable");
      }
    })();
    return () => { controller.abort(); queue.current?.stop(); };
  }, [bootNonce, refresh, setState]);

  const mutate = useCallback((mutation: Mutation): Promise<StudyState> => {
    if (!queue.current) return Promise.reject(new Error("Open the study session before saving."));
    return queue.current.enqueue(mutation).catch((error) => {
      if (error instanceof Error && /limit|Too many changes/.test(error.message)) setSave({ kind: "error", message: error.message });
      throw error;
    }) as Promise<StudyState>;
  }, []);

  useEffect(() => {
    const up = () => { setOnline(true); queue.current?.retry(); };
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    return () => { window.removeEventListener("online", up); window.removeEventListener("offline", down); };
  }, []);

  const retryNow = useCallback(() => queue.current?.retry(), []);
  const clearNotice = useCallback(() => setSave({ kind: "idle" }), []);
  const downloadRecovery = useCallback(() => {
    if (!recovery) return;
    const url = URL.createObjectURL(new Blob([recovery], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "unsaved-study-changes.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [recovery]);
  return { phase, state, save, online, mutate, refresh, retryNow, clearNotice, pendingCount: () => queue.current?.size ?? 0, hasRecovery: recovery !== null, downloadRecovery };
}
export type StudySession = ReturnType<typeof useStudySession>;
