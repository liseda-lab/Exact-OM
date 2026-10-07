"use client";

// One displayed page plus its continuations (19 F16). The base page comes from the entity
// context; later pages are read only on request through the active workspace source, so the
// main app, the study and the tutorial continue the same way. Items keep the service's order
// and identities; a failure stays local, keeps what is shown and never reads as absence.

import { useCallback, useEffect, useRef, useState } from "react";

import { appendPage, isComplete } from "@/lib/continuation";

export interface Continued<T> {
  items: T[];
  total: number | null;
  hasMore: boolean;
  complete: boolean;
  /** Whether the active source can read further pages. */
  available: boolean;
  loading: boolean;
  error: unknown;
  loadMore: () => void;
}

interface BasePage<T> {
  items: T[];
  next_cursor: string | null;
  total_count: number | null;
}

export function useContinuation<T>(base: BasePage<T> | null | undefined, load: ((cursor: string, signal: AbortSignal) => Promise<{ items: T[]; next_cursor: string | null }>) | null, key: (item: T) => string): Continued<T> {
  const [state, setState] = useState<{ base: BasePage<T>; items: T[]; cursor: string | null; loading: boolean; error: unknown } | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  // A new base page (another entity, scope or reload) starts again from that page.
  useEffect(() => {
    controller.current?.abort();
  }, [base]);
  const current = state && base && state.base === base ? state : null;
  const items = current ? current.items : base?.items ?? [];
  const cursor = current ? current.cursor : base?.next_cursor ?? null;
  const total = base?.total_count ?? null;
  const keyRef = useRef(key);
  keyRef.current = key;

  const loadMore = useCallback(() => {
    if (!base || !cursor || !load || current?.loading) return;
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    const before = items;
    setState({ base, items: before, cursor, loading: true, error: null });
    load(cursor, abort.signal).then(
      (page) => {
        if (abort.signal.aborted) return;
        setState({ base, items: appendPage(before, page.items, keyRef.current), cursor: page.next_cursor, loading: false, error: null });
      },
      (error) => {
        if (abort.signal.aborted) return;
        setState({ base, items: before, cursor, loading: false, error });
      },
    );
  }, [base, cursor, load, current?.loading, items]);

  const hasMore = Boolean(cursor);
  return { items, total, hasMore, complete: isComplete(items.length, total, hasMore), available: Boolean(load), loading: Boolean(current?.loading), error: current?.error ?? null, loadMore };
}
