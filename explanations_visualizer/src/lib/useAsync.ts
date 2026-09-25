"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { isAbort } from "@/lib/api";

export interface AsyncState<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

/**
 * Load data for a key; previous data stays visible while a new key loads so the layout
 * never collapses, and stale responses from an older key are ignored.
 */
export function useAsync<T>(key: string | null, loader: (signal: AbortSignal) => Promise<T>): AsyncState<T> {
  const [state, setState] = useState<{ key: string | null; data: T | undefined; error: unknown; loading: boolean }>({
    key: null,
    data: undefined,
    error: undefined,
    loading: Boolean(key),
  });
  const [nonce, setNonce] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    if (!key) {
      setState({ key: null, data: undefined, error: undefined, loading: false });
      return;
    }
    const controller = new AbortController();
    setState((prev) => ({ key, data: prev.key === key ? prev.data : undefined, error: undefined, loading: true }));
    loaderRef
      .current(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ key, data, error: undefined, loading: false });
      })
      .catch((error) => {
        if (isAbort(error) || controller.signal.aborted) return;
        setState({ key, data: undefined, error, loading: false });
      });
    return () => controller.abort();
  }, [key, nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);
  return { data: state.data, error: state.error, loading: state.loading, reload };
}
