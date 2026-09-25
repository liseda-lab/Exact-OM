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
 * Load data for a key, retaining data only when reloading that same key.
 * A changed key must never expose data or errors from the previous selection.
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
  return state.key === key
    ? { data: state.data, error: state.error, loading: state.loading, reload }
    : { data: undefined, error: undefined, loading: Boolean(key), reload };
}
