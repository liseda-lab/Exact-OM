"use client";

// Navigation state lives in the URL (source, candidate, focused entities, open tab) so
// reloads, resizes, back/forward and shared links preserve exactly what was selected.

import { useCallback, useSyncExternalStore } from "react";

const EVENT = "exact:urlchange";

function subscribe(listener: () => void) {
  window.addEventListener("popstate", listener);
  window.addEventListener(EVENT, listener);
  return () => {
    window.removeEventListener("popstate", listener);
    window.removeEventListener(EVENT, listener);
  };
}

export function useUrlState(): [URLSearchParams, (updates: Record<string, string | null | undefined>, mode?: "push" | "replace") => void] {
  const search = useSyncExternalStore(
    subscribe,
    () => window.location.search,
    () => "",
  );
  const update = useCallback((updates: Record<string, string | null | undefined>, mode: "push" | "replace" = "push") => {
    const params = new URLSearchParams(window.location.search);
    for (const [key, value] of Object.entries(updates)) {
      if (value === null || value === undefined || value === "") params.delete(key);
      else params.set(key, value);
    }
    const query = params.toString();
    const next = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
    if (next === `${window.location.pathname}${window.location.search}${window.location.hash}`) return;
    if (mode === "push") window.history.pushState(null, "", next);
    else window.history.replaceState(null, "", next);
    window.dispatchEvent(new Event(EVENT));
  }, []);
  return [new URLSearchParams(search), update];
}
