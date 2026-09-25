"use client";

// Server-owned deployment facts for the exploration app: which profile is serving, which
// bundle is open, its ontologies and saved runs. The client never chooses its own profile.

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { ApiError, getJson } from "@/lib/api";
import type { Health, OntologyMeta, Page, RunSummary } from "@/lib/types";

export interface ExploreState {
  phase: "loading" | "ready" | "no_bundle" | "legacy" | "unreachable";
  health: Health | null;
  ontologies: Record<string, OntologyMeta>;
  runs: RunSummary[];
  error: string | null;
  reload: () => void;
}

const ExploreContext = createContext<ExploreState | null>(null);

async function allPages<T>(path: string, params: Record<string, string> = {}): Promise<T[]> {
  const items: T[] = [];
  let cursor: string | null = null;
  for (let guard = 0; guard < 20; guard += 1) {
    const page: Page<T> = await getJson<Page<T>>(path, { ...params, limit: 100, cursor });
    items.push(...page.items);
    if (!page.next_cursor) break;
    cursor = page.next_cursor;
  }
  return items;
}

export function ExploreProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<Omit<ExploreState, "reload">>({
    phase: "loading",
    health: null,
    ontologies: {},
    runs: [],
    error: null,
  });
  const [nonce, setNonce] = useState(0);
  const reload = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      let health: Health;
      try {
        health = await getJson<Health>("/api/v1/health");
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          // A historical run-directory server exposes only /api/health and the old viewer.
          try {
            await getJson("/api/health");
            if (!cancelled) {
              setState((prev) => ({ ...prev, phase: "legacy" }));
              window.location.replace(`/legacy/${window.location.search}`);
            }
            return;
          } catch {
            /* fall through to unreachable */
          }
        }
        if (!cancelled) setState((prev) => ({ ...prev, phase: "unreachable", error: "The Exact service is not responding." }));
        return;
      }
      if (health.status !== "available") {
        if (!cancelled) setState({ phase: "no_bundle", health, ontologies: {}, runs: [], error: null });
        return;
      }
      try {
        const [ontologies, runs] = await Promise.all([
          allPages<OntologyMeta>("/api/v1/ontologies"),
          allPages<RunSummary>("/api/v1/runs").catch(() => [] as RunSummary[]),
        ]);
        if (cancelled) return;
        setState({
          phase: "ready",
          health,
          ontologies: Object.fromEntries(ontologies.map((item) => [item.ontology_version_id, item])),
          runs,
          error: null,
        });
      } catch {
        if (!cancelled) setState((prev) => ({ ...prev, phase: "unreachable", health, error: "The bundle could not be read." }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [nonce]);

  const value = useMemo(() => ({ ...state, reload }), [state, reload]);
  return <ExploreContext.Provider value={value}>{children}</ExploreContext.Provider>;
}

export function useExplore(): ExploreState {
  const value = useContext(ExploreContext);
  if (!value) throw new Error("ExploreProvider is missing");
  return value;
}

/** Display name for an ontology: its declared name, else its role in the run. */
export function ontologyName(state: ExploreState, ontologyId: string | null | undefined, run?: RunSummary | null): string {
  if (!ontologyId) return "Ontology";
  const meta = state.ontologies[ontologyId];
  if (meta?.name) return meta.name;
  if (run?.source_ontology_version_id === ontologyId) return "Source ontology";
  if (run?.target_ontology_version_id === ontologyId) return "Target ontology";
  return "Ontology";
}
