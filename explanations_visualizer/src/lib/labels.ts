"use client";

// Batched display-label cache over GET /api/v1/labels. Labels are presentation only:
// a missing label is shown as such and the IRI stays visible, never replaced by a guess.

import { useEffect, useSyncExternalStore } from "react";

import { buildUrl, request } from "@/lib/api";
import type { LabelItem } from "@/lib/types";

export interface LabelEntry {
  status: "loading" | "available" | "absent" | "failed" | "not_included";
  value: string | null;
}

export interface RemoteLabelSource {
  /** Session/publication/presentation namespace, independent of an opaque route locator. */
  key: string;
  base: string;
  sessionId?: string;
}
const defaultSource: RemoteLabelSource = { key: "exploration", base: "/api/v1" };
const sources = new Map<string, { source: RemoteLabelSource; ontology: string }>();

const cache = new Map<string, LabelEntry>();
const queue = new Map<string, Set<string>>();
const listeners = new Set<() => void>();
let version = 0;
let timer: ReturnType<typeof setTimeout> | null = null;

function key(ontology: string, iri: string, source: RemoteLabelSource = defaultSource): string {
  return `${source.key}\u0000${source.base}\u0000${source.sessionId ?? ""}\u0000${ontology}\u0000${iri}`;
}

function notify() {
  version += 1;
  listeners.forEach((listener) => listener());
}

async function flush() {
  timer = null;
  const batches = Array.from(queue.entries());
  queue.clear();
  for (const [batchKey, set] of batches) {
    const { source, ontology } = sources.get(batchKey)!;
    const iris = Array.from(set);
    for (let start = 0; start < iris.length; start += 100) {
      const chunk = iris.slice(start, start + 100);
      try {
        const page = await request<{ items: LabelItem[] }>(buildUrl(`${source.base}/labels`, { ontology_version_id: ontology, iri: chunk }), { headers: { Accept: "application/json", ...(source.sessionId ? { "X-Study-Session": source.sessionId } : {}) } });
        const seen = new Set<string>();
        for (const item of page.items) {
          const iri = item.entity?.iri ?? item.iri;
          if (!iri) continue;
          const label = item.preferred_label;
          // Prefer a class label when one IRI is punned across kinds.
          if (seen.has(iri) && item.entity?.kind !== "class") continue;
          seen.add(iri);
          cache.set(key(ontology, iri, source), label.value ? { status: "available", value: label.value } : { status: label.status === "not_exported" ? "not_included" : "absent", value: null });
        }
        chunk.forEach((iri) => {
          if (cache.get(key(ontology, iri, source))?.status === "loading") cache.set(key(ontology, iri, source), { status: "absent", value: null });
        });
      } catch {
        chunk.forEach((iri) => cache.set(key(ontology, iri, source), { status: "failed", value: null }));
      }
      notify();
    }
  }
}

export function requestLabels(ontology: string | null | undefined, iris: string[], source: RemoteLabelSource = defaultSource) {
  if (!ontology) return;
  const batchKey = key(ontology, "", source);
  sources.set(batchKey, { source, ontology });
  let queued = false;
  for (const iri of iris) {
    if (!iri || cache.has(key(ontology, iri, source))) continue;
    cache.set(key(ontology, iri, source), { status: "loading", value: null });
    if (!queue.has(batchKey)) queue.set(batchKey, new Set());
    queue.get(batchKey)!.add(iri);
    queued = true;
  }
  if (queued && !timer) timer = setTimeout(flush, 12);
  if (queued) notify();
}

/** Seed labels the page already knows (e.g. from an entity context) to avoid refetching. */
export function seedLabel(ontology: string, iri: string, value: string | null, source: RemoteLabelSource = defaultSource) {
  if (!cache.has(key(ontology, iri, source)) || cache.get(key(ontology, iri, source))?.status !== "available") {
    cache.set(key(ontology, iri, source), value ? { status: "available", value } : { status: "absent", value: null });
    notify();
  }
}

export function peekLabel(ontology: string | null | undefined, iri: string, source: RemoteLabelSource = defaultSource): LabelEntry | undefined {
  if (!ontology) return undefined;
  return cache.get(key(ontology, iri, source));
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useLabelVersion(): number {
  return useSyncExternalStore(subscribe, () => version, () => 0);
}

/** Returns a lookup for labels in one ontology, fetching any that are missing. */
export function useLabels(ontology: string | null | undefined, iris: string[], source: RemoteLabelSource = defaultSource): (iri: string) => LabelEntry | undefined {
  useLabelVersion();
  const signature = iris.join("\n");
  useEffect(() => {
    requestLabels(ontology, signature ? signature.split("\n") : [], source);
  }, [ontology, signature, source]);
  return (iri: string) => peekLabel(ontology, iri, source);
}
