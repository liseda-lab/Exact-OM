"use client";

// Batched display-label cache over GET /api/v1/labels. Labels are presentation only:
// a missing label is shown as such and the IRI stays visible, never replaced by a guess.

import { useEffect, useSyncExternalStore } from "react";

import { getJson } from "@/lib/api";
import type { LabelItem } from "@/lib/types";

export interface LabelEntry {
  status: "loading" | "available" | "absent" | "failed" | "not_included";
  value: string | null;
}

const cache = new Map<string, LabelEntry>();
const queue = new Map<string, Set<string>>();
const listeners = new Set<() => void>();
let version = 0;
let timer: ReturnType<typeof setTimeout> | null = null;

function key(ontology: string, iri: string): string {
  return `${ontology}\u0000${iri}`;
}

function notify() {
  version += 1;
  listeners.forEach((listener) => listener());
}

async function flush() {
  timer = null;
  const batches = Array.from(queue.entries());
  queue.clear();
  for (const [ontology, set] of batches) {
    const iris = Array.from(set);
    for (let start = 0; start < iris.length; start += 100) {
      const chunk = iris.slice(start, start + 100);
      try {
        const page = await getJson<{ items: LabelItem[] }>("/api/v1/labels", { ontology_version_id: ontology, iri: chunk });
        const seen = new Set<string>();
        for (const item of page.items) {
          const iri = item.entity?.iri ?? item.iri;
          if (!iri) continue;
          const label = item.preferred_label;
          // Prefer a class label when one IRI is punned across kinds.
          if (seen.has(iri) && item.entity?.kind !== "class") continue;
          seen.add(iri);
          cache.set(key(ontology, iri), label.value ? { status: "available", value: label.value } : { status: "absent", value: null });
        }
        chunk.forEach((iri) => {
          if (cache.get(key(ontology, iri))?.status === "loading") cache.set(key(ontology, iri), { status: "absent", value: null });
        });
      } catch {
        chunk.forEach((iri) => cache.set(key(ontology, iri), { status: "failed", value: null }));
      }
      notify();
    }
  }
}

export function requestLabels(ontology: string | null | undefined, iris: string[]) {
  if (!ontology) return;
  let queued = false;
  for (const iri of iris) {
    if (!iri || cache.has(key(ontology, iri))) continue;
    cache.set(key(ontology, iri), { status: "loading", value: null });
    if (!queue.has(ontology)) queue.set(ontology, new Set());
    queue.get(ontology)!.add(iri);
    queued = true;
  }
  if (queued && !timer) timer = setTimeout(flush, 12);
  if (queued) notify();
}

/** Seed labels the page already knows (e.g. from an entity context) to avoid refetching. */
export function seedLabel(ontology: string, iri: string, value: string | null) {
  if (!cache.has(key(ontology, iri)) || cache.get(key(ontology, iri))?.status !== "available") {
    cache.set(key(ontology, iri), value ? { status: "available", value } : { status: "absent", value: null });
    notify();
  }
}

export function peekLabel(ontology: string | null | undefined, iri: string): LabelEntry | undefined {
  if (!ontology) return undefined;
  return cache.get(key(ontology, iri));
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useLabelVersion(): number {
  return useSyncExternalStore(subscribe, () => version, () => 0);
}

/** Returns a lookup for labels in one ontology, fetching any that are missing. */
export function useLabels(ontology: string | null | undefined, iris: string[]): (iri: string) => LabelEntry | undefined {
  useLabelVersion();
  const signature = iris.join("\n");
  useEffect(() => {
    requestLabels(ontology, signature ? signature.split("\n") : []);
  }, [ontology, signature]);
  return (iri: string) => peekLabel(ontology, iri);
}
