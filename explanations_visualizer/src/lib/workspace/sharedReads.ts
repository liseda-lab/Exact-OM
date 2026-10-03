// Shares one presentation's validated focal reads between the readiness loader and the
// workspace components, so the initial contexts, descriptions and comparisons are fetched
// once and the components render exactly what readiness checked. Reads are bound to the
// presentation: `close()` aborts them all, and a failed read is forgotten so a retry
// fetches only what failed. Any read that reports a lost session or policy is passed to
// `onInvalidated`; ordinary lazy failures stay local to the panel that made them.

import { ApiError } from "../api";
import type { EntityContext, EntityRef } from "../types";
import type { ExplanationResult, WorkspaceSource } from "./types";

const signature = (entity: EntityRef) => `${entity.ontology_version_id}\u0000${entity.kind}\u0000${entity.iri}`;

/** A lost session, revoked access or changed policy, as opposed to a local read failure. */
export function invalidates(error: unknown): boolean {
  if (!(error instanceof ApiError)) return false;
  if (error.status === 401 || error.status === 403) return true;
  return error.status === 409 && error.code !== "stale_cursor" && error.code !== "invalid_cursor";
}

function followAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(new DOMException("Aborted", "AbortError"));
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", abort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", abort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}

export interface SharedSource extends WorkspaceSource {
  /** Abort every outstanding read; the source must not be used afterwards. */
  close: () => void;
  /** Forget a validated-but-unusable result so the next read refetches it. */
  forget: (kind: "context" | "explanation", key: string) => void;
}

export function shareReads(source: WorkspaceSource, onInvalidated: (error: unknown) => void): SharedSource {
  const controller = new AbortController();
  const contexts = new Map<string, Promise<EntityContext>>();
  const explanations = new Map<string, Promise<ExplanationResult>>();
  const watch = <T,>(promise: Promise<T>): Promise<T> =>
    promise.catch((error) => {
      if (!controller.signal.aborted && invalidates(error)) onInvalidated(error);
      throw error;
    });
  const memo = <T,>(cache: Map<string, Promise<T>>, key: string, load: () => Promise<T>) => {
    let promise = cache.get(key);
    if (!promise) {
      promise = watch(load());
      cache.set(key, promise);
      promise.catch(() => {
        if (cache.get(key) === promise) cache.delete(key);
      });
    }
    return promise;
  };
  return {
    ...source,
    entityContext: (entity, signal) => followAbort(memo(contexts, signature(entity), () => source.entityContext(entity, controller.signal)), signal),
    explanation: (task, entities, signal) => followAbort(memo(explanations, `${task}|${entities.map(signature).join("|")}`, () => source.explanation(task, entities, controller.signal)), signal),
    fact: (ref, signal) => watch(source.fact(ref, signal)),
    hierarchy: (entity, direction, basis, cursor, signal) => watch(source.hierarchy(entity, direction, basis, cursor, signal)),
    search: (ontology, term, cursor, signal) => watch(source.search(ontology, term, cursor, signal)),
    evidence: (pair, signal) => watch(source.evidence(pair, signal)),
    facts: source.facts ? (entity, category, cursor, signal) => watch(source.facts!(entity, category, cursor, signal)) : undefined,
    close: () => controller.abort(),
    forget: (kind, key) => (kind === "context" ? contexts : explanations).delete(key),
  };
}

export const contextKey = (entity: EntityRef) => signature(entity);
export const explanationKey = (task: "entity_profile" | "pair_comparison", entities: EntityRef[]) => `${task}|${entities.map(signature).join("|")}`;
