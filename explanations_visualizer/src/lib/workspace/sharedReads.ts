// Shares one presentation's validated focal reads between the readiness loader and the
// workspace components, so the initial contexts, descriptions and comparisons are fetched
// once and the components render exactly what readiness checked. Reads are bound to the
// presentation: `close()` aborts them all.
//
// A focal response is cached only after its structure validates (19 F18): an unusable 200
// response becomes a local failure for every consumer and is evicted, so the next read or a
// Retry fetches it again. `reset()` evicts everything when a render failure leaves no way to
// tell which cached response caused it. A read that may mean a lost session or policy is
// passed to `onSuspect`; the shell decides whether it really invalidates the case.

import { ApiError } from "../api";
import type { EntityContext, EntityRef } from "../types";
import type { ExplanationResult, WorkspaceSource } from "./types";

const signature = (entity: EntityRef) => `${entity.ontology_version_id}\u0000${entity.kind}\u0000${entity.iri}`;

/** A 200 response that is unusable for the view that requested it. */
export class InvalidResponseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidResponseError";
  }
}

/** A possible lost session, revoked access or changed policy (not a stale cursor). */
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
  /** Forget one cached result so the next read refetches it. */
  forget: (kind: "context" | "explanation", key: string) => void;
  /** Forget every cached focal result (after a render failure of unknown cause). */
  reset: () => void;
}

export interface SharedOptions {
  onSuspect: (error: unknown) => void;
  /** Problems that make a context unusable, or null. */
  validateContext?: (entity: EntityRef, value: EntityContext) => string | null;
  /** Problems that make an explanation result unusable, or null. */
  validateExplanation?: (task: "entity_profile" | "pair_comparison", entities: EntityRef[], value: ExplanationResult) => string | null;
}

export function shareReads(source: WorkspaceSource, options: SharedOptions | ((error: unknown) => void)): SharedSource {
  const settings: SharedOptions = typeof options === "function" ? { onSuspect: options } : options;
  const controller = new AbortController();
  const contexts = new Map<string, Promise<EntityContext>>();
  const explanations = new Map<string, Promise<ExplanationResult>>();
  const watch = <T,>(promise: Promise<T>): Promise<T> =>
    promise.catch((error) => {
      if (!controller.signal.aborted && invalidates(error)) settings.onSuspect(error);
      throw error;
    });
  const memo = <T,>(cache: Map<string, Promise<T>>, key: string, load: () => Promise<T>, validate: (value: T) => string | null) => {
    let promise = cache.get(key);
    if (!promise) {
      promise = watch(load()).then((value) => {
        const problem = validate(value);
        if (problem) throw new InvalidResponseError(problem);
        return value;
      });
      cache.set(key, promise);
      const settled = promise;
      settled.catch(() => {
        if (cache.get(key) === settled) cache.delete(key);
      });
    }
    return promise;
  };
  return {
    ...source,
    entityContext: (entity, signal) => followAbort(memo(contexts, signature(entity), () => source.entityContext(entity, controller.signal), (value) => settings.validateContext?.(entity, value) ?? null), signal),
    explanation: (task, entities, signal) =>
      followAbort(memo(explanations, `${task}|${entities.map(signature).join("|")}`, () => source.explanation(task, entities, controller.signal), (value) => settings.validateExplanation?.(task, entities, value) ?? null), signal),
    fact: (ref, signal) => watch(source.fact(ref, signal)),
    hierarchy: (entity, direction, basis, cursor, signal) => watch(source.hierarchy(entity, direction, basis, cursor, signal)),
    search: (ontology, term, cursor, signal) => watch(source.search(ontology, term, cursor, signal)),
    evidence: (pair, signal) => watch(source.evidence(pair, signal)),
    facts: source.facts ? (entity, category, cursor, signal) => watch(source.facts!(entity, category, cursor, signal)) : undefined,
    close: () => controller.abort(),
    forget: (kind, key) => (kind === "context" ? contexts : explanations).delete(key),
    reset: () => {
      contexts.clear();
      explanations.clear();
    },
  };
}

export const contextKey = (entity: EntityRef) => signature(entity);
export const explanationKey = (task: "entity_profile" | "pair_comparison", entities: EntityRef[]) => `${task}|${entities.map(signature).join("|")}`;
