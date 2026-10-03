"use client";

// Loads and validates one scored presentation, then reports its readiness (19 F11/F12):
//   loading → (required content validated) → rendered → usable
//   any required failure, incompatible service or lost session → blocked (retry keeps work)
// v2 explanation cases read only their authorized scope through the shared API adapter;
// a missing descriptor or incompatible capabilities block the case instead of selecting
// another adapter. v1 keeps its explicitly selected legacy resource adapter. Baseline cases
// make no workspace request. The focal reads are shared with the rendered workspace.

import { Component, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { ApiError, describeError, getJson, isAbort } from "@/lib/api";
import { createApiSource, scopedRead } from "@/lib/workspace/apiSource";
import { indexResources } from "@/lib/workspace/resourceIndex";
import { createResourceSource } from "@/lib/workspace/resourceSource";
import { contextKey, explanationKey, shareReads, type SharedSource } from "@/lib/workspace/sharedReads";
import type { WorkspaceSource } from "@/lib/workspace/types";
import { useEntityContext, useExplanation } from "@/lib/workspace/WorkspaceContext";
import type { EntityRef } from "@/lib/types";
import { contextBindingProblem, explanationProblem, inBatches, requiredContent, validateCapabilities, validateCase, workspaceCapabilities, type CaseComponent, type ScopeCapabilities } from "@/study/caseReadiness";
import type { ExplanationResource, StudyCase, StudyState } from "@/study/types";

export interface ContentFailure {
  label: string;
  detail: string;
}

export type CaseReadiness =
  | { kind: "loading"; done: number; total: number }
  | { kind: "rendering" }
  | { kind: "usable" }
  | { kind: "blocked"; cause: "service" | "content" | "session" | "render"; title: string; failures: ContentFailure[] };

const CONCURRENCY = 4;

function serviceFailure(error: unknown, what: string): CaseReadiness {
  if (error instanceof ApiError && error.status === 503)
    return { kind: "blocked", cause: "service", title: "The study service could not provide this case right now.", failures: [{ label: what, detail: error.message }] };
  if (error instanceof ApiError && (error.status === 401 || error.status === 403 || error.status === 409))
    return { kind: "blocked", cause: "session", title: "This page no longer has access to the case.", failures: [{ label: what, detail: error.message }] };
  return { kind: "blocked", cause: "content", title: "Part of this case could not be loaded.", failures: [{ label: what, detail: describeError(error) }] };
}

export function useCaseContent({ state, components, ontologyName, caseLabels }: { state: StudyState; components: Set<CaseComponent>; ontologyName: (id: string) => string; caseLabels: (current: StudyCase) => Record<string, string> }) {
  const binding = `${state.session_id}|${state.study_revision}|${state.current_case_id}|${state.current_presentation_id}`;
  const [studyCase, setStudyCase] = useState<StudyCase | null>(null);
  const [source, setSource] = useState<WorkspaceSource | null>(null);
  const [readiness, setReadiness] = useState<CaseReadiness>({ kind: "loading", done: 0, total: 1 });
  const [attempt, setAttempt] = useState(0);
  const shared = useRef<{ key: string; source: SharedSource } | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  const namesRef = useRef(ontologyName);
  namesRef.current = ontologyName;
  const labelsRef = useRef(caseLabels);
  labelsRef.current = caseLabels;
  const componentsRef = useRef(components);
  componentsRef.current = components;

  // A new presentation or session discards everything from the previous one.
  useEffect(() => {
    setStudyCase(null);
    setSource(null);
    setReadiness({ kind: "loading", done: 0, total: 1 });
    return () => {
      shared.current?.source.close();
      shared.current = null;
    };
  }, [binding]);

  useEffect(() => {
    const controller = new AbortController();
    const live = () => !controller.signal.aborted;
    const block = (next: CaseReadiness) => live() && setReadiness(next);
    const session = stateRef.current.session_id;
    (async () => {
      setReadiness((current) => (current.kind === "usable" ? current : { kind: "loading", done: 0, total: 1 }));
      let current: StudyCase;
      try {
        current = await scopedRead<StudyCase>("/api/v1/study", "/cases/current", undefined, controller.signal, session);
      } catch (error) {
        if (!isAbort(error)) block(serviceFailure(error, "The case"));
        return;
      }
      if (!live()) return;
      const { reject, block: blocking } = validateCase(stateRef.current, current);
      if (reject.length) {
        setStudyCase(null);
        block({ kind: "blocked", cause: "service", title: "The study service sent a case that does not match this session.", failures: reject.map((detail) => ({ label: "The case", detail })) });
        return;
      }
      setStudyCase(current);
      if (blocking.length) {
        block({ kind: "blocked", cause: "service", title: "This case cannot be shown with this version of the study service.", failures: blocking.map((detail) => ({ label: "The case", detail })) });
        return;
      }
      const v2 = (current.contract_version ?? "exact-study/1.0") === "exact-study/2.0";
      if (current.condition !== "explanation") {
        setReadiness({ kind: "rendering" });
        return;
      }
      if (!v2) {
        // Frozen v1 publications: the explicitly selected legacy adapter over frozen resources.
        try {
          const loaded = await Promise.all(current.explanation_refs.map((ref) => getJson<ExplanationResource>(`/api/v1/study/resources/${encodeURIComponent(ref)}`, undefined, controller.signal)));
          if (!live()) return;
          setSource(
            createResourceSource({
              key: `study|${session}|${current.study_revision}|${current.presentation_id}`,
              kind: "study_resource",
              index: indexResources(loaded, { extraLabels: labelsRef.current(current) }),
              navigation: "prepared",
              scopeNote: "Only the information prepared for this case is available here.",
              ontologyName: (id) => namesRef.current(id),
            }),
          );
          setReadiness({ kind: "rendering" });
        } catch (error) {
          if (!isAbort(error)) block(serviceFailure(error, "The prepared explanations"));
        }
        return;
      }
      const scope = current.workspace!.scope_id;
      const base = `/api/v1/study/workspace/${encodeURIComponent(scope)}`;
      let caps: ScopeCapabilities;
      try {
        caps = await scopedRead<ScopeCapabilities>(base, "/capabilities", undefined, controller.signal, session);
      } catch (error) {
        if (!isAbort(error)) block(serviceFailure(error, "The case's workspace"));
        return;
      }
      if (!live()) return;
      const incompatible = validateCapabilities(caps, { scopeId: scope, studyRevision: stateRef.current.study_revision, current, components: componentsRef.current });
      if (incompatible.length) {
        block({ kind: "blocked", cause: "service", title: "The study service described a workspace that does not match this case.", failures: incompatible.map((detail) => ({ label: "The case's workspace", detail })) });
        return;
      }
      const key = `study|${session}|${current.study_revision}|${current.presentation_id}|${scope}|${caps.policy_hash}`;
      if (shared.current?.key !== key) {
        shared.current?.source.close();
        const api = createApiSource({ key, base, sessionId: session, kind: "study_resource", capabilities: workspaceCapabilities(caps, (id) => namesRef.current(id)), ontologyName: (id) => namesRef.current(id) });
        shared.current = {
          key,
          source: shareReads(api, (error) =>
            setReadiness({ kind: "blocked", cause: "session", title: "This page no longer has access to the case.", failures: [{ label: "Your session", detail: error instanceof Error ? error.message : "Access changed." }] }),
          ),
        };
      }
      const reads = shared.current.source;
      const required = requiredContent(current, componentsRef.current, caps);
      let done = 0;
      setReadiness((value) => (value.kind === "usable" ? value : { kind: "loading", done, total: required.length }));
      const results = await inBatches(required, CONCURRENCY, async (item) => {
        if (item.kind === "context") {
          const ctx = await reads.entityContext(item.entities[0], controller.signal);
          const problem = contextBindingProblem(ctx, item.entities[0]);
          if (problem) {
            reads.forget("context", contextKey(item.entities[0]));
            throw new Error(problem);
          }
        } else {
          const task = item.kind === "profile" ? "entity_profile" : "pair_comparison";
          const result = await reads.explanation(task, item.entities, controller.signal);
          const problem = explanationProblem(result, task, item.entities);
          if (problem) {
            reads.forget("explanation", explanationKey(task, item.entities));
            throw new Error(problem);
          }
        }
        done += 1;
        if (live()) setReadiness((value) => (value.kind === "loading" ? { kind: "loading", done, total: required.length } : value));
      });
      if (!live()) return;
      const failures = results.flatMap((result, index): ContentFailure[] => (result.status === "rejected" && !isAbort(result.reason) ? [{ label: required[index].label, detail: result.reason instanceof ApiError ? describeError(result.reason) : (result.reason as Error).message }] : []));
      if (results.some((result) => result.status === "rejected" && isAbort(result.reason))) return;
      setSource(reads);
      if (failures.length) {
        const lost = results.find((result) => result.status === "rejected" && result.reason instanceof ApiError && [401, 403, 409].includes(result.reason.status));
        block({ kind: "blocked", cause: lost ? "session" : "content", title: lost ? "This page no longer has access to the case." : "Part of this case could not be loaded.", failures });
        return;
      }
      setReadiness((value) => (value.kind === "usable" ? value : { kind: "rendering" }));
    })();
    return () => controller.abort();
  }, [binding, attempt]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);
  const rendered = useCallback(() => setReadiness((value) => (value.kind === "rendering" ? { kind: "usable" } : value)), []);
  const renderFailed = useCallback((error: unknown) => setReadiness({ kind: "blocked", cause: "render", title: "This case's information could not be displayed.", failures: [{ label: "The case display", detail: error instanceof Error ? error.message : "The display did not finish." }] }), []);
  return { studyCase, source, readiness, retry, rendered, renderFailed, attempt };
}

/** Catches a render failure in the case workspace so it blocks readiness instead of passing. */
export class WorkspaceBoundary extends Component<{ onError: (error: unknown) => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: unknown) {
    this.props.onError(error);
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

/**
 * Confirms that the shown pair rendered from validated data: the same reads the cards use
 * have settled, and the cards are no longer marked busy. Reports once per mount.
 */
export function RenderProbe({ source, target, components, onRendered, onStalled }: { source: EntityRef; target: EntityRef; components: Set<CaseComponent>; onRendered: () => void; onStalled: () => void }) {
  const sourceCtx = useEntityContext(source);
  const targetCtx = useEntityContext(target);
  const pair = useMemo(() => [source, target], [source, target]);
  const one = useMemo(() => [source], [source]);
  const other = useMemo(() => [target], [target]);
  const profiles = components.has("entity_description");
  const sourceProfile = useExplanation("entity_profile", profiles ? one : null);
  const targetProfile = useExplanation("entity_profile", profiles ? other : null);
  const comparison = useExplanation("pair_comparison", components.has("pair_comparison") ? pair : null);
  const settled = [sourceCtx, targetCtx, sourceProfile, targetProfile, comparison].every((item) => !item.loading && !item.error) && Boolean(sourceCtx.data && targetCtx.data);
  const reported = useRef(false);
  useEffect(() => {
    if (!settled || reported.current) return;
    let frame = 0;
    const started = performance.now();
    const check = () => {
      const page = document.querySelector(".case-page");
      const cards = page?.querySelectorAll(".card-pair .entity-card").length ?? 0;
      const busy = page?.querySelector(".card-pair .skeleton-block, .card-pair [aria-busy='true'], .comparison [aria-busy='true']");
      if (cards === 2 && !busy) {
        reported.current = true;
        onRendered();
        return;
      }
      if (performance.now() - started > 10_000) {
        reported.current = true;
        onStalled();
        return;
      }
      frame = requestAnimationFrame(check);
    };
    frame = requestAnimationFrame(check);
    return () => cancelAnimationFrame(frame);
  }, [settled, onRendered, onStalled]);
  return null;
}
