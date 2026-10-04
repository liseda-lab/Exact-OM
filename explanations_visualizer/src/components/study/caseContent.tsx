"use client";

// Loads and validates one scored presentation, then reports its readiness (19 F11/F12/F18/F19):
//   loading → (required content validated) → rendering → (admitted components rendered) → usable
//   any required failure, incompatible service or lost session → blocked (retry keeps work)
// v2 explanation cases read only their authorized scope through the shared API adapter;
// a missing descriptor or incompatible capabilities block the case instead of selecting
// another adapter. v1 keeps its explicitly selected legacy resource adapter. Baseline cases
// make no workspace request. The focal reads are validated before they are cached and are
// shared with the rendered workspace. Every transition is tagged with its attempt, so a late
// result can neither overwrite a blocked state nor ready another attempt.

import { Component, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { ApiError, describeError, getJson, isAbort } from "@/lib/api";
import { createApiSource, scopedRead } from "@/lib/workspace/apiSource";
import { indexResources } from "@/lib/workspace/resourceIndex";
import { createResourceSource } from "@/lib/workspace/resourceSource";
import { shareReads, type SharedSource } from "@/lib/workspace/sharedReads";
import type { WorkspaceSource } from "@/lib/workspace/types";
import { useEntityContext, useExplanation } from "@/lib/workspace/WorkspaceContext";
import type { EntityRef } from "@/lib/types";
import { createAccessGuard, type AccessGuard } from "@/study/accessGuard";
import { contextProblem, explanationProblem, inBatches, renderExpectations, requiredContent, validateCapabilities, validateCase, workspaceCapabilities, type CaseComponent, type ScopeCapabilities } from "@/study/caseReadiness";
import type { ExplanationResource, StudyCase, StudyState } from "@/study/types";

export interface ContentFailure {
  label: string;
  detail: string;
}

export type CaseReadiness = { attempt: number } & (
  | { kind: "loading"; done: number; total: number }
  | { kind: "rendering" }
  | { kind: "usable" }
  | { kind: "blocked"; cause: "service" | "content" | "session" | "render"; title: string; failures: ContentFailure[] }
);

type Blocked = Extract<CaseReadiness, { kind: "blocked" }>;

const CONCURRENCY = 4;
const LOST_ACCESS = "This page no longer has access to the case.";

function detailOf(error: unknown): string {
  return error instanceof ApiError ? describeError(error) : error instanceof Error ? error.message : "The request failed.";
}

function failureFrom(error: unknown, what: string, attempt: number): Blocked {
  if (error instanceof ApiError && error.status === 503)
    return { attempt, kind: "blocked", cause: "service", title: "The study service could not provide this case right now.", failures: [{ label: what, detail: error.message }] };
  if (error instanceof ApiError && (error.status === 401 || error.status === 403 || error.status === 409))
    return { attempt, kind: "blocked", cause: "session", title: LOST_ACCESS, failures: [{ label: what, detail: error.message }] };
  return { attempt, kind: "blocked", cause: "content", title: "Part of this case could not be loaded.", failures: [{ label: what, detail: detailOf(error) }] };
}

export function useCaseContent({ state, components, ontologyName, caseLabels }: { state: StudyState; components: Set<CaseComponent>; ontologyName: (id: string) => string; caseLabels: (current: StudyCase) => Record<string, string> }) {
  const binding = `${state.session_id}|${state.study_revision}|${state.current_case_id}|${state.current_presentation_id}`;
  const [studyCase, setStudyCase] = useState<StudyCase | null>(null);
  const [source, setSource] = useState<WorkspaceSource | null>(null);
  const [readiness, setReadiness] = useState<CaseReadiness>({ attempt: 0, kind: "loading", done: 0, total: 1 });
  const [attempt, setAttempt] = useState(0);
  const attemptRef = useRef(0);
  attemptRef.current = attempt;
  const readinessRef = useRef(readiness);
  readinessRef.current = readiness;
  const shared = useRef<{ key: string; source: SharedSource; guard: AccessGuard } | null>(null);
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
    setReadiness({ attempt: attemptRef.current, kind: "loading", done: 0, total: 1 });
    return () => {
      shared.current?.guard.cancel();
      shared.current?.source.close();
      shared.current = null;
    };
  }, [binding]);

  /** Block the case, unless the blocking result belongs to an attempt that has been superseded. */
  const block = useCallback((next: Blocked) => {
    if (next.attempt !== attemptRef.current) return;
    setReadiness(next);
  }, []);

  useEffect(() => {
    const mine = attempt;
    const controller = new AbortController();
    const live = () => !controller.signal.aborted && attemptRef.current === mine;
    const session = stateRef.current.session_id;
    // Only advance a state that this attempt still owns; never overwrite a block.
    const advance = (next: CaseReadiness) =>
      setReadiness((value) => (value.attempt === mine && (value.kind === "loading" || value.kind === "rendering") ? next : value));
    (async () => {
      setReadiness({ attempt: mine, kind: "loading", done: 0, total: 1 });
      let current: StudyCase;
      try {
        current = await scopedRead<StudyCase>("/api/v1/study", "/cases/current", undefined, controller.signal, session);
      } catch (error) {
        if (!isAbort(error) && live()) block(failureFrom(error, "The case", mine));
        return;
      }
      if (!live()) return;
      const { reject, block: blocking } = validateCase(stateRef.current, current);
      if (reject.length) {
        setStudyCase(null);
        block({ attempt: mine, kind: "blocked", cause: "service", title: "The study service sent a case that does not match this session.", failures: reject.map((detail) => ({ label: "The case", detail })) });
        return;
      }
      setStudyCase(current);
      if (blocking.length) {
        block({ attempt: mine, kind: "blocked", cause: "service", title: "This case cannot be shown with this version of the study service.", failures: blocking.map((detail) => ({ label: "The case", detail })) });
        return;
      }
      const v2 = (current.contract_version ?? "exact-study/1.0") === "exact-study/2.0";
      if (current.condition !== "explanation") {
        advance({ attempt: mine, kind: "rendering" });
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
          advance({ attempt: mine, kind: "rendering" });
        } catch (error) {
          if (!isAbort(error) && live()) block(failureFrom(error, "The prepared explanations", mine));
        }
        return;
      }
      const scope = current.workspace!.scope_id;
      const base = `/api/v1/study/workspace/${encodeURIComponent(scope)}`;
      let caps: ScopeCapabilities;
      try {
        caps = await scopedRead<ScopeCapabilities>(base, "/capabilities", undefined, controller.signal, session);
      } catch (error) {
        if (!isAbort(error) && live()) block(failureFrom(error, "The case's workspace", mine));
        return;
      }
      if (!live()) return;
      const incompatible = validateCapabilities(caps, { scopeId: scope, studyRevision: stateRef.current.study_revision, current, components: componentsRef.current });
      if (incompatible.length) {
        block({ attempt: mine, kind: "blocked", cause: "service", title: "The study service described a workspace that does not match this case.", failures: incompatible.map((detail) => ({ label: "The case's workspace", detail })) });
        return;
      }
      const key = `study|${session}|${current.study_revision}|${current.presentation_id}|${scope}|${caps.policy_hash}`;
      if (shared.current?.key !== key) {
        shared.current?.guard.cancel();
        shared.current?.source.close();
        const api = createApiSource({ key, base, sessionId: session, kind: "study_resource", capabilities: workspaceCapabilities(caps, (id) => namesRef.current(id)), ontologyName: (id) => namesRef.current(id) });
        // A refused optional read (for example a restricted resource) is local; only a
        // refusal of the scope itself means the session or policy no longer permits the case.
        // Each re-check belongs to the attempt current when it is issued (19 F20).
        const guard = createAccessGuard({
          attempt: () => attemptRef.current,
          live: () => shared.current?.key === key,
          check: (signal) => scopedRead(base, "/capabilities", undefined, signal, session),
          needsCheck: (error) => error instanceof ApiError && error.status === 403,
          lost: (error) => error instanceof ApiError && [401, 403, 409].includes(error.status),
          onLost: (issuedFor, error) =>
            block({ attempt: issuedFor, kind: "blocked", cause: "session", title: LOST_ACCESS, failures: [{ label: "Your session", detail: error instanceof Error ? error.message : "Access changed." }] }),
        });
        shared.current = {
          key,
          guard,
          source: shareReads(api, {
            onSuspect: guard.suspect,
            validateContext: (entity, value) => contextProblem(value, entity),
            validateExplanation: (task, entities, value) => explanationProblem(value, task, entities),
          }),
        };
      }
      const reads = shared.current.source;
      const required = requiredContent(current, componentsRef.current, caps);
      let done = 0;
      advance({ attempt: mine, kind: "loading", done, total: required.length });
      const results = await inBatches(required, CONCURRENCY, async (item) => {
        if (item.kind === "context") await reads.entityContext(item.entities[0], controller.signal);
        else await reads.explanation(item.kind === "profile" ? "entity_profile" : "pair_comparison", item.entities, controller.signal);
        done += 1;
        if (live()) advance({ attempt: mine, kind: "loading", done, total: required.length });
      });
      if (!live() || results.some((result) => result.status === "rejected" && isAbort(result.reason))) return;
      setSource(reads);
      const failures = results.flatMap((result, index): ContentFailure[] => (result.status === "rejected" ? [{ label: required[index].label, detail: detailOf(result.reason) }] : []));
      if (failures.length) {
        const lost = results.find((result) => result.status === "rejected" && result.reason instanceof ApiError && [401, 403, 409].includes(result.reason.status));
        block({ attempt: mine, kind: "blocked", cause: lost ? "session" : "content", title: lost ? LOST_ACCESS : "Part of this case could not be loaded.", failures });
        return;
      }
      advance({ attempt: mine, kind: "rendering" });
    })();
    return () => controller.abort();
  }, [binding, attempt, block]);

  const retry = useCallback(() => {
    // After a render failure the culprit is unknown: refetch every focal read rather than
    // re-rendering a cached response that may be unusable.
    if (readinessRef.current.kind === "blocked" && readinessRef.current.cause === "render") shared.current?.source.reset();
    // Checks issued for the attempt being retried can no longer decide anything.
    shared.current?.guard.cancel();
    setAttempt((value) => value + 1);
  }, []);
  const rendered = useCallback((forAttempt: number) => setReadiness((value) => (value.kind === "rendering" && value.attempt === forAttempt ? { attempt: forAttempt, kind: "usable" } : value)), []);
  const renderFailed = useCallback(
    (error: unknown) => block({ attempt: attemptRef.current, kind: "blocked", cause: "render", title: "This case's information could not be displayed.", failures: [{ label: "The case display", detail: error instanceof Error ? error.message : "The display did not finish." }] }),
    [block],
  );
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
 * Confirms that the shown pair rendered from validated data, exactly as far as the frozen
 * condition displays it (19 F19): the question always, both entity cards only when original
 * context or descriptions are admitted, descriptions only when admitted, and the comparison
 * only when admitted. Reports once per mount, for the attempt that mounted it.
 */
export function RenderProbe({ source, target, components, attempt, onRendered, onStalled }: { source: EntityRef; target: EntityRef; components: Set<CaseComponent>; attempt: number; onRendered: (attempt: number) => void; onStalled: () => void }) {
  const expected = renderExpectations(components);
  const sourceCtx = useEntityContext(source);
  const targetCtx = useEntityContext(target);
  const pair = useMemo(() => [source, target], [source, target]);
  const one = useMemo(() => [source], [source]);
  const other = useMemo(() => [target], [target]);
  const sourceProfile = useExplanation("entity_profile", expected.profiles ? one : null);
  const targetProfile = useExplanation("entity_profile", expected.profiles ? other : null);
  const comparison = useExplanation("pair_comparison", expected.comparison ? pair : null);
  const settled = [sourceCtx, targetCtx, sourceProfile, targetProfile, comparison].every((item) => !item.loading && !item.error) && Boolean(sourceCtx.data && targetCtx.data);
  const reported = useRef(false);
  useEffect(() => {
    if (!settled || reported.current) return;
    let frame = 0;
    const started = performance.now();
    const check = () => {
      const page = document.querySelector(".case-page");
      const question = Boolean(page?.querySelector(".pair-question"));
      const cards = page?.querySelectorAll(".card-pair .entity-card").length ?? 0;
      const cardsBusy = Boolean(page?.querySelector(".card-pair .skeleton-block, .card-pair [aria-busy='true']"));
      const comparisonShown = Boolean(page?.querySelector(".comparison"));
      const comparisonBusy = Boolean(page?.querySelector(".comparison [aria-busy='true']"));
      const ready = question && (expected.cards ? cards === 2 && !cardsBusy : cards === 0) && (expected.comparison ? comparisonShown && !comparisonBusy : !comparisonShown);
      if (ready) {
        reported.current = true;
        onRendered(attempt);
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
  }, [settled, onRendered, onStalled, attempt, expected.cards, expected.comparison]);
  return null;
}
