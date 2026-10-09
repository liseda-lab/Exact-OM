"use client";

// The guided task bar of a tutorial lesson (21 F02). It stays beside the practice workspace
// and shows one immediate task: the next requirement the server has not acknowledged, with
// one or two short sentences of help and its status. "Show me" opens the right detail view,
// scrolls the control into view below the sticky bars and focuses it; it never performs the
// action, ticks a requirement or chooses an answer. The lesson's full steps sit behind "Why?".
// Targets are semantic `data-tour` hooks on real controls, never coordinates or nth elements.

import { useEffect, useRef, useState } from "react";

import { IconCheck } from "@/components/common/Icons";
import type { LessonRequirement, RequirementAction, TutorialLesson } from "@/study/types";

/** Frontend-authored short help per action; the requirement label (publication text) is the task itself. */
export const TASK_HINTS: Record<RequirementAction, string> = {
  inspect_other_candidate: "Select a different candidate in the list. Inspecting a candidate never ranks it.",
  return_to_candidate: "Select a candidate you already inspected (marked Viewed).",
  search_entity: "In Hierarchy, search the source ontology (for example “crate”) and choose a result.",
  navigate_parent: "In Hierarchy, select one of the parents listed above the focused concept.",
  navigate_child: "Select one of the children listed below the focused concept.",
  return_to_compared: "Come back to the compared concept: select it in the tree, or use “Back to compared concept”.",
  open_citation: "Under a generated description, select a citation to see the record it rests on.",
  open_original_axiom: "In a card, select “Show original axiom”.",
  locate_in_evidence_list: "Open Evidence and select “Show in graph” on a feature.",
  inspect_graph_or_list: "Select a feature in the graph’s item list or a dashed feature line. Comparison lines only explain themselves.",
  change_graph_view: "Zoom in or out, or select “Fit all”.",
  rank_with_details_open: "With Evidence or the graph open, add or move a candidate in your practice answer.",
  add_rank: "Add a candidate to your practice answer.",
  move_rank: "Move a ranked candidate up or down. Rank two candidates first if needed.",
  remove_rank: "Remove a candidate from your practice answer.",
  undo_rank: "Select Undo to reverse your last change.",
  keep_initial_order: "Select “Keep initial order”.",
  check_partial_ranking: "Rank one to four candidates, then select “Check practice answer”.",
  choose_none: "Select “None of these”.",
  choose_insufficient: "Select “Insufficient information”.",
  locate_downloads: "Find the two ontology files.",
  report_multiple_methods: "In practice report 1, answer Yes and select both methods described.",
  report_no_methods: "In practice report 2, answer No.",
  copy_iri: "Copy an entity’s IRI with its Copy IRI button.",
};

export interface TourTarget {
  /** Detail view that must be open first. */
  tab?: "hierarchy" | "evidence" | "graph";
  /** Candidate selectors, most specific first; the first visible, enabled match is used. */
  selectors: string[];
  /** Said when no matching control is on screen, instead of pointing into empty space. */
  missing: string;
}

export const TOUR_TARGETS: Record<RequirementAction, TourTarget> = {
  inspect_other_candidate: { selectors: ['[data-tour="inspect"]:not([data-position="1"]):not([aria-pressed="true"])'], missing: "The candidate list is not available right now." },
  return_to_candidate: { selectors: ['[data-tour="inspect"][data-viewed="true"]:not([aria-pressed="true"])'], missing: "Inspect a different candidate first; then come back to one you inspected." },
  search_entity: { tab: "hierarchy", selectors: ['[data-tour="search-source"]', '[data-tour="search-target"]'], missing: "The search box is not available right now." },
  navigate_parent: { tab: "hierarchy", selectors: ['[data-tour="parent-source"]', '[data-tour="parent-target"]'], missing: "The focused concept shows no parent. Search for another concept, or go back to the compared concept." },
  navigate_child: { tab: "hierarchy", selectors: ['[data-tour="child-source"]', '[data-tour="child-target"]'], missing: "The focused concept has no children. Open a parent first; its children are listed below it." },
  return_to_compared: { tab: "hierarchy", selectors: ['[data-tour="return-source"]', '[data-tour="return-target"]'], missing: "You are viewing the compared concept. Open a parent or child first, then come back." },
  open_citation: { selectors: ['[data-tour="citation"]'], missing: "No citation is shown yet. Wait for the generated description, or inspect another candidate." },
  open_original_axiom: { selectors: ['[data-tour="axiom"]'], missing: "No original axiom toggle is shown for these cards. Inspect another candidate." },
  locate_in_evidence_list: { tab: "evidence", selectors: ['[data-tour="evidence-select"]'], missing: "This candidate shows no selected feature. Inspect another candidate." },
  inspect_graph_or_list: { tab: "graph", selectors: ['[data-tour="graph-feature"]', '[data-tour="evidence-select"]'], missing: "This candidate shows no selected feature. Inspect another candidate." },
  change_graph_view: { tab: "graph", selectors: ['[data-tour="graph-zoom"]'], missing: "The graph controls are not available right now." },
  rank_with_details_open: { tab: "graph", selectors: ['[data-tour="rank-add"]', '[data-tour="rank-move"]:not(:disabled)'], missing: "Your answer controls are not available right now." },
  add_rank: { selectors: ['[data-tour="rank-add"]'], missing: "Every candidate is already in your answer. Remove one first." },
  move_rank: { selectors: ['[data-tour="rank-move"]:not(:disabled)', '[data-tour="rank-add"]'], missing: "Add two candidates first, then move one." },
  remove_rank: { selectors: ['[data-tour="rank-remove"]', '[data-tour="rank-add"]'], missing: "Add a candidate first, then remove it." },
  undo_rank: { selectors: ['[data-tour="rank-undo"]:not(:disabled)', '[data-tour="rank-add"]'], missing: "Make a change first; then Undo reverses it." },
  keep_initial_order: { selectors: ['[data-tour="rank-keep"]'], missing: "The answer controls are not available right now." },
  check_partial_ranking: { selectors: ['[data-tour="answer-submit"]:not(:disabled)', '[data-tour="rank-add"]'], missing: "Rank one to four candidates first." },
  choose_none: { selectors: ['[data-tour="explicit-none_of_these"]'], missing: "The answer controls are not available right now." },
  choose_insufficient: { selectors: ['[data-tour="explicit-insufficient_evidence"]'], missing: "The answer controls are not available right now." },
  locate_downloads: { selectors: ['[data-tour="downloads"]'], missing: "The ontology files are not listed right now." },
  report_multiple_methods: { selectors: ['[data-tour="report-multi"] input[type="radio"]'], missing: "The practice reports are shown in the last lesson." },
  report_no_methods: { selectors: ['[data-tour="report-none"] input[type="radio"]'], missing: "The practice reports are shown in the last lesson." },
  copy_iri: { selectors: ['[data-tour="copy-iri"]'], missing: "No Copy IRI button is shown right now." },
};

function usable(element: HTMLElement) {
  return element.offsetParent !== null && !(element as HTMLButtonElement).disabled && element.getAttribute("aria-disabled") !== "true";
}

/** The first visible, enabled control for a target, or null. */
export function findTarget(root: ParentNode, target: TourTarget): HTMLElement | null {
  for (const selector of target.selectors) {
    const match = Array.from(root.querySelectorAll<HTMLElement>(selector)).find(usable);
    if (match) return match;
  }
  return null;
}

/** Bring a control into view and focus it with a visible, non-colour-only highlight. */
export function highlight(element: HTMLElement) {
  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  element.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" });
  element.focus({ preventScroll: true });
  document.querySelectorAll(".tour-target").forEach((node) => node.classList.remove("tour-target"));
  element.classList.add("tour-target");
  const clear = () => element.classList.remove("tour-target");
  window.setTimeout(clear, 4000);
  element.addEventListener("blur", clear, { once: true });
}

export type RequirementState = "saved" | "saving" | "todo";

export function LessonCoach({
  lessonNumber,
  lessonCount,
  lesson,
  states,
  notice,
  onShowMe,
  onBack,
  onNext,
  nextLabel,
  canGoBack,
  extra,
  tools,
}: {
  lessonNumber: number;
  lessonCount: number;
  lesson: TutorialLesson;
  states: Record<string, RequirementState>;
  notice: { tone: "info" | "warn"; text: string } | null;
  onShowMe: (requirement: LessonRequirement) => void;
  onBack: () => void;
  onNext: () => void;
  nextLabel: string;
  canGoBack: boolean;
  extra?: React.ReactNode;
  /** Secondary tools kept with the task (for example the ontology files). */
  tools?: React.ReactNode;
}) {
  const [why, setWhy] = useState(false);
  const [chosen, setChosen] = useState<string | null>(null);
  const bar = useRef<HTMLElement>(null);
  // Sticky bars stack: publish this bar's height so scrolled-to controls clear it (WCAG 2.4.11).
  useEffect(() => {
    const element = bar.current;
    if (!element) return;
    const root = document.documentElement;
    const set = () => root.style.setProperty("--coach-h", `${Math.ceil(element.getBoundingClientRect().height)}px`);
    set();
    const observer = new ResizeObserver(set);
    observer.observe(element);
    return () => {
      observer.disconnect();
      root.style.removeProperty("--coach-h");
    };
  }, []);

  const open = lesson.requirements.filter((requirement) => states[requirement.requirement_id] !== "saved");
  const doneCount = lesson.requirements.length - open.length;
  const current = lesson.requirements.find((requirement) => requirement.requirement_id === chosen && states[requirement.requirement_id] !== "saved") ?? open.find((requirement) => states[requirement.requirement_id] === "todo") ?? open[0] ?? null;
  const allSaved = open.length === 0;
  const state = current ? states[current.requirement_id] : "saved";

  return (
    <section ref={bar} className={allSaved ? "coach coach-done" : "coach"} aria-labelledby="coach-task">
      <div className="coach-main">
        <div className="coach-text">
          <span className="coach-step meta">
            Lesson {lessonNumber} of {lessonCount}
            {lesson.optional ? " · optional" : ""} · {doneCount} of {lesson.requirements.length} steps saved
          </span>
          {current ? (
            <>
              <h2 id="coach-task" className="coach-task">
                {current.label}
              </h2>
              <p className="coach-hint">{TASK_HINTS[current.action]}</p>
            </>
          ) : (
            <>
              <h2 id="coach-task" className="coach-task">
                <IconCheck /> Done. Every step of this lesson is saved.
              </h2>
              <p className="coach-hint">Continue when you are ready. You can come back to any lesson.</p>
            </>
          )}
        </div>
        <div className="coach-actions">
          {current && (
            <>
              <span className={`coach-status coach-status-${state}`} aria-live="polite">
                {state === "saving" ? "Saving…" : "To do"}
              </span>
              <button type="button" className="btn btn-sm" onClick={() => onShowMe(current)}>
                Show me
              </button>
            </>
          )}
          <button type="button" className="btn btn-sm btn-quiet" aria-expanded={why} onClick={() => setWhy((value) => !value)}>
            {why ? "Hide details" : "Why?"}
          </button>
          {tools}
          <span className="coach-spacer" />
          <button type="button" className="btn btn-sm" disabled={!canGoBack} onClick={onBack}>
            Back
          </button>
          <button type="button" className={allSaved ? "btn btn-sm btn-primary" : "btn btn-sm"} onClick={onNext}>
            {nextLabel}
          </button>
        </div>
      </div>
      {notice && (
        <p className={notice.tone === "warn" ? "coach-notice note note-warn" : "coach-notice note"} role="status">
          {notice.text}
        </p>
      )}
      <ul className="lesson-checklist" aria-label="Steps in this lesson">
        {lesson.requirements.map((requirement) => {
          const status = states[requirement.requirement_id];
          return (
            <li key={requirement.requirement_id} className={status === "saved" ? "done" : status === "saving" ? "saving" : requirement === current ? "current" : ""}>
              <button type="button" className="checklist-step" aria-label={`Choose step: ${requirement.label}`} aria-current={requirement === current ? "step" : undefined} disabled={status === "saved"} onClick={() => setChosen(requirement.requirement_id)}>
                <span className="check-mark" aria-hidden="true">
                  {status === "saved" ? <IconCheck /> : null}
                </span>
                <span>{requirement.label}</span>
              </button>
              <span className="meta">{status === "saved" ? "Done · saved" : status === "saving" ? "Saving…" : "To do"}</span>
            </li>
          );
        })}
      </ul>
      {why && (
        <div className="coach-why">
          <ol className="lesson-steps">
            {lesson.steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          {extra}
        </div>
      )}
    </section>
  );
}
