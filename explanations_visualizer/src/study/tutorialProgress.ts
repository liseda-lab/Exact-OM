// Which lesson requirements an observed action satisfies. These are reports of actions,
// not proof of reading or understanding: the server stores them as lesson evidence and grades
// comprehension separately. Pure module (type-only imports) for Node's test runner.

import type { WorkspaceAction } from "../lib/workspace/types";
import type { EventType, RequirementAction, TutorialLesson } from "./types";

export type LessonSignal =
  | { kind: "rank"; event: EventType; element?: string; detailsOpen: boolean }
  | { kind: "inspect"; position: number; returning: boolean }
  | { kind: "workspace"; action: WorkspaceAction }
  | { kind: "practice_check"; partial: boolean }
  | { kind: "downloads" }
  | { kind: "report"; consulted: boolean; methods: number };

export function actionsFor(signal: LessonSignal): RequirementAction[] {
  switch (signal.kind) {
    case "rank": {
      const actions: RequirementAction[] = [];
      if (signal.event === "rank_add") actions.push("add_rank");
      if (signal.event === "rank_move") actions.push("move_rank");
      if (signal.event === "rank_remove") actions.push("remove_rank");
      if (signal.event === "revision" && signal.element === "undo") actions.push("undo_rank");
      if (signal.event === "keep_initial_order") actions.push("keep_initial_order");
      if (signal.event === "response_type_change" && signal.element === "none_of_these") actions.push("choose_none");
      if (signal.event === "response_type_change" && signal.element === "insufficient_evidence") actions.push("choose_insufficient");
      if (signal.detailsOpen) actions.push("rank_with_details_open");
      return actions;
    }
    case "inspect": {
      const actions: RequirementAction[] = [];
      if (signal.position !== 1) actions.push("inspect_other_candidate");
      if (signal.returning) actions.push("return_to_candidate");
      return actions;
    }
    case "practice_check":
      return signal.partial ? ["check_partial_ranking"] : [];
    case "downloads":
      return ["locate_downloads"];
    case "report":
      if (!signal.consulted) return ["report_no_methods"];
      return signal.methods >= 2 ? ["report_multiple_methods"] : [];
    case "workspace": {
      const action = signal.action;
      switch (action.type) {
        case "hierarchy_focus":
          return action.via === "search" ? ["search_entity"] : action.via === "child" ? ["navigate_child"] : action.via === "return" ? ["return_to_compared"] : ["navigate_parent"];
        case "citation_open":
          return ["open_citation"];
        case "axiom_open":
          return ["open_original_axiom"];
        case "evidence_select":
          return action.from === "list" ? ["locate_in_evidence_list", "inspect_graph_or_list"] : ["inspect_graph_or_list"];
        case "locate_fact":
          return action.inList ? ["locate_in_evidence_list"] : [];
        case "graph_edge":
          return ["inspect_graph_or_list"];
        case "graph_zoom":
        case "graph_fit":
        case "graph_reset":
          return ["change_graph_view"];
        case "copy_iri":
          return ["copy_iri"];
        default:
          return [];
      }
    }
  }
}

/** Requirement IDs of `lesson` that the signal satisfies (directly or as an accessible alternative). */
export function requirementsFor(lesson: TutorialLesson, signal: LessonSignal): string[] {
  const actions = actionsFor(signal);
  return lesson.requirements.filter((requirement) => actions.includes(requirement.action) || requirement.alternatives?.some((alternative) => actions.includes(alternative))).map((requirement) => requirement.requirement_id);
}
