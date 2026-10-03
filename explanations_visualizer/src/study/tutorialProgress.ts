// Which lesson requirements an observed action satisfies. These are reports of actions,
// not proof of reading or understanding: the server stores them as lesson evidence and grades
// comprehension separately. Pure module (type-only imports) for Node's test runner.

import type { WorkspaceAction } from "../lib/workspace/types";
import type { EventType, RequirementAction, TutorialLesson, TutorialPublic, ExplanationResource } from "./types";
import type { EntityRef } from "../lib/types";

export type LessonSignal =
  | { kind: "rank"; event: EventType; element?: string; detailsOpen: boolean }
  | { kind: "inspect"; position: number; returning: boolean }
  | { kind: "workspace"; action: WorkspaceAction; entity?: EntityRef }
  | { kind: "practice_check"; partial: boolean }
  | { kind: "downloads" }
  | { kind: "report"; consulted: boolean; methods: number; methodCodes?: string[] };

export interface TutorialActionEvidence {
  requirement_id: string; action: RequirementAction;
  candidate_id?: string; entity?: EntityRef; fact_id?: string;
  ranked_candidate_ids?: string[]; response_type?: string | null;
  methods?: string[]; asset_ids?: string[];
}

/** Preserve the observed action's typed payload for backend predicate validation. */
export function evidenceFor(lesson: TutorialLesson, signal: LessonSignal, tutorial: TutorialPublic, resources: ExplanationResource[], practice: Record<string, unknown> | null): TutorialActionEvidence[] {
  const observed = actionsFor(signal);
  return lesson.requirements.flatMap((requirement) => {
    const action = observed.includes(requirement.action) ? requirement.action : requirement.alternatives?.find((alternative) => observed.includes(alternative));
    if (!action) return [];
    const evidence: TutorialActionEvidence = { requirement_id: requirement.requirement_id, action };
    if (signal.kind === "inspect") evidence.candidate_id = tutorial.case.candidates.find((candidate) => candidate.display_position === signal.position)?.candidate_id;
    if (signal.kind === "rank" || signal.kind === "practice_check") {
      evidence.ranked_candidate_ids = (practice?.ranked_candidate_ids as string[] | undefined) ?? [];
      evidence.response_type = (practice?.response_type as string | null | undefined) ?? null;
    }
    if (signal.kind === "downloads") evidence.asset_ids = tutorial.case.ontology_resources.map((asset) => asset.asset_id);
    if (signal.kind === "report") evidence.methods = signal.methodCodes ?? [];
    if (signal.kind === "workspace") {
      const event = signal.action;
      if (event.type === "hierarchy_focus") evidence.entity = { ontology_version_id: event.side === "source" ? tutorial.case.source.ontology_version_id : tutorial.case.candidates[0].entity.ontology_version_id, iri: event.iri, kind: event.kind };
      if (event.type === "citation_open" || event.type === "axiom_open" || event.type === "locate_fact") evidence.fact_id = event.factId;
      if (event.type === "evidence_select" || event.type === "graph_edge") evidence.fact_id = resources.flatMap((resource) => resource.evidence).find((item) => item.evidence_id === event.evidenceId)?.fact_ids[0];
      if (event.type === "copy_iri") evidence.entity = signal.entity;
    }
    return [evidence];
  });
}

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
