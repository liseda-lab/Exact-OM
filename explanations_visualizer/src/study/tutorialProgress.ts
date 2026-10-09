// Which lesson requirements an observed action satisfies. These are reports of actions,
// not proof of reading or understanding: the server stores them as lesson evidence and grades
// comprehension separately. Pure module (type-only imports) for Node's test runner.
//
// An interaction is not evidence by itself (21 F01). Evidence that names an original record
// or entity is emitted only when that identity is admitted by the frozen practice resources,
// supports the action, and belongs to the candidate being inspected; otherwise the
// interaction stays telemetry/inspection only. Nothing is invented, reused from another
// record or relaxed to obtain a tick.

import type { WorkspaceAction } from "../lib/workspace/types";
import type { EventType, RequirementAction, TutorialLesson, TutorialPublic, ExplanationResource } from "./types";
import type { EntityRef } from "../lib/types";

export type LessonSignal =
  | { kind: "rank"; event: EventType; element?: string; detailsOpen: boolean }
  | { kind: "inspect"; position: number; returning: boolean }
  /** `candidateId`: the practice candidate inspected when the action happened (scopes evidence). */
  | { kind: "workspace"; action: WorkspaceAction; entity?: EntityRef; candidateId?: string | null }
  | { kind: "practice_check"; partial: boolean }
  | { kind: "downloads" }
  | { kind: "report"; consulted: boolean; methods: number; methodCodes?: string[] };

export interface TutorialActionEvidence {
  requirement_id: string; action: RequirementAction;
  candidate_id?: string; entity?: EntityRef; fact_id?: string;
  ranked_candidate_ids?: string[]; response_type?: string | null;
  methods?: string[]; asset_ids?: string[];
}

const FACT_ACTIONS = new Set<RequirementAction>(["open_citation", "open_original_axiom", "locate_in_evidence_list", "inspect_graph_or_list"]);
const ENTITY_ACTIONS = new Set<RequirementAction>(["search_entity", "navigate_parent", "navigate_child", "return_to_compared", "copy_iri"]);

const key = (entity: { ontology_version_id: string; iri: string; kind?: string }) => `${entity.ontology_version_id}|${entity.kind ?? ""}|${entity.iri}`;

/** What the frozen practice resources admit, for checking evidence before it is reported. */
function admitted(tutorial: TutorialPublic, resources: ExplanationResource[]) {
  const facts = new Set(resources.flatMap((resource) => [...resource.facts, ...(resource.referenced_labels ?? [])].map((fact) => fact.fact_id)));
  const cited = new Set(resources.flatMap((resource) => [...resource.entity_profiles, ...resource.pair_comparison].flatMap((claim) => claim.fact_ids)));
  const parents = new Set(resources.flatMap((resource) => resource.hierarchy.map((edge) => key(edge.parent))));
  const children = new Set(resources.flatMap((resource) => resource.hierarchy.map((edge) => key(edge.child))));
  const compared = new Set([key(tutorial.case.source), ...tutorial.case.candidates.map((candidate) => key(candidate.entity))]);
  const evidence = resources.flatMap((resource) => resource.evidence);
  return { facts, cited, parents, children, compared, evidence };
}

/**
 * The original record a selected matcher feature rests on, resolved against the active
 * candidate's own evidence (evidence IDs may repeat across candidates or resources). The
 * record must be available, admitted and linked to that feature; when several are shown, the
 * first one the action names (the one actually inspected) is used.
 */
function featureFact(scope: ReturnType<typeof admitted>, evidenceId: string | null | undefined, candidateId: string | null | undefined, named: string[] | undefined): string | undefined {
  if (!evidenceId) return undefined;
  const items = scope.evidence.filter((item) => item.evidence_id === evidenceId && (!candidateId || item.candidate_id === candidateId) && item.status === "available");
  for (const item of items) {
    const linked = item.fact_ids.filter((id) => scope.facts.has(id));
    const chosen = named ? named.find((id) => linked.includes(id)) : linked[0];
    if (chosen) return chosen;
  }
  return undefined;
}

/** Preserve the observed action's typed payload for backend predicate validation; drop anything it cannot support. */
export function evidenceFor(lesson: TutorialLesson, signal: LessonSignal, tutorial: TutorialPublic, resources: ExplanationResource[], practice: Record<string, unknown> | null): TutorialActionEvidence[] {
  const observed = actionsFor(signal);
  if (!observed.length) return [];
  const scope = admitted(tutorial, resources);
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
      if (event.type === "hierarchy_focus") {
        const ontology = event.side === "source" ? tutorial.case.source.ontology_version_id : (tutorial.case.candidates.find((candidate) => candidate.candidate_id === signal.candidateId) ?? tutorial.case.candidates[0]).entity.ontology_version_id;
        evidence.entity = { ontology_version_id: ontology, iri: event.iri, kind: event.kind };
      }
      if (event.type === "citation_open") evidence.fact_id = scope.cited.has(event.factId) ? event.factId : undefined;
      if (event.type === "axiom_open") evidence.fact_id = scope.facts.has(event.factId) ? event.factId : undefined;
      if (event.type === "locate_fact") evidence.fact_id = scope.evidence.some((item) => (!signal.candidateId || item.candidate_id === signal.candidateId) && item.fact_ids.includes(event.factId)) ? event.factId : undefined;
      if (event.type === "evidence_select") evidence.fact_id = featureFact(scope, event.evidenceId, signal.candidateId, event.factIds);
      if (event.type === "graph_edge") evidence.fact_id = event.edgeKind === "feature" && event.factId ? featureFact(scope, event.evidenceId, signal.candidateId, [event.factId]) : undefined;
      if (event.type === "copy_iri") evidence.entity = signal.entity;
    }
    // Reject what the server would reject, before it is reported or shown as done.
    if (FACT_ACTIONS.has(action) && !evidence.fact_id) return [];
    if (ENTITY_ACTIONS.has(action)) {
      if (!evidence.entity?.kind) return [];
      if (action === "navigate_parent" && !scope.parents.has(key(evidence.entity))) return [];
      if (action === "navigate_child" && !scope.children.has(key(evidence.entity))) return [];
      if (action === "return_to_compared" && !scope.compared.has(key(evidence.entity))) return [];
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
        case "hierarchy_focus": {
          const base: RequirementAction = action.via === "search" ? "search_entity" : action.via === "child" ? "navigate_child" : action.via === "return" ? "return_to_compared" : "navigate_parent";
          // Reaching the compared entity again by any real navigation is a return (21 F02).
          return action.returned && base !== "return_to_compared" ? [base, "return_to_compared"] : [base];
        }
        case "citation_open":
          return ["open_citation"];
        case "axiom_open":
          return ["open_original_axiom"];
        case "evidence_select":
          return action.from === "list" ? ["locate_in_evidence_list", "inspect_graph_or_list"] : ["inspect_graph_or_list"];
        case "locate_fact":
          return action.inList ? ["locate_in_evidence_list"] : [];
        case "graph_edge":
          // Bridges and ontology-context lines are inspection only, never original-feature evidence.
          return action.edgeKind === "feature" ? ["inspect_graph_or_list"] : [];
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

/** Requirement IDs of `lesson` whose action this kind of signal performs (before any identity check). */
export function requirementsFor(lesson: TutorialLesson, signal: LessonSignal): string[] {
  const actions = actionsFor(signal);
  return lesson.requirements.filter((requirement) => actions.includes(requirement.action) || requirement.alternatives?.some((alternative) => actions.includes(alternative))).map((requirement) => requirement.requirement_id);
}

/**
 * Lesson evidence waiting for the server (21 F01.4–5). Only valid, not-yet-acknowledged
 * evidence is held, one per requirement, so the store is bounded by the lesson definitions.
 * Acknowledged requirements are never resent or overwritten, and evidence the server rejected
 * is dropped so a later save cannot carry it again.
 */
export class LessonEvidence {
  private unsent = new Map<string, TutorialActionEvidence>();

  /** Add newly observed evidence; returns the requirement IDs it newly makes pending. */
  observe(evidence: TutorialActionEvidence[], acknowledged: ReadonlySet<string>): string[] {
    const added: string[] = [];
    for (const item of evidence) {
      if (acknowledged.has(item.requirement_id)) continue;
      if (!this.unsent.has(item.requirement_id)) added.push(item.requirement_id);
      this.unsent.set(item.requirement_id, item);
    }
    return added;
  }

  /** Forget evidence for requirements the server now holds. */
  acknowledge(acknowledged: ReadonlySet<string>) {
    for (const id of Array.from(this.unsent.keys())) if (acknowledged.has(id)) this.unsent.delete(id);
  }

  /** Drop evidence a terminal validation failure rejected, unless a newer observation replaced it. */
  reject(sent: TutorialActionEvidence[]): string[] {
    const removed: string[] = [];
    for (const item of sent) {
      if (this.unsent.get(item.requirement_id) === item) {
        this.unsent.delete(item.requirement_id);
        removed.push(item.requirement_id);
      }
    }
    return removed;
  }

  snapshot(): TutorialActionEvidence[] {
    return Array.from(this.unsent.values());
  }

  pending(): Set<string> {
    return new Set(this.unsent.keys());
  }

  get size() {
    return this.unsent.size;
  }
}
