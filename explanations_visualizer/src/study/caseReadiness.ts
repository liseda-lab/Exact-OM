// Required content for one scored presentation (19 F11/F12). Pure validation and planning,
// so the rules can be tested without a browser: which responses must be present and valid
// before ranking is enabled and `case_ready` may be reported, and which capability or case
// responses are incompatible with the presentation they claim to describe.

import type { EntityRef } from "../lib/types";
import type { ExplanationResult, WorkspaceCapabilities } from "../lib/workspace/types";
import type { PublicAsset, StudyCase, StudyState } from "./types";

/** Components a frozen study condition can display (final-form matrix codes). */
export type CaseComponent = "original_context" | "entity_description" | "hierarchy" | "evidence_table" | "evidence_graph" | "pair_comparison";

/** The workspace capability component that authorizes each displayed component. */
export const CAPABILITY_FOR: Record<CaseComponent, string> = {
  original_context: "context",
  entity_description: "profiles",
  hierarchy: "hierarchy",
  evidence_table: "evidence",
  evidence_graph: "evidence",
  pair_comparison: "comparison",
};

/** The scoped capabilities response (18 B11), as serialized by the study service. */
export interface ScopeCapabilities {
  contract_version: string;
  scope_id: string;
  study_revision: string;
  policy_hash: string;
  synthetic: boolean;
  focal_entities: EntityRef[];
  components: string[];
  page_limit: number;
  ontology_versions: string[];
  bases: string[];
  reasoner: string;
  reason: string | null;
  completeness: Record<string, { scope?: string; extraction?: string; imports_complete?: boolean; domain_completeness?: string }>;
  coverage: string;
}

export const identity = (entity: Pick<EntityRef, "ontology_version_id" | "iri" | "kind">) => `${entity.ontology_version_id}\u0000${entity.kind}\u0000${entity.iri}`;

const isEntity = (value: unknown): value is EntityRef => {
  const record = value as Record<string, unknown> | null;
  return Boolean(record) && typeof record!.ontology_version_id === "string" && typeof record!.iri === "string" && typeof record!.kind === "string" && Boolean(record!.iri);
};

const SHA256 = /^(sha256:)?[0-9a-f]{64}$/;

/**
 * Problems with a current-case response. `reject`: the response does not describe this
 * session's presentation and must not be shown. `block`: the case may be shown but not
 * answered (missing workspace descriptor, unverifiable downloads, cross-condition data).
 */
export function validateCase(state: Pick<StudyState, "contract_version" | "study_revision" | "current_case_id" | "current_presentation_id" | "ontology_resources">, current: StudyCase): { reject: string[]; block: string[] } {
  const reject: string[] = [];
  const block: string[] = [];
  const contract = state.contract_version ?? "exact-study/1.0";
  if (!current || typeof current !== "object") return { reject: ["The case response is unreadable."], block };
  if ((current.contract_version ?? "exact-study/1.0") !== contract) reject.push("The case belongs to a different study protocol version.");
  if (current.study_revision !== state.study_revision) reject.push("The case belongs to a different study revision.");
  if (current.case_id !== state.current_case_id || current.presentation_id !== state.current_presentation_id) reject.push("The case is not the presentation this session is on.");
  if (!isEntity(current.source)) reject.push("The source concept is missing its identity.");
  const candidates = Array.isArray(current.candidates) ? current.candidates : [];
  if (candidates.length !== 5) reject.push(`The case lists ${candidates.length} candidates instead of five.`);
  if (new Set(candidates.map((item) => item.candidate_id)).size !== candidates.length) reject.push("Two candidates share an identifier.");
  if (candidates.some((item) => !isEntity(item.entity))) reject.push("A candidate is missing its identity.");
  else if (new Set(candidates.map((item) => identity(item.entity))).size !== candidates.length) reject.push("Two candidates are the same entity.");
  const positions = candidates.map((item) => item.display_position).sort((a, b) => a - b);
  if (positions.some((position, index) => position !== index + 1)) reject.push("The candidates' display positions are not 1 to 5.");
  if (reject.length) return { reject, block };
  const assets = new Map<string, PublicAsset>(state.ontology_resources.map((asset) => [asset.asset_id, asset]));
  for (const id of current.ontology_resource_ids ?? []) {
    const asset = assets.get(id);
    if (!asset) block.push(`The ontology file “${id}” for this case is not listed for this session.`);
    else if (!SHA256.test(asset.sha256) || !(asset.size_bytes > 0)) block.push(`The ontology file “${id}” has no verifiable size and hash.`);
  }
  if (contract === "exact-study/2.0") {
    if (current.condition === "explanation" && !(typeof current.workspace?.scope_id === "string" && current.workspace.scope_id)) block.push("The study service did not identify this case's workspace.");
    if (current.condition !== "explanation" && (current.workspace || (current.explanation_refs ?? []).length)) block.push("A case without explanations was sent explanation resources.");
  }
  return { reject, block };
}

/** Problems with a capabilities response for this presentation; empty when compatible. */
export function validateCapabilities(caps: ScopeCapabilities, expected: { scopeId: string; studyRevision: string; current: StudyCase; components: Iterable<CaseComponent> }): string[] {
  const problems: string[] = [];
  if (!caps || typeof caps !== "object") return ["The workspace description is unreadable."];
  if (caps.contract_version !== "exact-explain/1.0") problems.push("The workspace uses an unsupported read contract.");
  if (caps.scope_id !== expected.scopeId) problems.push("The workspace describes a different scope.");
  if (caps.study_revision !== expected.studyRevision) problems.push("The workspace belongs to a different study revision.");
  if (typeof caps.policy_hash !== "string" || !caps.policy_hash) problems.push("The workspace has no information policy.");
  const versions = new Set(Array.isArray(caps.ontology_versions) ? caps.ontology_versions : []);
  const focal = [expected.current.source, ...expected.current.candidates.map((item) => item.entity)];
  if (focal.some((entity) => !versions.has(entity.ontology_version_id))) problems.push("The workspace does not cover this case's ontologies.");
  const listed = new Set((Array.isArray(caps.focal_entities) ? caps.focal_entities : []).filter(isEntity).map(identity));
  if (listed.size !== focal.length || focal.some((entity) => !listed.has(identity(entity)))) problems.push("The workspace is for a different source or candidates.");
  const admitted = new Set(Array.isArray(caps.components) ? caps.components : []);
  for (const component of expected.components) {
    if (!admitted.has(CAPABILITY_FOR[component])) problems.push(`The workspace does not authorize ${component.replace(/_/g, " ")}, which this condition shows.`);
  }
  if (!Array.isArray(caps.bases) || typeof caps.reasoner !== "string" || typeof caps.coverage !== "string") problems.push("The workspace description is incomplete.");
  return problems;
}

/** Describe the authorized scope honestly: complete for the frozen filtered scope, no more. */
export function workspaceCapabilities(caps: ScopeCapabilities, ontologyName: (id: string) => string): WorkspaceCapabilities {
  const has = (component: string) => (caps.components.includes(component) ? "available" : "not_exported");
  const limitations: string[] = [];
  if (caps.reasoner !== "per_ontology" && caps.reason) limitations.push(`Inferred hierarchy: ${caps.reason}. Only asserted and structural relations are shown.`);
  for (const [version, entry] of Object.entries(caps.completeness ?? {})) {
    if (entry.imports_complete === false) limitations.push(`${ontologyName(version)}: imported ontologies are not complete in this study copy.`);
    if (entry.extraction && entry.extraction !== "complete_for_policy") limitations.push(`${ontologyName(version)}: extraction is ${entry.extraction.replace(/_/g, " ")}.`);
  }
  if (Object.values(caps.completeness ?? {}).some((entry) => entry.domain_completeness === "not_established")) limitations.push("Completeness relative to the whole domain is not established; something absent here may still exist elsewhere.");
  return {
    search: "available",
    navigation: "available",
    bases: caps.bases.filter((basis): basis is WorkspaceCapabilities["bases"][number] => ["literal_asserted", "structural_navigation", "reasoner_inferred"].includes(basis)),
    reasoner: caps.reasoner,
    graphExpansion: true,
    profiles: has("profiles"),
    comparison: has("comparison"),
    evidence: has("evidence"),
    scopeNote:
      caps.coverage === "complete_declared_context_scope"
        ? "Search and browsing cover the complete declared context of the study's frozen, filtered ontology copies. That is the study's scope, not every ontology release."
        : `Coverage of this workspace: ${caps.coverage.replace(/_/g, " ")}.`,
    limitations,
    synthetic: caps.synthetic,
  };
}

export type RequiredKind = "context" | "profile" | "comparison";

export interface RequiredItem {
  key: string;
  kind: RequiredKind;
  entities: EntityRef[];
  /** Plain description used when this item fails. */
  label: string;
}

/** The bounded minimum for an explanation case: six contexts, and the admitted profiles and comparisons. */
export function requiredContent(current: StudyCase, components: Set<CaseComponent>, caps: ScopeCapabilities): RequiredItem[] {
  const sorted = [...current.candidates].sort((a, b) => a.display_position - b.display_position);
  const focal: { entity: EntityRef; name: string }[] = [{ entity: current.source, name: "the source concept" }, ...sorted.map((item) => ({ entity: item.entity, name: `candidate ${item.display_position}` }))];
  const items: RequiredItem[] = focal.map(({ entity, name }) => ({ key: `context|${identity(entity)}`, kind: "context", entities: [entity], label: `Ontology information for ${name}` }));
  if (components.has("entity_description") && caps.components.includes("profiles"))
    items.push(...focal.map(({ entity, name }): RequiredItem => ({ key: `profile|${identity(entity)}`, kind: "profile", entities: [entity], label: `Generated description of ${name}` })));
  if (components.has("pair_comparison") && caps.components.includes("comparison"))
    items.push(...sorted.map((item): RequiredItem => ({ key: `comparison|${identity(current.source)}|${identity(item.entity)}`, kind: "comparison", entities: [current.source, item.entity], label: `Comparison with candidate ${item.display_position}` })));
  return items;
}

/** A 200 response is still unusable when it describes another entity. */
export function contextBindingProblem(ctx: { entity?: unknown } | null | undefined, entity: EntityRef): string | null {
  if (!ctx || !isEntity(ctx.entity)) return "The response has no entity identity.";
  return identity(ctx.entity) === identity(entity) ? null : "The response describes a different entity.";
}

/** Terminal honest statuses are usable; anything else is a failure. */
export function explanationProblem(result: ExplanationResult | null | undefined, task: "entity_profile" | "pair_comparison", entities: EntityRef[]): string | null {
  if (!result || typeof result !== "object") return "The response is unreadable.";
  if (result.status === "unverified" || result.status === "not_requested" || result.status === "not_exported") return null;
  if (result.status !== "available") return "The response has an unknown status.";
  if (result.explanation.task !== task) return "The response is for a different task.";
  const wanted = new Set(entities.map(identity));
  const subjects = Array.isArray(result.explanation.entities) ? result.explanation.entities.filter(isEntity).map(identity) : [];
  return subjects.length && [...wanted].every((key) => subjects.includes(key)) ? null : "The response describes different entities.";
}

/** Run loaders with bounded concurrency; results keep input order. */
export async function inBatches<T, R>(items: T[], limit: number, run: (item: T) => Promise<R>): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const index = next++;
      try {
        results[index] = { status: "fulfilled", value: await run(items[index]) };
      } catch (reason) {
        results[index] = { status: "rejected", reason };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
  return results;
}
