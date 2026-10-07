import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { contextBindingProblem, contextProblem, explanationProblem, inBatches, renderExpectations, requiredContent, validateCapabilities, validateCase, workspaceCapabilities } from "./caseReadiness.ts";

// The backend's actual serialized HTTP examples (S1 handoff), not hand-written shapes.
const examples = JSON.parse(readFileSync(new URL("../../../docs/verification/explanation-integration-backend-examples.json", import.meta.url), "utf8")).examples;
const explanationCase = examples.explanation_case.body;
const baselineCase = examples.baseline_case.body;
const capabilities = examples.capabilities.body;
const ALL = new Set(["original_context", "entity_description", "hierarchy", "evidence_table", "evidence_graph", "pair_comparison"] as const);

const sha = "a".repeat(64);
const stateFor = (current: Record<string, any>) => ({
  contract_version: "exact-study/2.0",
  study_revision: current.study_revision,
  current_case_id: current.case_id,
  current_presentation_id: current.presentation_id,
  ontology_resources: current.ontology_resource_ids.map((id: string) => ({ asset_id: id, sha256: sha, size_bytes: 10 })),
});
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));

test("the recorded v2 explanation and baseline cases are accepted for their own presentation (J01)", () => {
  assert.deepEqual(validateCase(stateFor(explanationCase), explanationCase), { reject: [], block: [] });
  assert.deepEqual(validateCase(stateFor(baselineCase), baselineCase), { reject: [], block: [] });
});

test("a v2 explanation case without its descriptor is blocked, never re-routed (R01)", () => {
  const missing = clone(explanationCase);
  delete missing.workspace;
  const result = validateCase(stateFor(missing), missing);
  assert.deepEqual(result.reject, []);
  assert.match(result.block.join(" "), /did not identify this case's workspace/);
  const leaking = clone(baselineCase);
  leaking.workspace = { scope_id: "other" };
  assert.match(validateCase(stateFor(leaking), leaking).block.join(" "), /without explanations was sent explanation resources/);
});

test("cross-presentation, cross-revision and malformed candidate lists are rejected", () => {
  const state = stateFor(explanationCase);
  assert.match(validateCase({ ...state, current_presentation_id: "another" }, explanationCase).reject.join(" "), /not the presentation/);
  assert.match(validateCase({ ...state, study_revision: "another" }, explanationCase).reject.join(" "), /different study revision/);
  const four = clone(explanationCase);
  four.candidates.pop();
  assert.match(validateCase(state, four).reject.join(" "), /4 candidates instead of five/);
  const duplicate = clone(explanationCase);
  duplicate.candidates[1].entity = duplicate.candidates[0].entity;
  assert.match(validateCase(state, duplicate).reject.join(" "), /same entity/);
  const positions = clone(explanationCase);
  positions.candidates[4].display_position = 1;
  assert.match(validateCase(state, positions).reject.join(" "), /display positions/);
});

test("download metadata must be listed and verifiable", () => {
  const state = stateFor(explanationCase);
  assert.match(validateCase({ ...state, ontology_resources: state.ontology_resources.slice(1) }, explanationCase).block.join(" "), /not listed/);
  assert.match(validateCase({ ...state, ontology_resources: state.ontology_resources.map((asset: object) => ({ ...asset, sha256: "" })) }, explanationCase).block.join(" "), /verifiable size and hash/);
});

test("the recorded capabilities match their case; any mismatch is incompatible (F11)", () => {
  const expected = { scopeId: explanationCase.workspace.scope_id, studyRevision: explanationCase.study_revision, current: explanationCase, components: ALL };
  assert.deepEqual(validateCapabilities(capabilities, expected), []);
  assert.match(validateCapabilities({ ...capabilities, scope_id: "other" }, expected).join(" "), /different scope/);
  assert.match(validateCapabilities({ ...capabilities, study_revision: "other" }, expected).join(" "), /different study revision/);
  assert.match(validateCapabilities({ ...capabilities, focal_entities: capabilities.focal_entities.slice(0, 5) }, expected).join(" "), /different source or candidates/);
  assert.match(validateCapabilities({ ...capabilities, ontology_versions: [] }, expected).join(" "), /ontologies/);
  assert.match(validateCapabilities({ ...capabilities, components: ["context", "hierarchy"] }, expected).join(" "), /entity description/);
  assert.deepEqual(validateCapabilities({ ...capabilities, components: ["context", "hierarchy"] }, { ...expected, components: new Set(["original_context", "hierarchy"] as const) }), []);
});

test("scope status is honest: complete for the frozen filtered scope, no inference claimed", () => {
  const caps = workspaceCapabilities(capabilities, () => "Ontology");
  assert.match(caps.scopeNote ?? "", /declared context of the study's frozen, filtered/);
  assert.doesNotMatch(caps.scopeNote ?? "", /unrestricted|all ontologies/);
  assert.deepEqual(caps.bases, ["literal_asserted", "structural_navigation"]);
  assert.equal(caps.reasoner, "not_run");
  assert.ok(caps.limitations.some((item: string) => /Inferred hierarchy/.test(item)));
});

test("the bounded minimum is six contexts plus admitted descriptions and comparisons (F12)", () => {
  const all = requiredContent(explanationCase, ALL, capabilities);
  assert.equal(all.filter((item: { kind: string }) => item.kind === "context").length, 6);
  assert.equal(all.filter((item: { kind: string }) => item.kind === "profile").length, 6);
  assert.equal(all.filter((item: { kind: string }) => item.kind === "comparison").length, 5);
  assert.equal(new Set(all.map((item: { key: string }) => item.key)).size, all.length, "no duplicate reads");
  assert.equal(requiredContent(explanationCase, new Set(["original_context", "hierarchy"] as const), capabilities).length, 6);
  assert.equal(requiredContent(explanationCase, ALL, { ...capabilities, components: ["context"] }).length, 6, "only admitted components are required");
});

test("terminal absence is usable; a dangling, mismatched or unknown result is not", () => {
  const source = explanationCase.source;
  const target = explanationCase.candidates[0].entity;
  assert.equal(explanationProblem({ status: "not_exported", reason: "No resource was prepared for this selection" }, "entity_profile", [source]), null);
  assert.equal(explanationProblem({ status: "not_requested" }, "pair_comparison", [source, target]), null);
  assert.equal(explanationProblem({ status: "unverified" }, "pair_comparison", [source, target]), null);
  const available = (task: string, entities: object[]) => ({ status: "available", explanation: { task, entities, claims: [] }, provenance: {}, factOntologies: [] }) as never;
  assert.equal(explanationProblem(available("pair_comparison", [source, target]), "pair_comparison", [source, target]), null);
  assert.match(explanationProblem(available("entity_profile", [source]), "pair_comparison", [source, target]) ?? "", /different task/);
  assert.match(explanationProblem(available("pair_comparison", [source, explanationCase.candidates[1].entity]), "pair_comparison", [source, target]) ?? "", /different entities/);
  assert.match(explanationProblem({ status: "mystery" } as never, "entity_profile", [source]) ?? "", /unknown status/);
  assert.equal(contextBindingProblem({ entity: source }, source), null);
  assert.match(contextBindingProblem({ entity: target }, source) ?? "", /different entity/);
  assert.match(contextBindingProblem({}, source) ?? "", /no entity identity/);
});

test("required reads run with capped concurrency and keep their order", async () => {
  let active = 0;
  let peak = 0;
  const results = await inBatches([1, 2, 3, 4, 5, 6, 7], 3, async (value: number) => {
    active += 1;
    peak = Math.max(peak, active);
    await new Promise((resolve) => setTimeout(resolve, 5 * (8 - value)));
    active -= 1;
    if (value === 5) throw new Error("five failed");
    return value * 10;
  });
  assert.ok(peak <= 3);
  assert.deepEqual(results.map((item: PromiseSettledResult<number>) => (item.status === "fulfilled" ? item.value : "failed")), [10, 20, 30, 40, "failed", 60, 70]);
});

const pageOf = (items: unknown[] = [], next_cursor: string | null = null) => ({ items, next_cursor, total_count: items.length, returned_count: items.length, truncated: Boolean(next_cursor), status: "available", reason: null, scope: {} });
const validContext = (entity: object) => ({
  artifact_type: "entity_context",
  entity,
  preferred_label: { status: "available", value: "x" },
  labels: [],
  definitions: pageOf(),
  synonyms: pageOf(),
  parents: pageOf(),
  categories: { definitions: pageOf(), comments: pageOf([], "cursor") },
  completeness: { scope: "root", extraction: "complete_for_policy", imports_complete: true },
});

test("a 200 context missing a required collection is unusable, not absent (R08)", () => {
  const entity = explanationCase.source;
  assert.equal(contextProblem(validContext(entity), entity), null);
  for (const field of ["synonyms", "definitions", "parents", "categories", "completeness", "preferred_label", "labels"]) {
    const broken = validContext(entity) as Record<string, unknown>;
    delete broken[field];
    assert.ok(contextProblem(broken, entity), `missing ${field}`);
  }
  const badPage = validContext(entity) as Record<string, any>;
  badPage.categories.comments = { next_cursor: null };
  assert.match(contextProblem(badPage, entity) ?? "", /categories are incomplete/);
  assert.match(contextProblem(validContext(explanationCase.candidates[0].entity), entity) ?? "", /different entity/);
  assert.match(contextProblem(null, entity) ?? "", /unreadable/);
});

test("render expectations follow the admitted components exactly (R09)", () => {
  assert.deepEqual(renderExpectations(ALL), { cards: true, profiles: true, comparison: true });
  assert.deepEqual(renderExpectations(new Set(["pair_comparison"] as const)), { cards: false, profiles: false, comparison: true });
  assert.deepEqual(renderExpectations(new Set(["original_context"] as const)), { cards: true, profiles: false, comparison: false });
  assert.deepEqual(renderExpectations(new Set(["entity_description"] as const)), { cards: true, profiles: true, comparison: false });
  assert.deepEqual(renderExpectations(new Set(["hierarchy", "evidence_table", "evidence_graph"] as const)), { cards: false, profiles: false, comparison: false });
  // The required reads match: descriptions and comparisons only when admitted.
  const kinds = (components: Set<string>) => requiredContent(explanationCase, components as never, capabilities).map((item: { kind: string }) => item.kind);
  assert.deepEqual([...new Set(kinds(new Set(["pair_comparison"])))], ["context", "comparison"]);
  assert.deepEqual([...new Set(kinds(new Set(["entity_description"])))], ["context", "profile"]);
  assert.deepEqual([...new Set(kinds(new Set(["hierarchy"])))], ["context"]);
});

