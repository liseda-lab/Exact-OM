import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { actionsFor, requirementsFor, evidenceFor, LessonEvidence } from "./tutorialProgress.ts";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { LESSONS, TUTORIAL_RESOURCE, tutorialPublic } from "./v2/tutorialContent.ts";

const lesson = (id: string) => LESSONS.find((item: { lesson_id: string }) => item.lesson_id === id)!;

test("lesson evidence names actual actions, scoped to the current lesson (C11)", () => {
  assert.deepEqual(requirementsFor(lesson("context"), { kind: "workspace", action: { type: "hierarchy_focus", side: "source", iri: "x", kind: "class", via: "child" } }), ["context.child"]);
  assert.deepEqual(requirementsFor(lesson("identity"), { kind: "workspace", action: { type: "hierarchy_focus", side: "source", iri: "x", kind: "class", via: "child" } }), []);
  assert.deepEqual(requirementsFor(lesson("identity"), { kind: "inspect", position: 3, returning: false }), ["identity.inspect"]);
  assert.deepEqual(requirementsFor(lesson("identity"), { kind: "inspect", position: 1, returning: true }), ["identity.return"]);
});

test("the graph lesson accepts the evidence list as an equivalent", () => {
  const listSelect = requirementsFor(lesson("graph"), { kind: "workspace", action: { type: "evidence_select", evidenceId: "e", from: "list" } });
  assert.ok(listSelect.includes("graph.inspect") && listSelect.includes("graph.view"));
  assert.deepEqual(requirementsFor(lesson("graph"), { kind: "rank", event: "rank_add", detailsOpen: true }), ["graph.rank"]);
});

test("answer-control and report lessons distinguish none, insufficient, partial and No", () => {
  assert.deepEqual(actionsFor({ kind: "rank", event: "response_type_change", element: "none_of_these", detailsOpen: false }), ["choose_none"]);
  assert.deepEqual(actionsFor({ kind: "rank", event: "response_type_change", element: "insufficient_evidence", detailsOpen: false }), ["choose_insufficient"]);
  assert.deepEqual(actionsFor({ kind: "rank", event: "revision", element: "undo", detailsOpen: false }), ["undo_rank"]);
  assert.deepEqual(actionsFor({ kind: "practice_check", partial: false }), []);
  assert.deepEqual(actionsFor({ kind: "report", consulted: true, methods: 1 }), []);
  assert.deepEqual(actionsFor({ kind: "report", consulted: true, methods: 2 }), ["report_multiple_methods"]);
  assert.deepEqual(actionsFor({ kind: "report", consulted: false, methods: 0 }), ["report_no_methods"]);
  assert.deepEqual(actionsFor({ kind: "workspace", action: { type: "locate_fact", factId: "f", inList: false } }), []);
});

test("backend requirement evidence preserves typed identities and the actual answer", () => {
  const tutorial = tutorialPublic([]);
  const candidate = tutorial.case.candidates[2];
  assert.deepEqual(evidenceFor(lesson("identity"), { kind: "inspect", position: 3, returning: false }, tutorial, [TUTORIAL_RESOURCE], null), [{ requirement_id: "identity.inspect", action: "inspect_other_candidate", candidate_id: candidate.candidate_id }]);
  const entity = tutorial.case.source;
  assert.deepEqual(evidenceFor(lesson("context"), { kind: "workspace", action: { type: "hierarchy_focus", side: "source", iri: entity.iri, kind: entity.kind, via: "return" } }, tutorial, [TUTORIAL_RESOURCE], null), [{ requirement_id: "context.return", action: "return_to_compared", entity }]);
  assert.deepEqual(evidenceFor(lesson("answers"), { kind: "rank", event: "response_type_change", element: "none_of_these", detailsOpen: false }, tutorial, [TUTORIAL_RESOURCE], { response_type: "none_of_these", ranked_candidate_ids: [] }), [{ requirement_id: "answers.none", action: "choose_none", response_type: "none_of_these", ranked_candidate_ids: [] }]);
  const report = evidenceFor(lesson("baseline"), { kind: "report", consulted: true, methods: 2, methodCodes: ["queries_scripts", "reasoner"] }, tutorial, [TUTORIAL_RESOURCE], null);
  assert.deepEqual(report[0].methods, ["queries_scripts", "reasoner"]);
});

// --- 21 F01: interaction is not evidence; identities are checked against the frozen resources ---

const practice = tutorialPublic([]);
const graph = lesson("graph");
const evidenceOf = (candidate: string, suffix: string) => TUTORIAL_RESOURCE.evidence.find((item: { evidence_id: string; candidate_id: string }) => item.candidate_id === candidate && item.evidence_id.endsWith(suffix))!;
const ws = (action: object, candidateId: string | null = "practice-c1") => ({ kind: "workspace" as const, action: action as never, candidateId });
const forGraph = (signal: object) => evidenceFor(graph, signal as never, practice, [TUTORIAL_RESOURCE], null);

test("selecting a comparison bridge or an expanded ontology edge completes nothing (U01)", () => {
  assert.deepEqual(forGraph(ws({ type: "graph_edge", edgeKind: "bridge", evidenceId: null, factId: null })), []);
  assert.deepEqual(forGraph(ws({ type: "graph_edge", edgeKind: "asserted", evidenceId: null, factId: null })), []);
  assert.deepEqual(forGraph(ws({ type: "graph_edge", edgeKind: "structural", evidenceId: null })), []);
  assert.deepEqual(requirementsFor(graph, ws({ type: "graph_edge", edgeKind: "bridge", evidenceId: null })), []);
});

test("a selected feature line reports the exact inspected record of the active candidate", () => {
  const item = evidenceOf("practice-c1", ".h.s");
  const result = forGraph(ws({ type: "graph_edge", edgeKind: "feature", evidenceId: item.evidence_id, factId: item.fact_ids[0] }));
  assert.deepEqual(result.map((entry: { requirement_id: string; fact_id?: string }) => [entry.requirement_id, entry.fact_id]), [["graph.inspect", item.fact_ids[0]], ["graph.view", item.fact_ids[0]]]);
});

test("a feature from another candidate's evidence set, or a record it does not link, is not evidence", () => {
  const other = evidenceOf("practice-c2", ".r.t");
  assert.deepEqual(forGraph(ws({ type: "graph_edge", edgeKind: "feature", evidenceId: other.evidence_id, factId: other.fact_ids[0] }, "practice-c1")), []);
  const own = evidenceOf("practice-c1", ".h.s");
  assert.deepEqual(forGraph(ws({ type: "graph_edge", edgeKind: "feature", evidenceId: own.evidence_id, factId: other.fact_ids[0] })), []);
  assert.equal(forGraph(ws({ type: "graph_edge", edgeKind: "feature", evidenceId: other.evidence_id, factId: other.fact_ids[0] }, "practice-c2")).length, 2);
});

test("the list alternative uses the record shown first; unresolved or missing records cannot satisfy it", () => {
  const item = evidenceOf("practice-c1", ".l.t");
  const listed = forGraph(ws({ type: "evidence_select", evidenceId: item.evidence_id, from: "list", factIds: item.fact_ids }));
  assert.ok(listed.every((entry: { fact_id?: string }) => entry.fact_id === item.fact_ids[0]));
  assert.deepEqual(forGraph(ws({ type: "evidence_select", evidenceId: item.evidence_id, from: "list", factIds: ["sha256:not-a-record"] })), []);
  const unresolved = { ...TUTORIAL_RESOURCE, evidence: TUTORIAL_RESOURCE.evidence.map((entry: typeof item) => (entry.evidence_id === item.evidence_id ? { ...entry, status: "unresolved" } : entry)) };
  assert.deepEqual(evidenceFor(graph, ws({ type: "evidence_select", evidenceId: item.evidence_id, from: "list" }) as never, practice, [unresolved], null), []);
  const unlinked = { ...TUTORIAL_RESOURCE, evidence: TUTORIAL_RESOURCE.evidence.map((entry: typeof item) => (entry.evidence_id === item.evidence_id ? { ...entry, fact_ids: [] } : entry)) };
  assert.deepEqual(evidenceFor(graph, ws({ type: "evidence_select", evidenceId: item.evidence_id, from: "list" }) as never, practice, [unlinked], null), []);
});

test("opening a record is not opening a citation; each needs its own admitted identity", () => {
  const evidenceLesson = lesson("evidence");
  const cited = TUTORIAL_RESOURCE.entity_profiles[0].fact_ids[0];
  const uncited = TUTORIAL_RESOURCE.facts.find((fact: { fact_id: string }) => !TUTORIAL_RESOURCE.entity_profiles.concat(TUTORIAL_RESOURCE.pair_comparison).some((claim: { fact_ids: string[] }) => claim.fact_ids.includes(fact.fact_id)))!.fact_id;
  const run = (action: object) => evidenceFor(evidenceLesson, ws(action) as never, practice, [TUTORIAL_RESOURCE], null).map((entry: { action: string; fact_id?: string }) => [entry.action, entry.fact_id]);
  assert.deepEqual(run({ type: "citation_open", factId: cited }), [["open_citation", cited]]);
  assert.deepEqual(run({ type: "citation_open", factId: uncited }), []);
  assert.deepEqual(run({ type: "axiom_open", factId: uncited }), [["open_original_axiom", uncited]]);
  assert.deepEqual(run({ type: "axiom_open", factId: "sha256:unknown" }), []);
});

test("reaching the compared entity again by a child link is a return; page load and other entities are not (U03)", () => {
  const context = lesson("context");
  const source = practice.case.source;
  const viaChild = evidenceFor(context, ws({ type: "hierarchy_focus", side: "source", iri: source.iri, kind: source.kind, via: "child", returned: true }) as never, practice, [TUTORIAL_RESOURCE], null);
  assert.deepEqual(viaChild.map((entry: { action: string }) => entry.action), ["navigate_child", "return_to_compared"]);
  assert.deepEqual(viaChild[1].entity, source);
  const notReturned = evidenceFor(context, ws({ type: "hierarchy_focus", side: "source", iri: source.iri, kind: source.kind, via: "child" }) as never, practice, [TUTORIAL_RESOURCE], null);
  assert.deepEqual(notReturned.map((entry: { action: string }) => entry.action), ["navigate_child"]);
  const parent = TUTORIAL_RESOURCE.hierarchy.find((edge: { child: { iri: string } }) => edge.child.iri === source.iri)!.parent;
  const elsewhere = evidenceFor(context, ws({ type: "hierarchy_focus", side: "source", iri: parent.iri, kind: parent.kind, via: "parent", returned: true }) as never, practice, [TUTORIAL_RESOURCE], null);
  assert.deepEqual(elsewhere.map((entry: { action: string }) => entry.action), ["navigate_parent"], "a return claim for a non-compared entity is dropped");
  assert.deepEqual(evidenceFor(context, ws({ type: "hierarchy_focus", side: "source", iri: "https://example.org/unknown", kind: "class", via: "parent" }) as never, practice, [TUTORIAL_RESOURCE], null), [], "not a recorded parent endpoint");
});

test("pending lesson evidence is bounded, never overwrites saved work and forgets rejected payloads (U02)", () => {
  const store = new LessonEvidence();
  const bad = { requirement_id: "graph.inspect", action: "inspect_graph_or_list" as const, fact_id: "f1" };
  assert.deepEqual(store.observe([bad], new Set()), ["graph.inspect"]);
  assert.deepEqual(store.observe([{ requirement_id: "graph.rank", action: "rank_with_details_open" as const }], new Set(["graph.rank"])), [], "already acknowledged: never resent");
  const sent = store.snapshot();
  const newer = { ...bad, fact_id: "f2" };
  store.observe([newer], new Set());
  assert.deepEqual(store.reject(sent), [], "a newer observation is not discarded by an older rejection");
  assert.deepEqual(store.reject(store.snapshot()), ["graph.inspect"]);
  assert.equal(store.size, 0, "a later save cannot carry rejected evidence");
  store.observe([newer], new Set());
  store.acknowledge(new Set(["graph.inspect"]));
  assert.deepEqual(store.snapshot(), []);
});
