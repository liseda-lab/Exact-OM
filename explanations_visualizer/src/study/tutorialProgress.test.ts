import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { actionsFor, requirementsFor, evidenceFor } from "./tutorialProgress.ts";
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
