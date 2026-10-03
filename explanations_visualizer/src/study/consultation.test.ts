import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { consultationProblem, EMPTY_CONSULTATION, normalizeConsultation } from "./consultation.ts";

test("No clears every method detail and the optional scope (C13)", () => {
  const answer = { ...EMPTY_CONSULTATION, consulted_external_ontologies: false, methods: ["protege" as const], other_editor: "x", resource_scope: "unsure" as const };
  assert.deepEqual(normalizeConsultation(answer), { ...EMPTY_CONSULTATION, consulted_external_ontologies: false });
  assert.equal(consultationProblem(normalizeConsultation(answer)), null);
});

test("Yes keeps several methods and only the names of selected methods", () => {
  const answer = { ...EMPTY_CONSULTATION, consulted_external_ontologies: true, methods: ["protege", "queries_scripts", "other_method", "protege"] as never[], other_editor: "hidden", other_method: "  notebook  ", resource_scope: null };
  const normalized = normalizeConsultation(answer);
  assert.deepEqual(normalized.methods, ["protege", "queries_scripts", "other_method"]);
  assert.equal(normalized.other_editor, null);
  assert.equal(normalized.other_method, "notebook");
  assert.equal(normalized.resource_scope, null, "an unanswered optional scope stays unknown, not supplied-only");
});

test("final reports need Yes with a method or No; drafts may be incomplete", () => {
  assert.match(consultationProblem(EMPTY_CONSULTATION)!, /Yes or No/);
  assert.match(consultationProblem({ ...EMPTY_CONSULTATION, consulted_external_ontologies: true })!, /at least one method/);
  assert.equal(consultationProblem({ ...EMPTY_CONSULTATION, consulted_external_ontologies: true, methods: ["reasoner" as const] }), null);
});
