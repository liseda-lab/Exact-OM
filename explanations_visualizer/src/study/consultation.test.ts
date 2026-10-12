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
  assert.match(consultationProblem(EMPTY_CONSULTATION)!.message, /Yes or No/);
  assert.match(consultationProblem({ ...EMPTY_CONSULTATION, consulted_external_ontologies: true })!.message, /at least one method/);
  assert.equal(consultationProblem({ ...EMPTY_CONSULTATION, consulted_external_ontologies: true, methods: ["reasoner" as const] }), null);
});

test("consultation/3 carries exactly its own fields; Other needs a name only when selected (22 B06)", () => {
  const yes = { ...EMPTY_CONSULTATION, consulted_external_ontologies: true, methods: ["protege", "other_method", "reasoner"] as const, other_method: "  owlready2, a SPARQL notebook  ", other_editor: "kept from an old draft", resource_scope: "supplied_only" as const };
  const normalized = normalizeConsultation({ ...yes, methods: [...yes.methods] }, true);
  assert.deepEqual(normalized, { consulted_external_ontologies: true, methods: ["protege", "other_method"], other_method: "owlready2, a SPARQL notebook", resource_scope: "supplied_only" });
  assert.equal("other_editor" in normalized, false, "no consultation/2 field is ever sent to a consultation/3 service");
  assert.equal(consultationProblem(normalized, true), null);
  assert.equal(consultationProblem({ ...normalized, other_method: "  " }, true)?.field, "other_method");
  // Deselecting Other clears its name; No clears every detail.
  assert.equal(normalizeConsultation({ ...normalized, methods: ["plain_files"] }, true).other_method, null);
  assert.deepEqual(normalizeConsultation({ ...normalized, consulted_external_ontologies: false }, true), { consulted_external_ontologies: false, methods: [], other_method: null, resource_scope: null });
  assert.equal(consultationProblem({ consulted_external_ontologies: false, methods: [], other_method: null, resource_scope: null }, true), null);
});
