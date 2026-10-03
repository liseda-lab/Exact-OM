import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { ASSESSMENT, ASSESSMENT_KEYS, gradeResponse, LESSONS, TUTORIAL_CANDIDATES, TUTORIAL_RESOURCE } from "./tutorialContent.ts";

test("practice material is synthetic and disjoint from real identities", () => {
  for (const entity of TUTORIAL_RESOURCE.entities) assert.match(entity.iri, /^https:\/\/example\.org\/practice\//);
  for (const fact of TUTORIAL_RESOURCE.facts) assert.match(fact.fact_id, /^practice\./);
  for (const candidate of TUTORIAL_CANDIDATES) assert.match(candidate.candidate_id, /^practice-/);
  assert.match(TUTORIAL_RESOURCE.limitations[0], /invented/);
});

test("every claim, edge and evidence item cites records that exist", () => {
  const ids = new Set(TUTORIAL_RESOURCE.facts.map((fact: { fact_id: string }) => fact.fact_id));
  for (const claim of [...TUTORIAL_RESOURCE.entity_profiles, ...TUTORIAL_RESOURCE.pair_comparison]) assert.ok(claim.fact_ids.every((id: string) => ids.has(id)), claim.claim_id);
  for (const edge of TUTORIAL_RESOURCE.hierarchy) assert.ok(edge.fact_ids.every((id: string) => ids.has(id)));
  for (const item of TUTORIAL_RESOURCE.evidence) assert.ok(item.fact_ids.every((id: string) => ids.has(id)));
  const keys = TUTORIAL_RESOURCE.facts.map((fact: { fact_id: string; subject: { iri: string } }) => `${fact.fact_id}|${fact.subject.iri}`);
  assert.equal(new Set(keys).size, keys.length, "one copy per fact and subject");
});

test("comparison claims use the backend's fixed template wording", () => {
  for (const claim of TUTORIAL_RESOURCE.pair_comparison) {
    assert.equal(claim.grounding, "semantic_template");
    assert.match(claim.text, /(Shared wording alone does not establish equivalence|Different wording alone does not establish incompatibility|One-sided information does not establish incompatibility)\.$/);
  }
});

test("six lessons, the graph lesson optional, and the five items in the frozen order (14)", () => {
  assert.deepEqual(LESSONS.map((lesson: { lesson_id: string }) => lesson.lesson_id), ["identity", "context", "evidence", "graph", "answers", "baseline"]);
  assert.deepEqual(LESSONS.filter((lesson: { optional?: boolean }) => lesson.optional).map((lesson: { lesson_id: string }) => lesson.lesson_id), ["graph"]);
  assert.deepEqual(ASSESSMENT.map((item: { question_id: string }) => item.question_id), ["score_meaning", "equivalence_scope", "response_states", "evidence_origin", "external_methods"]);
  for (const item of ASSESSMENT) assert.ok(LESSONS.some((lesson: { lesson_id: string }) => lesson.lesson_id === item.lesson_id), `${item.question_id} links back to a lesson`);
});

test("pass predicates match the programme's assessment table, independent of option position", () => {
  assert.equal(gradeResponse("score_meaning", { choice: "advice" }), true);
  assert.equal(gradeResponse("score_meaning", { choice: "probability" }), false);
  assert.equal(gradeResponse("equivalence_scope", { choice: "broader" }), true);
  assert.equal(gradeResponse("response_states", { matches: { judged_none: "none_of_these", cannot_judge: "insufficient_evidence", not_answered: "unanswered" } }), true);
  assert.equal(gradeResponse("response_states", { matches: { judged_none: "insufficient_evidence", cannot_judge: "none_of_these", not_answered: "unanswered" } }), false);
  assert.equal(gradeResponse("evidence_origin", { matches: { statement: "original", feature: "matcher", summary: "generated" }, part_b: "yes" }), false);
  assert.equal(gradeResponse("evidence_origin", { matches: { statement: "original", feature: "matcher", summary: "generated" }, part_b: "no" }), true);
  assert.equal(gradeResponse("external_methods", { choices: ["per_case", "optional", "combine_change"] }), true);
  assert.equal(gradeResponse("external_methods", { choices: ["optional", "combine_change", "per_case", "protege_required"] }), false);
  assert.equal(gradeResponse("external_methods", { choices: ["optional", "combine_change"] }), false);
  for (const [id, key] of Object.entries(ASSESSMENT_KEYS) as [string, { right: string; wrong: string }][]) assert.ok(key.right && key.wrong, id);
});
