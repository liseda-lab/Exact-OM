import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { FALLBACK_PRACTICE, practiceActionComplete } from "./practice.ts";
test("fallback practice is explicitly synthetic, ordered and separate from study identities", () => {
  assert.deepEqual(FALLBACK_PRACTICE.map((item) => item.kind), ["simple", "complex", "partial_ranking", "none_of_these"]);
  for (const item of FALLBACK_PRACTICE) {
    assert.equal(item.synthetic, true);
    assert.equal(item.candidates.length, 5);
    assert.equal(new Set(item.candidates.map((candidate) => candidate.candidate_id)).size, 5);
    assert.ok(item.source.description && item.candidates.every((candidate) => candidate.description));
    assert.ok(!("score" in item.candidates[0]));
  }
});
test("practice completion requires intentional partial and explicit-none actions", () => {
  assert.equal(practiceActionComplete("simple", { responseType: null, ranked: [] }), false);
  assert.equal(practiceActionComplete("partial_ranking", { responseType: "ranked_candidates", ranked: ["a", "b", "c", "d", "e"] }), false);
  assert.equal(practiceActionComplete("partial_ranking", { responseType: "ranked_candidates", ranked: ["a", "b"] }), true);
  assert.equal(practiceActionComplete("none_of_these", { responseType: "insufficient_evidence", ranked: [] }), false);
  assert.equal(practiceActionComplete("none_of_these", { responseType: "none_of_these", ranked: [] }), true);
});
