import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { FormOrderError, formOrderProblems, orderedEntries, orderedKeys } from "./formOrder.ts";
import type { Question } from "./types";

const sorted = (record: Record<string, string>) => Object.fromEntries(Object.entries(record).sort(([a], [b]) => a.localeCompare(b)));
const question = (id: string, options: Record<string, string> | null, extra: Partial<Question> = {}): Question => ({ id, label: id, options, multiple: false, required: true, show_if: null, matrix: null, ...extra });

const HELPFUL = { not_helpful: "Not helpful", slightly: "Slightly helpful", moderately: "Moderately helpful", very: "Very helpful", cannot_judge: "Did not use / cannot judge" };
const EFFORT = { very_low: "Very low", low: "Low", moderate: "Moderate", high: "High", very_high: "Very high", cannot_judge: "Cannot judge" };

test("legacy v1 forms stored with sorted keys render in their declared ordinal order (C05)", () => {
  const helpful = question("component_usefulness", sorted(HELPFUL), { matrix: sorted({ original_context: "a", entity_description: "b", hierarchy: "c", evidence_table: "d", evidence_graph: "e", pair_comparison: "f" }) });
  assert.deepEqual(Object.keys(helpful.options!), ["cannot_judge", "moderately", "not_helpful", "slightly", "very"]);
  assert.deepEqual(orderedKeys(helpful, "options", "exact-study-forms/1"), ["not_helpful", "slightly", "moderately", "very", "cannot_judge"]);
  assert.deepEqual(orderedKeys(helpful, "rows", "exact-study-forms/1"), ["original_context", "entity_description", "hierarchy", "evidence_table", "evidence_graph", "pair_comparison"]);
  const effort = question("mental_effort", sorted(EFFORT));
  assert.deepEqual(orderedKeys(effort, "options", "exact-study-forms/1"), ["very_low", "low", "moderate", "high", "very_high", "cannot_judge"]);
  assert.deepEqual(orderedEntries(effort.options, orderedKeys(effort, "options", "exact-study-forms/1"))[0], ["very_low", "Very low"]);
});

test("a publication with fewer components keeps the declared order of those present", () => {
  const most = question("most_helpful_components", sorted({ hierarchy: "c", original_context: "a", all_equally: "x", none_helpful: "y", cannot_judge: "z" }), { multiple: true });
  assert.deepEqual(orderedKeys(most, "options", "exact-study-forms/1"), ["original_context", "hierarchy", "all_equally", "none_helpful", "cannot_judge"]);
});

test("explicit v2 order wins over scrambled object keys and must be an exact permutation", () => {
  const scrambled = question("q", { c: "C", a: "A", b: "B" }, { option_order: ["b", "c", "a"] });
  assert.deepEqual(orderedKeys(scrambled, "options", "anything"), ["b", "c", "a"]);
  assert.throws(() => orderedKeys(question("q", { a: "A", b: "B" }, { option_order: ["a"] }), "options", "v"), FormOrderError);
  assert.throws(() => orderedKeys(question("q", { a: "A", b: "B" }, { option_order: ["a", "b", "b"] }), "options", "v"), FormOrderError);
  assert.throws(() => orderedKeys(question("q", { a: "A", b: "B" }, { option_order: ["a", "c"] }), "options", "v"), FormOrderError);
});

test("a form with no declared order fails visibly instead of sorting labels", () => {
  const unknown = question("new_question", { b: "B", a: "A" });
  assert.throws(() => orderedKeys(unknown, "options", "exact-study-forms/1"), FormOrderError);
  assert.deepEqual(formOrderProblems([unknown], "exact-study-forms/1").length, 1);
  assert.deepEqual(formOrderProblems([question("comments", null, { required: false })], "exact-study-forms/1"), []);
  const extra = question("cs_experience_years", { none: "None", invented: "Invented" });
  assert.throws(() => orderedKeys(extra, "options", "exact-study-forms/1"), FormOrderError);
});
