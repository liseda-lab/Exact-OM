import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { formsGeneration, isReviewProtocol, unsupportedProtocol } from "./types.ts";

test("forms families include frozen component variants; unknown protocols fail visibly (23 C04)", () => {
  assert.equal(formsGeneration("exact-study-forms/2"), 2);
  assert.equal(formsGeneration("exact-study-forms/2-cmp"), 2);
  assert.equal(formsGeneration("exact-study-forms/3"), 3);
  assert.equal(formsGeneration("exact-study-forms/30x"), null);
  const state = (forms: string, versions: Record<string, string> = {}) => ({ forms: { version: forms }, protocol_versions: versions }) as never;
  assert.deepEqual(unsupportedProtocol(state("exact-study-forms/2-cmp")), []);
  assert.deepEqual(unsupportedProtocol(state("exact-study-forms/3", { setup: "setup/3", consultation: "consultation/3", methods: "study-methods/3" })), []);
  assert.deepEqual(unsupportedProtocol(state("exact-study-forms/4")), ["forms exact-study-forms/4"]);
  assert.deepEqual(unsupportedProtocol(state("exact-study-forms/3", { consultation: "consultation/4" })), ["consultation consultation/4"]);
  assert.equal(isReviewProtocol(state("exact-study-forms/3-cmp", { methods: "study-methods/3" })), true);
  assert.equal(isReviewProtocol(state("exact-study-forms/2", { methods: "study-methods/3" })), false);
});
