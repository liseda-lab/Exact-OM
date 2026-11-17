import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { formatScore } from "./scorePresentation.ts";

test("the final neuroaxonal ranking does not appear to contain exact endpoints", () => {
  assert.deepEqual([0.9999269198113435, 0.000021137052289676113, 0.00001183403142896325, 0.000008187251058416894, 0.000008139792035560275].map(formatScore),
    ["≈1.00", "≈0.00", "≈0.00", "≈0.00", "≈0.00"]);
});

test("recorded endpoints and ordinary scores keep their existing presentation", () => {
  assert.deepEqual([0, 1, 0.5, 0.126, 125].map(formatScore), ["0.00", "1.00", "0.50", "0.13", "125"]);
  assert.equal(formatScore(0.0049), "≈0.00");
  assert.equal(formatScore(0.0051), "0.01");
  assert.equal(formatScore(0.994), "0.99");
});
