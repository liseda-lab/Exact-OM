import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { retainCurrentTiming, splitTiming } from "./timing.ts";
const open = { stage: "case", caseId: "c", presentationId: "p", start: 1000 };
test("submit freezes timing and a later acknowledgement cannot extend the case", () => {
  const result = splitTiming(open, 2500, true);
  assert.equal(result.interval?.end, 2500);
  assert.equal(result.next, null);
  assert.equal(splitTiming(result.next, 15000).interval, null);
});
test("consecutive observations do not overlap and preserve the ready timestamp", () => {
  const first = splitTiming(open, 61000);
  const second = splitTiming(first.next, 121000);
  assert.equal(first.interval?.start, 1000);
  assert.equal(second.interval?.start, first.interval?.end);
});
test("overnight timer suspension is an unknown gap rather than ordinary task time", () => {
  const resumed = splitTiming(open, 12 * 60 * 60 * 1000);
  assert.equal(resumed.interval, null);
  const observed = splitTiming(resumed.next, 12 * 60 * 60 * 1000 + 60000);
  assert.equal(observed.interval!.end - observed.interval!.start, 60000);
});

test("a quick valid ranking retains its observed interval below 250 milliseconds", () => {
  const result = splitTiming(open, 1042.5, true);
  assert.equal(result.interval?.end, 1042.5);
  assert.equal(result.interval?.start, 1000);
  assert.equal(result.next, null);
});

test("a parent stage effect preserves the same case's already usable interval", () => {
  assert.equal(retainCurrentTiming(open, "case", "p"), open);
  assert.equal(retainCurrentTiming(open, "case", "next-presentation"), null);
  assert.equal(retainCurrentTiming(open, "paused", "p"), null);
  assert.equal(retainCurrentTiming(null, "case", "p"), null);
});
