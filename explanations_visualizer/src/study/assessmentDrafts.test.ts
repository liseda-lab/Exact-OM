import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { AssessmentDrafts, canonical } from "./assessmentDrafts.ts";

const empty = {};

test("a visible edit survives leaving the question before its save is acknowledged (S0-R2)", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q1", { choice: "a" });
  drafts.sent("q1");
  // Next/Previous: the card remounts and reads the store, not the stale server draft.
  assert.deepEqual(drafts.value("q1", undefined, empty), { choice: "a" });
  drafts.reconcile({});
  assert.deepEqual(drafts.value("q1", undefined, empty), { choice: "a" }, "an unrelated refresh keeps the pending edit");
  drafts.settled("q1", { q1: { choice: "a" } });
  assert.equal(drafts.isPending("q1"), false);
  assert.deepEqual(drafts.value("q1", { choice: "a" }, empty), { choice: "a" });
});

test("an older acknowledgement never replaces a newer local edit", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q1", { choice: "a" });
  drafts.sent("q1");
  drafts.edit("q1", { choice: "b" });
  drafts.sent("q1");
  drafts.settled("q1", { q1: { choice: "a" } });
  assert.deepEqual(drafts.value("q1", { choice: "a" }, empty), { choice: "b" });
  // Editing back to the saved value while a save is in flight is not acknowledged early.
  drafts.edit("q1", { choice: "a" });
  drafts.sent("q1");
  drafts.reconcile({ q1: { choice: "a" } });
  assert.equal(drafts.isPending("q1"), true, "a save is still outstanding");
  drafts.settled("q1", { q1: { choice: "b" } });
  drafts.settled("q1", { q1: { choice: "a" } });
  assert.equal(drafts.isPending("q1"), false);
});

test("with nothing pending, the server's draft wins (another tab, replay)", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q2", { choices: ["x"] });
  drafts.sent("q2");
  drafts.settled("q2", { q2: { choices: ["x"] } });
  assert.deepEqual(drafts.value("q2", { choices: ["y"] }, empty), { choices: ["y"] });
});

test("a conflict keeps the local edit pending so it can be saved again", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q3", { matches: { r1: "none", r2: "insufficient" }, part_b: "x" });
  drafts.sent("q3");
  drafts.settled("q3", { q3: { matches: { r1: "other" } } });
  assert.deepEqual(drafts.pending(), [{ questionId: "q3", response: { matches: { r1: "none", r2: "insufficient" }, part_b: "x" } }]);
});

test("queued drafts restored after a reload show until the server holds them; key order does not matter", () => {
  const drafts = new AssessmentDrafts();
  drafts.seed([{ questionId: "q4", response: { matches: { b: "2", a: "1" } } }]);
  assert.deepEqual(drafts.value("q4", undefined, empty), { matches: { b: "2", a: "1" } });
  drafts.reconcile({ q4: { matches: { a: "1", b: "2" } } });
  assert.equal(drafts.isPending("q4"), false);
  assert.equal(canonical({ b: 1, a: [2, { d: 3, c: undefined }] }), canonical({ a: [2, { d: 3 }], b: 1 }));
});
