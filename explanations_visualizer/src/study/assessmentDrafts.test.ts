import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { AssessmentDrafts, canonical, MAX_AUTOMATIC_RESAVES } from "./assessmentDrafts.ts";

const empty = {};
const A = { choice: "a" };
const B = { choice: "b" };
const C = { choice: "c" };

test("a visible edit survives leaving the question before its save is acknowledged (S0-R2)", () => {
  const drafts = new AssessmentDrafts();
  assert.equal(drafts.edit("q1", A, undefined), true);
  drafts.sent("q1");
  assert.deepEqual(drafts.value("q1", undefined, empty), A, "the card remounts and reads the store");
  assert.deepEqual(drafts.observe({}), []);
  assert.deepEqual(drafts.value("q1", undefined, empty), A, "an unrelated refresh keeps the pending edit");
  drafts.acknowledged("q1", { q1: A });
  drafts.observe({ q1: A });
  assert.equal(drafts.isPending("q1"), false);
});

test("an older acknowledgement never replaces a newer local edit, even when it equals the saved answer", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q1", A, undefined);
  drafts.sent("q1");
  drafts.acknowledged("q1", { q1: A });
  drafts.observe({ q1: A });
  // Saved A; B goes out, then the participant returns to A while B is still outstanding.
  drafts.edit("q1", B, A);
  drafts.sent("q1");
  drafts.edit("q1", A, A);
  drafts.sent("q1");
  assert.deepEqual(drafts.observe({ q1: A }), [], "equality with the saved answer is not an acknowledgement");
  assert.equal(drafts.isPending("q1"), true);
  drafts.acknowledged("q1", { q1: B });
  drafts.observe({ q1: B });
  assert.deepEqual(drafts.value("q1", B, empty), A, "the intermediate B is never shown");
  drafts.acknowledged("q1", { q1: A });
  drafts.observe({ q1: A });
  assert.equal(drafts.isPending("q1"), false);
});

test("restored writes count until each settles (S0-FU-R2): saved A, queued B, latest A, reload", () => {
  const drafts = new AssessmentDrafts();
  drafts.seed([
    { questionId: "q1", response: B, base: canonical(A) },
    { questionId: "q1", response: A, base: canonical(B) },
  ]);
  assert.deepEqual(drafts.value("q1", A, empty), A);
  drafts.observe({ q1: A });
  assert.equal(drafts.isPending("q1"), true, "not cleared because it equals the saved answer");
  drafts.acknowledged("q1", { q1: B });
  drafts.observe({ q1: B });
  assert.deepEqual(drafts.value("q1", B, empty), A);
  drafts.acknowledged("q1", { q1: A });
  drafts.observe({ q1: A });
  assert.equal(drafts.isPending("q1"), false);
});

test("a terminal replay failure re-saves only what is still based on the saved answer", () => {
  // Saved A, then B and C (latest) restored; a change to another question makes them 409.
  const drafts = new AssessmentDrafts();
  drafts.seed([
    { questionId: "q1", response: B, base: canonical(A) },
    { questionId: "q1", response: C, base: canonical(B) },
  ]);
  drafts.failed("q1");
  assert.deepEqual(drafts.observe({ q1: A }), [], "nothing is decided while a write is outstanding");
  drafts.failed("q1");
  assert.deepEqual(drafts.observe({ q1: A, q2: { choice: "other" } }), [{ questionId: "q1", response: C }]);
  assert.deepEqual(drafts.value("q1", A, empty), C, "the latest answer stays visible");
  // When the latest answer is already the saved one, nothing needs to be written.
  const same = new AssessmentDrafts();
  same.seed([
    { questionId: "q1", response: B, base: canonical(A) },
    { questionId: "q1", response: A, base: canonical(B) },
  ]);
  same.failed("q1");
  same.failed("q1");
  assert.deepEqual(same.observe({ q1: A }), []);
  assert.equal(same.isPending("q1"), false);
});

test("a conflicting same-question answer is never written automatically (S0-FU-R1)", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q1", A, undefined);
  drafts.sent("q1");
  drafts.failed("q1"); // 409: another tab saved B meanwhile
  assert.deepEqual(drafts.observe({ q1: B }), []);
  assert.deepEqual(drafts.value("q1", B, empty), B, "the saved answer is shown");
  assert.deepEqual(drafts.conflict("q1"), A, "this tab's answer is kept");
  assert.deepEqual(drafts.observe({ q1: B }), [], "and stays unwritten");
  assert.deepEqual(drafts.conflicts(), [{ questionId: "q1", response: A, base: canonical(undefined) }]);
  // An explicit choice: write this tab's answer on top of the saved one it was shown.
  assert.deepEqual(drafts.resolve("q1", "mine", B), A);
  assert.equal(drafts.baseOf("q1"), canonical(B));
  assert.equal(drafts.conflict("q1"), null);
});

test("a conflict on a multi-part answer is resolved only by the participant", () => {
  const mine = { matches: { r1: "none", r2: "insufficient" }, part_b: "x" };
  const theirs = { matches: { r1: "other" } };
  const drafts = new AssessmentDrafts();
  drafts.edit("q3", mine, undefined);
  drafts.sent("q3");
  drafts.failed("q3");
  drafts.observe({ q3: theirs });
  assert.deepEqual(drafts.value("q3", theirs, empty), theirs);
  assert.equal(drafts.resolve("q3", "saved", theirs), null);
  assert.equal(drafts.isPending("q3"), false);
  assert.deepEqual(drafts.value("q3", theirs, empty), theirs);
});

test("editing the saved answer shown in a conflict is a deliberate new edit based on it", () => {
  const drafts = new AssessmentDrafts();
  drafts.restoreConflicts([{ questionId: "q1", response: A, base: canonical(undefined) }]);
  assert.deepEqual(drafts.value("q1", B, empty), B);
  assert.equal(drafts.edit("q1", C, B), true);
  assert.equal(drafts.baseOf("q1"), canonical(B));
  assert.equal(drafts.conflict("q1"), null);
});

test("an edit made after the saved answer changed elsewhere is kept, not written", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q1", A, undefined);
  drafts.sent("q1");
  drafts.failed("q1");
  // Before the failure is evaluated the participant types again; the refreshed state shows B.
  assert.equal(drafts.edit("q1", C, B), false);
  assert.deepEqual(drafts.conflict("q1"), C);
});

test("independent questions: a failed answer whose saved value is unchanged is re-saved, a bounded number of times", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q1", A, undefined);
  for (let attempt = 1; attempt <= MAX_AUTOMATIC_RESAVES; attempt += 1) {
    drafts.sent("q1");
    drafts.failed("q1"); // 409 caused by another tab's answer to q2
    assert.deepEqual(drafts.observe({ q2: B }), [{ questionId: "q1", response: A }], `re-save ${attempt}`);
  }
  drafts.sent("q1");
  drafts.failed("q1");
  assert.deepEqual(drafts.observe({ q2: B }), [], "no endless loop: the participant decides");
  assert.deepEqual(drafts.conflict("q1"), A);
});

test("an older outbox without a recorded base never re-saves on its own", () => {
  const drafts = new AssessmentDrafts();
  drafts.seed([{ questionId: "q1", response: C, base: null }]);
  drafts.failed("q1");
  assert.deepEqual(drafts.observe({}), []);
  assert.deepEqual(drafts.conflict("q1"), C);
});

test("a replaced (coalesced) write only stops counting", () => {
  const drafts = new AssessmentDrafts();
  drafts.edit("q1", A, undefined);
  drafts.sent("q1");
  drafts.edit("q1", B, undefined);
  drafts.sent("q1");
  drafts.replaced("q1");
  drafts.observe({});
  assert.equal(drafts.isPending("q1"), true);
  drafts.acknowledged("q1", { q1: B });
  drafts.observe({ q1: B });
  assert.equal(drafts.isPending("q1"), false);
});

test("key order does not matter", () => {
  assert.equal(canonical({ b: 1, a: [2, { d: 3, c: undefined }] }), canonical({ a: [2, { d: 3 }], b: 1 }));
});
