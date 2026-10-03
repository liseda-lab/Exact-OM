import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { assessmentPosition, followAcknowledged, lessonPosition, restoredPosition, samePosition } from "./tutorialPosition.ts";

const tutorial = { lessons: [{ lesson_id: "identity" }, { lesson_id: "context" }], assessment: [{ question_id: "score_meaning" }] } as never;

test("restores the server's lesson, assessment landing and focused question (R03)", () => {
  assert.deepEqual(restoredPosition(tutorial, lessonPosition("context")), { view: "lesson", lesson_id: "context", question_id: null });
  assert.deepEqual(restoredPosition(tutorial, assessmentPosition()), { view: "assessment", lesson_id: null, question_id: null });
  assert.deepEqual(restoredPosition(tutorial, assessmentPosition("score_meaning")), { view: "assessment", lesson_id: null, question_id: "score_meaning" });
});

test("an unknown lesson or question never becomes a screen of its own", () => {
  assert.deepEqual(restoredPosition(tutorial, lessonPosition("gone")), lessonPosition("identity"));
  assert.deepEqual(restoredPosition(tutorial, assessmentPosition("gone")), assessmentPosition());
  assert.deepEqual(restoredPosition(tutorial, null), lessonPosition("identity"), "only a pre-extension receipt lacks a position");
});

test("an acknowledgement never pulls a newer local navigation back", () => {
  const one = lessonPosition("identity");
  const two = lessonPosition("context");
  const quiz = assessmentPosition();
  // Navigated 1 → 2 locally, then 2 is acknowledged: stay.
  assert.deepEqual(followAcknowledged(two, one, two, tutorial), two);
  // Navigated 1 → 2 → assessment quickly; an older receipt for 2 arrives: stay on the assessment.
  assert.deepEqual(followAcknowledged(quiz, one, two, tutorial), quiz);
  // Reloaded on 1 while an unsent move to 2 replays: follow the acknowledged position.
  assert.deepEqual(followAcknowledged(one, one, two, tutorial), two);
  assert.ok(samePosition(followAcknowledged(two, two, two, tutorial), two));
});
