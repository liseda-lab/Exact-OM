// Durable tutorial position (18 B12). Pure module (type-only imports) for Node's test runner.
// The server normalizes historical progress; the client only checks that a position names
// this frozen tutorial's lessons/questions and never infers one from other fields.

import type { TutorialPosition, TutorialPublic } from "./types";

export const lessonPosition = (lessonId: string): TutorialPosition => ({ view: "lesson", lesson_id: lessonId, question_id: null });
export const assessmentPosition = (questionId: string | null = null): TutorialPosition => ({ view: "assessment", lesson_id: null, question_id: questionId });

export function samePosition(a: TutorialPosition | null | undefined, b: TutorialPosition | null | undefined): boolean {
  return Boolean(a && b && a.view === b.view && a.lesson_id === b.lesson_id && a.question_id === b.question_id);
}

/** The screen for a server position, checked against the frozen tutorial. */
export function restoredPosition(tutorial: Pick<TutorialPublic, "lessons" | "assessment">, position: TutorialPosition | null | undefined): TutorialPosition {
  if (position?.view === "lesson" && tutorial.lessons.some((lesson) => lesson.lesson_id === position.lesson_id)) return lessonPosition(position.lesson_id);
  if (position?.view === "assessment") return assessmentPosition(position.question_id && tutorial.assessment.some((item) => item.question_id === position.question_id) ? position.question_id : null);
  return lessonPosition(tutorial.lessons[0].lesson_id);
}

/**
 * After a server acknowledgement: follow the acknowledged position only when the screen still
 * shows the previous acknowledged one (for example an entry replayed after a reload), so a
 * newer local navigation is never pulled back by an older receipt.
 */
export function followAcknowledged(current: TutorialPosition, previous: TutorialPosition | null, acknowledged: TutorialPosition | null, tutorial: Pick<TutorialPublic, "lessons" | "assessment">): TutorialPosition {
  if (!acknowledged || samePosition(previous, acknowledged)) return current;
  return samePosition(current, previous) ? restoredPosition(tutorial, acknowledged) : current;
}
