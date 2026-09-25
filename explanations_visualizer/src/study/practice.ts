import type { PracticeCase, ResponseType } from "./types";

// Synthetic nonmedical controls practice only. This is never a scored case bank or matcher output.
export const FALLBACK_PRACTICE: PracticeCase[] = [
  {
    practice_id: "controls-simple-v1", kind: "simple", title: "1 · Read a definition and choose",
    instructions: "Inspect the source and the five candidate descriptions. Add a candidate you consider equivalent, then check your practice answer. A familiar name or a high position alone is not evidence of equivalence.", synthetic: true,
    source: { label: "Filled circular shape", description: "A flat shape consisting of a circle and all the area inside it. Parent: flat shape." },
    candidates: [
      { candidate_id: "simple-a", label: "Circle boundary", description: "Only the curved boundary, with no interior area." },
      { candidate_id: "simple-b", label: "Disk", description: "A flat shape consisting of a circle and all the area inside it. Parent: flat shape." },
      { candidate_id: "simple-c", label: "Sphere", description: "A three-dimensional round surface." },
      { candidate_id: "simple-d", label: "Triangle", description: "A flat shape with three straight sides." },
      { candidate_id: "simple-e", label: "Shape", description: "A general geometric form; it need not be circular or flat." },
    ],
  },
  {
    practice_id: "controls-complex-v1", kind: "complex", title: "2 · Keep every qualifier",
    instructions: "Inspect the descriptions before ranking. Notice the parent, the closed-lid condition, “only round” and “at least two”. Broader, narrower and related concepts need not mean the same thing. Try moving or removing a candidate, then check a ranking.", synthetic: true,
    source: { label: "Closed crate of round pieces", description: "Parent: crate. Its lid is closed, all its contents are round pieces, and it contains at least two pieces. “At least two” allows three or more; “only round” restricts every piece." },
    candidates: [
      { candidate_id: "complex-a", label: "Crate with a round piece", description: "A crate containing some round piece. Its other contents and lid position are not specified." },
      { candidate_id: "complex-b", label: "Closed two-piece crate", description: "A closed crate containing exactly two round pieces, with no other contents." },
      { candidate_id: "complex-c", label: "Crate", description: "A storage container. No lid position, number of pieces or piece shape is specified." },
      { candidate_id: "complex-d", label: "Closed round-piece collection crate", description: "Parent: crate. Its lid is closed; it holds at least two pieces, and every piece it holds is round." },
      { candidate_id: "complex-e", label: "Open round-piece crate", description: "A crate with its lid open, holding at least two round pieces and no other contents." },
    ],
  },
  {
    practice_id: "controls-partial-v1", kind: "partial_ranking", title: "3 · Leave candidates unranked",
    instructions: "Create a partial answer with one to four candidates. Add the most plausible first and leave the rest unranked. An omitted candidate is not automatically added later. Try Undo after removing a candidate.", synthetic: true,
    source: { label: "Object holder", description: "An object whose purpose is to hold other objects. No shape, size or material is specified." },
    candidates: [
      { candidate_id: "partial-a", label: "Container", description: "An object whose purpose is to hold other objects. Shape, size and material are unrestricted." },
      { candidate_id: "partial-b", label: "Red container", description: "A container whose colour is red; other colours are excluded." },
      { candidate_id: "partial-c", label: "Receptacle", description: "An object for holding other objects, with no restriction on shape, size or material." },
      { candidate_id: "partial-d", label: "Contained object", description: "An object being held inside a container; it is not necessarily itself a container." },
      { candidate_id: "partial-e", label: "Container label", description: "A tag that identifies a container." },
    ],
  },
  {
    practice_id: "controls-none-v1", kind: "none_of_these", title: "4 · Use an explicit none response",
    instructions: "These descriptions deliberately provide no equivalent for the source. Choose None of these and check it. An empty answer is not None. Insufficient information is a different choice for a case you cannot judge; you can try it, then switch back to None for this lesson.", synthetic: true,
    source: { label: "Three-sided polygon", description: "A flat closed shape with exactly three straight sides." },
    candidates: [
      { candidate_id: "none-a", label: "Square", description: "A flat closed shape with exactly four equal straight sides and four right angles." },
      { candidate_id: "none-b", label: "Pentagon", description: "A flat closed shape with exactly five straight sides." },
      { candidate_id: "none-c", label: "Circle", description: "A curved circular boundary with no straight sides." },
      { candidate_id: "none-d", label: "Sphere", description: "A three-dimensional round surface." },
      { candidate_id: "none-e", label: "Open line segment", description: "A single straight segment with two endpoints; it is not a closed shape." },
    ],
  },
];

/** Check the requested control action, not semantic correctness or a study answer key. */
export function practiceActionComplete(kind: PracticeCase["kind"], answer: { responseType: ResponseType | null; ranked: string[] }): boolean {
  if (kind === "none_of_these") return answer.responseType === "none_of_these" && answer.ranked.length === 0;
  return answer.responseType === "ranked_candidates" && answer.ranked.length > 0 && answer.ranked.length <= (kind === "partial_ranking" ? 4 : 5);
}
