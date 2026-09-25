// Candidate ordering and score presentation. Order comes from recorded ordinal ranks; a
// score is only a fallback and the interface says which basis was used.

import type { Candidate, Score } from "@/lib/types";

export type OrderBasis = "joint_rank" | "retrieval_rank" | "score" | "pair_id";

export function orderCandidates(items: Candidate[]): { ordered: Candidate[]; basis: OrderBasis } {
  const hasJoint = items.length > 0 && items.every((item) => item.ordinal_ranks.candidate_joint_rank != null);
  const hasRetrieval = items.length > 0 && items.every((item) => item.ordinal_ranks.retrieval_rank != null);
  const score = (item: Candidate) => primaryScore(item)?.value ?? Number.NEGATIVE_INFINITY;
  const basis: OrderBasis = hasJoint ? "joint_rank" : hasRetrieval ? "retrieval_rank" : items.some((item) => primaryScore(item)) ? "score" : "pair_id";
  const ordered = [...items].sort((a, b) => {
    if (basis === "joint_rank") return (a.ordinal_ranks.candidate_joint_rank ?? 0) - (b.ordinal_ranks.candidate_joint_rank ?? 0) || a.pair_id.localeCompare(b.pair_id);
    if (basis === "retrieval_rank") return (a.ordinal_ranks.retrieval_rank ?? 0) - (b.ordinal_ranks.retrieval_rank ?? 0) || a.pair_id.localeCompare(b.pair_id);
    if (basis === "score") return score(b) - score(a) || a.pair_id.localeCompare(b.pair_id);
    return a.pair_id.localeCompare(b.pair_id);
  });
  return { ordered, basis };
}

export function orderDescription(basis: OrderBasis): string {
  switch (basis) {
    case "joint_rank":
      return "In Exact's recorded candidate order";
    case "retrieval_rank":
      return "In recorded retrieval order";
    case "score":
      return "Sorted by matching score (no rank was recorded)";
    default:
      return "No order was recorded; listed by identifier";
  }
}

/** The final decision score when present, else the first recorded score. */
export function primaryScore(item: { scores: Score[] }): Score | undefined {
  return item.scores.find((score) => score.name === "S_final") ?? item.scores[0];
}

export function formatScore(value: number): string {
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);
}

export function scoreLabel(score: Score): string {
  return score.calibration_status === "validated" ? "calibrated score" : "matching score";
}
