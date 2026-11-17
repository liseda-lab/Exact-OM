import type { Candidate } from "./types";

export type OrderBasis = "joint_rank" | "retrieval_rank" | "ranking_score" | "final_score" | "pair_id";

/** Recorded ranks take precedence. Score fallback uses one field shared by the whole pool. */
export function orderCandidates(items: Candidate[]): { ordered: Candidate[]; basis: OrderBasis } {
  const every = (predicate: (item: Candidate) => boolean) => items.length > 0 && items.every(predicate);
  const score = (item: Candidate, name: string) => item.scores.find((value) => value.name === name)?.value;
  const hasJoint = every((item) => item.ordinal_ranks.candidate_joint_rank != null);
  const hasRetrieval = every((item) => item.ordinal_ranks.retrieval_rank != null);
  const hasRanking = every((item) => item.ranking?.mode === "local_ranking" && Number.isFinite(score(item, "S_rank")));
  const hasFinal = every((item) => Number.isFinite(score(item, "S_final")));
  const basis: OrderBasis = hasJoint ? "joint_rank" : hasRetrieval ? "retrieval_rank" : hasRanking ? "ranking_score" : hasFinal ? "final_score" : "pair_id";
  const ordered = [...items].sort((a, b) => {
    let difference = 0;
    if (basis === "joint_rank") difference = a.ordinal_ranks.candidate_joint_rank! - b.ordinal_ranks.candidate_joint_rank!;
    else if (basis === "retrieval_rank") difference = a.ordinal_ranks.retrieval_rank! - b.ordinal_ranks.retrieval_rank!;
    else if (basis === "ranking_score" || basis === "final_score") {
      const name = basis === "ranking_score" ? "S_rank" : "S_final";
      difference = score(b, name)! - score(a, name)!;
    }
    return difference || a.pair_id.localeCompare(b.pair_id);
  });
  return { ordered, basis };
}

export function orderDescription(basis: OrderBasis): string {
  switch (basis) {
    case "joint_rank":
      return "In Exact's recorded candidate order";
    case "retrieval_rank":
      return "In recorded retrieval order";
    case "ranking_score":
      return "Sorted by ranking score (no rank was recorded)";
    case "final_score":
      return "Sorted by final matching score (no rank was recorded)";
    default:
      return "No common rank or score is available; listed by identifier";
  }
}
