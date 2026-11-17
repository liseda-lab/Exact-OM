import type { Candidate, Score } from "./types";

/** Keep rounded endpoints distinct from recorded zero and one. No score is changed. */
export function formatScore(value: number): string {
  const rounded = Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);
  if ((Number(rounded) === 0 && value !== 0) || (rounded === "1.00" && value !== 1)) {
    return `≈${rounded}`;
  }
  return rounded;
}

/** Display the saved output score; an intermediate ranking must not replace a final score. */
export function primaryScore(item: Pick<Candidate, "scores"> & Partial<Pick<Candidate, "ranking" | "output_mode" | "saved_alignment_member">>): Score | undefined {
  const ranking = item.scores.find((score) => score.name === "S_rank");
  if (item.ranking?.mode === "local_ranking" && ranking) return ranking;
  if (item.output_mode === "global_acceptance" && item.saved_alignment_member === true) {
    const acceptance = item.scores.find((score) => score.name === "P_match");
    if (acceptance) return acceptance;
  }
  return item.scores.find((score) => score.name === "S_final") ?? ranking ?? item.scores[0];
}

export function scoreLabel(score: Score): string {
  if (score.name === "S_rank") return "Ranking score";
  if (score.name === "P_match") return "Acceptance score";
  return score.calibration_status === "validated" ? "Calibrated score" : "Matching score";
}

export function scoreHelpText(score: Score): string {
  if (score.name === "S_rank") return "A ranking score orders candidates within this query. It is not a probability of being correct or a saved alignment decision.";
  if (score.name === "P_match") return score.calibration_status === "validated"
    ? "The acceptance model's calibrated score for this pair. Final alignment membership is shown separately."
    : "The acceptance model's score for this pair. Final alignment membership is shown separately. It is not a probability of being correct.";
  return score.calibration_status === "validated" ? "A calibrated score." : "A matching score ranks candidates for this source. It is not a probability of being correct.";
}
