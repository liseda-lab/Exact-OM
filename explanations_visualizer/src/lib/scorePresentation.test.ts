import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { formatScore, primaryScore, scoreHelpText, scoreLabel } from "./scorePresentation.ts";
import type { Candidate, Score } from "./types";

function score(name: string, value: number, calibration_status: Score["calibration_status"] = "not_established"): Score {
  return { name, value, calibration_status, calibration_artifact: null, stage: "selection", meaning: "Recorded score", range: null };
}

const localRanking: Candidate["ranking"] = { mode: "local_ranking", query_id: 0, candidate_count: 2, rank: 1, saved_output_member: true, artifact_ref: "ranking" };

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

test("explicit local ranking displays S_rank over final and acceptance scores, including zero", () => {
  const ranking = score("S_rank", 0);
  const final = score("S_final", 0.9);
  const acceptance = score("P_match", 0.99, "validated");
  for (const scores of [[acceptance, final, ranking], [ranking, final, acceptance], [final, ranking, acceptance]]) {
    assert.equal(primaryScore({ scores, ranking: localRanking }), ranking);
    assert.equal(scoreLabel(primaryScore({ scores, ranking: localRanking })!), "Ranking score");
  }
});

test("ordinary alignments display the final score rather than an intermediate pre-NIL ranking", () => {
  const ranking = score("S_rank", 0.99);
  const final = score("S_final", 0);
  for (const scores of [[ranking, final], [final, ranking]]) {
    assert.equal(primaryScore({ scores }), final);
    assert.equal(scoreLabel(primaryScore({ scores })!), "Matching score");
  }
});

test("explicit staged global output displays acceptance only for saved alignment members", () => {
  const ranking = score("S_rank", 0.99);
  const final = score("S_final", 0.8);
  const acceptance = score("P_match", 0);
  for (const scores of [[ranking, final, acceptance], [acceptance, final, ranking], [final, acceptance], [ranking, acceptance]]) {
    const staged = { scores, output_mode: "global_acceptance" as const, saved_alignment_member: true };
    assert.equal(primaryScore(staged), acceptance);
    assert.equal(scoreLabel(primaryScore(staged)!), "Acceptance score");
  }
  assert.equal(primaryScore({ scores: [final, ranking], output_mode: "global_acceptance", saved_alignment_member: true }), final, "missing acceptance scores retain the final score");
});

test("legacy rejected candidates retain S_final despite higher ranking and acceptance model scores", () => {
  const ranking = score("S_rank", 0.99);
  const acceptance = score("P_match", 0.2);
  const final = score("S_final", 0.03);
  const scores = [ranking, acceptance, final];
  assert.equal(primaryScore({ scores, saved_alignment_member: false }), final);
  assert.equal(primaryScore({ scores, saved_alignment_member: false })?.value, 0.03);
  for (const output_mode of [undefined, "native_scores", "local_ranking", "global_acceptance"] as const) {
    for (const saved_alignment_member of [undefined, null, false, true]) {
      if (output_mode === "global_acceptance" && saved_alignment_member === true) continue;
      assert.equal(primaryScore({ scores, output_mode, saved_alignment_member }), final, `${output_mode}/${saved_alignment_member}`);
    }
  }
});

test("ranking remains a display fallback when no final or acceptance score exists", () => {
  const base = score("S_base", 0.8);
  const ranking = score("S_rank", 0.25);
  assert.equal(primaryScore({ scores: [base, ranking] }), ranking);
  assert.equal(primaryScore({ scores: [base] }), base);
  assert.equal(primaryScore({ scores: [] }), undefined);
});

test("ranking labels and help describe query order, regardless of calibration metadata", () => {
  for (const calibration of ["not_established", "validated", "not_applicable"] as const) {
    const ranking = score("S_rank", 0.5, calibration);
    assert.equal(scoreLabel(ranking), "Ranking score");
    assert.match(scoreHelpText(ranking), /within this query/);
    assert.match(scoreHelpText(ranking), /not a probability.*or a saved alignment decision/);
  }
  assert.equal(scoreLabel(score("S_final", 0.5)), "Matching score");
  assert.equal(scoreLabel(score("S_final", 0.5, "validated")), "Calibrated score");
  assert.equal(scoreHelpText(score("S_final", 0.5, "validated")), "A calibrated score.");
});

test("acceptance labels and help describe the model score without implying final alignment membership", () => {
  for (const calibration of ["not_established", "validated", "not_applicable"] as const) {
    const acceptance = score("P_match", 0.75, calibration);
    assert.equal(scoreLabel(acceptance), "Acceptance score");
    assert.match(scoreHelpText(acceptance), /acceptance model's.*score/);
    assert.match(scoreHelpText(acceptance), /Final alignment membership is shown separately/);
    assert.doesNotMatch(scoreHelpText(acceptance), /within this query|saved global alignment|recorded.*alignment/);
    if (calibration === "validated") assert.match(scoreHelpText(acceptance), /calibrated/);
    else assert.match(scoreHelpText(acceptance), /not a probability/);
  }
});
