import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { orderCandidates, orderDescription } from "./candidateOrdering.ts";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { primaryScore, scoreLabel } from "./scorePresentation.ts";
import type { Candidate } from "./types";

function candidate(pair_id: string, values: Record<string, number>, local = false): Candidate {
  return {
    pair_id,
    source: { ontology_version_id: "source", iri: "urn:source", kind: "class" },
    target: { ontology_version_id: "target", iri: `urn:${pair_id}`, kind: "class" },
    scores: Object.entries(values).map(([name, value]) => ({ name, value, stage: "selection", meaning: "Recorded score", range: null, calibration_status: "not_established", calibration_artifact: null })),
    ranking: local ? { mode: "local_ranking", query_id: 0, candidate_count: 2, rank: 1, saved_output_member: true, artifact_ref: "ranking" } : undefined,
    ordinal_ranks: { candidate_joint_rank: null, nil_rank: null, joint_ordering: null, joint_tie_rule: null, retrieval_rank: null, retrieval_ordering: null, retrieval_tie_rule: null, retrieval_provenance: null },
    retrieval_channels: [], evidence_counts: {}, nil: {}, saved_alignment_member: false,
    membership_provenance: { status: "derived_from_saved_artifacts", artifact_hash: null }, status: "available", decision_status: "available",
  };
}

test("global fallback sorts by common S_final, never selected-only acceptance versus native scores", () => {
  const accepted = candidate("accepted", { S_final: 0.1, P_match: 0.99, S_rank: 0.95 });
  const other = candidate("other", { S_rank: 0.05, S_final: 0.8 });
  accepted.output_mode = other.output_mode = "global_acceptance";
  accepted.saved_alignment_member = true;
  const input = [accepted, other];
  const result = orderCandidates(input);
  assert.deepEqual(result.ordered, [other, accepted]);
  assert.deepEqual(input, [accepted, other], "sorting preserves the caller's list");
  assert.equal(result.basis, "final_score");
  assert.equal(orderDescription(result.basis), "Sorted by final matching score (no rank was recorded)");
  assert.equal(scoreLabel(primaryScore(accepted)!), "Acceptance score");
  assert.equal(primaryScore(accepted)?.value, 0.99, "the displayed output is independent of the sort score");
});

test("local fallback uses common S_rank only when every candidate declares local ranking", () => {
  const a = candidate("a", { S_rank: 0, S_final: 0.9, P_match: 1 }, true);
  const b = candidate("b", { S_rank: 0.8, S_final: 0.1 }, true);
  const local = orderCandidates([a, b]);
  assert.deepEqual(local.ordered, [b, a]);
  assert.equal(local.basis, "ranking_score");
  assert.equal(orderDescription(local.basis), "Sorted by ranking score (no rank was recorded)");
  b.ranking = undefined;
  const mixed = orderCandidates([b, a]);
  assert.equal(mixed.basis, "final_score");
  assert.deepEqual(mixed.ordered, [a, b]);
});

test("incomplete or incomparable score pools use identifiers without mixing score fields", () => {
  const pools = [
    [candidate("b", { P_match: 0.9 }), candidate("a", { S_final: 0.1 })],
    [candidate("b", { S_rank: 0.9 }), candidate("a", { S_rank: 0.1 })],
    [candidate("b", { S_rank: 0.9 }, true), candidate("a", { S_final: 0.1 }, true)],
    [candidate("b", { S_final: 0.9 }), candidate("a", {})],
    [candidate("b", { S_final: 0.9 }), candidate("a", { S_final: Number.NaN })],
  ];
  for (const pool of pools) {
    const result = orderCandidates(pool);
    assert.equal(result.basis, "pair_id");
    assert.deepEqual(result.ordered.map((item) => item.pair_id), ["a", "b"]);
    assert.match(orderDescription(result.basis), /No common rank or score/);
  }
  assert.deepEqual(orderCandidates([]), { ordered: [], basis: "pair_id" });
});

test("recorded joint and retrieval ranks take precedence over any display or sort scores", () => {
  const a = candidate("a", { S_final: 0.1, P_match: 0.99, S_rank: 0.9 }, true);
  const b = candidate("b", { S_final: 0.9, S_rank: 0.1 }, true);
  a.ordinal_ranks.candidate_joint_rank = 2;
  b.ordinal_ranks.candidate_joint_rank = 1;
  a.ordinal_ranks.retrieval_rank = 1;
  b.ordinal_ranks.retrieval_rank = 2;
  const joint = orderCandidates([a, b]);
  assert.equal(joint.basis, "joint_rank");
  assert.deepEqual(joint.ordered, [b, a]);
  b.ordinal_ranks.candidate_joint_rank = null;
  const retrieval = orderCandidates([b, a]);
  assert.equal(retrieval.basis, "retrieval_rank");
  assert.deepEqual(retrieval.ordered, [a, b]);
});

test("common zero scores are valid and ties keep deterministic pair identity order", () => {
  for (const local of [false, true]) {
    const values: Record<string, number> = local ? { S_rank: 0 } : { S_final: 0 };
    const result = orderCandidates([candidate("b", values, local), candidate("a", values, local)]);
    assert.equal(result.basis, local ? "ranking_score" : "final_score");
    assert.deepEqual(result.ordered.map((item) => item.pair_id), ["a", "b"]);
  }
});
