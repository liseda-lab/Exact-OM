# XR-E05 — Guard and endpoint candidate proposal coverage

**Status:** planned<br>
**Depends on:** XR-E00; XR-WP3 deterministic generators; XR-WP4 for learned proposer arms<br>
**Question/hypothesis:** RQ6/H6<br>
**Co-primary outcomes:** top-k candidate recall and accepted recovery gain

## Authorized claim

Whether bounded generators/retrievers propose semantically useful guard, qualification, and
replacement candidates at practical k, and whether missed recovery is driven primarily by candidate
coverage rather than safety rejection or utility selection.

## Candidate tasks

- target and source endpoint replacement;
- source-side guards for guarded-forward states;
- target-side guards for guarded-backward states;
- qualification components;
- closed-inventory composite state construction.

Evaluate each task separately before aggregating. A candidate is relevant by a frozen hierarchy:
exact reference expression/entity, logical equivalence, bidirectional subsumption above threshold,
query-basis similarity above threshold, or expert adjudication. The rule cannot be chosen per result.

## Methods

1. local structural/asserted deterministic generation;
2. Exact-OM top-k alternatives only;
3. lexical/embedding retrieval;
4. deterministic combined diversity-aware ranker;
5. learned retrieval/ranking;
6. optional constrained LLM ranking over the identical closed inventory;
7. candidate oracle containing known relevant candidate(s), for ceiling analysis only.

Compare at frozen k values and equal total inventory budgets. Report prefilter rejection and
diversity pruning separately from rank truncation.

## Corpora

- complex/relation-aware reference cases;
- mechanism-aware corruptions made by endpoint shift or guard/qualification removal;
- exact small components with known recovery-improving states;
- expert-adjudicated captured conflicts where public references are incomplete.

Split by ontology pair and clean conflict group. Test reference expressions/entities are not used to
construct their candidate pools beyond ontology-native legal inventory generation.

## Failure attribution

For every missed best recovery, assign the earliest exclusive stage:

1. relevant expression/entity not representable by XR-1 grammar/inventory;
2. representable but not generated;
3. generated then rejected by a deterministic prefilter;
4. present before ranking but outside retained top-k/budget;
5. present in the master but rejected by safety/non-vacuity;
6. safe and present but not selected by utility/global conflicts;
7. selected but receives no semantic recovery credit.

Re-run stages 5–6 with the candidate-oracle and reference-derived utility on tractable cases. This
distinguishes H6 from a generic low-recovery observation.

## Outcomes

- recall@k/MRR by candidate task and ontology pair;
- retained state-space size and generation/inference cost;
- accepted non-vacuous/non-redundant rate;
- recovery gain and semantic correctness among accepted candidates;
- candidate-oracle headroom;
- failure-attribution distribution;
- expression complexity and distance of replacements;
- uncertainty/recall curves for adaptive budgets.

## Decision rule

RQ6 is supported only if both candidate recall and accepted recovery clear G4 thresholds without an
eligibility failure or impractical state-space cost. H6 is supported if candidate/representability
stages account for the preregistered dominant share of oracle-recoverable misses and the
candidate-oracle materially raises recovery.

High top-k recall with negligible accepted recovery does not pass. A high recovery rate obtained
only from impossible/redundant guards receives zero credit.

## Non-claims

- The candidate oracle is an upper bound, not a deployable method.
- Reference-expression exact match is not the only valid semantics, but alternative metrics must be
  frozen.
- Constrained LLM ranking success does not authorize free-form axiom generation.
- This experiment does not compare downstream utility training except for failure attribution.
