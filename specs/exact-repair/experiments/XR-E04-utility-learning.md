# XR-E04 — Pointwise, pairwise, and decision-focused repair utility

**Status:** planned<br>
**Depends on:** XR-E00; XR-WP4; XR-E01 eligibility gate<br>
**Question/hypothesis:** RQ5/H4<br>
**Primary outcome:** held-out pair-level downstream repair regret

## Authorized claim

Whether decision-focused training improves final safe repair decisions relative to provisional
confidence, hand-coded utility, and locally trained state models on ontology pairs excluded from
training, calibration, and model selection.

Safety is held constant by the oracle and is not a model outcome.

## Compared utilities

1. uniform active-state/cardinality utility;
2. provisional confidence-weighted utility;
3. frozen hand-coded state utility;
4. pointwise tabular model;
5. pairwise within-object/component ranker;
6. decision-focused/structured model;
7. optional graph model, only if preregistered before confirmatory testing;
8. reference-derived oracle utility ceiling on tractable components.

All arms receive identical state inventories, policy, exact solver, and candidate evidence. Utility
is integerized using a frozen scale. Model selection cannot inspect confirmatory test regret.

## Data and splits

Train using reference positives/confirmed negatives or positive-unlabeled objectives, synthetic
corruptions, exact small-component labels, and training-only expert preferences. Primary split is by
ontology pair. Model seeds and pair folds are crossed where feasible. Set-valued optimal labels are
preserved.

Run a deliberately local-accuracy-matched analysis where models with similar state classification
accuracy are compared on final regret. This tests the predict-then-optimize motivation directly.

## Outcomes

Primary:

- regret of the selected verified assignment under a frozen external evaluation utility, aggregated
  at ontology-pair level.

Secondary:

- weighted semantic preservation and relation-aware F1;
- local state accuracy/ranking metrics;
- calibration, score margins, and uncertainty-quality curves;
- optimum sensitivity to utility perturbations;
- fallback/unknown rates and eligibility (must not differ through acceptance);
- inference/training cost and exact-label generation cost.

## Decision rule

H4 requires a positive paired regret improvement of decision-focused over pointwise learning above
the G4 margin on held-out pairs, with confidence interval/test/effect size reported and no safety
eligibility loss. A local accuracy improvement without regret improvement does not support H4.

If decision-focused wins only on synthetic corruption but not public/captured outputs, bound the
claim accordingly. If a graph model fails to beat simpler models, retain the simpler deployment
candidate.

## Ablations

- remove conflict-context features;
- remove global pair/matcher features;
- no positive-unlabeled treatment;
- single-target versus set-valued supervision;
- local pretraining only versus structured fine-tuning;
- pairwise utility terms off where implemented.

These form one corrected comparison family; they are secondary unless promoted at G4.

## Non-claims

- Learned utility never proves safety.
- Results do not establish transfer to excluded domains/matchers; XR-E06 does.
- Calibration quality does not imply causal correctness of the utility objective.
- Training cost is not a scalability claim for inference/repair.
