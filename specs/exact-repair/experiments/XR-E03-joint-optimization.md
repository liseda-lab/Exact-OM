# XR-E03 — Joint state optimization versus sequential diagnosis and recovery

**Status:** planned<br>
**Depends on:** XR-E00; XR-WP2; XR-WP3 for the full-action arm<br>
**Question/hypothesis:** RQ4/H3<br>
**Primary outcome:** regret against exact external-utility optimum

## Authorized claim

Whether choosing final mapping states jointly yields lower downstream regret than first choosing a
deletion diagnosis and then recovering semantic content, when both methods receive the same frozen
candidate actions, policy, information, and total resource budget.

## Methods

1. **Joint:** the XR master directly maximizes frozen final external-aligned utility over all states.
2. **Sequential-cost:** minimize diagnosis/deletion cost, freeze that diagnosis, then maximize
   recovery within it.
3. **Sequential-lexicographic:** diagnosis cost first and recovery second; reported to show the
   effect of deliberately choosing that objective, not as an implementation error.
4. **Sequential-multidiagnosis sensitivity:** retain top-k diagnoses before recovery under the same
   total solve budget.
5. **Oracle optimum:** exhaustive enumeration or exact solve under the external evaluation utility.

The sequential baseline may not receive a smaller candidate inventory or lose evidence through an
implementation shortcut. Diagnosis definitions, costs, and whether direction weakening counts as
diagnosis are frozen at G4.

## Corpus

Primary units are small/medium corruption conflict components with several safe diagnoses and
different recovery potential. Include neutral cases where sequential and joint should tie, plus
adversarial cases like a cheap deletion with no recovery versus a costlier change enabling high
value. Public/captured components are included where an external utility can be defined without
test leakage.

## Procedure

- Freeze one state inventory and external utility per instance.
- Compute oracle-optimal safe assignment(s).
- Run each method with identical oracle/solver versions and a common aggregate budget.
- Verify all outputs independently.
- Measure regret, semantic preservation, diagnosis cost, recovery value, and compute use.
- Repeat a matched exact/no-budget subset and a fixed-budget subset; do not conflate algorithmic
  objective differences with timeout behavior.

## Decision rule

H3 requires lower paired external-utility regret for joint optimization by the G4 margin on the
exact subset, with no eligibility degradation. Fixed-budget results explain operational tradeoffs.
If sequential top-k closes the gap only with materially larger cost, report the Pareto tradeoff.

## Secondary analyses

- proportion of instances tied, helped, or harmed;
- regret by number/overlap of diagnoses and available recovery primitives;
- objective mismatch versus search-budget effects;
- top-k repair diversity and multiple optimal assignments;
- time/reasoner calls as descriptive outcomes.

## Non-claims

- The study does not assert joint optimization is better under a truly lexicographic
  diagnosis-first user objective.
- It does not compare learned utility models.
- A regret advantage under synthetic external utility must not be generalized to expert preference
  without XR-E09 or replication on curated public cases.
