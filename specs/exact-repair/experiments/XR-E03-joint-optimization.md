# XR-E03 — Global selection and interactions

Research question: RQ1/RQ2/RQ5.

## Data

Small fully enumerated cases plus hub/overlap and complementary/redundant consequence families.

## Comparison

Compare greedy verified selection, exact unary MaxSAT and exact bounded-pairwise MaxSAT. On small cases use exhaustive enumeration as truth. Give controls the same pool and costs.

## Measurements

Measure decoded teacher utility/regret, feasibility, solver time, cut counts, lower/upper bounds, gap and pending assignments.

## Interpretation and acceptance

Validate pair encoding and quantisation. Distinguish optimality of a learned surrogate from optimality of teacher semantics. Never report zero gap after dropping a higher-scored unknown assignment.

Shared controls, split rules, resources and status reporting are defined in [02](../02-experimental-protocol.md).
