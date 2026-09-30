# XR-E03 — Action interactions and verification order

**Target:** XR-2.1, 30 September 2026. **Question:** RQ2, RQ5. Requirements for new runs; archived XR-2 results remain separate.

## Data

Fully enumerated small plans plus support-overlap, hub, cyclic-incidence and candidate-induced conflict families. Include three-way contradictions with every pair feasible and higher-order semantic effects.

## Comparisons

Greedy, exact unary and sparse-pair MaxSAT on common pools. Add quartet supervision to whole-plan value/ranking. Cross no-risk versus plan-risk shortlist scheduling with full-assignment versus proved support cuts. The primary ordering ablation uses the same shortlist size/window/construction budget; shortlist size one is a separately named serial-master baseline.

## Measurements

Teacher/independent semantic utility and cost separately; risk calibration on a representative decided validation set; first verified and matched-quality time/queries; solver time; cut types; all untested/unknown bounds and certified optimum time.

## Acceptance and interpretation

Risk never changes the frozen semantic objective or produces clauses. A three-way negative is not three pair negatives. Report gains from cuts separately from gains from learning. A lower-value early incumbent may require more work to prove optimality; preserve every deferred high-value plan in the bound.

Shared splits, resources, failure accounting and statistics follow [02](../02-experimental-protocol.md). Dependency gates are in [08](../08-experiment-matrix-and-handoff.md); code changes and tests are in [12](../12-implementation-migration.md). Resolve every numerical setting in a new [run manifest](../protocol/README.md) before execution.
