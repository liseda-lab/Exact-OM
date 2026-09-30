# XR-E04 — Supervised proposals and value learning

**Target:** XR-2.1, 30 September 2026. **Question:** RQ2. Requirements for new runs; archived XR-2 results remain separate.

## Data

Generated structural parents, exact tiny teacher universes, newly generated pools with verified sampled plans, and declared training-side real adaptation.

## Comparisons

Symbolic rich-action baseline, MLP/no-graph, HGT and matched R-GCN; unary/pair factors; exact-only versus exact plus sampled collection; independent versus explicitly plan-conditioned proposals. Separate proposal ranking/imitation from benefit regression and risk training.

## Measurements

Generated-pool verified coverage/quality/effort as primary development outcomes; fixed-inventory benefit error/ranking/regret as diagnostics; new candidate coverage; reachable teacher mass; unknown/missing label rates and full label/training cost.

## Acceptance and interpretation

Use the identical graph/pair transformation in all paths. Complete finite teacher distributions require a decided universe; sampled repair rankings are approximate supervision. Unknown probes cannot be dropped from the scalar denominator. Select checkpoints by the frozen generated-pool rule, never by hidden test regret.

Shared splits, resources, failure accounting and statistics follow [02](../02-experimental-protocol.md). Dependency gates are in [08](../08-experiment-matrix-and-handoff.md); code changes and tests are in [12](../12-implementation-migration.md). Resolve every numerical setting in a new [run manifest](../protocol/README.md) before execution.
