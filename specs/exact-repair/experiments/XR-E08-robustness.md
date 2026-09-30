# XR-E08 — Evidence, omission and infrastructure robustness

**Target:** XR-2.1, 30 September 2026. **Question:** RQ2, RQ5. Requirements for new runs; archived XR-2 results remain separate.

## Data

Grouped evidence perturbations, dedicated final-candidate removal and vocabulary/grammar omission controls, import changes, candidate vocabulary expansion and backend/worker failures.

## Comparisons

Score noise/missingness, misleading text/explanations, provenance uncertainty, pair-selector budget truncation and selected-term absence. Compare cached/fresh execution under changed semantic dependencies.

## Measurements

Paired quality/coverage degradation; retrieval versus grammar versus sampling versus value error; final candidate identities after every producer; masked feature states; support/cache invalidation; valid bounds and retained proof events.

## Acceptance and interpretation

Candidate-removal identities belong to the evaluator and do not become ordinary generator features. A separate missing-symbol intervention changes the declared menu. Assert final exclusion even when deterministic generation or fallback runs. A stale cache or unknown delivery cannot produce a new trusted logical/semantic label.

Shared splits, resources, failure accounting and statistics follow [02](../02-experimental-protocol.md). Dependency gates are in [08](../08-experiment-matrix-and-handoff.md); code changes and tests are in [12](../12-implementation-migration.md). Resolve every numerical setting in a new [run manifest](../protocol/README.md) before execution.
