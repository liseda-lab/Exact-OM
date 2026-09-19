# XR-E08 — Evidence and infrastructure robustness

Research question: RQ2/RQ5.

## Data

Grouped generated perturbations and permissible real evidence perturbations.

## Comparison

Vary score noise/calibration, missing labels/descriptions, misleading explanations, provenance quality, absent useful terms, import duplication and changed import content. Inject solver/verifier/circuit failure.

## Measurements

Measure paired degradation, uncertainty/unknown coverage, semantic damage, calibration where labels support it, cache invalidation and bound correctness.

## Interpretation and acceptance

Keep soft evidence out of logical assertions. Cached proofs must not survive changed dependencies. Failures cannot turn into safe labels or permanent conflict cuts.

Shared controls, split rules, resources and status reporting are defined in [02](../02-experimental-protocol.md).
