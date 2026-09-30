# XR-E09 — Grounded semantic-fidelity supervision

**Target:** XR-2.1, 30 September 2026. **Question:** RQ4. Requirements for new runs; archived XR-2 results remain separate.

## Data

Verified feasible alternatives with grounded definitions and consequence evidence; meaningful synthetic contrasts and training-side real examples. Independently frozen held-out cases/annotator roles for evaluation.

## Comparisons

Symbolic consequences only versus symbolic plus LLM weak fidelity. Reuse existing OpenRouter client/profiles/ledger. Compare criterion labels, pair rankings and calibrated anchors; repeat order-swapped judgements and independent judge/prompt controls. Personal cost preference fitting is optional future work.

## Measurements

Pairwise agreement with ties/abstentions and evidence coverage; calibration/anchor error; independent held-out fidelity and symbolic consequences; explicit edit costs; judge disagreement/order effects; label calls/tokens/cost and cache reuse.

## Acceptance and interpretation

Follow [13](../13-semantic-fidelity-supervision.md). Coherence and preservation of original erroneous axioms are not meaning quality. Hide learned scores/method identity; costs are outside the meaning rubric. Held-out labels are not fitting data, and the learned value head cannot judge its own output. With no experts, claim controlled synthetic meaning and AI-labelled fidelity only. LLM access is already available; annotation datasets and validation remain new work.

Shared splits, resources, failure accounting and statistics follow [02](../02-experimental-protocol.md). Dependency gates are in [08](../08-experiment-matrix-and-handoff.md); code changes and tests are in [12](../12-implementation-migration.md). Resolve every numerical setting in a new [run manifest](../protocol/README.md) before execution.
