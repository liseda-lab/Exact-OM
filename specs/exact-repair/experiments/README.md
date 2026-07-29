# Exact-Repair experiment programme

**Namespace:** `XR-E`<br>
**Status:** planned; separate from `specs/experiments/E*.md`

These studies validate result-changing Exact-Repair capabilities. They run on captured provisional
alignments after their implementation dependencies pass. No `XR-E` study changes product defaults;
promotion follows the gates in the parent README.

All studies inherit [`../02-experimental-protocol.md`](../02-experimental-protocol.md). XR-E00
freezes the shared protocol before confirmatory test runs.

## Study matrix

| Study | Primary question/hypothesis | Implementation dependency | Primary outcome |
|---|---|---|---|
| [XR-E00](XR-E00-harness.md) | Infrastructure for all RQs | XR-WP1; captured Exact-OM artifacts | Frozen, replayable protocol and harness |
| [XR-E01](XR-E01-safety-and-certificates.md) | RQ1/H1 | XR-WP2 | False-safe count/rate |
| [XR-E02](XR-E02-state-language.md) | RQ2/RQ3, H2/H5 | XR-WP3; XR-E01 pilot gate | Pair-level semantic preservation at equal eligibility |
| [XR-E03](XR-E03-joint-optimization.md) | RQ4/H3 | XR-WP2; XR-WP3 for full actions | Regret against exact external-utility optimum |
| [XR-E04](XR-E04-utility-learning.md) | RQ5/H4 | XR-WP4 | Held-out downstream repair regret |
| [XR-E05](XR-E05-candidate-proposals.md) | RQ6/H6 | XR-WP3/WP4 | Top-k recall plus accepted recovery gain |
| [XR-E06](XR-E06-generalization.md) | RQ7/H8 quality portion | XR-WP4/WP5 | Transfer degradation in regret/preservation |
| [XR-E07](XR-E07-scalability-anytime.md) | RQ8/H7 | XR-WP5 | Time-to-zero-gap and incumbent regret over time |
| [XR-E08](XR-E08-robustness.md) | RQ9/H8 safety portion | XR-WP2/WP5 | False-safe/status correctness under perturbation |
| [XR-E09](XR-E09-expert-explanations.md) | RQ10 | XR-WP5; human-study approval | Review accuracy and time |

## Execution waves

1. **Protocol/pilot:** XR-E00, then the XR-E01 safety pilot.
2. **Mechanism validation:** XR-E01 confirmatory, XR-E02, XR-E03, and XR-E05 deterministic arm.
3. **Learning:** XR-E04 and XR-E05 learned arm.
4. **System behavior:** XR-E06, XR-E07, and XR-E08 with the frozen selected candidate/model variants.
5. **Human evaluation:** XR-E09 only after schemas/UI and participant protocol are approved.

The waves prevent tuning rich states or learned models on confirmatory safety/transfer outcomes.
XR-E02 and XR-E03 may execute in parallel after their shared inventory is frozen, but they cannot
change that inventory in response to each other's test results.

## Claim coverage rule

- A study supports only the RQs/hypotheses listed in its spec.
- Safety eligibility is checked in every study, but only XR-E01/XR-E08 support broad safety and
  failure-handling claims.
- Performance observations in quality studies are descriptive unless replicated under XR-E07.
- Expert preference cannot replace logical or reference/query validation.
- Exploratory mechanism strata are labelled and do not retroactively expand a primary claim.

## Required outputs from every study

- frozen protocol/config/split/baseline manifest;
- captured-input hashes and per-run repair certificates;
- final replay/eligibility table;
- long-form per-unit metrics and failures;
- analysis script/environment hash;
- generated report following the common template;
- a machine-readable decision record: `supported`, `not_supported`, `inconclusive`, or
  `ineligible_due_to_safety`, with rationale.
