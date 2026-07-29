# E17 — Promoted-Stack Integration Audit

**Motivation** (audit obs. 17): individual experiments isolate causal changes, but the product
ships their defaults together. Retrieval changes candidate pools; new channels compete through
σ-mixing and alter selector features; extraction interacts with thresholds, NIL, and label-free
acceptance. Independent improvements therefore do not imply that the combined matcher is
better, calibrated, coherent, or affordable.

E17 is a release gate, not a mechanism for rescuing an experiment that failed its own test.

## Research questions

- **RQ17.1**: Does the complete proposed default stack improve on the current rolling baseline
  `R_n`, and what is its cumulative change relative to historical `B0`?
- **RQ17.2**: Are the observed combined gains consistent with additivity, or do promoted changes
  exhibit diminishing, synergistic, or sign-reversing interactions?
- **RQ17.3**: Does each promoted flag retain a beneficial or non-inferior marginal contribution
  when removed from the otherwise complete stack?
- **RQ17.4**: Do the known high-risk boundaries—retrieval×downstream matching,
  channels×fusion/selector, extraction×calibration/NIL/label-free selection,
  listwise×NIL, and the supervised boundaries below—remain valid in combination?
- **RQ17.5**: Does the proposed stack preserve calibration, abstention/coverage, typed alignment
  coherence, explanation reconstruction, runtime/memory, and LLM cost?
- **RQ17.6**: Does the stack behave correctly under every supervision resolution — do the
  label-free resolutions still compose into a working matcher when no training reference is
  supplied, and do supervised components resolve correctly around E22's effective-training-unit
  and profile-policy boundaries?

## Hypotheses

The frozen proposed stack should improve its macro quality metric over `R_n` with the standard
CI gate and preserve every task×kind×relation guard. A batch containing only declared
cost/robustness promotions instead uses its pre-registered aggregate objective with the README's
quality non-inferiority guard. Some sub-additivity is expected because channels share evidence,
but no promotion should reverse its declared objective or become harmful beyond the slice MDE
when embedded in the stack.

## Inputs and freeze

The harness reads promotion-eligibility manifests produced since `R_n`. Each manifest names the
experiment arm, locked baseline, config overlay, fitted-artifact hashes, candidate-pool
fingerprint, power declaration, and promotion decision. Before reporting data is opened:

1. Apply all eligible overlays to `R_n` in dependency order, regenerate config and candidate
   caches, and refit any selector/calibrator/relation head on permitted training splits. Reusing
   a fitted artifact from an incompatible feature or pool fingerprint is an error.
2. Run the arms below on development tasks. Remove or revise a flag only here, then freeze the
   release-candidate stack and all interaction contrasts.
3. Produce one E17 reporting matrix. Reporting leave-one-out results diagnose interactions but
   cannot be used to prune and rerun a more favorable stack on the same test data; a failure
   blocks release and requires a separately pre-registered integration revision.

## Arms

- `rolling`: exact current defaults from `R_n` (**primary baseline**).
- `historical`: immutable `B0` (**longitudinal diagnostic**, never the primary promotion
  comparator).
- `stack_all`: every frozen proposed promotion applied and all derived artifacts refit.
- `stack_minus_EXX`: one arm per promoted result, removing only that result from `stack_all`.

On development data, also run `R_n + EXX` single-change arms and mandatory 2×2 factorials for
every pair that shares a candidate pool, score mixture, selector feature/threshold, extraction,
NIL, or relation-typing boundary. Among supervised promotions the mandatory pairs are
E20×{E18, E19} (a fitted head over a changed pool), E19×{E18, E10} (fused score versus ranking
and acceptance heads competing for the same headroom), E19×{E04, E07, E21} (learned fusion moves
`U`, which moves LLM gating and NIL together), and E23×{E02, E13, E18, E19} (learned graph
evidence versus anchor propagation/materialized closure, plus the ranking and learned-fusion
heads that consume its changed feature schema). An E23×E19 arm always refits E19 with the graph
channel present; loading a pre-E23 artifact is an error. Every supervised promotion is additionally
run in both of its resolutions, since a stack is only shippable if it composes under
`label_free` as well as `supervised`. Report the interaction residual
`Δ(A+B) − Δ(A) − Δ(B)`. Any pair whose development interaction changes sign or exceeds the
relevant MDE is pre-registered as a confirmatory reporting contrast; Holm-adjust those
contrasts. LLM arms use E00's model fingerprint/drift guard.

## Validation

Run the full eligible reporting matrix, three paired seeds, with the exact source/entity-kind,
relation, supervision, candidate-recall, cost, and incomplete-reference protocols from the
programme README. Primary comparison: `stack_all` versus `rolling`. Report `stack_all` versus
`historical` separately and never attribute that cumulative delta to the newest batch alone.

For every leave-one-out arm report marginal ΔP/R/F1 or MRR, coverage/abstention, ECE/Brier,
typed-relation macro F1 and cross-hierarchy cycle/coherence counts, channel importance, exact
explanation reconstruction, wall time, peak memory, and LLM tokens/calls. Compare observed
marginals with the individual experiment's effect and MDE, noting baseline or pool changes.

## Release decision and baseline stamp

The proposed defaults may ship only if:

1. `stack_all` passes the standard promotion criteria against `rolling`, including powered
   per-slice guards and any already-approved capability-specific overrides.
2. No reporting leave-one-out result shows that removing a flag improves the primary endpoint
   by more than that slice's MDE, reverses that flag's declared cost/robustness objective, or
   repairs a guard failure. Such a result blocks the batch; it is not permission to tune on
   reporting data.
3. All pre-registered high-risk interaction contrasts avoid sign reversal and their costs and
   calibration shifts are within their declared gates.
4. Candidate-pool/artifact compatibility, explanation reconstruction, NIL scale, LLM identity,
   and typed-hierarchy coherence checks pass.
5. The stack runs end to end under `supervision.mode: label_free` on a track with training
   labels present, producing a valid alignment with the expected supervision labels — a release
   may not depend on labels being available.

On success, append `R_{n+1}` to E00's lineage with parent `R_n`, the complete resolved config,
promotion manifests, fitted artifacts, candidate pools, and E17 result hashes. `B0` and all
earlier `R_*` entries remain immutable. Multiple sequential flag-flip PRs for the same release
all cite this one integration stamp; splitting the PRs does not split the statistical gate.

**Effort**: M, mostly compute. **Risks**: combinatorial interactions cannot all be exhaustively
tested; LOO measures marginal contribution near the full stack rather than every higher-order
interaction; repeated integration attempts can leak reporting results. The dependency-derived
factorials cover the most plausible boundaries, and a failed reporting gate requires a new
pre-registration rather than iterative test-set pruning.
