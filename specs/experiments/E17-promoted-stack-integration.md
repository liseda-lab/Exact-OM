# E17 — Frozen-Stack Integration Experiment

**Motivation** (audit obs. 17): individual experiments isolate causal changes, but the paper and
product present those changes as one matcher. Retrieval changes candidate pools; channels compete
through fusion; selection, calibration, NIL behavior, and LLM gating share scores. Independent
improvements therefore do not establish that their combination is better, coherent, or affordable.

E17 is mandatory when the paper claims a final combined Exact-OM system and when a release proposes
more than one result-changing default. It is not a mechanism for rescuing an arm that failed its
own confirmation.

## Research questions

- **RQ17.1**: Does the frozen combined stack improve on the paper's current `R_n` baseline?
- **RQ17.2**: Does every claimed component retain a beneficial or non-inferior marginal
  contribution inside the combined stack?
- **RQ17.3**: Do the small number of predeclared high-risk component pairs show synergy,
  cancellation, or sign reversal?
- **RQ17.4**: Does the stack preserve candidate recall, calibration, coverage/abstention, typed
  coherence, exact explanation reconstruction, runtime/memory, and LLM cost?
- **RQ17.5**: Does the combined matcher run correctly in each supervision regime claimed by the
  paper, including label-free execution when training references happen to be present?

## Inputs and freeze

E17 consumes only candidates that survived their experiment's development screen and confirmatory
comparison. A component with only exploratory evidence may appear in a separately labelled
diagnostic stack but not in the primary paper stack. Before E17 reporting data is opened:

1. Apply surviving overlays to the same frozen `R_n` baseline in dependency order. Retrieval
   changes run first; regenerate candidate pools and refit downstream artifacts whose pool or
   feature fingerprint changed.
2. On development data, execute the E17 screening matrix below, resolve integration defects, and
   freeze the final stack, leave-one-out arms, required interaction contrasts, reporting tasks,
   seeds, endpoints, and cost/non-inferiority bounds.
3. Write one immutable E17 selection/design record. Reporting results may diagnose a failed stack
   but may not be used to prune it and rerun a more favorable combination on the same test data.

## Screening matrix

Run on development data, normally with one seed:

- `rolling`: exact current defaults from the frozen `R_n` baseline.
- `stack_all`: every individually confirmed paper component, with compatible artifacts refit.
- `stack_minus_EXX`: remove one claimed component at a time from `stack_all`.
- `rolling_plus_EXX`: optional single-addition controls when the individual result used a
  different baseline or candidate pool and cannot support an interaction calculation directly.
- targeted 2×2 contrasts only for component pairs that share a causal boundary.

The default high-risk boundaries are retrieval×selector/reranker, fusion×acceptance, fusion or
uncertainty×LLM gating, contrastive evidence×quality/fusion, extraction×NIL, and representation or
graph structure×a fitted downstream head. Instantiate a contrast only when both components are in
`stack_all`. The E17 config names and justifies every included pair before screening. Do not run a
complete factorial.

Use the screening results to repair implementation incompatibilities, not to manufacture a better
test result. Once stable, freeze `stack_all`. A component that becomes harmful on development may
be removed before the freeze, with the removal and reason recorded; it remains part of the
individual experiment record.

## Confirmatory matrix

Run on untouched reporting data with at least three paired seeds:

- `rolling` (**primary baseline**);
- frozen `stack_all` (**primary candidate**);
- one `stack_minus_EXX` arm for every component whose contribution is claimed in the paper;
- only those 2×2 interaction contrasts named as central or high-risk in the frozen design record;
- historical `B0` optionally, as a longitudinal diagnostic and never the primary comparator.

This bounded matrix replaces an exhaustive factorial. Leave-one-out measures marginal contribution
near the final system; the selected 2×2 contrasts test the most plausible shared boundaries.

## Validation and reporting

Use the full eligible reporting tasks declared for the paper, the same candidate pools and paired
seeds where retrieval is held fixed, and all applicable supervision/incomplete-reference rules
from the programme README. The primary comparison is `stack_all` versus `rolling` on the frozen
macro quality endpoint. Report cumulative change versus `B0` separately when available and never
attribute it solely to the newest stack.

For `stack_all`, baseline, and every leave-one-out arm, report:

- candidate recall before end-to-end quality;
- per-task and macro P/R/F1 or MRR/Hits@1;
- coverage/abstention and ECE/Brier where applicable;
- entity-kind and typed-relation slices claimed by the paper;
- exact explanation reconstruction;
- wall time, peak memory when material, and LLM calls/tokens;
- the marginal delta from removing each component.

For each retained 2×2, report the interaction residual
`delta(A+B) - delta(A) - delta(B)` with its paired interval. Apply the frozen multiplicity rule when
more than one interaction is confirmatory.

## Paper and release decision

The combined-system claim passes only if:

1. `stack_all` improves the frozen primary endpoint over `rolling`, or meets a predeclared
   quality non-inferiority bound for a cost/robustness claim, with the required paired interval.
2. No reporting task or claimed entity-kind/relation slice breaches its frozen regression bound.
3. No leave-one-out result shows that removing a claimed component materially improves the
   primary endpoint or repairs a guard failure. Such a result makes the combined claim
   `inconclusive` or failed; it is not permission to retune on reporting data.
4. Predeclared high-risk interactions avoid an unexplained sign reversal and stay within frozen
   calibration and cost bounds.
5. Candidate-pool/artifact compatibility, supervision resolution, explanation reconstruction,
   NIL scale, LLM identity, and typed-coherence checks pass where applicable.

For a paper-only result, the immutable E17 manifest and result hashes are sufficient. If the same
stack will become product defaults, append `R_{n+1}` with parent `R_n`, resolved configuration,
artifacts, candidate-pool fingerprint, and E17 result hashes; earlier baselines remain immutable.

**Effort**: M, primarily compute. **Risks**: leave-one-out does not identify every higher-order
interaction, while an exhaustive factorial is impractical. The bounded matrix answers the combined
system question and tests the few causal boundaries most likely to invalidate it without turning
E17 into a second full experiment programme.
