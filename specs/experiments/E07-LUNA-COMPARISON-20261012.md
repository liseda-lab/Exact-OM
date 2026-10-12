# E07 Luna sensitivity on saved development evidence

The user requested a small Luna comparison on liseda-t2 on 2026-10-12. This is
E07's existing optional `second_model` sensitivity, separate from the active G4
composition and its downstream queue. It does not replace the selected judge or
authorize a final-study model change.

## Frozen comparison

- Control: completed `facts_binary` in `e07-selection-recovery-02`, NCIT–DOID
  D0_E03 public development split, seed 17: 200 sources, 4,000 candidate pairs,
  106 exact anchors and 1,000 invoked binary judgments (five per source).
- Treatment: `openai/gpt-6-luna-decisions` through OpenRouter's
  `POST /api/alpha/decisions`, using one text `noul` predicate per saved judged
  pair on the OpenAI-only route. This changes both the model and probability
  interface; it cannot isolate model capacity alone. Seed 17 fixes the saved
  population and paired bootstrap, not the hosted model: Decisions has no seed
  parameter. Retain the actual response model and raw responses.
- Reuse the saved source population, candidates, ontology evidence, non-LLM
  scores, fusion weights, exact anchors and extraction policy. No ontology
  loading, projection, retrieval, embedding generation or new annotation is
  required. No private test references are accessible to the comparison.
- Keep the original label-free, uncalibrated fusion and configured 0.7
  threshold/cardinality policy. Do not tune them against comparison outcomes.
  Before any paid call, replay the saved control probabilities and require
  exact reproduction of its emitted mappings and reference-based metrics.
- Freeze the input panel and dependency checksums before dispatch. Verify the
  historical judged evidence and record any unavoidable prompt/interface
  adaptation. A changed evidence population is a different experiment.

Report precision, recall, F1, paired corrections/harms, changed mappings,
coverage/refusals, actual token usage and measured API wall time. The public
reference is **known incomplete**: reference-relative false positives are not
verified semantic errors. Historical complete wall-time/cost measurements are
unavailable, so do not claim a matched-runtime or matched-cost experiment.

## Narrow transport and resource amendment

This optional diagnostic uses the existing OpenRouter credential and the alpha
Decisions route, verified by a separate repair project's October 12 pilot.
Credentials come from the current repository `api_key` at remote worker startup;
discard stale inherited credentials and never store their values in receipts.
Ordinary campaign routing and frozen G4 workers remain unchanged. Refusals and malformed
probabilities are unresolved outcomes, never fabricated negative decisions.
Rationales remain disabled and there is no local generative fallback.
The observed alpha contract differs from OpenAI's direct API: `state` and named
`questions` are mappings, `answers.equivalent.noul` is the probability, the
response identifies provider OpenAI and the model alias or `-20261006` revision,
and usage has input/output tokens and cost without a mandatory total-token field.
A refusal can surface as HTTP 502; retained rejected responses cannot be resent
automatically. The pilot's bundled semantic results do not qualify our distinct
single-pair predicate scientifically. Freeze endpoint metadata and its reported
input/output prices for estimates, separately from actual charged usage.

Use the existing content-bound prompt policy, durable raw-response ledger and
central campaign spending authority. Count Decisions input usage in the same
`exact-om-focused-v2` / `E07` scope, including all prior E07 spending. Its 25M
family pause, 200M campaign pause and existing notification thresholds remain
unchanged. Reserve input exposure before transmission, preserve uncertain
attempts and never automatically repeat an unknown paid send. Decisions' zero
generated-output reservation must not weaken ordinary chat output bounds.
Published-price cost estimates must remain separate from reported billed costs.
Keep one compact prediction log and the durable response ledger. A missing
initialized ledger blocks resumption; saved predictions cannot justify losing
accounting or resending previously charged requests. Duplicate invocations must
not overwrite another worker's status or exit receipt.

This replay is CPU/API work. The intended t2 job needs a small CPU allocation,
not its GPU. The requested independent t2 comparison permits one sequential Luna
sender alongside G4, as a narrow exception to the single hosted lane in
RUN-ONCE-EXECUTION. Both still reserve against the same central campaign limits.
Preserve allocation 14372 and all unrelated jobs. A cross-node paid
worker must share the authoritative spending store and verify mounted-path and
locking access before sending; copying/resetting that store is prohibited.

## Readiness and continuation

Offline tests and successful control replay establish implementation/preparation
readiness only. Real endpoint access, a usable direct credential, frozen launch
inputs and supervised ownership are separate dispatch gates. A lightweight real
14372 coordination step owns the separate t2 `srun`, propagates its exit and
verified completion, and is registered with the existing supervisor. Record the
actual remote step separately; never pretend it belongs to 14372. Account its
2 CPUs/8 GiB under separate t2 capacity, with zero local GPUs, so local GPU
successors remain eligible. Reuse the existing storage wrapper and shared hard
spending policy; preserve failed/uncertain ownership before any recovery.

Luna results remain descriptive development sensitivity evidence. Promotion
requires a separate decision, any necessary training-only recalibration and
revalidation of the G4 comparisons affected by the changed judge. Neither G4
nor the three queued GPU throughput checks waits for this optional experiment.

API contract: [Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
and [direct request reference](https://developers.openai.com/api/reference/resources/decisions/methods/create).
The OpenRouter alpha wire contract is separately backed by the saved
`data/exact-repair-corrective-20261009/qualification/decisions-pilot-20261012-001`
endpoint metadata and successful raw responses; do not substitute the direct
API's schema for it.
