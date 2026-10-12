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
readiness only. Real endpoint access, a usable OpenRouter credential, frozen launch
inputs and supervised ownership are separate dispatch gates. A lightweight real
14372 coordination step owns the separate t2 `srun`, propagates its exit and
verified completion, and is registered with the existing supervisor. Record the
actual remote step separately; never pretend it belongs to 14372. Account its
2 CPUs/8 GiB under separate t2 capacity, with zero local GPUs, so local GPU
successors remain eligible. Reuse the existing storage wrapper and shared hard
spending policy; preserve failed/uncertain ownership before any recovery.

This E07 replay remains descriptive development sensitivity evidence, not the
model-selection comparison on the eventual G4 winner. The subsequent approved
plan below supersedes the earlier proposal to recalibrate for Luna or reopen
G4 comparisons. Neither G4 nor the three queued GPU throughput checks waits for
this optional replay.

API contract: [Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
and [direct request reference](https://developers.openai.com/api/reference/resources/decisions/methods/create).
The OpenRouter alpha wire contract is separately backed by the saved
`data/exact-repair-corrective-20261009/qualification/decisions-pilot-20261012-001`
endpoint metadata and successful raw responses; do not substitute the direct
API's schema for it.

## Verification and launch, 2026-10-12

Implementation: `44b63fdc` introduced the bounded adapter; `a0a6ecf8` bound it to
the observed OpenRouter alpha contract; `e411274e` froze the replay and remote
ownership implementation. All 150 transport/accounting/pilot, 17 replay and
9 launch tests passed, with relevant lint/type checks and independent review.
The offline real-data check matched all 1,000 historical requests/probabilities,
all 4,000 scores and all 197 emitted mappings (reference F1 0.8232044199).

The frozen package is
`data/experiments-v2/e07-luna-comparison-20261012-01/`, with a 14 MB input panel
and 5 MB source archive footprint; no ontology/cache copies are required.
Estimated input is 4,167,983 tokens (~$0.417 at the captured route rate), not an
actual bill. Maximum request admission is 9,646 tokens/26,533 bytes, within the
unchanged prompt limits. Native usage/cost is recorded in the durable ledger.

The live t2 worker is `14466.0`, owned by supervised coordination step
`14372.274`; initial health was healthy and 51/1,000 judgments were complete at
21:13 UTC. G4 remains `14372.273`, supervisor `14372.269`. This is live launch/API
verification, not a completed sensitivity result or scientific promotion.
Read the package's `HANDOFF.md` and `dispatch-01/supervisor-handoff.json` for
actual ownership, safe continuation and current artifact paths.

An initial automatic launch review classified the evidence as sensitive. Before
resubmission, local provenance established public NCIT/DOID inputs and verified
that every payload's evidence exactly matched its historical OpenRouter/OpenAI
request. No reference labels or private test data are sent. Review then accepted
the launch; the earlier rejected command made no submissions or paid calls.

## Approved continuation: select the configuration, then compare judges

The user approved this sequence on 2026-10-12. Finish the development programme,
including G4 and its existing correctness/cost corrections, with the historical
`openai/gpt-4o-mini-2024-07-18` decision model. Retain completed experiments;
do not repeat them or reopen rejected E07/E21/E25 arms because Luna is available.
This amendment schedules a prospective comparison before final scientific work;
it does not change an active configuration, complete G4, or authorize final rollout.

1. Freeze the corrected G4-selected effective configuration, resolved supervision,
   runtime `deployment-artifacts.json` and actual fitted artifact hashes; a YAML
   with unpopulated artifact paths is insufficient.
   Preserve the historical configuration selection and its evidence. If the
   selected configuration has no active LLM decisions, record that applicability
   issue; enabling an LLM would be an additional treatment, not a model swap.
2. Compare exactly two judge arms on that same configuration: the historical
   GPT-4o mini decision route and Luna's approved Decisions route. Hold ontology
   evidence, source/query membership, candidates, anchors, all non-decision LLM
   roles, routing/gating, non-LLM scores, fusion, calibration, fitted weights,
   thresholds, extraction and seeds fixed. No model-specific refitting, threshold
   search, prompt/evidence optimization or student retraining is part of this
   comparison. Recompute downstream outputs that consume changed probabilities
   with those same weights and rules; they are outputs, not additional factors.
3. Use the admitted D0/D1 public development cohorts, or deterministic subsets
   bounded by the pre-call work/token forecast. Freeze exact membership, query
   pools, permitted evidence, selection endpoints/guards and cost boundaries
   before examining new judge outcomes. Preserve complete source/query groups
   and the competing-candidate scope required by ranking and global extraction.
   Use the existing development selection
   rules where applicable, record uncertainty and retain mini if no challenger
   qualifies. Do not substitute the exposed 200-source E07 result for this check,
   adaptively resample for a better result, inspect private answers, or use final
   reporting populations to choose the model. Local ranking pools remain local.
4. Reuse native preparation, embeddings and compatible scores. A cached mini
   control is valid only when the selected recipe, actual fitted artifacts,
   population/query pools, prompts, model revision and extraction identities
   match. Reconstruct its saved probabilities through the comparison path and
   reproduce scores, rankings and mappings before paying for Luna. Missing
   matching control requests require declared, accounted mini calls; changing
   the configuration to fit an available cache is prohibited.
5. Implement the smallest compatible replay through the actual selected pipeline.
   The existing `compare_luna_e07.py` supports only its label-free scalar/greedy
   recipe; it cannot silently stand in for G4's fitted selector/calibration and
   extraction. Preserve fit provenance. Any explicit evaluation-only binding
   for the swapped judge must verify identical fitted bytes and authorize only
   this comparison; never relabel a mini-trained fit as Luna-trained or relax
   general cache/deployment compatibility checks.
6. Record both actual model identities, the required wire/predicate adaptation,
   coverage/refusals, paired quality, correction/harm and cost. Because mini uses
   chat token probabilities and Luna uses Decisions, this estimates the effect
   of the deployed judge interface as well as the model. It answers which judge
   works better with the mini-selected configuration, not which independently
   optimized system is best. Unlisted reference pairs remain unlabelled; they are
   not confirmed negatives.
7. Freeze the resulting judge choice before final scientific runs and rebind
   their prospective descriptors. Preserve the original G4 decision and report
   this as a separate model choice; do not overwrite old metrics. A model change
   alone cannot be credited as an architectural gain: final controls/contrasts
   must preserve the declared attribution or explicitly delimit the claim.
   If Luna is chosen, its final application manifest must bind the same unchanged
   artifacts and inherited training provenance to the chosen judge. The comparison's
   evaluation-only binding does not itself authorize final deployment.

Preparation belongs after corrected G4 selection/fitting evidence is available.
The live supervisor has a metadata/code-only preparation successor, not an
already implemented scientific worker. Its predecessor must replace preparation
dependencies with real admitted verification/selection successors before retiring
its row; the comparison preparer must do the same for the actual paired worker
and final admission. No fictitious completed run satisfies either dependency.
GPU throughput checks can proceed independently. Keep the old E17 rows held and
carry the comparison dependency into every corrected final descriptor.

Use the existing E07/campaign spending authority, including all historical spend;
forecast both arms before dispatch. This approval does not raise the 25M E07 or
200M campaign pauses, change prompt limits, or reopen the completed t2 run.
Rationales remain off. Submit admitted workers as detached supervised Slurm steps,
preserve the interactive allocation and existing storage guards, and hand off
monitoring without babysitting. No new paid calls are made by plan preparation.
