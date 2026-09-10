# Shared implementation clarifications — v2

**This replaces the v1 clarification/handoff.** It incorporates the accepted review and the
single-node broad-search constraint. Read RUN-PLAN and CHECKPOINT-RECOVERY first. Experimental
switches remain off in ordinary product defaults until a separate promotion/release action.

## 1. Explicit task semantics

Add a strict execution_mode with global_alignment and local_ranking. Candidate provenance is
generated, frozen_generated, or benchmark_supplied and is independent of mode. A frozen
unlabeled candidate pool can feed the real global selector/extractor. Global F1 and local MRR
must come from the corresponding execution paths, not two evaluators applied to local output.
Shared pair evidence may be reused. Assert executed stages and fixed eligible source universe.

Training features come from the training pool. Reporting pools cannot stand in for disjoint
training sources. Strip gold/reference columns from scorer/LLM inputs, including implicit gold
position in candidate lists. The E17 retrieval removal changes the pool policy, never task mode.

## 2. Baseline and readiness

R_0 is the immutable historical production configuration. First reproduce its intended
production path. Freeze a v2 baseline with parent R_0 if accepted correctness fixes change
numerical behavior; record the delta. Never edit R_0 hashes to make a modified implementation
pass the old identity guard.

Readiness is per arm, stage, and capability. Distinguish implemented primitive, integrated path,
fixture validation, data/model binding, screen-ready, and confirm-ready. A missing confirm
artifact must not prevent an independently valid screen. A declaration or an intentional
NotImplementedError is not implementation. The v1 experiment-wide ready flags are superseded.

The user confirms OAEI and BioKG-Align data are available. Resolve actual paths/descriptors and
verify capabilities. In particular, replace the repository's stale BioKG stub with a descriptor
of the available data; do not invent a publication URL or claim the data are unavailable because
the old descriptor says so. Available data and sufficiently complete labels are different checks.

## 3. Single node and all LLM roles

The target is one RTX 5090 and either 64 GB or 128 GB host RAM. Probe actual usable memory and
VRAM. The normal profile must fit 64 GB; use memory-mapped/chunked vectors, bounded batches, one
heavy GPU worker, and checkpointed source groups. The 128 GB profile may increase throughput or
admit a larger graph; it must not silently change a treatment's candidate/evidence semantics.

All generative LLM work uses OpenRouter: ontology/triple verbalization when enabled, entity
profiles, briefs, binary/listwise judgments, evidence-acquisition decisions, rationales,
exemplars/teacher calls, and model-sensitivity diagnostics. Local SapBERT/BGE/MiniLM-style
encoders and trained non-generative heads use the RTX 5090. No local generative-model fallback,
download, or VRAM allocation is permitted. An OpenRouter failure pauses that stage and preserves
work, rather than changing the model/provider or falling back locally.

Pin OpenRouter model, upstream provider where routing permits, requested/resolved revision,
tokenizer and prompt identity, decoding, seed, and logprob capability. Validate all returned
identities and record endpoint metadata without credentials. A model/provider change creates
new request identities; paired comparisons must remain compatible. Use versioned model IDs and disable provider fallback where supported. If a provider cannot
expose an immutable weight revision, record that limitation, provider identity and observation
window, interleave paired requests in bounded blocks, and retain raw responses. A frozen,
prospective comparison may support a result conditional on that observed hosted service; it
cannot establish exact fresh-call reproduction or a time-invariant deployment claim. An actual
model/provider change or unresolved drift splits the evidence into separate blocks; unpaired
or post-hoc comparisons are exploratory until a new predeclared paired block is collected.
Persistent response caching supports exact replay of observed responses, not model immutability.

The user grants broad OpenRouter spending discretion. Record that authorization and phase
request/token allocations in the campaign lock; do not require another arbitrary USD cap.
Track projected and actual USD, and never hard-code pricing.
Resolve current rates before execution, include input/output/retry cost, and reserve final calls.
Rate-limit retries use bounded backoff and request-ledger recovery. No OpenRouter model family
sweep: one primary model and at most one predeclared sensitivity model in the extended profile.

## 4. LLM off, routing, and trust

The decision-off control sets an actual off gate and disables primary decision/brief/rationale
generation. llm.experiment.enabled=false alone is invalid because it restores the normal gate.
Shared upstream verbalization artifacts remain identical across decision arms and their cost
is reported separately. A fully non-generative pipeline is a distinct diagnostic when changing
verbalization also changes evidence. Assert zero calls for every role declared disabled.

Implement quantile policies as two separately named contracts:

- source_top_fraction: freeze p and statistic/tie rule on development; on each unlabeled
  inference population select exactly ceil(p*N) eligible source groups. This is the primary
  budget policy. Persist inference selection for replay, never replay development pair IDs.
- transferred_threshold: fit a numeric cutoff on development and apply it unchanged; reporting
  call rate may vary. It is a diagnostic about distribution transfer, not an exact-budget policy.

For pair-level diagnostic parity, implement analogous pair_top_fraction separately. Canonical
ties sort by descending statistic, source IRI, then target IRI where applicable. Source-level
statistics include top-two score margin, candidate entropy, cross-model disagreement, NIL
uncertainty, and collisions. No gate uses reporting gold. A source need not have two pair gates
fire before entering a comparative call. The initial source-fraction policy uses
1-clip(top1_score-top2_score,0,1) for at least two candidates, and 1-clip(top1_score,0,1)
for a singleton; empty pools use an explicit no-candidate fallback, not an invented score.
Other statistics are recorded diagnostic features or inputs to E21, not an undeclared grid.

Measure routing and trust separately. Fit any fusion weight/calibrator only on training groups,
with development selection. Include the shipped beta*U mixture, a frozen constant-weight
comparison, and source-level decision integration; no optimal reporting-set weight. E25 names three
cached trust replays, followed by at most one selected integration at G4. Its source-first
variant uses the comparative choice within the displayed alternatives, including none, then
the frozen acceptance/cardinality rules; it does not turn displayed-none into ontology NIL. Keep
both pre-LLM and post-LLM decisions and correction/harm counts. Missing or malformed model output
uses the same predeclared deterministic abstain/base fallback in paired arms and is counted.

## 5. Evidence presentation and categorical probabilities

Keep raw ontology facts with stable IDs and full provenance. Compare the current five-section
64-token brief with a deterministic structured packet and, only as a controlled alternative,
a longer brief. Exclude numeric channel scores from the score-blind packet. Include relevant
source and target definitions, qualifier differences, relation directions, and explicit unknowns.
Do not label unmatched evidence as contradiction merely because it is absent on the other side.

Comparative input is the same top five or fewer candidates plus none, with frozen selection and
seeded order. Required binary controls see exactly the same candidates and facts. Record
overflow/omitted ranks. Use constrained letters and retain the full categorical P(A..E,Z), with Z=none in
this displayed pool. Probe the provider output at the actual decision token, including whitespace
and tokenization variants. Complete valid-option scores are required before calling the output
a joint categorical distribution; missing top-logprob entries are not zero-probability evidence.
If the bound provider cannot expose these scores, retain a separately declared hard-choice
comparison and evaluate accuracy/correction/harm without probability/calibration claims. Do not
silently substitute self-reported confidence or another provider. This capability gap must not
cancel direct-evidence/choice experiments. None in the displayed pool is not ontology-wide NIL.

Primary categorical representation is raw_joint P(letter). A pairwise_vs_none diagnostic uses
P(letter)/(P(letter)+P(Z)); it receives its own permitted calibration. conditional_real and
max_normalized are extended diagnostics only; max_normalized is not a probability and cannot
justify calibration claims. Zero denominators abstain deterministically and are logged.

One deterministic call per source is primary. Optional order sensitivity uses two permutations;
self-consistency uses two permutations times three samples at temperature 0.7 only if admitted
by the extended budget. Aggregate in canonical candidate coordinates, never by displayed letter.
Report list-length/order sensitivity and total costs, not an assumed saving from fewer calls.

Bounded evidence acquisition may retrieve at most two additional ontology evidence packets per
source, then one final judgment. It cannot fabricate ontology facts, access reference labels,
or expand the candidate pool after G1. Query/candidate rescue is an E05 treatment before G1 or
a later explicitly exploratory design. Record any pretrained/world-knowledge claim separately.

## 6. Quality, fusion, and explanations

E26's five mandatory decomposition arms are full, constant_q, no_sharpening, suppression_only,
and uniform. Constant_q and suppression_only apply only to active channels; uniform retains
the historical deliberately naive comparison and is identified as such. Do not attribute its
entire gap to q. Persist all quality components, active flags, label counts/top-m similarities,
and contributions. Absence from a reference is not a reliability-training target by default.

Test duplicate/near-duplicate synonym invariance and singleton behavior. Existing singleton
margin q=1 and singleton entropy q=0 are explicit controls, not assumed reliability truths.
A candidate-margin quality variant measures ambiguity between target concepts and is bound to
the candidate pool. Entropy over synonyms needs a frozen temperature and count-aware baseline;
encoder score agreement is not automatically agreement about which target is best. Reliability
models assess error conditional on score, evidence availability, and source difficulty.

analytic_fitted uses exactly the shipped nested analytic family with fitted tau/gamma and
nonnegative multipliers of arithmetic mean one. Neutral parameters and unit multipliers must
replay shipped scores and explanations before any fit. A flat structural-authority sum is a
separate family, never an unadvertised consequence of selecting analytic_fitted.
learned_global may use its explicitly different normalized nonnegative weighted-score family.
Adaptive weights are optional and must retain exact decomposition. Weight entropy is reported
as a diagnostic; there is no mandatory multi-channel entropy floor.

Pair scores, feature-additive selector logits, calibrated probabilities, final thresholds,
exact-anchor decisions, and collision competitors must each be reconstructible. Add actual
counterfactual replay for evidence removal and runner-up comparisons where claimed; algebraic
importance is not a causal effect or a logical proof. Feature-additive diagnostic models retain
their own declared explanation schema and may not masquerade as channel decompositions.

## 7. Difference, polarity, and hard anchors

Use supported/contradicted/unobserved states. One-sided absence supplies no signed contradiction
unless the declared semantics justify it; its contradiction quality is zero. Keep the current
normalized difference and clarified absolute score as diagnostics. Absolute is
1 - clip(0.5*(mean_source_unsupported+mean_target_unsupported),0,1); expose empty-side treatment
explicitly. Activity alone is never its success criterion.

Signed identifiers require pinned property-IRI allowlists, namespaces, normalization, and a
semantic rule making mismatch incompatible. Definitions, synonyms, and cross-references must
retain distinct provenance and be deduplicated across evidence banks. Test relation deletion,
irrelevant insertion, duplication, equivalent encodings, and source/target reversal.
Asymmetric difference is diagnostic until it is bound to a valid typed relation interpretation.

Protect historical exact matches only in the hard-anchor control. The soft-anchor treatment
keeps them eligible for rescoring/competition and records kind, collisions, and qualifier checks.
Conflicting hard constraints fail with IDs. Never infer equivalence solely from matching labels
across entity kinds. Predicted anchors must not confirm themselves; retain their provenance and
measure controlled 1%/5% anchor-noise propagation on development data.

## 8. Assignment and NIL

Primary E01 assignment first makes scores below threshold infeasible, assigns utility
score-threshold to eligible edges, and permits a private zero-utility unmatched target per
source. It optimizes this declared accepted-edge objective. Exact hard anchors are preassigned
only in that control. Component cap is 500 source-plus-target vertices; fallback is the frozen
threshold-first greedy method, with component/fallback diagnostics. Stable marriage is
source-proposing with canonical tie breaks.

Keep raw-score assignment followed by thresholding as the legacy diagnostic. The three-edge
case 0.90/0.69/0.69 at threshold 0.70 must keep the 0.90 edge under the primary assignment.
Do not claim that greedy solves the paper's global argmax. Track target collisions and recall
loss for explicit one-to-one versus other declared benchmark cardinality regimes.

NIL candidates share a joint probability scale with real candidates. Keep P(ontology NIL),
pool-miss, and abstention/unknown distinct where the data support those labels. Artificial
gold removal is a pool-miss diagnostic, not a natural NIL benchmark. E04 must use a real
NIL-bearing DISO or eligible BioKG/OAEI task rather than declaring NCIT–DOID to have NIL labels.

## 9. Fitting, transfer, and metrics

Use grouped out-of-fold training artifacts and permitted validation for selection. Persist
feature/explanation schema, negative policy, pool, data/model locks, seed, and training recipe.
All fitted consumers reject mismatched artifacts. Gold-absent source groups cannot become
ordinary negative rows without a valid completeness/NIL policy. Filter known positives from
hard/in-batch negatives. Artifacts cannot learn from reporting-reference candidate columns.

For a reference s->t1 and prediction s->t2, set-based F1 has one FP and one FN. E03's FP-only
alternative is a changed decision-loss surrogate, not a correction of faulty double counting.
Score calibration and a bounded cosine are different things. Report proper scores and
reliability diagrams with base rates; a quality-bin Brier alone does not isolate calibration.

E14 internal directions: <= is written < and means source subclass/subproperty of target;
> reverses that direction; = is equivalence. Entailment through predicted bridges is conditional
on those bridges. Exclude a queried pair's own asserted bridge from independent evidence.
Mutual subsumption can imply equivalence; a graph cycle alone is not OWL inconsistency.
Separate new SCC/hierarchy collapse, policy conflicts, and reasoner-proven unsatisfiability.
No unrestricted safety claim follows from this audit.
