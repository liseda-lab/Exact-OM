# XR-2 protocol archive and XR-2.1 run contract

[pilot.json](pilot.json) covers the historical XR-2 methodology: all action families, HGT with matched controls, conditioned probabilistic generation, unary/pairwise optimisation, typed symbolic supervision, generated-to-real adaptation, Conference and separate Bio-ML releases. Values are starting settings for an exploratory pilot, not empirically selected defaults.

[smoke.json](smoke.json) inherits the pilot and reduces counts/budgets. Resolve inheritance before validation or hashing. Dictionaries merge recursively; arrays replace. Unknown fields, cycles and obsolete XR-P1 versions are errors.

[schema.json](schema.json) is the strict schema for a resolved protocol. [validate_protocol.py](../reference/validate_protocol.py) checks the schema subset used here plus cross-field invariants without external dependencies. It prints the resolved hash and requested generated-case counts. It does not download data, reason over OWL, label cases or train a model.

The 16 generated families request 576 pilot cases: 384 training, 96 development and 96 test. Smoke requests 96: 32 per split. Generation/label deadlines can produce fewer; log requested, generated, verified and labelled counts separately.

The requested split above is the configuration request. The archived executed corpus used grouped membership of 360 training, 90 development and 126 test cases; see [evidence](../11-benchmark-evidence.md). These counts are different facts, not values to reconcile by editing historical files.

## Historical semantic conventions behind the numbers

Each group is a clean structural parent, with independently recorded corruptions and clean controls. All siblings stay in one split. The mixed family is additionally evaluated in a composition-holdout regime: omit mixed training/development cases in that named arm rather than leaking the held-out mechanism.

The graph uses width 128, three HGT layers and four attention heads. The matched R-GCN control uses the same target/candidate attention readouts. Four circuit mixture components are compared with one component. These settings do not prescribe a specific graph or circuit library.

Profile costs count features of the actual selected edit once: mapping deletion; original inclusion directions removed; endpoints changed; subclass expressions specialised; new necessary conditions; new intersection/existential constructors; ontology objects changed; and changed human-authored ontology objects. Do not count a constructor merely for retaining an original axiom. Ontology keep has zero edit cost; human ontology edits incur both the general and human increments. A composite is costed by its emitted changes, not by adding duplicate labels for its component trace. Provenance-unknown edits use the general increment and are reported separately. These numbers are declared preferences, not inferred probabilities.

Desired and unwanted query families have frozen weights before comparing predictions. Normalise within each nonempty family as specified in [07](../07-corpus-and-training.md); the pilot uses equal desired-family weights and the stated unwanted-family penalty. Store the full resolved query list, labels and weights per case. Required/prohibited hard queries contain identifiers resolved by the case manifest, not arbitrary free text passed to a reasoner.

A protocol is supplemented by a run manifest containing exact input hashes, per-pair train/development/test membership, provenance/eligibility, hard query definitions, qualified backend and library versions, hardware and actual command arguments. The whole-ontology ekaw holdout constrains the Conference test split; all remaining pair assignments must be frozen before training. Bio-ML shared-ontology overlap must be disclosed.

## Preserved operational invariants

- Compile and verification budgets are per call; the run/stage/campaign supervisors impose their own deadlines.
- No full initial classification is mandatory before diagnosis can make progress.
- Unknown assignments are pending and contribute to the global bound; finite retries are not logical exclusions.
- Exact teacher distributions require complete caches. Partial verified comparisons may train explicitly named losses.
- A missing complete development subset cannot silently switch checkpoint selection to test results. Record the unavailable criterion and stop that arm or use a predeclared development-only fallback.
- Ontology changes are returned as patches. Repair remains disabled by default in production.

The preserved schema encodes the historical XR-2 pilot, not the XR-2.1 additions. Ablations are derived, separately named experiment manifests with explicit changed factors; do not relax validation silently to accommodate an old protocol.


## XR-2.1 configuration boundary

**Normative revision: 30 September 2026.** Preserve `pilot.json`, `smoke.json`, `schema.json` and `batches.json` byte-for-byte for reproducibility. Their loader and static tests validate XR-2 only. New requirements below must be implemented in a distinct `exact-repair/protocol/v3` schema/loader and separately named run files before an XR-2.1 campaign. This document is the configuration design; it is not an executable run request.

Reject unknown fields, duplicate JSON keys, unsupported versions, cyclic inheritance, nonfinite numbers and implicit inheritance from XR-2. Fully resolve and hash a run before use. File names may be chosen by the implementation, but schema identity and content determine compatibility. A compatibility converter may carry over explicitly listed unchanged values; it cannot default away new guarantees or relabel an old checkpoint.

### Required sections

| Section | Fields that must be resolved before execution |
|---|---|
| identity | Research revision, record/feature/label schemas, code/dirty hash, run/parent IDs, purpose (smoke/development/confirmatory), no automatic campaign resumption |
| input | Immutable ontology/import/alignment/evidence manifests, relation interpretation, licensing/availability, matcher source, split parent and evidence cutoff |
| policy | Monitored public/private signatures, source-exception mode/proofs, hard query basis, activated obligations, ontology occurrence eligibility and authorship |
| generation | Grammar/templates/constructors, per-side menus, endpoint alternatives, omission controls, bounds/expansion schedule, elementary guarantees, candidate caps and tie rules |
| circuit | Per-family partition, variable-tree policy, ref/GC policy/version, allocated/live/root/element/RSS limits, per-call and aggregate deadline, cache directory/version/reuse policy, semantic constraints and proof dependencies, exact distribution identity |
| model | Backbone/readouts/dimensions, feature masks, pair selector and limits, proposal context, unary/pair benefit, risk-head configuration and optional cost predictor explicitly disabled initially |
| teacher | Typed frozen consequence basis, non-vacuity, family weights, exact enumeration ceiling/deadline, per-query masks, complete-versus-sampled eligibility |
| collection | Number of frozen rounds, cases, plan-attempt strata/budgets, generator/exploration mix, solver diversity, quartet context, deduplication and retry policy |
| losses | Target basis/scale, symbolic/AI eligibility, anchors, regression/ranking/proposal/quartet/risk weights, per-loss masks/normalisation, sampled target approximation name |
| llm_labels | Enabled flag, existing OpenRouter profile and explicit teacher/evaluator role binding, request-ledger identity, frozen rubric/model/prompt/evidence schema, criterion applicability, tie/abstention policy, call/token/cost ceilings, cache/reuse, independent evaluator split; explicit approval for a new run |
| objective | Semantic calibration identity, explicit nonnegative edit weights, integer scale/rounding, pair factor bounds and frozen epoch rules |
| selection | Shortlist size, nonnegative integer utility window, construction deadline, deterministic tie rule, matched utility-first control, risk ordering, support cuts, unknown/untested ledgers, exact upper bounds, retry budget, first-incumbent versus optimality stopping objective |
| reasoning | Qualified capability matrix/routing, detector supported rules, support budget, complete query coverage, safe session/module/cache policy, unknown causes |
| resources | Whole-case and per-stage wall/CPU/RSS limits, worker startup/input transfer supervision, cleanup grace, concurrency and campaign ceiling |
| evaluation | Cohorts, exact train/dev/test hashes, metrics, accounting/cache policy, checkpoint rule, independent judge, grouping/interval method, all scheduled-case denominator |

Record chosen values even when a component is disabled. Never turn an omitted field into “unlimited”. Per-call deadlines cannot exceed the remaining whole-case budget. An increased menu/language or changed target/feature/schema must change the appropriate identity and downstream epoch/checkpoint compatibility.

### Development starting settings

The following are explicit starting settings for the **new generated-data development pilot**, not measured optima or automatic Bio-ML budgets. Create them in a new v3 configuration only after the corresponding loader/runtime exists. A smoke profile reduces counts, not semantic safeguards.

| Component | Initial setting and controls |
|---|---|
| Graph | HGT width 128, three layers, four attention heads, dropout 0.1; matched R-GCN readouts; two-hop context, 4,096 nodes, 32,768 edges, 64 discovered supports and 128 text tokens per retained description; omissions explicit |
| Interactions | At most 16 object neighbours per object using the shared selector; preserve bounded candidate-factor accounting separately; unary control uses no pair utility |
| Menus/grammar | Start at classes/properties/endpoints per side 8/4/8 with depth/constructor bounds 1/1; next 16/8/16 with 2/2; final 32/16/32 with 3/3. At most three declared stages, subject to remaining budget; never claim unreached stages were searched |
| Candidates | Up to 64 unique candidates/object in stages one/two and 128 in stage three; fixed-cap controls 32/64/128 use compatible menus or report invalid budgets. The final 32-endpoint menus can require 66 elementary choices for an equivalence even before rich replacements, so stage three must not inherit cap64. Elementary alternatives are reserved first; if they exceed the cap, reject the cap or raise it explicitly before freezing rather than dropping them |
| Sampling | Four mixture components with one-component control; 32 draws/object/stage, maximum 256 total draws/object; count duplicates and all failed attempts; exact family-mass weighting |
| Compilation | At most 20 seconds/family call and remaining aggregate generation budget; development worker RSS limit 8,192 MB. Start live/reachable and allocated-node guards separately at 1,000,000 and 4,000,000, measured with correct lifetimes. These are resource guards, not target circuit sizes; compare tree policies under identical limits |
| Case budget | 300 seconds end-to-end, initial diagnosis allocation 20 seconds, generation allocation up to 60 seconds, each full verification call up to 30 seconds, cleanup grace 2 seconds reserved in supervision. All are clipped by remaining case budget; no unrestricted retries |
| Search | Shortlist eight for both risk-order and matched utility-order arms; serial master K=1 is a separate baseline. Matched sensitivities K=4/16. Integer objective scale 1,000, window delta 250 (controls 0/1,000), at most 10 seconds per shortlist construction, stable canonical assignment-ID ties; at most 100 master solves and 100 candidate checks per case, all clipped by remaining budget. One scheduled attempt before optional single retry with an explicitly larger remaining budget; frozen coefficients per epoch |
| Collection | Three supervised collection rounds, up to 128 plan attempts/case/round: 32 utility solutions, 32 diverse alternatives, 32 quartet members (eight quartets), 32 proposal-driven plans. Deduplicate records and retain stratum origin/attempt cost; overlaps do not create extra labels |
| Optimisation | AdamW learning rate 0.001, weight decay 0.0001, gradient norm cap 1, batch size eight cases, at most 100 epochs, development decoding every five epochs, patience ten evaluations; seeds 13/37/73 on fresh held-out data |
| Loss coefficients | Benefit 1, ranking 0.2, proposal 1, quartet 0.2, risk 1, each normalised by eligible labelled terms. Disabled/missing heads have zero eligible terms, not fabricated labels. No unanchored AI scalar/cost tradeoff |
| LLM | Disabled in initial correctness and symbolic-training pilots. Enable only under a new annotation manifest with fixed model/prompt, evidence, independent evaluation and explicit bounded expenditure; no default unlimited label job |

Compilation budgets apply across all attempted families/stages; the generator must allocate the aggregate budget deterministically and report families never attempted. Increasing the allocated-node ceiling does not fix retained-node leaks or justify silent language reduction. Acceptance tests compare support/probability correctness before performance conclusions.

The shortlist grid changes scheduling, not utility or feasibility. Risk versus utility ordering uses the same K, delta, construction budget and tie rule. With scale 1,000, delta 250 is a quarter of the declared calibrated utility unit; it is a development choice, not a semantic constant. Development checkpoint selection follows [07](../07-corpus-and-training.md), not the historical fixed-pool regret rule. Freeze numerical tie tolerances and any fallback before the first evaluated epoch.

Whole-ontology biomedical runs need a separately declared resource profile based on capability and development measurements. These generated settings do not promise Bio-ML verification in 300 seconds. A bounded unknown is an expected possible outcome.
