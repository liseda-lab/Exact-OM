# Exact-Repair: corrections required after the October preliminary study

**Date:** 9 October 2026. **Target:** XR-2.1 corrective revision; implementation requirements, not a completion report. **Execution plan:** [16 Single-node experiments](16-liseda05-experiment-plan.md). **Evidence:** [review record](implementation/preliminary-review-20261009.json).

The objective is to make the agreed repair method operate reliably and efficiently on `liseda-05`, and obtain the main experimental results before the user-supplied abstract deadline of 18 October. The full-paper deadline is 25 October. Both dates are in 2026; planning uses Europe/London time. Additional CPU nodes or six A100 GPUs are optional capacity, never prerequisites.

This document specifies corrections to the implementation. It does not reject the method because preliminary runs failed. Preserve the shared graph encoder, constrained probabilistic generation, unary/pair semantic benefit, whole-plan risk, weighted MaxSAT and qualified symbolic verification. Do not introduce RL, replace MaxSAT, replace OWL semantics with an unqualified approximation, or begin a large architecture search as part of these corrections.

For the requirements explicitly amended here, this document and [16](16-liseda05-experiment-plan.md) supersede conflicting implementation or campaign defaults in documents 02, 07, 08, 10, 12 and 14, and the experiment breadth/prioritisation in 13. Unchanged formal/action contracts remain in force. Historical protocols, receipts and preliminary results remain immutable. This specification is not an experiment launch or a restart of the stopped T2 campaign.

## 1. Source and evidence baseline

Remote refs were fetched on 9 October. Local `dev` was fast-forwarded from `092c7fe` to **`70c2c73189f4856c438827d896dae667a271bbcf`**, without overlapping the user's local edits. The repair campaign is separately published at **`origin/xr21-t2-20261001`, `9fa4c369d0d8a99a7f7272bc24835d033d80e5aa`**. Focused qualification is at **`origin/xr21-qualification-005`, `f8d8223f83b3b2f0426a03e8cc2b928a98c52c4d`**. Fetching current `dev` does not merge either repair branch.

The review used `exact-repair-preliminary-results-20261007.tar.gz`, its 1,304 checksummed members, row-level reports and the published campaign source. The compact export omits raw datasets, model weights and some worker logs; this was not a replay. A source-bound correction receipt proves only its recorded checks, not conformance of a later merged implementation.

| Observation | Corrective implication |
|---|---|
| Fresh generated-pool quality: 18/576 rows; learned arms 4/384, all coherent controls; learned corrupted inputs 0/192 | Establish bounded repair of corrupted inputs before multiplying runs; separate clean/corrupt outcomes everywhere. |
| Fresh graph arms each have 48/64 schema-unavailable rows | Declare and qualify the full schema; train new compatible models. |
| Expanded TRAIN acquisition: 14/128 cases with usable semantics, all overlap; DEV collection: 33 unique usable labels/256 slots | Repair collection throughput and cross-family coverage; do not increase epochs on the same narrow labels. |
| Primary scaling: 200/240 rows fall back to the original inventory; only 8/240 generated-pool quality outcomes | Preserve useful partial generation, but report fallback separately from successful rich generation. |
| Successful native circuit compilation: 351 calls/13.09 seconds; supervised compiler calls: 658/1,018.25 seconds | Measure orchestration and failed calls before blaming SDD compilation. The populations differ; this is not a per-call speedup estimate. |
| Real projections: 8/54 usable outcomes, all coherent, one exposed ontology pair | Qualify actual Conference inputs; these projections do not establish real repair performance. |

Detailed failure counts, original source regimes and the distinction between completed processes and scientific outcomes remain in the archive. Neither timeouts nor unavailable results are logical negatives.

## 2. Methodological changes, explicitly delimited

| Decision | Classification and required disclosure |
|---|---|
| Schema, lifecycle, transport, provenance, denominator and cost-accounting repairs | Implementation of the existing method. |
| Reuse of immutable context, compiled circuits, prepared reasoning and committed labels | Performance corrections; accepted candidates, target values and logical guarantees must remain equivalent. |
| Complete typed endpoint substitution for supported composite bundles | Extension of implemented action coverage already intended by the action contract. Version the candidate/cost identities; do not relabel old runs. |
| Early verified elementary repair, followed by bounded complex expansion | **Explicit change to search scheduling.** Enable as `staged_verified_repair`; evaluate against the matched one-stage policy. It does not change the final feasibility definition or invent a completeness claim over ungenerated actions. |
| Informative TRAIN-only acquisition and desired/unwanted semantic probes | Implements the agreed supervised approach, but changes the pilot sampling distribution and target basis. Version both; old scalar labels cannot be silently reused under the new target. |
| Primary symbolic versus symbolic-plus-LLM comparison and scheduled use of both GPUs | Restores semantic fidelity as a primary research component; revises resource and experiment priorities. Additive weak-loss routing corrects implementation, not logical semantics. Unary/extra encoders become bounded ablations. |
| A future restricted-fragment reasoning study | A separately named scope/claim change, **not required for the primary campaign**. Never substitute it silently for complete acceptance. |

An implementation agent must record any additional semantic change in its handoff. Ordinary engineering choices within these contracts do not require a new user approval checkpoint.

## 3. Required implementation work

Each `COR-*` requirement is closed only by a source-bound acceptance receipt listing actual checks, results and remaining scope. An empty test selection or a synthetic mocked reasoner test cannot close native correctness or throughput requirements.

### COR-01 — Integrate the published fixes before making new ones

**Priority:** P0. **Locations:** campaign changes under `exact/repair/`, `tools/repair/`, focused repair tests and their resource helpers.

Start from current `origin/dev` and integrate the necessary campaign changes and qualification tests in an isolated implementation branch. Review conflicts against current native-stack, experiment-budget, hosted-spending and checkpoint code. Do not copy the older campaign checkout over newer unrelated code or require a wholesale merge of its historical orchestration scripts.

Retain the already corrected file/hash transport for large labels, final-pool vocabulary omissions, canonical activation-sensitive removals, elementary endpoint recovery, declared graph schema and their focused tests. Port only the study tooling required by the new plan; reuse the existing runner rather than create another supervisor framework. Preserve original unknown/failure rows and old STOP/PAUSE records.

**Acceptance:** the merged code passes the applicable repair regressions and focused campaign corrections, including payloads larger than 16 MiB, omission after every candidate producer, exact original-bundle protection and removal of an activation-bearing bundle with an otherwise identical axiom set. Record native package versions and the new source revision. Explicitly distinguish fixes imported unchanged from new fixes.

### COR-02 — Make the graph contract independent of the observed mini-corpus

**Priority:** P0. **Locations:** `graph_schema.py`, `graph.py`, `model.py`, training preparation and checkpoint loading.

Declare all supported node and relation types, feature layouts, missingness conventions and reversal rules before reading splits. The campaign declaration has 13 node types and 553 relation types; this is a compatibility contract, not a requirement to allocate dense tensors for absent edges. Keep sparse graphs and qualify actual parameter/optimizer memory. Never add randomly initialised parameters while evaluating a frozen model.

Bind the same schema identity to all encoder/value arms, generated pools, dataset releases and checkpoints. Exercise every declared type with tiny typed fixtures independent of held-out data. Training must include varied constructor/axiom contexts; a schema-compatible but never-trained relation does not establish generalisation. Any parameter-sharing redesign is a separately recorded architecture change, not an invisible memory optimisation.

**Acceptance:** clean and corrupted development cases, unseen-in-pilot constructors, composite mappings and generated candidates run through every enabled encoder without missing-key or shape errors. Perform actual forward, backward, optimizer update, save/resume and generated-pool inference. Old incompatible checkpoints are rejected with a typed reason, not silently padded.

### COR-03 — Repair process lifecycle, partial progress and measurements

**Priority:** P0. **Locations:** `workers.py`, `compilation.py::_compile_bounded_cached`, `proposals.py::enumerate_grammar`, `tools/repair/scaling.py`, label acquisition.

Propagate `cleanup_complete` from nested calls to their consumers and receipts. The current compiler/enumerator consumers can discard that flag. An operationally incomplete call cannot be marked a clean completed stage. Preserve previously committed, independently verified outputs, but quarantine the affected worker and do not admit new work into its reservation until owned descendants are accounted for.

Use bounded reusable workers or batched work to reduce repeated interpreter/import/transport costs. First instrument the existing path; do not build a distributed service. Workers have task-count, RSS and lifetime limits, per-task remaining deadlines, single ownership and deterministic restart. No shared mutable reasoner across concurrent theories; no fork of a live CUDA context. File/hash payload transport is retained.

Persist phase-start and phase-finish records for retrieval, grammar construction, immutable-context checking, compile/cache lookup, sampling, graph/model work, materialisation, reasoning, query scoring, proof extraction and cleanup. Write effective menu and case identities before potentially blocking work. Journal completed object/family/label results atomically; later failure must not erase earlier results.

Store failed compiler `failure_telemetry` when no circuit exists; the scaling exporter currently drops it although the profile exporter handles it. Distinguish never attempted, timed out, unsupported, resource exhausted, software failure and completed. Count full process-tree CPU/RSS, queue/service wall time, bytes transported, cold/warm cache status and cleanup. Do not add overlapping phase totals to campaign elapsed time.

**Acceptance:** terminate at each major phase, including nested success, timeout, native hang, large payload transfer and receipt acknowledgement. All calls return within declared work plus cleanup bounds, preserve committed evidence and cumulative budgets, and leave no unaccounted owned descendants. Injected cleanup failure reaches compiler/enumerator callers. Compare repeated tasks with isolated execution for identical scientific results.

### COR-04 — Reuse immutable reasoning and compile once per exact language

**Priority:** P0. **Locations:** `grammar.py::with_immutable_context`, `detection.py`, `compilation.py`, `pipeline.py`.

`with_immutable_context` currently invokes a fresh detector per candidate and repeats immutable classification even for shared activation tuples. Deduplicate canonical activation tuples and reuse completion/proof work within a request. Where practical, prepare the fixed theory once and request active-expression obligations without repeatedly enumerating irrelevant monitored-class obligations.

Cache keys include immutable asserted theory/import identity, policy, activation tuple, detector rules/version, scope and bounds. Validate replayed proofs against that exact context. Candidate exclusions still identify all corresponding encodings separately. Preserve the predeclared logical check-budget semantics when deduplicating work; record both requested checks and actual expensive calls. Detector silence, unsupported expressions and timeout never become proof of satisfiability. Do not persist transient unknowns as permanent completed decisions.

Compiled structure is independent of neural weights. Reuse it across draws, epochs and compatible models with keys covering grammar, typed vocabulary, contextual exclusions, variable/vtree order, canonical encoding and compiler version. Sampling recomputes the appropriate weighted counts and mixture/family masses. Preserve fixed-literal weights, mixture conditioning and probability sums over aliases with identical axiom **and activation** content.

An unfinished circuit is never usable. Completed family circuits may define an explicitly restricted support, with normalisation over precisely that support and a new support identity. Such output is not a full-language likelihood. Exact proposal targets requiring omitted support remain masked and missing teacher mass remains reported. Do not narrow a declared language invisibly to claim a timeout recovery.

**Acceptance:** small exhaustive supports agree before/after caching in candidate identity, excluded assignments, replayed proofs, partition functions and probabilities. Test changed imports/policy/activations, empty families, aliases, partial compilation and exhaustion. End-to-end timed profiles must show where savings occur; an isolated compile speedup is not sufficient.

### COR-05 — Support complete composite endpoint replacement and correct costs

**Priority:** P0. **Locations:** `candidates.py::materialize_retrieved_endpoints`, `replacement_cost_features`, grammar templates and candidate provenance.

Do not reduce a composite mapping to an inferred elementary relation. Bind source/target anchors to the observed mapping record. Apply a declared, type-preserving substitution to all bound endpoint occurrences in the complete original axiom bundle and its activation expressions. Preserve the other structure, unaffected axioms and provenance. If anchors share an IRI or have ambiguous binding, require a supported explicit binding rather than global text replacement. Ambiguous alternatives are reported unavailable; they do not crash the entire case or remove keep/delete/other valid actions.

Example: replacing bound target `T` by `T'` in `{S intersection E subclassOf T, T subclassOf S}`, with activated `S intersection E`, must emit `{S intersection E subclassOf T', T' subclassOf S}` and the unchanged activation. Replacing bound source `S` must also update its occurrences inside the activation. This is one complete replacement, not two separately selectable edits.

Initially extend composite substitution for the class-mapping/axiom kinds already admitted by the repair API. Preserve already supported elementary object-property, data-property and individual replacements; unsupported composite non-class substitutions remain explicit, not an unrequested expansion or silent regression. Update cost extraction together with generation: charge actual bound endpoint changes once, preserve ontology/human-authorship costs and avoid charging retained constructors as new ones. A validated edit descriptor may explain the structural delta, but an arbitrary producer tag cannot set costs. Equivalent outputs for the same editable object and original occurrence have equivalent cost and canonical identity regardless of construction path.

**Acceptance:** clean and corrupt composite bundles, both equivalence directions, repeated bound occurrences, activations, unchanged context, duplicate axiom emitters, locked/ineligible objects, ambiguous anchors and canonical duplicates. Check expected emitted axioms and costs explicitly. Qualify the intended enabled language; an endpoint-disabled arm remains an ablation, not completion of this requirement.

### COR-06 — Obtain an incumbent before expensive expansion when possible

**Priority:** P0 for reliable staged operation; the one-stage control remains available. **Locations:** `pipeline.py`, `kernel.py`, candidate/score epoch records.

The existing kernel already returns a completely verified coherent input unchanged. Perform this bounded check before expensive rich generation when possible. On unknown initial verification, continue under the declared repair policy without asserting coherence.

The new `staged_verified_repair` schedule is:

1. Freeze input evidence, edit policy, monitored signature and hard query basis for the declared language. Construct permitted keep/delete/directional and other inexpensive deterministic alternatives without circuit work.
2. Spend a reserved, bounded selection/verification allowance seeking an incumbent. No feasible elementary repair is promised when hard obligations or locks forbid one.
3. Generate complex candidates in small committed object/family batches. At a declared boundary, freeze the resulting inventory and objective and run the same MaxSAT/verifier kernel.
4. Preserve the best **verified** incumbent under the current objective; return it with generation coverage and search gap when the remaining budget cannot support another useful epoch. Otherwise return explicit unresolved/no-incumbent status.

Carry assignments by canonical candidate IDs, not array positions. Reuse a verification certificate only for identical materialised theory, activations, policy, imports and qualified verifier contract. If vocabulary/policy must grow, revalidate before retaining the incumbent as verified. Never carry an upper bound between different inventories/objectives.

Prefer fixed input graph/context and candidate-independent factor definitions within a stage schedule. If expansion changes predictions or eligible interaction pairs, rescore all retained candidates, recompute the incumbent score and restart the optimiser with a new objective identity. Reuse only cuts whose proof premises still validate. Preserve old verified certificates as scoped evidence, not as current optimality certificates.

Reserve time for final verification and cleanup; a generation timeout must not consume the entire case budget or erase committed candidates. Elementary fallback and rich generated repair have distinct provenance. All logical guarantees remain conditional on completed acceptance. `OPTIMAL_IN_POOL` names the realised frozen pool only; later ungenerated candidates remain outside the claim.

**Acceptance:** a deliberately stalled generator still returns an already verified feasible incumbent; invalid elementary plans are never accepted; coherent input remains unchanged; complex-only feasible cases can progress to expansion; index reorder, activation change, new vocabulary and score change do not reuse stale certificates/bounds. Compare one-stage/staged schedules under equal total budgets.

### COR-07 — Share a qualified reasoning path across inference and supervision

**Priority:** P0. **Locations:** `owl.py`, `kernel.py`, `tools/repair/train.py::_assignment_label`, `_verify_intended`, `acquisition.py::RecordingVerifier`.

Replace hard-coded HermiT/Python selection in the training/recording path with the same resolved capability-aware routing contract used by inference. Route on the complete asserted constructs, imports, activated expressions and actual query kinds, then measured cost. Size alone cannot establish backend completeness. Log backend, version, route attempt and obligation coverage. Qualify the installed ELK/HermiT/native routes; do not introduce another ontology representation or a production JVM requirement.

Within an assignment, reuse materialisation and a prepared reasoner session across hard verification and soft teacher queries where its qualified API permits this. The current soft-query batch is already batched; the correction concerns duplicate preparation between batches and assignments with identical theory. Share only immutable cached results across workers. A session receiving changes must have qualified update semantics or be rebuilt.

Stop a feasibility-only call after one decisive sound violation when additional labels are not requested. Persist that violation and mark remaining obligations unchecked. Do not mark its undecided semantic value as zero. Collect additional obligations only under an explicit training/explanation allowance. For a feasible label, all required hard obligations still need complete results. A failed required entailment is not automatically a monotone support cut; preserve exact-assignment fallback where appropriate.

Use the proof-producing incomplete detector for early rejection. Cache proof supports and completed query results with exact theory/policy/query/backend identities. A changed axiom, source exception, query or activation invalidates incompatible reuse. Approximate modules/projections cannot authorise full-ontology repair unless preservation of every checked obligation is established.

**Acceptance:** differential checks against the existing qualified complete route on the supported small corpus; mixed supported/unsupported constructs; consistency versus coherence; hard required/prohibited entailments; activation satisfiability; source exceptions; unknowns and early rejection. Cached and cold labels agree. Completed negative evidence survives a later timeout without fabricating unchecked obligations.

### COR-08 — Make training preparation productive under partial outcomes

**Priority:** P0. **Locations:** `train.py::label_case`, intended-parent checks, `learning.py`, acquisition scheduler.

Separate generated-parent validity, assignment feasibility and semantic-query completeness. The current intended-parent gate couples feasibility and every semantic query. A verified valid generated parent with some unresolved soft queries can still produce feasibility/risk supervision. An unverified intended parent remains unqualified; do not certify it merely because the generator intended it to be coherent.

Keep desired/unwanted probes and their non-vacuity obligations frozen independently of proposed repairs. Train scalar semantic benefit/ranking only when the declared target is justified; never normalise over the subset of queries that happened to finish. Existing per-obligation masks may support separately declared auxiliary losses. Infeasible assignments provide decided risk labels, not invented negative semantic values. All unknowns retain their masks.

Use exhaustive teachers only where the actual finite product and query costs fit. Else acquire unique, verified sampled repairs containing newly generated candidates. Preserve sampler identity, selection probability where known, stratum and acquisition cost. The resulting ranking/sample-conditioned likelihood is not the exact distribution over unobserved repairs. Deduplicate repeated observations of the same complete assignment while retaining acquisition origins and costs. Reuse theory/query results for identical materialisations, but retain distinct assignment identities, edit costs and probability mass: two assignments producing the same ontology need not be the same repair decision.

Allocate acquisition by missing family/parent coverage and marginal useful labels per time, using TRAIN-only evidence. Prioritise distinct feasible semantic contrasts and decided infeasible plans; stop repeatedly collecting duplicate or zero-information variants while other strata lack labels. Record attempted, verified, semantically usable and novel counts separately. A fixed-case timeout is not retried indefinitely.

Audit positive proposal mass outside observable grammar support. Repair encoding/canonicalisation discrepancies and legal action omissions first. Do not insert hidden clean/reference candidates into deployment menus to force coverage. Genuine retrieval misses remain misses; mask incompatible likelihood targets and report their mass. TRAIN-only labels may supervise retrieval improvements without becoming predictor features.

**Acceptance:** valid parent plus one unknown soft query retains justified risk labels; unknown parent is not qualified; v3 partial-cache losses operate with correct masks; new candidates receive their own checked labels; a changed target rejects scalar-label reuse; duplicate samples do not inflate usable counts. No development/test labels enter optimizer updates.

### COR-09 — Train and measure the heads for their distinct purposes

**Priority:** P0. **Locations:** shared pair selection, `interaction_loss`, `risk_loss`, proposal losses and training reports.

Acquire counterfactual quartets for eligible model interaction pairs with common remaining assignments. Verify all four plans before assigning a semantic mixed-difference target. Include complementary, redundant and opposing semantic effects on multiple independent parents, plus zero-effect controls. Retain zero contrasts for calibration but do not count them as informative nonzero interaction coverage. No pair factor means no claim that its quartets trained that factor.

Whole-plan infeasibility and validated support/obligation outcomes supervise risk. A multi-mapping conflict does not give every contained pair a negative semantic label. Cases with genuinely higher-order dependencies should be reported as limits of pairwise utility rather than solved by arbitrary pair penalties.

Report per head: eligible labels, masks, independent parent count, target variance, gradient/update counts, loss, proposal reachability/novelty, positive/negative/zero quartet contrasts, held-out contrast error, risk discrimination/calibration and a constant-risk baseline fitted on TRAIN. Evaluate risk ordering on/off with the same candidate pool, integer score, shortlist, initial proof set and cut-generation/validation rules. Subsequent proved cuts may differ because verification order differs; do not leak one arm's future proofs into the other. Calibrate using DEV only. Checkpoint selection still serves complete repair performance, not whichever auxiliary metric looks best.

**Acceptance:** tiny controlled cases show nonzero intended gradients and parameter changes; interaction targets equal independently computed finite differences; infeasible cases never enter benefit quartets; risk ordering changes verification order but neither objective nor reachable assignments. The final report states when a head has insufficient evidence.

### COR-10 — Preserve development denominators and real optimizer state

**Priority:** P0. **Locations:** `train.py` development construction around line 2773 in the campaign revision, `generated_checkpoint_criterion`, optimizer/acquisition checkpoints.

Construct DEV evaluation from the frozen case schedule, independently of whether a cache exists. Every scheduled case produces a report; missing caches, generation failures, unknown verification and unavailable semantic targets remain in the denominator. The current entry point filters missing caches before computing coverage; fix this before expanded fitting. Reject any evaluation whose case-ID set differs from its manifest.

Retain the declared coverage-aware checkpoint rule, grouped by structural parent, with fixed tie-breaking and an explicit unavailable criterion. If no checkpoint qualifies, record that outcome; a predeclared exploratory fallback cannot be renamed primary success. Test metrics never choose checkpoints or stopping times.

Qualify real minibatches and AdamW state on the full schema, not only isolated gradients. Save model, optimizer, scheduler if any, RNG, minibatch position, actual update count, acquisition/model identities, validation phase, best-model state and cumulative budgets atomically. Interrupted validation does not consume patience. Resume processes each committed update/label exactly once. A changed schema or target is a new training run, not an exact resume.

**Acceptance:** remove an entire DEV cache and verify unchanged denominator; all-failed/no-label DEV remains explicit; interrupt before/after updates and during development and compare with uninterrupted deterministic execution; measure VRAM and process-tree memory after optimizer state exists. No assertion of fitting readiness from zero-update probes.

### COR-11 — Correct controls, measurement scope and real-data admission

**Priority:** P0. **Locations:** study schedules/reporters, [16](16-liseda05-experiment-plan.md).

Provide distinct matched generation, fixed-pool scoring, risk-order and end-to-end comparisons. A deletion-only baseline directly constructs its elementary pool and does not pay rich-generation costs. Retain deletion-after-rich-generation only as a separately labelled action-language ablation. Strong symbolic scoring uses declared observable ontology/matcher evidence and query targets available at inference; a hidden clean-parent oracle is an upper reference, not a deployable baseline.

A local support-retention surrogate and a budgeted exact observable-query search are separate symbolic controls. For the latter, do not insert non-additive entailment credit into independent MaxSAT weights and call it exact. On tractable shared pools, evaluate complete plans externally; on larger pools use bounded search with explicitly justified bounds or report the best checked result without an optimality claim. Charge all scoring reasoner work.

Record clean/corrupt, family, independent parent, theory scope, initial status, source revision and hardware strata. Low runtime caused by skipped inputs is not an efficiency result. Report coverage and conditional quality together; risk efficiency is measured to first verified repair and matched external quality, separately from proving optimality. Include startup/generation/teacher/query work in end-to-end and total learning-cost accounts, with shared work charged once.

The prior fresh/robustness/projection cases are now exposed development/regression material. Preserve genuinely unopened held-out releases. Use ontology-level disjoint Conference splits where specified; disclose pair-level overlap if a separate protocol permits it. Qualify original ontology/import support and actual matcher alignments. Synthetic corruption of real ontologies is a separate condition. Do not filter coherent or difficult test pairs after seeing outcomes.

**Acceptance:** the frozen expected row set exactly matches the report, including missing outcomes; each matched comparison has identical case/policy/action/budget identities except its declared factor; intervention no-ops are counted; native deletion works when rich generation deliberately stalls; real release hashes, provenance and split records are present.

### COR-12 — Make LLM semantic supervision a primary, correctly attached target

**Priority:** P0 for the primary combined-supervision condition. **Locations:** `tools/repair/train.py::case_loss`, semantic annotation/evaluation adapters, existing OpenRouter router/ledger, [13](13-semantic-fidelity-supervision.md) and [16](16-liseda05-experiment-plan.md).

The current mixed-benefit path blends symbolic and weak scalars only when both exist, masking usable symbolic labels when weak values are absent. Its explicit comparison/tie loss is restricted to the LLM-only mode. Do not treat that flag as the required combined programme. Version the loss contract and checkpoint configuration; historical mixed runs retain their original semantics.

Use six primary HGT-pair fits: symbolic versus symbolic-plus-LLM supervision at the same three seeds. Match initial weights, common input/candidate/plan support, symbolic labels, optimizer settings and update opportunities within each seed. All observed textual/evidence features are identical across conditions. The combined condition adds separately masked auxiliary semantic-fidelity losses:

`L_combined = L_symbolic + alpha * L_weak_anchor + beta * L_weak_comparison`.

`L_symbolic` includes the existing qualified symbolic benefit, proposal, interaction and risk terms with their own masks. Normalize each source/task term by its declared eligible counts, not by the union/intersection of all labels; a term with no eligible labels contributes zero and records zero coverage. Missing/invalid/abstained LLM labels must not remove a usable symbolic training signal. Weak labels are never copied into missing symbolic fields. Anchored whole-plan ratings and strict/tied comparisons follow SF-010/011/019. The auxiliary losses backpropagate through the existing benefit factors and shared encoder, without new semantic authority or per-action pseudo-labels. This is a calibrated multi-target objective, not a claim that both teachers measure identical ground truth. Freeze source scale calibration, loss weights and reduction rules from TRAIN/DEV before test; record target disagreement and per-source losses. Scalar costs remain fixed and absent from judge prompts.

Primary proposal and risk targets remain the existing symbolic targets in both conditions; LLM evidence enters the semantic benefit loss only. Shared encoder updates may indirectly change proposal/risk predictions, which is part of E1 and is separated by the fixed-pool E2 comparison. No uncalibrated ordinal preference becomes a cardinal interaction target: a weak quartet requires all four feasible plans and complete, same-basis anchored values. Infeasible plans remain useful symbolic risk examples even when semantic annotation is ineligible.

Attach every annotation to its exact case, complete assignment, materialised theory/activations, policy, evidence, query/rubric, model and parser identities. Only verified feasible pairs with complete required semantic evidence enter global fidelity supervision. Reuse verified sampled repairs, including new candidates; do not transfer a label from a similar expression or repair. Keep TRAIN, DEV selection and held-out TEST judge namespaces and roles separate. Version DEV admission explicitly: existing weak-label validation paths can require an evaluator different from the teacher. The primary protocol may reuse the TRAIN teacher family for DEV selection only under a declared `development_selection` use, split-specific evidence/labels and no independent-assessment claim. Preserve legacy independent-DEV modes; do not falsify an independence flag or bypass a guard globally. Final TEST admission still rejects every judge configuration used for TRAIN or DEV selection. Bind the use policy to label/cache/checkpoint identities. A completed response is revalidated locally; a new plan needs a new eligible annotation, not interpolation from the offline label inventory. Generate/verify DEV/TEST outputs first, annotate committed batches within the reserved stage budget, then attach the exact labels. Training never reads test judgements or final-assessor feedback.

For repeated ratings of the same plan, replace exact score-equality admission with a frozen aggregation rule. Group only identical complete plans under the same plan-relevant evidence, policy, applicability, rubric and calibrated judge basis; counterpart/order changes alone need not create a new scalar target. Retain every raw rating and comparison-specific preference. Use the declared per-criterion median of eligible complete ratings as the starting scalar convention, followed by the frozen criterion weights; freeze minimum coverage and disagreement-to-abstention thresholds during calibration. Incompatible evidence contexts remain separate and cannot be averaged. Version and commit each aggregate before training/selection consumes it; later annotations cannot mutate an active target. Do not select the most favourable score or mistake ordinary valid rater variation for corrupted identity.

Reuse the existing OpenRouter runtime and ledger as in SF-037–042; no new annotation service is required. Treat definitions, evidence and model rationales as data. Enforce grounded citations, strict schema validation, abstention, blinding, order-swap audits, finite request/token/spend limits and safe replay. Purely structural A/B/C examples do not become semantic examples through invented descriptions. Teacher/DEV profiles cannot double as the claimed independent TEST evaluator, including provider fallback. LLMs supply a fallible proxy for intended meaning; symbolic verification retains final feasibility authority and MaxSAT retains plan selection. No online LLM is introduced into repair inference.

**Acceptance:** mixed-loss fixtures show that removing an LLM record leaves the same symbolic contribution/gradient; a valid weak strict preference, tie and anchored rating each reach the intended parameters, with source counts and weights independently recomputed. Invalid/abstained weak labels add no fidelity gradient, do not erase risk labels, and cannot create cuts. Both paired arms consume identical symbolic IDs/features/update schedules and fixed costs. Mock post-decode annotation attaches a previously unseen exact plan, rejects a nearby-plan cache hit, and preserves complete DEV denominators; missing labels cannot silently choose a checkpoint through a favourable denominator. A same-family DEV selection fixture is accepted only with the new declared use policy and cannot be relabelled independent. A held-out evaluator used in DEV selection or reached through teacher fallback invalidates independent-evaluation admission. Verify known repeated-rating aggregates, disagreement abstention and rejection of incompatible contexts; keep pairwise targets comparison-specific. Exercise ledger/schema/replay fixtures from document 13, plus actual backward/optimizer/checkpoint checks. Live labels and semantic performance require their own separately reported evidence. No paid calls are performed by the specification task.

## 4. Implementation order and completion record

Run independent engineering work concurrently, but respect these dependencies:

1. **Integrate and instrument:** COR-01, COR-03 and COR-10; establish clean source and trustworthy records.
2. **Restore coverage:** COR-02, COR-05 and COR-07; complete action/schema/native regression checks.
3. **Remove repeated work:** COR-04 and shared reasoning/session work in COR-07; retain differential correctness evidence.
4. **Make bounded progress:** COR-06; compare against the existing one-stage execution path.
5. **Train productively:** COR-08, COR-09 and COR-12; perform actual optimizer/resume qualification, then collect TRAIN-only shards.
6. **Freeze and execute:** COR-11 and the dated resource plan, after its automated correctness/resource checks pass.

Every change should be a small reviewable implementation commit with focused tests; avoid infrastructure rewrites unrelated to a measured bottleneck. Do not wait for optional architecture or Bio-ML breadth before starting qualified primary work.

Publish one conformance record with entries for COR-01 through COR-12: source/test/backend identities; commands; passed/failed/skipped results; measured performance; known limitations; compatibility migrations; and affected study arms. P0 logical/schema/lifecycle failures block only dependent arms. Throughput failures trigger the predeclared scope reduction in document 16, never weaker logical acceptance or silently larger budgets. Record actual methodological amendments separately from imported bug fixes.

The specification commit itself closes no implementation requirement and starts no jobs. The next implementation agent should implement these requirements, validate them and prepare the resolved campaign artifacts. A later run instruction can execute those artifacts without asking the user to approve routine engineering choices already fixed here.
