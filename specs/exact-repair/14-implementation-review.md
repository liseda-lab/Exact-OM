# XR-2.1 implementation review and corrective specification

**Review date:** 30 September 2026. **Reviewed implementation:** `6f6a3e0038dbd5898126a34b28dee1faa06ef2f3` (`feat(repair): implement XR-2.1 generation, learning and verified search`). **Design baseline:** specification commit `d918cf623a93a3fa649920bec0a12fe09ef246e5`. **Status:** open corrective requirements; this document does not implement or certify their fixes.

The implementation follows the main research direction, but several connections between generation, graph construction, training, persistence and verification are incomplete or inconsistent. These defects must be addressed before relying on the affected experimental results. They do not justify replacing the circuit generator, HGT, MaxSAT or symbolic verification with a different architecture.

This review supplements [12 Implementation migration](12-implementation-migration.md), whose findings describe the earlier XR-2 implementation. The owning methodology contracts in [the suite index](README.md) still define the method. The `REV-*` requirements below define the corrective work for the reviewed XR-2.1 implementation. Historical evidence remains historical; an implementation commit or an existing conformance receipt does not close a new finding automatically.

## 1. Scope, evidence and preserved behavior

### 1.1 What is already implemented

Preserve the following working architecture while fixing the defects:

- Complete replacement bundles over mappings and explicitly eligible ontology-axiom occurrences.
- Family-factored probabilistic circuits with grammar constraints and restricted, proof-supported immutable-context constraints. These are not global coherence certificates.
- Generation of new candidate pools, symbolic labeling of sampled complete assignments, and inclusion of those examples in the optimizer's training order. The new acquisition path is not merely writing unused caches.
- Unary and pairwise semantic-benefit factors, explicit edit costs, feasible counterfactual interaction losses, and a separate whole-plan risk readout.
- MaxSAT construction of a complete repair plan; learned risk orders a bounded verification shortlist rather than changing the objective or authorizing logical exclusions.
- Qualified proof-support cuts, complete public-signature checking, and conservative treatment of unknown and deferred assignments in bounds.
- Offline semantic-fidelity annotation through the existing OpenRouter routing and request ledger, with distinct teacher/evaluator roles and grounded evidence checks.

No detector false-positive or incorrect shortlist optimality claim was demonstrated by this bounded review. That is a statement about the evidence obtained, not a proof that every implementation path is correct. Reinforcement learning remains outside the main method.

### 1.2 Evidence categories

Each finding records one of these evidence levels:

1. **Reproduced:** a focused execution demonstrates the behavior. Where the execution isolates a component, the limitation is stated.
2. **Source-confirmed:** the relevant control flow establishes the defect, but no complete end-to-end reproduction was obtained.
3. **Implementation gap:** a specified capability or measurement is absent or disconnected.
4. **Concurrency concern:** an unsafe interleaving is visible in the code, but the race has not been reproduced. Add a deterministic interleaving test before claiming a reproduced failure.

Source locations refer to the reviewed commit. Module and function names are authoritative if lines subsequently move. The fixtures below are self-contained requirements; implementation must not depend on a reviewer's temporary environment or private reproduction scripts.

### 1.3 Validation observed during the review

- The checked-in [validation record](implementation/xr21-validation.json) reports **330 passing tests, zero failures and zero skips** on Linux, Python 3.12.3 and Torch 2.7.0. All 39 listed source-file digests matched the reviewed checkout. This is existing recorded evidence, not a new execution by this review.
- A separate local run passed **65 tests** from `repair_semantic_fidelity_test.py`, `repair_protocol_v3_test.py` and `openrouter_ledger_test.py`.
- The broader macOS rerun used a different environment, including Torch 2.2.2. It did not finish, showed unresolved failures, and was stopped. It is not a conformance pass, and those incomplete results do not establish that the recorded Linux result is false. CPU/RSS worker supervision also has explicit Linux-specific requirements.
- Focused reproductions below exposed cases absent from the passing tests. Some used an isolated control-flow harness or an encoder with no message passing; those do not stand in for full HGT or OWL integration tests.
- No new scientific experiment, biomedical-scale benchmark, live LLM annotation or human semantic evaluation was performed.

For this documentation change, validate references, requirement coverage and the diff. Runtime regression tests below belong to the subsequent implementation changes; they are not claimed to have passed already.

## 2. Findings and priority

`P1` means a blocker for ordinary training or the intended large-ontology verification path. `P2` means a defect or capability gap in a specified configuration or study arm. `P3` means additional operational/reporting work required for the corresponding claim. Priorities do not change the logical acceptance policy.

| ID | Priority | Finding | Evidence | Main owner |
|---|---|---|---|---|
| REV-01 | P1 | A fixed event-count ceiling prevents complete large-signature verification | Reproduced event failure; source-confirmed verification consequence | `workers.py`, `kernel.py` |
| REV-02 | P2 | Synchronous event processing can overrun the advertised deadline | Reproduced callback failure; production path inspected | `workers.py`, `kernel.py` |
| REV-03 | P1 | Risk reads supports excluded from the graph; disabled risk is still evaluated | Reproduced readout failure; source-confirmed call paths | `graph.py`, `model.py`, `pipeline.py`, `train.py` |
| REV-04 | P2 | Acquisition omits declared graph/context limits | Source-confirmed | `train.py`, shared preparation |
| REV-05 | P1 | Acquired and generated-development labels lose configured semantic weights | Source-confirmed calls; reproduced teacher arithmetic | `train.py`, `learning.py`, protocol identities |
| REV-06 | P2 | Sampling strata are starved and sampler identities omit settings | Reproduced collector behavior | `learning.py`, `train.py`, protocol |
| REV-07 | P2 | Contextual filtering can remove an entire protected family representative | Reproduced | `grammar.py`, `pipeline.py` |
| REV-08 | P2 | Progressive stages retain candidates excluded by their declared language | Reproduced with a minimal scoring model | `pipeline.py`, protocol validation |
| REV-09 | P2 | Deadline checkpoints skip unfinished minibatches on resume | Reproduced isolated epoch control flow | `train.py`, checkpointing |
| REV-10 | P2 | Standalone resume loses validated source-exception evidence | Source-confirmed | `records.py`, `kernel.py` |
| REV-11 | P2 | Replayed LLM judgments can count as independent quorum votes | Reproduced | `semantic_fidelity.py` |
| REV-12 | P2 | Support-level supervision is not connected | Implementation gap | `model.py`, training labels/losses |
| REV-13 | P2 | Concurrent cache eviction can interrupt successful compilation | Concurrency concern, not reproduced | `compilation.py` |
| REV-14 | P3 | Compiler reports do not expose all required phase/cold-cache measurements | Implementation gap | compilation reports and study export |

REV-01 through REV-12 cover the review's principal correctness and methodology findings. REV-13 and REV-14 retain supplementary audit findings with their weaker or narrower evidence clearly identified.

## 3. Required fixes and acceptance cases

### REV-01 — Complete large verification without a fixed total-event ceiling

**Evidence and impact.** [workers.py](../../exact/repair/workers.py), `bounded_call`, lines 269–272 rejects an event once 10,000 events have been retained. [kernel.py](../../exact/repair/kernel.py), `verify_theory`, lines 136–156 emits an event per completed verification obligation. Consistency plus 10,000 non-exempt monitored classes already exceeds the ceiling; required/prohibited queries, active expressions and backend attempts can add more events. The result is an operational failure or unknown, not false acceptance.

A worker emitting 10,001 valid small events and then returning was stopped with `status="error"`, `detail="invalid or excessive worker events"`, and 10,000 retained events. This establishes the transport defect; it is not a measured Bio-ML verification run.

**Required change.** Remove the fixed total-obligation obstruction while preserving bounded memory, frame sizes and backpressure. Use bounded batches or incremental durable storage with an explicit coverage index. A capacity derived from the frozen obligation set may be part of validation, but simply replacing 10,000 by a larger constant is insufficient. Bound duplicate/fallback attempts independently of distinct obligation coverage. Transport the final coverage/report through a bounded representation as well; do not move the same failure into a final oversized message.

Successful verification still requires validated coverage of every expected obligation. Preserve assignment, theory, policy, obligation and backend qualification identities. Repeated events cannot manufacture missing coverage. If a proof and a positive report disagree, do not accept the plan; preserve the sound failure and record the integrity discrepancy.

**Acceptance.** `REV-01-T1`: transport 10,001 valid events and complete without excessive memory. `T2`: exercise the real verification producer with more than 10,000 obligations and a controlled complete adapter; independently test native-adapter coverage on a small ontology. `T3`: missing, duplicate, malformed, out-of-order and conflicting events cannot manufacture feasibility. `T4`: a completed acknowledged failure survives a later hang or broken frame. Record peak buffering and final-report transport, not only successful status.

**Contract:** RE-15, RE-20; K-12, K-13. **Reuse:** independently valid completed proofs remain valid; partial positive streams must not be promoted. Record a new transport/coverage implementation identity.

### REV-02 — Keep the deadline supervisor responsive during event handling

**Evidence and impact.** `workers.py:269–274` invokes the event handler synchronously. The production handler in `kernel.py:675–696,746–752` reconstructs/hashes theory dependencies, validates events and persists the ledger through `kernel.py:438–449`. These operations run outside the killable worker execution. A one-second call with a synthetic handler that sleeps for 2.2 seconds returned after 2.438 seconds, exceeding the documented 0.2-second cleanup allowance. Actual slow-storage behavior was not measured.

**Required change.** Keep potentially blocking event validation, proof checking, serialization and persistence under an independently responsive supervisor. Reuse frozen theory/query indexes where safe instead of rebuilding them for every event. Use the existing worker/ledger infrastructure; a new generic execution framework is unnecessary. A thread alone is not a termination guarantee for blocked native or filesystem operations.

Acknowledgment means the event has been validated and durably committed. If this cannot finish within budget, do not acknowledge it or trust a partial record. Return the last committed evidence plus an explicit operational status. Charge event processing, storage and cleanup to the declared stage/total budgets. Preserve the distinction between sampled resource monitoring and OS-enforced limits.

**Acceptance.** `REV-02-T1`: stall validation and separately stall persistence; the controlling call returns within the stated deadline plus cleanup allowance. `T2`: CPU/RSS monitoring remains active while a callback is blocked. `T3`: kill between validation and durable commit; replay accepts only committed records. `T4`: previously acknowledged failures and incumbents survive. Use controllable barriers and injected storage delays, with a documented platform allowance rather than a brittle microsecond performance assertion.

**Contract:** RE-15, RE-20; K-12, K-13. **Reuse:** retain valid durable evidence; identify and exclude incomplete transactions. This requirement does not permit weakening evidence validation to save time.

### REV-03 — Make risk consume exactly the supports admitted to the graph

**Evidence and impact.** [graph.py](../../exact/repair/graph.py), lines 565–576 omits explanations when a limit is reached. [train.py](../../tools/repair/train.py):1023 and `pipeline.py:1034` still pass all retrieved explanations to [model.py](../../exact/repair/model.py), `plan_risk_logit`, lines 455–463. Encoding a support-only class absent from memory raises `ValueError("Candidate symbol was not retrieved into observable graph memory")`.

The reproduction used three revision objects, an explanation containing one otherwise absent class, and `max_explanations=0`. Risk without that support succeeded; risk with the omitted support raised. Graph construction and the PyTorch readout were real; unused PyG convolution imports were replaced for an `encoder="none", layers=0` fixture. Training calls the failing path directly. Inference catches a risk-ordering failure and falls back, so this is not evidence of unsound repair authorization. If all symbols happen to remain, excluded explanations can still affect the score despite being reported omitted.

Additionally, `train.py:1017–1032` computes risk logits before checking `plan_risk`; disabling the loss therefore does not avoid this failure.

**Required change.** Derive one canonical admitted-support set from the completed bounded graph. Pass exactly that set to training risk, inference risk and frozen replay. Bind the full admitted records and admission-policy version into the risk identity. Retain omission IDs/reasons for reporting, separately from neural inputs. Validate that every structure read by an admitted support is encodable. Do not reintroduce excluded supports through an unbudgeted side channel or silently expand graph limits.

When risk is disabled, do not invoke its readout or loss. A missing/failed risk estimate may use the declared utility-order fallback; it must not change objective coefficients, feasible-set membership or hard clauses.

**Acceptance.** `REV-03-T1`: both explanation-cap and node/edge-budget truncation with a support-only entity succeed through training and frozen inference. `T2`: with the graph, admitted-support set and effective omission features fixed, changing only excluded records outside that projection leaves the risk score unchanged; an admitted support remains observable. Rebuilding the graph may legitimately change admission and is a different comparison. `T3`: a risk stub that raises if called is never invoked when risk is disabled. `T4`: serialization/replay preserves admitted identities and rejects incompatible memory. `T5`: imperfect or unavailable risk never removes an unverified alternative from the upper bound.

**Contract:** NN-010, NN-022, NN-023, NN-026; TR-027. **Reuse:** invalidate affected risk descriptors/scores. Unchanged, independently qualified feasibility proofs and primary objective factors need not be discarded solely because scheduling inputs changed.

### REV-04 — Use the same effective preparation settings on every path

**Evidence and impact.** Active acquisition's `_freeze_training_model` call in `train.py:1229–1262` omits the configured graph node, edge, explanation and text limits. Initial training, reconstructed sampled graphs and generated development pass those limits. Nondefault configurations therefore collect examples under different contexts from those declared for training/deployment.

**Required change.** Centralize the effective preparation configuration used by training, acquisition, generated development and inference. Reuse existing configuration/record types where possible. Forward and record all graph/text, retrieval and interaction limits explicitly, including defaults after resolution. Do not infer the effective configuration from an enclosing manifest that the invocation did not actually consume.

Compare equivalent stages: generation may legitimately change the inventory, so an initial graph and a later expanded-pool graph are not required to have the same hash. Given the same observable input, final inventory and effective settings, all paths must construct the same admitted support set, graph and ordered interaction set.

**Acceptance.** `REV-04-T1`: deliberately nondefault node, edge, explanation and text limits reach every entry point. `T2`: equivalent frozen requests produce matching graph/support/pair identities and omission records. `T3`: changing one limit invalidates affected prepared artifacts. Include acquisition and resumed acquisition, not only direct model tests.

**Contract:** TR-027; NN-010, NN-017, NN-026. **Reuse:** regenerate affected graph, pair and risk artifacts and re-evaluate collection claims. Logical/semantic observations can be reused only under their own matching theory/policy/query dependencies.

### REV-05 — Preserve one declared semantic target across labeling paths

**Evidence and impact.** `protocol.py:542–543` resolves configured desired/unwanted family weights; initial teacher preparation passes them at `train.py:2101–2102`. Active labels at `train.py:1288–1294` and generated-development labels at `train.py:462–468` omit them, taking `_assignment_label` defaults of 1.0. Acquired-cache identities at `train.py:1335–1344` also omit the teacher weights.

The teacher computes a weighted sum of within-family means, not a normalization that cancels these weights. With one fully preserved desired family, a declared weight of 2 gives benefit 2; the defaulted call gives benefit 1. The same repaired theory can thus enter training twice on different scales, and checkpoint selection can use a different objective from its manifest. All-1 configurations hide the defect. Edit costs remain unchanged, making the tradeoff inconsistent as well.

**Required change.** Resolve an explicit immutable semantic-target specification, containing the query basis, desired/unwanted family weights, aggregation convention and version. Pass it through original teachers, active acquisition, generated development and resumed collection. Reuse the same target evaluator. Bind it into scalar-label, training and checkpoint-selection identities and reject mismatches. If different semantic tasks are intentionally trained together, represent their task conditioning and declared combination explicitly; do not mix them through default arguments.

Keep semantic benefit, edit cost and feasibility separate. Correct propagation must not subtract costs inside benefit or convert unknown query outcomes into decided values.

**Acceptance.** `REV-05-T1`: use distinct nonunit desired and unwanted weights and a feasible theory with both types of outcome; all labeling paths return the same vector, masks and scalar. `T2`: direct benefit aggregation matches the declared formula. `T3`: changing either weight changes scalar-label and selection identities. `T4`: costs are unchanged and subtracted once. `T5`: generated development ranks checkpoints using the configured target, not the defaults.

**Contract:** TR-001, TR-008, TR-011, TR-017, TR-025, TR-028. **Reuse:** complete retained query outcomes may be reweighted without another reasoner call if their dependencies still match. Preserve unknown masks. Historical labels must retain their actual default-weight semantics; do not rewrite old results. Models trained on mixed scales require renewed training/selection qualification rather than silent continuation.

### REV-06 — Allocate collection quotas explicitly and retain every attempt's origin

**Evidence and impact.** [learning.py](../../exact/repair/learning.py), `collect_sampled_repairs`, lines 721–749 concatenates uniform rows, proposed MaxSAT rows, quartets and counterfactuals, then truncates globally. With 128 attempts, exploration fraction 0.5, 32 utility attempts, 32 quartet rows and 32 diversity attempts, the first three groups consume the entire budget. A reproduction with eight four-choice objects retained 64 uniform rows, 32 MaxSAT rows and 13 unique quartet rows, with 19 duplicate attempts and no counterfactual rows.

`proposal_attempts` is used to cap per-object candidate draws at `train.py:1235–1240`, rather than implementing the declared complete-plan proposal stratum. `generator_fraction` has no training consumer. The sampler hash at `learning.py:795` omits exploration/quartet/diversity settings; changing the exploration fraction changed the sample but not that hash. Deduplication also discards duplicate origins rather than preserving them.

**Required change.** Resolve an explicit finite attempt schedule before verification. Distinguish complete-plan sampling quotas from candidate draws per object. Implement utility-driven, proposal-driven, diverse/counterfactual and quartet strata with stated meanings. Declare whether a quartet budget counts four-assignment groups or assignment attempts, and normalize it once.

Existing fractions and quotas must not compete silently. Prefer explicit complete-plan stratum quotas as the authoritative schedule. A legacy fraction can be translated into a resolved allocation by a versioned adapter; incompatible simultaneous declarations must be rejected. Uniform controls, if requested, need their own declared share of the same finite budget, not an extra prepended block. With four declared strata of 32 attempts, the resolved schedule totals 128 and cannot starve the fourth because of list order. Interleave scheduled strata deterministically if useful so a deadline does not systematically favor the first one; record any unvisited suffix.

Record requested, scheduled, attempted, duplicated, verified, unknown, unavailable and unvisited counts per stratum. Deduplicate complete assignments by inventory-bound identity for labeling, while retaining all attempt origins, order and available sampling probabilities. A duplicate may consume a scheduled attempt without another verification; it must not erase its origin. No retry-until-feasible or replacement of unknown outcomes to improve apparent coverage. Preserve complete quartet grouping and explicitly mask an interrupted/incomplete quartet.

Hash all effective sampler settings, resolved quotas, ordering/deduplication rules, seed, frozen model/inventory and algorithm version. A per-draw probability is not the inclusion probability after deduplication; leave unavailable propensities unknown and do not invent importance weights.

**Acceptance.** `REV-06-T1`: four-by-32 declared quotas receive exactly those scheduled attempts; an extra uniform fraction is resolved explicitly or rejected. `T2`: changing a valid effective exploration, quartet or diversity setting changes sampler identity; contradictory legacy declarations may instead be rejected. `T3`: repeated assignment proposals produce one qualified label and all origins. `T4`: unavailable proposal mass, exhausted finite pools and deadlines produce honest denominators. `T5`: a fixed seed/configuration reproduces the schedule, including resume. `T6`: three-way infeasible plans remain plan-risk examples and do not become three invented pairwise semantic negatives.

**Contract:** TR-014, TR-015, TR-016, TR-019, TR-023; sampled-record provenance in [01](01-architecture-and-contracts.md). **Reuse:** independently valid assignment labels may be reused by their semantic dependencies. Invalidate old sampler-compliance claims and derived selection weights; missing historical origins cannot be reconstructed as facts.

### REV-07 — Find an admissible representative before closing an action family

**Evidence and impact.** `grammar.py:154–161` accepts the first syntactically bounded representative and breaks before considering contextual bans. `pipeline.py:746–752` filters that candidate later without trying another one.

For a mapping `S ⊑ T`, a source-expression menu containing `A` and `B`, and immutable `Disjoint(A,S)`, the first representative can be `S ⊓ A ⊑ T`. That active expression is forbidden by the contextual constraint. `S ⊓ B ⊑ T` is still admissible, but zero-draw controls in the reproduction contained neither a subclass-expression-specialisation nor a composite-family representative. Random sampling may happen to recover one; that does not satisfy protected family coverage.

**Required change.** Search for a representative satisfying the same bounded grammar and contextual constraints as ordinary generation before marking the family represented. Prefer querying a satisfying assignment from the already compiled family circuit, or a bounded deterministic traversal that tests full admissibility. Do not enumerate an unbounded grammar. Distinguish proved empty support from exhausted representative-search/compilation resources. Preserve elementary alternatives and report any family that could not be protected.

Canonical bundle deduplication may attach multiple family origins to one candidate. It must preserve family coverage provenance and must not distort circuit probability or count the same plan as several independent alternatives. Check that the final candidate cap and mandatory-preservation policy actually retain the chosen representatives; do not silently exceed the cap or claim coverage after eviction.

**Acceptance.** `REV-07-T1`: the disjointness fixture above retains the admissible `B` representative with zero random draws. `T2`: making every expression inadmissible produces an explicit empty-family result. `T3`: resource exhaustion is reported differently from logical emptiness. `T4`: an editable disjointness support does not become a permanent immutable ban. `T5`: final pool caps and deduplication preserve all feasible protected coverage or report its precise limitation.

**Contract:** CG-010, CG-013, CG-015, CG-023, CG-025, CG-039; MIG-06. **Reuse:** regenerate affected pools and their scores; do not compare corrected coverage against old rows under an unchanged generation identity.

### REV-08 — Make progressive language declarations agree with preserved candidates

**Evidence and impact.** `pipeline.py:780–788` reinserts prior candidates as mandatory. `freeze_progressive_rounds`, lines 1234–1239 validates only depth, constructor count, candidate cap and draw count. It does not validate that action menus, omitted symbols or contextual constraints are compatible with preservation.

A two-stage reproduction generated endpoint replacement `B ≡ C` in stage 0. Stage 1 declared only keep/delete and omitted `C`, but still retained `B ≡ C` while reporting the narrower language. The reproduction used real retrieval, grammar and freezing with a minimal zero-valued scoring model. This compromises missing-symbol/action ablations even if final verification remains sound.

**Required change.** Treat a progressive schedule as nested expansion of one declared language. Validate semantic compatibility before starting it: effective action families, typed vocabulary, omission controls, grammar bounds, immutable-context constraints and candidate filters must permit each preserved bundle. Prefer rejecting a non-nested schedule; if narrowing is a required experiment, make it an explicit new generation epoch with validated filtering and no inherited objective bounds. A changed policy or background identity is not the same progressive epoch.

Apply final-candidate interventions after every producer, including preservation, then verify the final frozen pool against the effective declaration. Report the actual preserved candidates and language identity. Do not repair the discrepancy merely by changing a report to hide the requested intervention.

**Acceptance.** `REV-08-T1`: the endpoint/omitted-`C` schedule is rejected or executed as a separately declared filtered epoch in which `C` is absent. `T2`: nested expansion preserves valid prior candidates. `T3`: changing contextual constraints or action families is checked, not only four numeric bounds. `T4`: final-candidate-removal and missing-symbol controls remain distinct and survive preservation. `T5`: changed inventories/objectives cannot inherit stale optimum/bound certificates.

**Contract:** CG-005, CG-007, CG-023, CG-024, CG-036; MIG-02; K-01. **Reuse:** regenerate the affected pool and re-evaluate omission results. Reuse proof supports only through their qualified semantic identities, never old candidate indices.

### REV-09 — Resume the actual partial training phase

**Evidence and impact.** `train.py:1430–1431` exits the minibatch loop on deadline but continues into development/history and unconditional `checkpoint(epoch + 1)` at line 1592. That overwrites the correct partial checkpoint written at line 1442.

An execution of the actual epoch control flow with an inert optimizer and controlled clock scheduled three cases and expired after one update. It wrote `(epoch=1, offset=1, optimized=1)` and then `(epoch=2, offset=0, optimized=0)`. The latter skips the two unprocessed cases on resume. This was a control-flow reproduction, not a full neural training run. The existing test interrupts after a completed epoch and does not cover this case.

**Required change.** Advance the epoch only when all scheduled minibatches have completed. Persist the exact phase, epoch, order/offset, cumulative loss/update counts, optimizer/model/RNG state, active acquisition state, best-model state and pending development work. If time expires, return through an interruption path that retains that state. An incomplete development evaluation cannot consume patience or be reported as a completed model-selection decision.

Avoid replaying an optimizer update or acquisition label already durably committed. Keep cumulative resource accounting across interruption. Reaching the configured total budget permits a clean resumable state but does not itself authorize extra work beyond that budget.

**Acceptance.** `REV-09-T1`: interrupt after one of three minibatches and resume remaining work exactly once. `T2`: interrupt after the final update but before development, and during development; resume the correct phase. `T3`: with deterministic settings and sufficient predeclared remaining budget, resumed optimizer/model/RNG state, update order and semantic selection results match an uninterrupted run. Timing, resource and interruption history fields must reflect the actual executions rather than be byte-identical. `T4`: changed data/target/preparation identities reject exact resume. `T5`: an exhausted cumulative budget is not replenished by loading the checkpoint.

**Contract:** TR-025, TR-T16; versioned recovery in [01](01-architecture-and-contracts.md). **Reuse:** an old checkpoint that cannot establish its true processed boundary cannot claim exact continuation. Recover an earlier demonstrably valid boundary or declare a separate warm-start run with renewed selection.

### REV-10 — Persist validated source-exception evidence for standalone resume

**Evidence and impact.** [records.py](../../exact/repair/records.py), `SearchLedgerV3`, lines 713–738 stores baseline reports but not `BaselineReportV3.exception_checks`. In `kernel.py:887–925`, standalone `repair(..., resume=True)` with default diagnosis, nonempty exceptions and no separately supplied baseline cannot restore those checks. It sets them to unknown with `"exception budget exhausted"`, then skips baseline collection because this is resume. New candidates become unknown through `_verify_with_exceptions` even when budgets remain.

This is source-confirmed; a complete reproduction was not obtained. The shared-study path can avoid it by supplying its separate baseline artifact again. The defect does not permit accepting invalid exceptions.

**Required change.** Store the complete immutable baseline/exception artifact, or a resolvable integrity-checked reference, in durable search state. Bind original-source consistency, each exception's completed class-unsatisfiability evidence, source/import closure, policy/query and backend qualification identities. Validate and reuse that artifact on resume. Missing or incompatible evidence requires explicitly budgeted revalidation or unknown with the real cause; it must not be mislabeled budget exhaustion merely because diagnosis was skipped.

**Acceptance.** `REV-10-T1`: stop after source-exception acquisition, resume with default standalone arguments and no external baseline, then verify another candidate and obtain a new incumbent. `T2`: changed source, policy, exception query or qualification identity prevents reuse. `T3`: missing/corrupt evidence never removes an obligation silently. `T4`: shared-study and standalone paths agree for the same qualified baseline. `T5`: spent budgets and pending/deferred bounds survive unchanged.

**Contract:** RE-06; K-01, K-13. **Reuse:** matching complete source proofs can be retained. A legacy ledger lacking evidence requires the explicit recovery path; do not fill its missing fields with unverified assumptions.

### REV-11 — Count unique annotation attempts, not parsed copies

**Evidence and impact.** [semantic_fidelity.py](../../exact/repair/semantic_fidelity.py), `aggregate_comparisons`, lines 700–713 counts every eligible list element. Passing the same comparison twice with quorum 2 returns decision `A`, eligible count 2 and the same comparison ID twice. Parser-only revalidation also creates multiple label records for a single wire response, so comparison-ID deduplication alone is not sufficient.

**Required change.** Bind each vote to a declared annotation attempt and the underlying durable request/response identity. Replaying or reparsing the same response is one judgment. In the current adapter, the wire identity is exposed in annotator metadata as `parameters_hash`; use a validated canonical identity rather than a rater-provided field or output text. Preserve distinct genuinely scheduled repetitions, presentation swaps and annotators even if their content happens to agree. Corrections and superseding parser versions must follow a declared rule and cannot multiply votes from one scheduled slot.

Validate duplicate observations against the frozen annotation schedule. Either reject duplicates explicitly or collapse them to one authoritative observation with an audit record. Preserve the scheduled denominator: if two attempts were scheduled but only one distinct response arrived, quorum 2 fails. Deduplication must not hide a missing, dissenting or invalid scheduled attempt. Different model repetitions are not independent human experts.

**Acceptance.** `REV-11-T1`: `[c,c]` cannot meet quorum 2. `T2`: two parser revisions of one request cannot meet it either. `T3`: two distinct scheduled requests with identical content count as two observations, with their provenance retained. `T4`: corrections, A/B swaps, disagreement, abstention and invalid output respect the aggregation manifest. `T5`: completed-response replay makes no new mock transport request, and aggregation makes no live calls. `T6`: training/evaluation artifacts reference the validated aggregation revision and exact unique observations used.

**Contract:** SF-016, SF-033, SF-034, SF-040, SF-041. **Reuse:** preserve raw responses and request accounting; recompute affected aggregates without paying for duplicate calls. Re-evaluate labels, checkpoint selection or metrics that depended on false quorum. Missing genuinely scheduled judgments remain missing.

### REV-12 — Implement support supervision or declare the disabled ablation

**Evidence and impact.** `model.py:133` creates `support_head`, but it has no call site. `_assignment_label` at `train.py:181–204` reduces verification to whole-plan feasibility and semantic outcomes, discarding obligation/support information needed by a separate support loss. Whole-plan BCE is implemented; support-level learning is not.

TR-020 requires the support-supervision capability, while TR-024 permits its loss to be optional. A valid explicitly disabled arm need not train this head. Merely constructing the layer must not be reported as implemented support learning or satisfy the enabled arm's conformance requirement.

**Required change.** For the enabled arm, retain qualified support/violation labels with complete assignment, asserted-axiom/occurrence identities, witness/obligation, activation, policy, backend provenance and availability information. Define the target precisely: predicting that a known sufficient support remains jointly active is different from predicting that its witness is still violated after some premises are removed. Absence of one sufficient support does not prove feasibility or supply a negative witness label.

Connect the declared readout and masked loss to the shared model using only permitted pre-decision inputs. Newly discovered post-decision proofs are supervision, not same-decision graph evidence. A subsequent declared repair round may use them once they are available. No extracted explanation and unknown verification are not negative support labels. Preserve higher-order support structure; a three-action contradiction cannot be converted into three pairwise incompatibilities.

For a disabled arm, skip the readout/loss, retain available/eligible label counts, record `support_enabled=false` and zero optimized support terms, and disable the corresponding learning claim. Disabling a loss must not erase evidence of label availability. Unused randomly initialized weights are not evidence of trained support prediction. Support/risk outputs remain soft predictions; only qualified symbolic proofs authorize hard cuts.

Known sufficient-support retention is already decidable symbolically from a selected assignment. Use the resulting proof-based cut directly; do not replace it with a neural estimate or present learning that Boolean conjunction as a new reasoning capability. The optional auxiliary experiment must test whether more specific supervision improves generalization to unverified combinations, or helps predict a declared witness violation beyond one already known proof. Compare against whole-plan risk alone and the same symbolic cuts. A support-disabled corrected system is a valid first experiment; the extra head is not a prerequisite for repair soundness.

**Acceptance.** `REV-12-T1`: a verified three-action conflict with feasible pairs produces one higher-order target and no false pair negatives. `T2`: eligible targets produce gradients in the support head. `T3`: unknown/unexplained rows produce no support gradient; justified negative controls follow the declared target definition. `T4`: disabled mode bypasses the head. `T5`: post-decision proof contents cannot change same-decision input features. Report each auxiliary's enabled state, label coverage and loss separately.

**Contract:** TR-020, TR-024; NN-008, NN-023; SF-020. **Reuse:** mark historical support weights as untrained. Version the enabled label/loss schema and retrain before claiming this capability. Shared auxiliary gradients may change the encoder; inference still keeps semantic benefit and logical risk as separate outputs.

### REV-13 — Make compiler-cache eviction safe under concurrent writers

**Evidence and impact.** [compilation.py](../../exact/repair/compilation.py), lines 341–342 gathers files and calls `stat()` outside the later `FileNotFoundError` handler. Writers use per-identity locks, so another writer can evict an entry between directory enumeration and `stat()`. The subsequent exception can turn an otherwise successful compilation into a worker error and remove generated families. This is a source-level concurrency concern; no live race was reproduced.

**Required change.** Add a deterministic concurrent/interleaving test first. Make enumeration, metadata collection and deletion tolerate disappearance of other immutable cache entries. Use a short bounded maintenance lock only if needed for the declared capacity semantics; do not serialize all native compilation unnecessarily. Cache maintenance failure must not change an already valid in-memory circuit into a logical empty family. Record cache/resource failures honestly, preserve atomic publication and integrity validation, and never mask an invalid compiled artifact as success.

**Acceptance.** `REV-13-T1`: delete an entry between enumeration and metadata access; the current valid result remains usable. `T2`: two independent identities publish/evict concurrently without an unhandled exception or corrupt artifact. `T3`: corrupt cached bytes are rejected. `T4`: capacity and cleanup limits remain enforced or yield an explicit resource status. Existing cold/warm semantic equivalence tests still pass.

**Contract:** CG-030, CG-032, CG-040. **Reuse:** valid immutable cache artifacts remain reusable under matching identities. Race protection alone does not require recomputing their probabilities or neural weights.

### REV-14 — Expose the compiler measurements needed for fair comparisons

**Evidence and impact.** The implementation records several compilation counters and stores cold-limit metadata, but the returned reports do not fully expose CPU and save/restore phase measurements or the originating cold-limit receipt. This limits the proposed comparison of family circuits, compilation strategies and warm-cache operation. It is a reporting gap, not evidence that the circuit probabilities are wrong. Alpha-renamed schema caching is optional and is not a missing mandatory feature.

**Required change.** Extend the existing compilation report/export with measured phase wall time, available CPU/RSS data, serialization/save/restore work, artifact size, cache hit/miss and the original compilation resource limits/receipt. Keep root-reachable, live, dead, manager-allocated and process memory counts distinct where the backend exposes them. Mark unavailable measurements explicitly. Do not substitute a warm-load duration for cold compilation or reconstruct unobserved historical timings.

**Acceptance.** `REV-14-T1`: cold compile and separate-worker warm load produce distinguishable complete receipts linked to the same structural identity. `T2`: the warm report exposes the original cold-limit receipt and current load/admission limits. `T3`: unsupported metrics are null/unavailable rather than zero. `T4`: study export retains the fields and charges work under the declared cold/warm comparison policy.

**Contract:** CG-027, CG-029, CG-030, CG-032. **Reuse:** compatible structural artifacts can remain cached; new performance claims require new complete measurements, not edited historical receipts.

## 4. Implementation order and acceptance gates

Prefer narrow changes to existing modules. Centralize preparation and target configuration because they already have multiple callers. Do not introduce a second ontology representation, solver, provider stack or a generic RL environment.

| Gate | Required work | Evidence before proceeding |
|---|---|---|
| G1: verification and recovery | REV-01, REV-02, REV-10 | Large-coverage transport, responsive deadlines, durable evidence and standalone exception-resume tests |
| G2: consistent model inputs and labels | REV-03, REV-04, REV-05, REV-09 | Identical effective preparation/targets, disabled-head behavior and exact partial-phase recovery |
| G3: declared candidate and sample coverage | REV-06, REV-07, REV-08; REV-13 for concurrent cache use | Quota/origin tests, admissible family representatives, language-preservation checks and cache interleaving test |
| G4: semantic and auxiliary supervision | REV-11; REV-12 for the enabled support arm | Unique-request aggregation, label provenance, masks and measured auxiliary gradients |
| G5: qualified experimental reporting | REV-14 and all applicable preceding gates | New source-bound conformance record, complete protocol/artifact identities and measured development coverage |

Independent fixes can be developed in parallel. A small fixture run may isolate a disabled component, but it must not be presented as completion of a gate it bypasses. Full XR-2.1 conformance includes the support capability; a named support-disabled ablation is a narrower arm.

Every corrective change must supply the implementation commit, affected `REV-*` IDs, actual test commands/environment/results and the new artifact identities. A passing old suite does not close a finding unless the new regression case is included. Mark a finding resolved only after its acceptance evidence is reviewed. Never replace the original review/evidence record to make an old revision appear corrected.

## 5. Artifact compatibility and historical results

Apply invalidation by dependency, not by deleting every cache or silently accepting everything. Keep historical artifacts immutable and record replacements separately.

| Changed dependency | Recompute or requalify | May be reused when independently valid |
|---|---|---|
| Grammar/context controls or protected candidate selection | Pools, generation coverage, learned factors for changed inventory, solver epoch and omission-control metrics | Matching source snapshots and qualified proofs by semantic identity |
| Admitted graph/supports or preparation settings | Graph/pair/risk artifacts, affected proposal scores and unary/pair coefficients, frozen objectives and training/selection behavior | Raw evidence and theory-bound verification results; primary factors only if a correction is confined to risk admission and leaves their inputs/values identical |
| Semantic weights/aggregation | Scalar targets, rankings, selection and affected model training | Complete query outcomes under matching theory/policy/query identities |
| Sampler schedule/provenance | Collection identity and compliance claims; affected sampling weights | Complete verified labels for the same inventory-bound assignment |
| Partial-epoch checkpoint semantics | Resume compatibility and any claim of exact continuation | A demonstrably correct earlier checkpoint or explicitly declared warm-start weights |
| Source-exception persistence | Ledger schema/adapter and baseline resolution | Qualified immutable source consistency/unsatisfiability evidence |
| Annotation deduplication/aggregation | Aggregates, downstream labels, affected selection/metrics | Raw responses and durable paid-request accounting |
| Optional support supervision | Auxiliary schema, label coverage and enabled-arm training | Compatible backbone weights as an explicitly recorded warm start |
| Compiler telemetry only | New performance measurements and report schema | Intact, compatible structural circuit artifacts |

Change component/schema identities wherever semantics or required fields change; do not silently reinterpret an existing artifact. Use explicit adapters where possible. A blanket version-4 migration is not required for every fix, but an unchanged version-3 label does not excuse missing semantic dependency checks. Protocols, cache keys, checkpoints and reports must identify the actual effective implementation/configuration.

In particular, raw semantic observations can often be reused without new reasoning or paid LLM calls. Reuse is justified by matching dependencies, not by proximity to an old candidate or similarity of names. Never replenish cumulative budgets as a side effect of migration or restart.

## 6. Verification plan for the corrective implementation

Implement the `REV-xx-Tn` cases in the existing repair tests, using the smallest relevant fixture. Use deterministic fault injection for resume, timeout and concurrency cases; use actual native adapters where the assertion concerns OWL semantics. A controlled backend is acceptable for large transport cardinality, but must not be described as a large native reasoning benchmark.

Run focused regressions first, then the repair/OpenRouter suite in the qualified Linux environment recorded by the implementation. The broad suite should include the following groups:

- `repair_v3_kernel_test.py`, `repair_v3_resources_test.py` and resume tests for transport, supervision and source exceptions.
- `repair_learning_v3_test.py`, training-completion/checkpoint tests and CLI tests for graph admission, target propagation, quotas, auxiliary modes and recovery.
- `repair_generation_v3_test.py`, grammar, circuit, proposals and compilation tests for protected representatives, nested languages and concurrent caching.
- `repair_semantic_fidelity_test.py` and `openrouter_ledger_test.py` with mock transport for annotation identity, aggregation and replay.
- Protocol/schema tests verifying effective settings, changed identities and rejection of conflicting quota declarations.

Record platform limitations rather than silently disabling required supervision. Retain new JUnit/log/source-digest evidence separately from `implementation/xr21-validation.json`. Recheck that source hashes bind the tested commit. A documentation-only commit does not justify regenerating runtime success receipts.

After conformance passes, freeze a new development protocol and measure generated-pool coverage, feasible/unknown rates, semantic quality, total verification time/calls and failure reasons. Include large conflict supports, hubs, overlapping/circular conflict structures and coherent-cycle controls as already required by [07](07-corpus-and-training.md). Then qualify real Conference/Bio-ML cohorts under [11](11-benchmark-evidence.md), preserving licensing, source-exception and backend scope. Do not infer those empirical results from this review.

The main research claim to test remains whether constrained learned generation reaches useful repairs under a finite budget, and whether learned semantic benefit and interactions improve plan quality or verification effort at matched coverage and cost. Circuit admissibility, MaxSAT optimality on a frozen pool, and verified logical feasibility remain separate claims.
