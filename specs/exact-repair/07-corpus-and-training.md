# XR-2.1 corpus, supervision and training

**Revision:** XR-2.1, 30 September 2026. **Status:** normative research specification; the additions below are not an implementation or experimental result. **Runtime baseline inspected:** `b4c1ed0d5e12c45974bdcb4d230fb2ab6c6deb04`.

This contract owns data construction, label eligibility, training objectives and checkpoint selection. [09](09-graph-and-neural-model.md) owns graph and model interfaces; [13](13-semantic-fidelity-supervision.md) owns LLM semantic-fidelity annotation and independent evaluation. The kernel and circuit contracts own authorisation and probability calculations. Requirement identifiers `TR-*` are stable acceptance references.

## 1. Supervision objectives and boundaries

**TR-001 — Four separate quantities.** Keep logical feasibility, semantic fidelity, proposal likelihood and explicit edit cost separate. A reasoner establishes supported logical obligations; it cannot determine intended domain meaning from coherence alone. A semantic-fidelity label assesses whether a feasible repaired theory retains or restores the intended meaning supported by the supplied evidence. The original alignment can contain errors, so preserving every original assertion is not the target. A learned value is an estimate of a declared semantic target, not logical authority. Proposal likelihood is neither value nor correctness probability.

The principal learned-supervision programme uses LLM weak judgments about semantic fidelity, supplemented by controlled symbolic teachers. It does not learn personal user preferences as its main research question. Declared edit-cost weights remain explicit. Optional future personalization requires a separate dataset, study and claim; it is not a substitute for semantic evaluation.

**TR-002 — Supervision provenance.** Every target identifies one of: `symbolic_intended`, `llm_weak`, `human_adjudicated`, or an explicitly named mixture with separately reported components. Generated intended theories provide controlled ground truth only for their declared construction and query basis. LLM judgments are weak labels, never reasoner proofs or domain ground truth. No human adjudication means no claim of expert-validated semantic correctness. [13](13-semantic-fidelity-supervision.md) specifies this distinction in detail.

**TR-003 — Existing versus revised training.** At the inspected baseline, `exact/repair/learning.py::benefit_losses` already trains anchored value/ranking losses from usable feasible rows of partial caches. `tools/repair/train.py::case_loss` restricts the exact marginal proposal loss to complete caches. XR-2.1 retains that capability and adds newly generated candidate inventories, supervised active sampling, interaction contrasts, separate conflict-risk targets and LLM fidelity supervision. It must not describe partial-cache value learning itself as newly implemented.

## 2. Dataset families and independent grouping

**TR-004 — Controlled mechanisms.** Generate verified intended theories and a query basis before producing corrupted observations or repair outputs. Required mechanisms include directional overstrengthening; subclass-expression specialisation on both ontology sides; necessary conditions; complex equivalence; endpoint confusion; removal/revision of disjointness and superclass conjuncts; ontology subclass revision; domain/range and existential-filler generalisation; redundant and complementary semantic benefit; higher-order conflict supports; and mixed compositions. Include consistent/coherent controls, intrinsically ambiguous observations and cases with no feasible repair in the frozen pool.

Vary connected path length, branching, overlap of explanation supports, irrelevant structure, expression depth, score calibration, missing evidence and misleading labels. Mirror roles. Vary human/generated/unknown authorship independently of correctness. A longer or renamed fixture alone does not establish structural diversity. Record generation mechanisms and all variation axes in the evaluator store.

**TR-005 — Corpus hierarchy.** Use four separately reported cohorts: controlled generated structure; controlled corruptions on training-side real ontology structure; actual held-out Conference matcher outputs; and Bio-ML transfer/scale cohorts under the release and licensing distinctions in [11](11-benchmark-evidence.md). Matchers and permitted upstream evidence are captured once and reused across repair arms. Controlled corruption and naturally erroneous alignments answer different questions. An LLM cannot infer useful domain semantics from opaque names such as `C17` unless an explicit grounded definition package is supplied; such examples remain structural/symbolic controls.

**TR-006 — Splits before enrichment.** Partition clean structural parents and real ontology-pair groups before corruption, annotation, proposal expansion or active sampling. All mirrored/renamed siblings, matcher outputs, explanation variants, candidate pools, counterfactuals and LLM comparisons derived from one parent retain its split. Distinguish pair holdout from whole-ontology and matcher holdout. Preserve a whole-ontology holdout when required by the common protocol. Shared biomedical ontology releases/contents are identified across cohorts. Test-derived labels, ontology statistics fitted across test inputs, annotation calibration, budgets or checkpoints cannot enter training or development selection.

Development may supply declared checkpoint labels; test labels remain sealed until all choices are frozen. A test case used to motivate redesign becomes exploratory/regression material for that revision. Active rounds operate on training groups only. The complete sampling and annotation ledger remains available for audit.

**TR-007 — Identifiability and ambiguity.** Identical observable inputs with different hidden intended theories cannot support a forced unique semantic label. Preserve alternative intended interpretations, an ambiguity flag and any missing discriminating evidence. Judges must be allowed to abstain. Do not encode hidden intent in candidate IDs, source names, generated descriptions, action tags or corruption-specific evidence fields.

## 3. Record schema and storage boundaries

**TR-008 — Immutable identities.** Proposed XR-2.1 records use explicit v3 schema identifiers and canonical hashes; baseline v2 records are not silently reinterpreted. Candidate identities include emitted syntax, active obligations and other feasibility-relevant restrictions. Assignment identity is a canonical object-ID to candidate-ID map, not a positional tuple without an inventory binding. Cross-inventory comparisons require matching case, policy, query and semantic-rubric identities; their candidate maps remain distinct. The following are logical dataset sections, not a requirement for a new storage framework: they are joined through `PlanSampleV3` and reuse `RepairInputV3`, `GraphInputV3`, `VerificationReportV3` and `ProofSupportV3` from [01](01-architecture-and-contracts.md).

| Record | Required fields and meaning |
|---|---|
| `CaseRecord` | `case_id`, `parent_group_id`, split, cohort/release, source/target/import hashes, observed alignment/evidence manifest, eligibility/policy/baseline hashes, accessible query manifest, budgets, generated-mechanism reference if applicable |
| `EvidenceManifest` | Evidence ID, entity/statement scope, source/document identity, content hash, evidence kind, reliability provenance, availability stage, licence/access status, explicit missingness and truncation status |
| `InventoryRecord` | Inventory hash, ordered object/candidate IDs, complete bundles/activations, fixed controls, retrieval menus and omissions, grammar/config hash, proposal/checkpoint revision, family coverage, cap decisions, initial versus final omission controls |
| `AssignmentRecord` | Case/inventory hashes, complete object-to-candidate map, reconstructed theory hash, proposal origin, active-round/checkpoint ID, random seed, selection stratum, inclusion probability when known; `unknown` when it is not |
| `VerificationLabel` | Assignment/theory/policy/backend hashes; `FEASIBLE`, `INFEASIBLE` or `UNKNOWN`; verdict and support scope for every policy obligation; unknown cause; cache status; measured calls/time/memory; explanation references and completeness |
| `QueryOutcome` | Query ID/type/family, desired/unwanted/unspecified role, declared weight and fixed denominator, entailment `true/false/unknown`, each typed non-vacuity condition and its status, backend/evidence references, evaluation time and mask |
| `SemanticLabel` | Target provenance, rubric/query versions, complete criterion vector with statuses, scalar benefit or null, scale/anchor version, required-outcome completeness, label confidence/disagreement metadata; LLM record references where applicable |
| `ComparisonLabel` | Two assignment IDs, same evidence/policy/rubric basis, semantic preference `A/B/tie/abstain`, criterion judgments, cited evidence/consequence IDs, completeness mask, annotator provenance, order/repetition metadata |
| `SupportLabel` | Violation/query type, explanation/support ID, selected candidate literals and exact fixed/editable axiom occurrences, theory/policy dependencies, proof scope, sufficient/minimal/unknown support status, future-availability boundary |
| `TrainingManifest` | All dependency and split hashes; architecture/loss versions; round schedule; example and loss eligibility counts; sample strata/weights; scale/cost versions; selected checkpoint rule; failures and requested denominators |

The inference feature view and evaluator label store are separate projections with allowlisted fields. A graph-builder function receives the inference view, not an unrestricted serialized training record. Hashes establish identity, not truth. Cache reuse validates theory, inventory, query, policy, backend, rubric and evidence dependencies appropriate to the cached object.

**TR-009 — Explicit statuses.** Required evidence/query/criterion states distinguish `observed`, `missing`, `truncated`, `unsupported`, `timeout`, `error`, `not_applicable` and `unvisited`. A numeric zero is an observed value. `not_applicable` is a manifest-level criterion decision, not a post hoc way to remove an inconvenient outcome. Unknown whole-policy status does not become a negative example. Per-obligation decided facts may train a separately named per-obligation head while the whole-plan label remains unknown.

## 4. Typed symbolic labels

**TR-010 — Non-vacuity.** For a supported consistent feasible theory T, desired query success combines entailment with the query's typed non-vacuity obligations:

~~~text
success_T(C ⊑ D) = entails_T(C ⊑ D) AND satisfiable_T(C)
success_T(Disjoint(C,D)) = entails_T(Disjoint(C,D))
                          AND satisfiable_T(C) AND satisfiable_T(D)
success_T(C ⊑ exists r.D) = entails_T(C ⊑ exists r.D) AND satisfiable_T(C)
~~~

A complex subclass query uses its complete antecedent. Desired disjointness does not require a satisfiable intersection. Property/instance probes define their own typed obligations; role-use non-vacuity can require satisfiability of `exists r.Thing`. Unknown constituent results propagate to the query status unless a three-valued logical result is already determined. Preserve constituent statuses even when the combined result is known.

**TR-011 — Fixed target basis.** Partition queries into desired, unwanted and unspecified before seeing repair outputs. Public-reference absence and synthetic non-entailment are not automatically unwanted real-world consequences. For complete symbolic targets:

~~~text
B_symbolic(R) = sum_f gamma_f mean_(q in Qplus_f) success_R(q)
                - sum_g delta_g mean_(q in Qminus_g) entails_R(q)
~~~

Only declared nonempty families enter the manifest. Record unwanted raw entailment separately; emptying an antecedent does not earn positive restoration credit. An inconsistent/infeasible theory has no semantic-benefit scalar and receives no credit through explosion. Hard obligations cannot be compensated by high soft benefit.

**TR-012 — Unknown denominators.** Compute a scalar only if every outcome required by its target definition is decided. Never replace a five-query mean by a four-query mean because one query timed out. For outcomes `[true,true,false,true,unknown]`, retain denominator 5, the complete vector and its mask; the scalar is null. A declared component loss may train on the four known components, reporting numerator and denominator separately; it is not whole-plan benefit or a complete query result.

## 5. Exhaustive and sampled teachers

**TR-013 — Exact small teachers.** Exhaustively enumerate complete assignments only for declared small frozen inventories. Record the Cartesian-product size before labeling, all assignment/query budgets, coverage and stop reason. Exact feasible-set utility normalization, marginals, optimum and regret require every relevant assignment and required target outcome decided. Discovery of disconnected conflict supports does not prove separability of the full policy/semantic objective. Never assume the all-delete assignment is feasible.

For a complete finite reference set F and declared utility `U*(R)=B*(R)-lambda^T c(R)`:

~~~text
rho(R) = exp(U*(R)/tau) / sum_(Rprime in F) exp(U*(Rprime)/tau)
t_i(a) = sum_(R in F) rho(R) 1[a_i = a]
~~~

Exactness refers to this frozen universe and declared target, not the whole grammar or universal intended meaning. A numerical distribution over exhaustive weak LLM labels is still a weak-label proxy and must not be presented as an exact symbolic/domain teacher.

**TR-014 — Verified sampled repairs.** For larger cases, generate candidate pools containing genuinely newly proposed bundles as well as controls. Sample complete assignments using declared mixtures of initial MaxSAT solutions, bounded diverse/near-optimal alternatives, uniform or stratified control proposals, learned proposals, counterfactual edits and disagreement/uncertainty strata. Record which mechanism selected each row. Do not label only the current model's preferred feasible solutions; include decided infeasible alternatives and structural/semantic diversity.

Verify every sampled assignment before admitting a whole-plan target. Label required symbolic outcomes and, for the fidelity cohort, grounded judgments under [13](13-semantic-fidelity-supervision.md). Incomplete labels remain in the ledger and denominator. A new candidate absent from an older cache triggers new labeling; it does not inherit the label of a syntactically similar candidate. Store full labels under the new inventory, even when an identical theory permits explicitly validated cache reuse.

**TR-015 — Supervised active rounds.** Freeze model, sampler and budgets at round start; acquire a bounded training-only batch; verify and annotate it; validate records; then retrain for the next round. Record model/sampler versions and all exclusions. The label is an external supervised observation, not a policy-gradient reward. No reinforcement learning, differentiable solver or online model mutation during a frozen solve is required. Budget and stopping rules are fixed on development data. Unknown/unsupported strata are reported rather than resampled until they disappear.

**TR-016 — Selection bias.** Exact sampling probabilities are recorded only when available; a deterministic optimizer's output is not assigned an invented uniform propensity. Sample-conditioned likelihood/ranking targets are named approximate. Importance weighting is allowed only with valid propensities, support and a declared variance-control rule. Evaluation uses independently scheduled cases and a representative held-out assignment sample, not the active sampler's optimistic training distribution.

## 6. Learning semantic value and interactions

**TR-017 — Frozen factorization.** Use the model in [09](09-graph-and-neural-model.md): shared unary semantic factors plus bounded pair factors. Explicit structural/edit costs are subtracted once outside semantic benefit. Train on whole-plan targets; do not give each candidate the maximum total utility of a repair containing it and then sum those values. That double-counts the remaining actions' contributions.

For a verified feasible anchor R0 with a complete target on the same semantic basis:

~~~text
d_hat(R) = B_hat(R) - B_hat(R0)
d_star(R) = B_star(R) - B_star(R0)
L_value = mean_R SmoothL1_beta(d_hat(R) - d_star(R))
L_rank  = mean_(R,S) softplus(-sign(B_star(R)-B_star(S))
                             * (B_hat(R)-B_hat(S)) / t_rank)
~~~

Absolute offsets cancel within a case; anchored numeric differences fix the scale relative to explicit costs. Choose actual feasible anchors, using stable lowest-cost then candidate-ID ties for the symbolic cohort. The baseline action used to identify pair factors need not itself form a feasible assignment and is not automatically a label anchor. LLM ordinal comparisons alone cannot identify a cardinal value/cost trade-off; use the declared anchored weak numeric rubric and calibration protocol in [13](13-semantic-fidelity-supervision.md), or report ranking only without a calibrated utility claim.

**TR-018 — Ranking masks and ties.** Whole-plan semantic ranking requires both assignments verified feasible and all outcomes required by that semantic target decided. Symbolic unequal-benefit ranking excludes exact ties. LLM ranking uses `A/B/tie/abstain` records: ties have an explicitly named tie loss; abstentions produce no global preference gradient. Partially known criteria can supervise a named criterion head, but cannot silently train the whole-plan rank. Use weights for declared label provenance/quality, never fabricated correctness probabilities from an LLM's self-confidence.

**TR-019 — Counterfactual quartets.** Construct four assignments differing only at objects i and j, with identical other choices, policy, evidence and target basis. Let their labels be B00, B10, B01 and B11. Only when all four are feasible and complete:

~~~text
Delta_ij = B11 - B10 - B01 + B00
L_interaction = SmoothL1(Delta_hat_ij - Delta_ij)
~~~

If any member is infeasible/unknown, do not substitute a huge negative benefit or zero. Its decided feasibility can train the separate risk target. For a pure unary-plus-pair model, this contrast isolates the selected pair's discrete difference. Different backgrounds can yield different true contrasts in a higher-order domain; retain those contexts and report residual error rather than claim an exact pairwise decomposition of OWL semantics.

Example: keep/delete choices for `s:A ⊑ t:B` and `t:B ⊑ s:C`, with desired `s:A ⊑ s:C` and no alternative path. Benefits `(B00,B10,B01,B11)=(0,0,0,1)` require a positive interaction of 1. If either of two actions independently restores the same benefit 1, `(0,1,1,1)` requires an interaction of -1. Costs remain a separate table.

**TR-020 — Feasibility negatives and support hyperedges.** Train a whole-plan failure probability from complete-policy `FEASIBLE=0` and soundly decided `INFEASIBLE=1`; unknown has no binary whole-plan target. Keep task/query type and applicable reasoning scope. Explanations supervise a separate support/violation head with exact selected candidate literals, emitted axiom occurrences and dependencies. A sufficient three-action conflict is a hyperedge; it does not label its three constituent pairs incompatible. Minimality is stored only when established. An unexplained infeasible plan provides a whole-plan negative without invented support labels. Explanations observed after a training assignment belong to labels for that decision; later rounds may use them as input only with explicit temporal/dependency provenance.

Absence of an extracted support is not a negative support label. A subset can be labeled non-conflicting only under a separately defined verified counterfactual/support task; feasibility after removing other axioms does not establish global compatibility in every extension. Learned risk/support predictions never create hard clauses.

## 7. Proposal learning

**TR-021 — Independent proposal baseline.** The baseline proposal distribution is per object. Its exact finite-teacher loss is `-sum_i sum_a t_i(a) log p_theta(a | context_i,K_i)`, including the exact circuit normalizer and the sum over all encodings of a canonical bundle. Independent per-object likelihoods do not learn cross-object correlations. Diagonal plans `(A,A)/(B,B)` and anti-diagonal plans `(A,B)/(B,A)` can have identical marginals. Within-object mixture dependence is a different capability.

**TR-022 — Optional plan conditioning.** A separately identified extension conditions each object proposal on a frozen hypothetical partial-plan context. An ordered conditional model may train the exact joint loss `-sum_R rho(R) sum_i log p_theta(a_i | a_<i,graph,K_i)` on a complete small teacher. This is a real conditional factorization, not an independent product renamed joint. Record order, prefix representation, constraints and context hash; all logits for each circuit call are computed before sampling. Mask unavailable future decisions. Randomized-order training, if used, records its order distribution.

For inference, union deduplicated candidates from several bounded hypothetical plans with mandatory controls, then freeze the final pool and re-score it independently under one declared factor objective. Generation contexts guide coverage; they are not different objective definitions for identical final candidates. Non-autoregressive iterative refinement can be an ablation, but its likelihood is not called a normalized joint plan distribution without an explicit probability model.

**TR-023 — Sampled proposal loss.** Verified sampled plans can train supervised likelihood or listwise preferences using a declared empirical distribution over the observed sample. Label it `sample_conditioned`, retain ties and provenance, and do not call it an exact teacher marginal. Select targets only from eligible feasible fully labeled plans for the chosen utility/fidelity target. A risk auxiliary may use infeasible plans independently. Record duplicates and sample weights so repeated acquisition does not silently overweight a plan.

A positive target outside the grammar/menu is a coverage failure. Do not inject the missing answer from a label into inference retrieval. The exact loss is unavailable for that target distribution; report missing mass and loss exclusion. A separately named reachable-subset loss may be studied, but its renormalization and changed target are explicit. Label/semantic quality, retrieval recall and generator recall are distinct outcomes.

## 8. Loss routing and optimization

**TR-024 — Eligibility matrix.** Loss counts below are mandatory per case, cohort, round and epoch.

| Label state | Whole-plan value/rank | Counterfactual contrast | Proposal target | Whole-plan risk | Support head |
|---|---|---|---|---|---|
| Feasible; required semantic outcomes complete | Yes, with valid same-basis anchor/comparison | Only a complete four-member group | Exact only with complete universe; otherwise named sampled loss | Feasible target | Only separately justified support/non-support targets |
| Feasible; required semantic outcome unknown | No complete scalar/rank; named known-component loss only | No | No utility-derived whole-plan target | Feasible target | Independently eligible decided targets |
| Infeasible, soundly decided | No semantic scalar/rank | No | No positive feasible-plan target | Infeasible target | Available valid support; otherwise masked |
| Whole-policy unknown | No | No | No | Mask whole-plan target; optionally train decided obligations separately | Only independently justified supported facts |
| LLM tie with complete eligible basis | Declared tie loss, numeric target only if rubric complete | Numeric contrast only with all four numeric targets complete | Preserve declared tie mass | Determined by verifier, not judge | Determined by symbolic support, not judge |
| LLM abstain/incomplete evidence | No global fidelity target | No | No fidelity-derived target | Determined independently | Determined independently |

The total loss is a versioned weighted sum of value, semantic comparison/tie, eligible interaction contrast, exact-or-sampled proposal, whole-plan risk, optional obligation/support, and optional criterion losses. Weights, temperatures, class/stratum weighting, regularization, clipping, optimizer, learning rate, update counts and early stopping are frozen in a new revision-specific run manifest. The [protocol migration notes](protocol/README.md) give provisional generated-development starting settings, not validated optima or executable annotation authorization. Do not amend frozen XR-2 pilot JSON or imply that its former `1/.2/1` weights cover new heads. Normalize each task by its eligible-example count, report empty tasks, and control losses so very large inventories do not dominate merely through pair count.

**TR-025 — Training sequence.** Each batch loads only allowed inference fields; builds the shared graph and menus; constructs the pair index under the same routine as inference; encodes candidate syntax; forms frozen-inventory semantic factors; computes complete-plan predictions and masked losses; evaluates applicable differentiable circuit likelihoods; and backpropagates through the shared encoder/readouts. Grammar topology, verifier outputs and annotation records are fixed targets. Risk gradients may update shared representations but cannot change the hard feasibility definition. MaxSAT need not be differentiable. Record task-gradient ablations if shared auxiliary losses harm semantic quality.

Generated symbolic pretraining, fidelity weak-label training, and training-side real adaptation are separate named stages/arms. Warm starts record exactly which weights, optimizer state and schema are reused. Test labels and test-selected models cannot migrate into adaptation. All replayable training checkpoints preserve optimizer/RNG state, data/split identities, loss eligibility, numeric scale and stop state.

## 9. Candidate controls, deployment parity and checkpoint selection

**TR-026 — Missing-action controls.** Distinguish `initial_inventory_omission`, `retrieval_symbol_omission`, `grammar_action_omission` and `final_pool_omission`. Apply final-pool ablation filters after every materialization, representative, sampling, deduplication and refill stage; assert the excluded bundle/action is absent from the frozen inventory. Do not silently remove protected keep/lock requirements to create an invalid control. If the experiment tests recovery from initial omission, allow regeneration but name and measure that recovery. The baseline corpus removed an intended candidate only from the initial pool; grammar regeneration could restore it, so historical missing-candidate labels are insufficient evidence for final-pool absence.

**TR-027 — Shared preparation.** Training, development and production must use the same explanation-aware pair selection, retrieved endpoint materialization, typed menus, candidate bounds, canonicalization and feature-status encoding. The baseline training call omitted retrieved explanations while production supplied them. Endpoint alternatives must be materialized for directly supplied and already prepared inputs, idempotently. An existing captured retrieval is reused only if all relevant identities/configuration match. Pair-selection and candidate-materialization outputs are hash-compared across the three paths.

**TR-028 — Checkpoint criterion.** Select on a frozen development schedule that evaluates actual generated pools, verified repairs and external semantic labels. Supplied-cache regret remains a small-case diagnostic, not the primary deployment criterion. Predeclare a coverage requirement and uncertainty-aware semantic-quality comparison; among eligible models meeting those criteria, compare verification effort at matched quality and coverage. Freeze the numeric rule/tolerances before the run. If no model meets the requirement, report selection failure or a clearly named exploratory fallback; do not quietly reduce the threshold.

Report all scheduled cases: useful retrieval/proposal coverage, generated-pool feasible coverage, semantic fidelity on eligible independently labeled outcomes, unknown/abstain rates, edits, time/calls to first verified feasible repair, time/calls to a declared equal-quality target, total work, final objective bound/gap and certification status. Include cold/warm cache accounting and label acquisition cost. Quality from the model being selected is not its own validation target. Fewer verifier calls achieved by dropping useful candidates is not a successful efficiency result. Partial/timeout runs remain in the denominator.

## 10. Acceptance and migration gates

**TR-029 — Required executable acceptance.** These tests are future implementation obligations, not claims that the specification edit ran them.

| Test ID | Required observation |
|---|---|
| TR-T01 | Typed disjointness/non-vacuity succeeds without requiring a nonempty disjoint intersection; inconsistency yields no benefit scalar. |
| TR-T02 | One unknown of five required queries preserves denominator 5 and produces a null scalar/global-rank mask. |
| TR-T03 | A partial cache with two eligible feasible labels produces value/rank gradients but no exact marginal teacher claim. |
| TR-T04 | Complementarity and redundancy quartets recover positive/negative contrasts; an infeasible/unknown quartet member masks its contrast. |
| TR-T05 | A three-literal conflict trains a hyperedge/whole-plan target without creating pair-negative labels or learned hard cuts. |
| TR-T06 | Exact tiny teacher likelihood matches enumeration; sampled-cache targets remain separately named; unreachable mass is visible. |
| TR-T07 | Correlated diagonal/anti-diagonal teachers give equal independent marginals; only the conditional extension can distinguish their joint targets. |
| TR-T08 | A candidate created after an older cache is verified/labeled under its new inventory; no copied nearest-candidate target is used. |
| TR-T09 | An overlap-family final-pool omission remains absent after representatives and refill; initial-omission recovery is reported separately. |
| TR-T10 | Explanation-only and candidate-induced object links yield identical pair indexes in train/dev/inference. |
| TR-T11 | Direct/prepared inputs materialize the same typed endpoint alternatives without duplicates. |
| TR-T12 | Absent/zero/unknown/truncated score channels and opaque identifiers cannot be mistaken for observed semantic evidence. |
| TR-T13 | Every sibling, sampled pool, active round and annotation comparison inherits its parent split; label-store fields fail the inference allowlist. |
| TR-T14 | A model with better cached regret but worse generated verified quality is not selected by an undeclared cached-first fallback. |
| TR-T15 | Unknown/abstaining LLM outcomes never become negative policy labels, fabricated scalars or expert-ground-truth claims. |
| TR-T16 | Resume rejects changed inventory, split, rubric, scale, pair-selection or loss schema; matching resume preserves optimizer/RNG state. |

**TR-030 — Revision evidence.** Preserve baseline artifacts and frozen protocol files. New datasets/checkpoints/results declare XR-2.1 manifest/schema identities and the code revision that actually implements them. A completed document, component test or symbolic toy example does not establish trained-model quality, real-domain semantic validity or fewer verifier calls.
