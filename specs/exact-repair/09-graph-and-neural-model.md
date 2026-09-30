# XR-2.1 observable graph and neural architecture

**Revision:** XR-2.1, 30 September 2026. **Status:** normative design, with implementation gaps relative to runtime `b4c1ed0d5e12c45974bdcb4d230fb2ab6c6deb04`. Planned interfaces use v3 semantics; existing v2 runtime/checkpoints remain historical artifacts.

This contract owns graph construction, features, shared model heads, sparse interaction selection and frozen scoring. [07](07-corpus-and-training.md) owns training masks and losses; [10](10-constrained-generation.md) owns circuit probability; [04](04-minimal-exact-kernel.md) owns exact optimization and verification; [13](13-semantic-fidelity-supervision.md) owns intended-meaning weak labels. `NN-*` requirements specify what the revised implementation must do.

## 1. Model boundary and outputs

**NN-001 — Placement.** Use one observable graph encoder and shared readouts. No explanation node owns a separately trained neural network. HGT attention heads are internal message-passing components, distinct from proposal, semantic-value and risk prediction heads. The main encoder is standard HGT, with matched R-GCN and no-message-passing controls using the same downstream readouts. Reuse the established optional neural runtime; a new generic graph/model framework is unnecessary.

~~~text
asserted snapshots + provisional alignment + available evidence
  -> bounded observable context, typed menus and endpoint alternatives
  -> HGT/R-GCN/no-message-passing node memory H
       -> object attention c_i
            -> proposal logits -> grammar/context circuit -> complete candidate bundles
       -> candidate syntax + attention e_ia, h_ia
            -> unary semantic head
            -> shared pair semantic head for frozen observable pair set P
       -> selected-plan / support aggregation
            -> whole-plan risk and optional support/effort auxiliaries
  -> freeze unary/pair semantic factors + explicit costs in ObjectiveV3
  -> MaxSAT assignments -> bounded risk scheduling -> exact verification
~~~

The semantic objective and risk scheduling outputs have separate types, hashes and consumers. A risk prediction, natural-language judgment or high proposal probability cannot authorize a repair.

**NN-002 — Interface.** A model invocation consumes `GraphInputV3`, the final canonical candidate inventory, a declared semantic-task/rubric identifier and any allowed conditioning, explicit cost settings for the generator, and fixed resource budgets. It returns graph/context memory, proposal records, per-candidate semantic coefficients, per-selected-pair candidate matrices, optional risk/effort predictions, and provenance/omission records. Shape and meaning are explicit:

| Value | Shape / scope | Meaning |
|---|---|---|
| H | N by d | One contextual embedding per admitted node; typed ID lookup retained |
| c_i | d per editable object | Attention readout from that object's observed statement/evidence context |
| e_ia | d per complete bundle | Canonical emitted syntax and active obligations, shared across heads |
| h_ia | d per object/candidate | Candidate plus object/context readout |
| proposal mixture logits | M per circuit call | Unnormalized component weights before grammar conditioning |
| proposal Boolean logits | M by V_i | Bernoulli-choice parameters for the fixed circuit invocation |
| u_i | C_i | Unary semantic contributions on the declared anchored scale |
| v_ij | C_i by C_j for (i,j) in P | Pair semantic interaction contributions, with recorded gauge |
| risk(R) | Scalar in [0,1] when calibrated | Estimated policy-failure risk under its training scope, not a verdict |
| support(H_s,R) | Scalar / typed target | Optional prediction about a declared support hyperedge and selected literals |
| effort(R) | Declared runtime/call target | Optional verification-effort estimate with censoring/support metadata |

`C_i` is the candidate count, `V_i` the Boolean circuit width, and M the mixture count. Memory/layout choices may vary; public meaning does not. Reject unknown node/edge schemas and incompatible checkpoint heads rather than silently initialize them for evaluation.

## 2. Observable graph schema

**NN-003 — Node identity and types.** Build from asserted shared-core snapshots, the full provisional alignment, allowed baseline diagnosis and evidence available before the current decision. Preserve `(IRI, entity kind)` under punning. A syntactic axiom is distinct from an editable occurrence and from its mapping endpoints.

| Node type | Required content / identity | Typical outgoing roles |
|---|---|---|
| Class, object/data property | Typed entity identity, ontology side(s), permitted descriptions | Syntactic participation through incident statements; evidence support |
| Individual, literal, datatype | Typed identity and supported literal/datatype structure | Assertion positions, property values, datatype roles |
| Constructor/expression | Canonical syntax identity, operator, scalar restrictions where relevant | `operand`, `property`, `filler`, `cardinality`, typed argument roles |
| Ontology axiom | Canonical semantic axiom identity, supported syntax | `subclass`, `superclass`, `domain`, `range`, disjoint operands, assertion roles |
| Editable occurrence | Exact source/import occurrence, authorship, eligibility/lock status | `asserts` original axiom; source document / occurrence provenance |
| Mapping object | Revision-object identity, original bundle and relation | `source`, `target`, `asserts`, evidence support |
| Explanation support | One witnessed query/violation plus its complete known support | `supports_statement`, `supports_occurrence`, `witnesses_query` |
| Query/task descriptor | Only declared inference-visible competency task/criterion identity and allowed syntax | Query operands and typed non-vacuity requirements |
| Evidence | Content/provenance identity, type, availability and status | `describes`, `supports`, or `matcher_evidence_for` a specific entity/object |

Add reverse message edges with distinct typed roles and explicit self-relations if required by the encoder. Preserve unordered operand symmetry; an arbitrary input list order must not change an intersection or disjointness embedding. A query descriptor is allowed only if that task/query basis is genuinely part of the inference input. Hidden intended queries, reference membership and annotation answers remain outside this graph.

**NN-004 — Explanations.** One explanation node denotes one discovered support with its witnessed obligation, not an arbitrary mapping pair and not necessarily an unsatisfiable class. Multiple supports may share a witness. Store whether support is sufficient, minimal, partial or unavailable, with reasoning provenance. A witness without extracted support gets a missing-support indicator. An unsupported/hypothetical explanation is soft evidence and cannot masquerade as a proven conflict. Neural message passing over a support does not itself prove its entailment.

**NN-005 — Occurrences and verification.** Graph sampling never changes the complete theory checked by the verifier. Duplicate/imported axioms retain occurrence provenance even if their syntactic node is shared. ABox, datatype and other supported logical constraints remain in verification even when their neural representation is bounded or their repair action set is limited. Unsupported neural/action coverage is recorded; it is not permission to discard logical input.

## 3. Features, missingness and leakage

**NN-006 — Typed feature channels.** Use a versioned feature schema rather than arbitrary evaluator-record flattening. Channels include syntax/type, ontology side, edit eligibility, original relation and object kind, authorship/provenance, matcher identity, available numeric scores, text/definition representations, observed diagnostic context and channel-specific missingness. Fit normalization or learned textual encoders only through the declared training regime. Constant-score, missing-score, unseen-matcher and misleading-text controls are required.

For every optional channel distinguish an observed zero from absence, unsupported acquisition, truncation, timeout/error and unknown provenance. Keep values plus a typed state mask; do not infer absence from a zero-filled array. Record score calibration provenance; confidence is not assumed calibrated. An omitted graph neighborhood is missing context, not evidence of no connection. Give the model the declared scope/coverage flags where appropriate, without exposing hidden counts of true conflicts or true useful candidates.

**NN-007 — Text and identifiers.** Entity labels, local names and definitions may be semantic evidence, with their origin and reliability. Raw namespaces, file paths, structural hashes, split IDs, candidate IDs and generated parent/seed encodings are lookup/provenance fields, not trainable lexical tokens. Permit meaningful local names through a declared extraction rule and evaluate opaque-ID renaming. Opaque synthetic identifiers cannot acquire invented domain meaning. Candidate ID ordering is used only for deterministic replay, never as a correctness feature.

Use shared entity/text projections; no ontology-specific IRI output parameter is required. A hashed-feature baseline can remain an explicit ablation, but it must follow the same allowlist/status semantics. Numeric hashing collisions and text truncation are measured. LLM rationales and label confidence from the evaluator store are not deployment features. If an independently available language-model description is used as evidence, it is a separately authorized upstream channel with its own leakage and reliability controls, not recycled judgment text.

**NN-008 — Forbidden inputs.** Reject clean hidden theories, corruption traces/positions, intended-action indexes, gold/reference membership, teacher utilities, pairwise semantic verdicts, test statistics, split/group identifiers, and explanations obtained only after the decision. Availability is checked per evidence record and round, not merely by a blacklist of field names. Schema projection and tests guard against aliases/nested leakage. Training records may contain all labels; graph constructors receive only the validated inference projection.

## 4. Retrieval and bounded context

**NN-009 — Vocabulary and endpoints.** Retrieve bounded per-side typed class/property menus from visible definitions/labels, structural neighborhoods, matcher alternatives and valid discovered supports. Materialize retrieved endpoint alternatives into complete candidates using the same action/eligibility checks on every entry path. This operation is idempotent, works for direct `RepairInput` construction as well as preparation adapters, and retains source/target type and side. Merely listing an endpoint in retrieval metadata does not establish that its replacement exists in the final inventory.

The baseline `prepare_repair` materialized alternatives, while direct neural preparation could retrieve them without wiring them into grammar controls. XR-2.1 closes that path difference. Keep retrieval recall, grammar representability, sampling recall and final-cap retention as separate denominators. The model cannot recover an out-of-menu symbol through more draws. Any learned retrieval extension is separately supervised and compared with deterministic retrieval; labels never inject missing test answers.

**NN-010 — Context budgets.** Make editable objects, original syntax, required active-candidate symbols and their minimal typed identities mandatory. Include explanation supports atomically when admitted; if the complete support exceeds the budget, record it as omitted or use an explicitly labeled partial-evidence representation. Do not retain an unlabeled fragment as a complete proof support. After mandatory content, budget additional fixed axioms/neighborhoods and text deterministically. Record node, edge, token, support and candidate-symbol omissions and stop causes.

A final candidate cannot be scored using a symbol absent from memory. Either admit its minimal observable identity/description within a predeclared reserve, or report candidate-context unavailability and the consequent coverage loss. No zero/random embedding is silently treated as complete evidence. The selected graph is frozen before the corresponding coefficients are computed; extending it requires recomputation/new-round identity.

## 5. Encoder and shared readouts

**NN-011 — Encoder.** HGT provides typed attention over admitted graph edges. It is an approximate representation model, not an OWL reasoner. A schematic head is:

~~~text
q_v = W_Q[type(v)] h_v
k_u = W_K[type(u)] h_u
attention_uvr = softmax_incoming(q_v^T W_A[r] k_u / sqrt(d_head))
message_v = sum_(u,r) attention_uvr W_M[r] W_V[type(u)] h_u
~~~

Use standard multihead/type-specific output projections, residuals, normalization and feed-forward layers, retaining per-node memory H. Record the actual library/operator/version. Match hidden width, depth, readouts, training budget and evidence for HGT/R-GCN/no-message-passing comparisons. The historical d=128, three layers, four heads and dropout 0.1 are retained only as development starting settings in [protocol guidance](protocol/README.md), not established optima.

**NN-012 — Object attention.** For each editable mapping/axiom occurrence i, compose a role-aware target query from its node and original syntax; attend to the relevant admitted node rows; produce shared context c_i:

~~~text
q_i = target_MLP(h_i || role_aggregate(original_arguments_i))
r_i = object_attention(q_i, H[context(i)])
c_i = context_MLP(q_i || r_i || allowed_status_features_i)
~~~

Read individual statements/supports and evidence directly. Avoid making a single pooled ontology vector the sole source for every decision. Context scopes include complete admitted explanation supports and declared query/task context. Cached contexts bind to graph and model revision.

**NN-013 — Complete candidate composition.** Compose every emitted axiom and active non-vacuity obligation with shared syntax modules. Named entities read their contextual memory; restrictions distinguish property and filler; directed inclusions distinguish subclass/superclass; intersection/disjoint operands are canonical and permutation invariant. Bundle aggregation preserves emitted-versus-activated roles and does not silently discard repeated logical effects with distinct obligations.

~~~text
e_ia = bundle_encoder(emitted_axioms_ia, active_obligations_ia, H)
q_ia = candidate_MLP(c_i || e_ia)
r_ia = candidate_attention(q_ia, H[context(i) union context(candidate_ia)])
h_ia = candidate_context_MLP(c_i || e_ia || r_ia || allowed_candidate_features)
~~~

The complete-bundle representation distinguishes deletion, one retained direction, specialization and complex equivalence. Explicit structural cost features are routed to the cost calculation; they must not be added to semantic targets as hidden penalties. Syntax features that also correlate with cost remain legitimate semantic context, but the direct penalty is subtracted once.

## 6. Proposal and semantic heads

**NN-014 — Proposal placement and probability.** Shared post-encoder proposal heads consume c_i, retrieved menu embeddings, template/slot embeddings and declared conditioning. They output mixture logits and all Boolean-choice logits before a circuit invocation. Entity selection shares parameters across observed vocabulary, for example `(W_slot c_i)^T W_entity h_C`. The circuit conditions this within-object distribution on fixed grammar and encoded contextual constraints; its normalizer and bundle inverse-encoding sum follow [10](10-constrained-generation.md).

A mixture can capture dependencies between Boolean choices inside an object. Independent per-object calls do not produce learned cross-object action correlation, even if all read the same graph or use the same mixture count. Shared random seeds are not a joint model. Log probabilities, semantic values and risk predictions have separate output fields.

**NN-015 — Optional plan context.** A plan-conditioned extension adds a representation of a hypothetical already selected prefix or explicitly masked partial assignment to the proposal context. Record the order, complete chosen bundles, unassigned mask and per-call context hash. Freeze that context before computing all circuit logits for the call; do not feed a partial Boolean circuit traversal back into them while claiming the original conditioned-mixture likelihood. Training and inference use the same available-prefix convention.

Bound refinement rounds and pool growth. Generate several hypothetical plans, union candidates with protected controls, apply declared ablations/caps and freeze the final inventory. A proposed prefix is neither verified nor a constraint on the final MaxSAT repair. Re-score final candidates/pairs in the shared final graph/context, not using incompatible scores inherited from different hypothetical plans. The main independent generator remains a required control.

**NN-016 — Semantic task and costs.** Unary and pair heads estimate the declared benefit/fidelity target on its calibrated scale. A semantic-task descriptor may condition both proposal and value heads when the competency queries/criterion weighting genuinely change and are available at inference. Explicit edit-cost settings may condition proposals to cover useful trade-offs. Cost-only changes do not redefine semantic benefit: the value target remains invariant and `lambda^T cost` is subtracted outside the heads. Personal user-profile learning is deferred; it is not the main supervision task in this revision.

## 7. Shared sparse interactions

**NN-017 — One selector.** Implement one deterministic pair-selection routine used by training, development, direct inference and prepared-input inference. Its inputs are the exact observable graph/evidence revision, admitted initial explanations, canonical **final** candidate inventory, query/task context if visible, and frozen limits. Its output is an ordered object-pair index P with inclusion reasons, considered channels, budget/omission counts and content hash. Match hashes when those inputs match. The baseline training path omitted retrieved explanations; the production path supplied them. An explanation-only pair must not remain untrained and appear silently at deployment.

**NN-018 — Selection channels and caps.** Consider pairs linked by original shared entities/axioms; complete admitted explanation supports; declared visible query dependencies; and entities/dependencies introduced by candidate bundles or active obligations. Candidate-induced connections are derived from observable syntax, not teacher knowledge that a pair is useful. Use candidate signature indexes and bounded support traversal; do not materialize an unbounded all-object/all-candidate Cartesian graph before truncation.

The run manifest declares deterministic channel precedence, tie rules, per-object degree, total object pairs, candidate-factor count and any support-incidence limits. Prioritize explanation support before generic high-degree neighborhood expansion unless the frozen manifest explicitly compares another policy. Preserve full hyperedge identity even when its pair expansion is capped. Report each omitted channel/link rather than imply all interactions are modeled. A pair matrix requires `C_i*C_j` coefficients; reduce pair/candidate budgets explicitly before freezing if this exceeds the factor budget. New graph/candidates/supports after freezing require a new selection/score round.

**NN-019 — Pair head and symmetry.** The shared pair head consumes both complete candidate representations and an observable joint context containing support/path/query information when available. A symmetric construction can combine sums/products of projected h_ia/h_jb with an order-invariant support readout; canonical object ordering preserves source/target roles inside each candidate. Swapping the two objects/candidates together must preserve the coefficient. This head is applied for every candidate combination in each selected object pair, not once per explanation node.

~~~text
raw_pair_ij(a,b) = pair_MLP(sym(h_ia,h_jb) || joint_context_ij)
B_hat(R) = beta_case + sum_i u_i(a_i) + sum_(i,j in P) v_ij(a_i,a_j)
U_hat(R) = B_hat(R) - lambda^T sum_i cost_i(a_i)
~~~

The bias beta_case cancels for within-case selection and anchored losses. If omitted from the objective, record the offset convention; do not portray the displayed shifted score as an absolute semantic percentage.

**NN-020 — Baseline gauge.** Choose a deterministic reference action a_i0 for each object, normally its required keep action. Define gauge-fixed pair effects by:

~~~text
v_ij(a,b) = raw_pair_ij(a,b) - raw_pair_ij(a,a_j0)
            - raw_pair_ij(a_i0,b) + raw_pair_ij(a_i0,a_j0)
~~~

This gives zero interaction whenever either choice is its reference. Gauge-fix unary contributions relative to their references too. Train that architecture directly, or transfer removed row/column terms into unary factors and the case offset when converting an already learned unrestricted decomposition; simply deleting those terms changes the objective. The reference assignment is a parameterization device and need not be feasible. Verified feasible anchors for semantic scale are a separate training requirement. Record gauge/version in checkpoints/objectives.

**NN-021 — Expressiveness and supervision.** Train through whole-plan semantic losses and the eligible quartet contrasts in [07](07-corpus-and-training.md). A GNN-conditioned unary sum cannot generally represent nonadditivity among the chosen actions. Pairwise factors represent complementarity/redundancy on the selected pair set, not arbitrary higher-order semantics. Use the two-mapping chain `(0,0,0,1)` and redundant-benefit `(0,1,1,1)` tables as required counterexamples. Report omitted-pair and higher-order residuals. A future higher-order utility head is a separate architecture requiring a bounded exactly encoded objective; a whole-plan neural reranker is not silently an exact MaxSAT objective.

## 8. Risk, support and effort auxiliaries

**NN-022 — Whole-plan input.** The risk head receives a proposed complete assignment as selected candidate embeddings plus graph/task context and typed aggregation of applicable support hyperedges. It can use a permutation-invariant set/attention readout and does not need a new optimizer. Its supervised target is decided policy failure under the recorded scope; unknown is masked. Its input includes all selected bundles, so an ontology edit can alter compatibility. A candidate cannot be globally marked unusable merely because one plan containing it failed.

**NN-023 — Support hyperedges.** Retain a support as one typed set of candidate/axiom-occurrence literals with witnessed obligation and dependencies. Aggregate the whole support before predicting its target; do not replace a three-way conflict by three pair negatives. Support extraction incompleteness produces a missing label, not a non-conflict label. The learned support prediction is soft risk information. The kernel accepts only its separately validated symbolic evidence for hard exclusions. Inputs obtained after the decision remain labels for that decision and acquire inference eligibility only in an explicitly new round with valid dependencies.

**NN-024 — Calibration and effort.** Report raw scores as uncalibrated until development-only calibration is measured on a representative labeled assignment sample. Calibrate separately for changed policy/cohort/backend scope when supported; out-of-domain risk is flagged. Report reliability/coverage by feasibility stratum, not just aggregate classification accuracy. Optional effort prediction records completed versus censored verifier time, actual underlying calls versus cache hits, backend/hardware/context and prediction target. A timeout is not a negative runtime or failed-feasibility label.

## 9. Scheduling and exact selection

**NN-025 — Bounded shortlist only.** MaxSAT owns assignment generation and the primary frozen objective. An optional scheduler obtains a bounded shortlist of distinct assignments under the current proof state, using temporary scheduling blocks that do not enter the logical clause set. Risk/effort scores may reorder verification within that shortlist and propose verified incumbents sooner. Record shortlist size, original objective ranks/scores, scheduling rule, chosen order and every unverified/pending assignment. Scheduling exclusions are reversible bookkeeping, never learned infeasibility clauses.

Reordering lower-utility plans ahead of the current primary optimum is permitted only if the kernel retains every unverified alternative in its outstanding-bound calculation. A good incumbent does not establish optimality until higher-scoring possibilities are ruled out/resolved under the kernel contract. Risk may break equal-primary-utility ties without changing the primary objective. Subtracting risk or verification cost from U is an explicit different objective/ablation and cannot claim the original semantic optimum. Bounded scheduling without completion may improve time to incumbent but must retain an unresolved gap/status.

**NN-026 — Freeze and replay.** Before a solve, freeze candidate inventory, graph/features/menus, semantic task/scale, pair index/gauge, checkpoint, explicit costs, integer quantization and coefficient tensors into `ObjectiveV3`. Risk checkpoint/scheduling policy are frozen separately for reproducibility; they cannot mutate coefficients during verification. Training/dropout are disabled for inference. A graph/proposal/model refresh begins a new round with a new objective hash and valid evidence transfer. Replay of feasibility does not require the model; replay of optimization consumes the recorded coefficients and proof state.

## 10. Architecture acceptance

**NN-027 — Required checks.** These are implementation acceptance obligations, not results of this document revision.

| Test ID | Required observation |
|---|---|
| NN-T01 | Typed node/edge and every head shape matches its schema; unseen schema/checkpoint combinations fail explicitly. |
| NN-T02 | Operand permutation and pair exchange invariance hold; subclass/superclass and property/filler swaps remain distinguishable. |
| NN-T03 | Every emitted axiom/active obligation affects the complete candidate representation; adding explicit cost alone does not alter semantic target. |
| NN-T04 | Explanation-only, original-signature and candidate-induced pairs match across training, development and inference; omissions and cap hits are reported. |
| NN-T05 | Pair gauge is zero in each reference row/column; a converted decomposition preserves all assignment scores up to the recorded constant. |
| NN-T06 | Unary cannot fit the nonadditive fixture; enabled pair factors can; an omitted pair is not claimed learned. |
| NN-T07 | Decided infeasible plans give risk gradients; unknown plans do not give whole-policy gradients; three-way supports remain hyperedges. |
| NN-T08 | Independent proposal products cannot distinguish equal-marginal different-joint teachers; conditional prefix inputs are recorded and logits remain fixed per call. |
| NN-T09 | Endpoint materialization is idempotent and identical on direct/prepared paths; out-of-memory candidate symbols never receive silent dummy context. |
| NN-T10 | Feature tests distinguish zero/missing/truncated/unsupported values; nested evaluator aliases and post-decision evidence are rejected. |
| NN-T11 | Opaque renaming preserves structural-control behavior; parent/seed/label identifiers do not become lexical features. |
| NN-T12 | Risk shortlist ordering may alter verification order but preserves the primary optimum/bound on a fully decidable exhaustive fixture. |
| NN-T13 | A pending high-utility assignment remains in the bound when risk favors a lower-utility verified incumbent; no learned hard cut is emitted. |
| NN-T14 | Rescoring the same final pool is independent of the hypothetical generation path; graph/model changes require a new frozen round. |
| NN-T15 | HGT, R-GCN and no-message-passing controls share readouts/evidence; unmatched parameter or runtime differences are disclosed. |
| NN-T16 | Scalar fidelity calibration, risk calibration and explicit cost subtraction are separately verified and provenance-tagged. |

**NN-028 — Claim boundary.** Passing architecture checks establishes contract conformance on those tests. It does not establish semantic-fidelity generalization, optimal real-world meaning, a smaller verifier workload or calibration outside the measured distribution. Such claims require the held-out generated-pool evaluation and independent semantic judgments in [07](07-corpus-and-training.md) and [13](13-semantic-fidelity-supervision.md).
