# XR-2.1 constrained generation and contextual semantic circuits

Revision: **XR-2.1 — 30 September 2026**. This is a target specification, not a declaration that the revised generator is implemented. The inspected runtime baseline is commit `b4c1ed0d5e12c45974bdcb4d230fb2ab6c6deb04`. The frozen XR-2 protocol JSON files retain their historical meaning and bytes; this revision does not silently change a completed experiment.

## 1. Purpose, evidence and guarantees

**CG-001 — Role.** Retain small per-template or per-family SDDs as the preferred initial implementation. Their purpose is to sample complete finite replacement bundles under typed grammar constraints and explicitly encoded, proof-supported contextual constraints. The graph model supplies weights; the circuit represents the admissible event. The circuit is not the ontology graph, a complete ontology reasoner, or a circuit for every coherent alignment.

**CG-002 — Evidence boundary.** The first generated-only pilot recorded generation failures on 79 of 90 development cases at each selected checkpoint: 66 allocated-SDD-node-limit failures and 13 compilation deadlines. Successful cached-inventory decoding on those cases does not establish successful generation. At the inspected baseline, `grammar.py` uses a right-linear vtree, disables automatic collection/minimization, checks all allocated manager nodes, and collects only after referencing the final root. `compilation.py` has a process-local cache; `pipeline.py` starts separate supervised generation workers. These are reasons to measure and repair the implementation, not proof that collection, a different vtree, or splitting will solve every case. The audit and migration obligations are in [12](12-implementation-migration.md).

**CG-003 — Three distinct guarantees.** Every result must state separately:

1. **Grammar validity:** well-typed, canonical replacement syntax, applicable templates, declared finite symbols and bounds.
2. **Encoded contextual admissibility:** satisfaction of the particular semantic conditions whose dependencies and proofs are captured for this language.
3. **Global repair feasibility:** satisfaction of the complete policy after all selected bundles are asserted together, established by the qualified verifier in [01](01-architecture-and-contracts.md) and [04](04-minimal-exact-kernel.md).

Exact circuit conditioning establishes the first two only to the extent that their conditions are encoded correctly. A sound incomplete detector's failure to find a contradiction is not proof of satisfiability. Even individually admissible candidates can conflict when combined. A complete supported fragment may justify a stronger claim only for its explicitly stated theory, queries and assignment scope; it does not silently extend to arbitrary OWL.

**CG-004 — Comparative justification.** Retain circuits on empirical evidence of useful contextual filtering, exact conditioning, reusable compilation, or better verified repair quality per total budget. Syntax alone is insufficient evidence of an advantage over a matched grammar decoder. Any alternative is evaluated after the repaired small-SDD route, using the same semantic constraints and candidate language; an adverse result is an acceptable research outcome. Prior SPL/knowledge-compilation work motivates the representation, not a claim of novel tractability; see [references.md](references.md).

## 2. Inputs, outputs and identities

**CG-005 — Frozen generation request.** One request must capture, before model inference:

| Input | Required content |
|---|---|
| Revision object | Stable object/occurrence identity, original asserted axiom bundle, entity kinds and roles, eligibility/lock state, side provenance |
| Background | Exact immutable asserted theory/import closure identity; declared editable occurrences; duplicate-occurrence provenance |
| Grammar | Enabled actions, explicit retention modes, canonicalization version, expression/template bounds, side-specific class/property menus, typed endpoint alternatives |
| Context constraints | Formula/rule identifiers, proof records and immutable supports, qualification scope; unresolved proposals remain labelled unresolved |
| Retrieval | Observed evidence and configuration, ranking/truncation records, selected and omitted symbols, no evaluator-only answers |
| Neural state | Checkpoint and graph identities, observable features, semantic-task and explicit cost-setting identities, mixture count and all logits fixed before this draw batch |
| Execution | Expansion schedule, draw and pool budgets, compiler/vtree policy, compilation/evaluation/RSS deadlines, seed and schema versions |

**CG-006 — Output envelope.** Use the target `ProposalRecordV3` / `GenerationReportV3` boundary from [01](01-architecture-and-contracts.md), without rewriting runtime/archive V2 records. Return object and language identities; status per family and expansion round; immutable SDD/vtree artifact identities; symbol/variable binding; proof dependencies; numeric normalizer status; draw provenance; all canonical candidate bundles with active-expression obligations, costs and merged derivation provenance; attempted/accepted/unique/retained counts; mandatory-control accounting; truncations; and resource telemetry from section 7. Distinguish an exact distribution over a declared support from an incomplete sample of that support. Do not publish a partially computed objective as a complete frozen objective.

**CG-007 — Identity separation.** Keep distinct hashes for observable input, retrieved menus, grammar, contextual proof constraints, Boolean encoding, compiler artifact, neural checkpoint/logits, sampled pool and final frozen objective. A compiler artifact contains no teacher labels or neural weights. A changed background, activation rule, proof support, canonicalization convention, candidate cost identity or binding must invalidate every dependent artifact, without needlessly invalidating unrelated structural artifacts.

## 3. Complete replacement language

**CG-008 — Finite expression grammar.** For finite observed class and object-property menus `C_i` and `R_i`:

~~~text
E ::= C | E AND E | EXISTS r.E
C in C_i; r in R_i
~~~

Root depth is zero; named leaves have zero constructor cost; intersection/existential nodes add one plus child costs. Flatten, deduplicate and sort intersections and give them a fixed binary association. Bounds apply to the canonical generated argument and every activated expression after attaching fixed endpoints. In particular, a small argument does not excuse an oversized `S AND E`. Preserve all complete-bundle action semantics and side mirrors in [06](06-first-experiment-implementation.md); do not reduce complex equivalence to a single specialised inclusion.

**CG-009 — Typed slots.** Slots contain constructor, class, property and a unique unused choice. One-hot choices, active-child structure, exact inactive-field values, depth/size, canonical order, side restrictions and template/retained-direction semantics define the grammar event. Class/property/individual mapping kinds are not interchangeable. Unsupported constructors remain in the verification theory; they are not fabricated by this initial generator. Observed menus may include a previously unused combination of existing terms; arbitrary new domain concepts are outside this language. Private verifier probes are separate from generated public concepts.

**CG-010 — Elementary availability.** Construct keep, eligible delete, applicable retained directions, and every declared enabled typed endpoint alternative deterministically, independently of sampling and compiler success. Keep preserves the original complete bundle. Locked/ineligible objects permit only their authorized unchanged state. Deterministic availability does not force selection: the exact master still chooses one candidate per object and the verifier may reject a candidate or combination. A proof that a mandatory control cannot meet the policy is recorded as a justified feasibility restriction, not hidden disappearance from the inventory. Do not claim that elementary availability guarantees a feasible repair.

**CG-011 — Candidate identity and aliases.** A candidate's semantic content for deduplication includes its complete canonical emitted axiom set and activated expressions. Preserve occurrence identity and explicit retention semantics. Merge compatible action/provenance aliases without double-counting a decision or losing the origin of a cost; conflicting metadata requires resolution under the records contract, not an arbitrary winner. Canonical syntax is not a decision procedure for logical equivalence. If more than one valid encoding emits the same bundle/activation, candidate probability is the sum of all those encodings' probabilities.

## 4. Proof-supported contextual constraints

**CG-012 — Immutable background.** Let `B` be the captured asserted theory whose relevant occurrences cannot be removed or replaced in this solve, including fixed imported copies and locked facts. A closure used to derive constraints must be proved from that `B` and bound to the reasoner/detector qualification and query semantics. Baseline diagnosis must expose inconsistent or policy-incompatible backgrounds; vacuous entailment from an inconsistent background must not be presented as useful evidence of meaningful weakening. A whole repaired theory may still be globally infeasible even when the local proof computation is valid.

**CG-013 — Admissible hard local conditions.** Examples include an intrinsically valid weakening template, a proved fixed subsumption required for a generalisation action, or a proved-unsatisfiable active expression relative to `B`. Every nontrivial semantic restriction records its conclusion, sufficient asserted support, activation conditions if any, background/import hash, engine/rule version, and qualification scope. Minimal justifications are optional. Learned scores, lexical similarity, observed co-occurrence, missing evidence and an unqualified solver response never become logical hard constraints. A proved entailment can authorize a weakening claim; inability to prove it yields `weakening_unproved`, not the opposite entailment.

**CG-014 — Compilation boundary.** Compile `K = G AND H_B`, where `G` is the declared grammar and `H_B` consists solely of qualified local conditions whose supports remain immutable. Restrict an action using a proof only when the declared policy makes the proven condition relevant. Candidate construction may retain a syntactically valid general revision without a weakening label only through a separately authorized general-revision template; it must not bypass an action whose prerequisite is the missing proof. A timeout or unsupported semantic check supplies no ban and no positive admissibility proof; report whether the resulting support is grammar-only or has the stated partial contextual filters. Syntactic filtering is always mandatory.

**CG-015 — Editable support.** If a proof uses an editable axiom, do not permanently remove the expression from an unconditional local language. Submit the support and relevant activation to the global verifier/master interface. A valid global constraint has the form `NOT (all supporting axioms present AND triggering activation/selection)`. Axiom presence means present from any selected bundle or immutable duplicate, not merely that the original object was kept. If sampling is explicitly conditioned on a hypothetical partial assignment, dependent filtering is permitted only under that captured condition; the candidate remains available outside it. Recompute/withdraw conditional restrictions when their condition changes. Ordinary presence cuts do not justify excluding a plan merely because a required entailment is absent.

**CG-016 — Worked contextual example.** Suppose immutable `B` contains:

~~~text
Range(hasDecision, Accepted)
Accepted AND Rejected SubClassOf Nothing
~~~

Then `B` entails `EXISTS hasDecision.Rejected SubClassOf Nothing`: every required successor would be both Accepted and Rejected. Consequently, a candidate activating `Paper AND EXISTS hasDecision.Rejected` violates an obligation requiring that expression to be satisfiable. The corresponding local branch may be excluded by this proof. Without the satisfiability obligation, the inclusion `Paper AND EXISTS hasDecision.Rejected SubClassOf T` could hold vacuously; emptiness alone does not make that inclusion inconsistent. If either support axiom is editable, replacing it may make the activated expression satisfiable, so use the conditional global restriction from CG-015. If fixed duplicate copies remain, their presence keeps the proof applicable. The argument uses OWL object-property range semantics, not the absence of a decision record.

**CG-017 — Limits.** A circuit accepting a candidate means that it satisfies the exact encoded conditions, not that the candidate is an intended or globally coherent repair. In a finite decidable setting one could classify every complete assignment and compile the safe set, but that does not avoid the expense of discovering the set or guarantee a compact circuit. The complete global verifier remains the acceptance authority.

## 5. Probability contract and per-family factorization

**CG-018 — Reference distribution.** For full Boolean encoding `z`, captured observable context `c` and declared task/cost conditioning `u`, compute all weights before sampling:

~~~text
q_m(z | c,u) = product_j Bernoulli(z_j; sigmoid(logit_mj))
q(z) = sum_m pi_m q_m(z), where pi = softmax(component_logits)
Z_m = sum_{z:K(z)} q_m(z)
Z = sum_m pi_m Z_m
p(z | c,u,K) = 1[K(z)] q(z) / Z
~~~

`M=1` is the independent-product control; a small mixture can express limited correlations. This is not an arbitrary autoregressive distribution with a tractable global normalizer. Freezing the logits is required for this formula; adaptive draws using new logits are separate identified distributions. Conditioning follows [09](09-graph-and-neural-model.md): semantic task descriptors and explicit cost settings are separate; personalisation is optional future work. Explicit edit costs must not be counted twice.

**CG-019 — Disjoint family partition.** Split `K` into mutually exclusive template/retention branches or disjoint families of such branches. For branch `f`, eliminate variables fixed by that branch and retain only its actual expression/menu variables. Let `a_mf` be the product of every eliminated fixed literal's weight, including false template alternatives, retained-direction choices, unused slots and out-of-menu Boolean choices. Let `Z_mf` be the weighted count of the remaining branch constraint under component `m`. Then:

~~~text
Z_m = sum_f a_mf Z_mf
Z = sum_m sum_f pi_m a_mf Z_mf
P(m,f | K) = pi_m a_mf Z_mf / Z
P(z_remaining | m,f,K) = q_m(z_remaining) 1[K_f] / Z_mf
~~~

A genuinely unconstrained eliminated variable sums to one under normalized Bernoulli weights; preserve its conditional draw if the full assignment is later reconstructed. Do not confuse such a free variable with a uniquely unused field, whose fixed-bit mass must be included. Families with zero valid mass receive zero probability. The partition is disjoint over encodings even if different branches emit the same canonical candidate.

**CG-020 — Preserve or declare a new distribution.** Dropping inactive-bit factors, selecting a family uniformly, retaining `pi_m` instead of its conditioned posterior, or normalizing every family independently without the branch-mass correction generally changes the distribution. A hierarchical learned family model is permitted only as a new named/versioned model with its own likelihood and supervision. The initial refactoring must reproduce the reference distribution on identical languages and weights. Removing redundant deterministic variables must preserve both support and mass, not just successful decoding.

**CG-021 — WMC and sampling.** Use a qualified existing SDD compiler. Decomposable AND children have disjoint scopes; deterministic OR alternatives have disjoint satisfying assignments. Multiply child masses and sum alternatives; account for missing scopes correctly. Compute all normalizers in stable log arithmetic, including gradients through `log Z`. Sample `(m,f)` using CG-019, then conditioned local branches. Finally evaluate a candidate's probability as the sum over all valid inverse encodings, across families/components as required. An incomplete inverse map is a likelihood error even when every sampled candidate is syntactically valid.

**CG-022 — Complexity and zero mass.** Compilation can be exponential. Evaluation is linear in the compiled DAG's edge/element count per component; reporting only its number of nodes can hide large decision fan-out. Exact normalization/sampling does not imply efficient exact mixture MAP. Distinguish logically empty support from nonempty support assigned zero weight. `Z=0` is not a distribution and never authorizes hard-constraint relaxation; elementary inventory construction still stands. Nonfinite or underflowed arithmetic requires a numeric failure or corrected stable evaluation, not treating a tiny nonzero event as logically impossible.

## 6. Preferred generation algorithm

**CG-023 — Ordered stages.** The target pipeline is:

1. Validate revision eligibility, complete original bundles, immutable/editable occurrence partition and budgets; construct deterministic elementary alternatives.
2. Capture observed side-specific retrieval, typed endpoint alternatives and explicit omissions. Build the graph and compute the frozen neural parameters.
3. Build applicable template/family schemas and the proof-supported contextual constraints. Compile or load each required small SDD under its explicit budget. A family may share a circuit only when its schema and binding are demonstrably equivalent.
4. Evaluate branch/component masses, validate support and normalizers, and sample according to the named distribution. Keep every family status even when another family succeeds.
5. Decode and independently validate emitted bundles/activations, aggregate duplicate probability/provenance, and combine samples with mandatory controls and authorized family representatives.
6. Apply the predeclared candidate cap. Preserve mandatory elementary states and an available representative per enabled family before rank filling; an insufficient cap is `invalid_budget`. A family with no applicable/retrieved/admissible candidate has its explicit reason, not a fabricated representative.
7. Freeze the final inventory and recompute every unary/pair factor against that exact inventory. Submit it to the master and qualified verifier. Changing effective coverage or the inventory starts a new identified epoch/solve with a newly frozen objective and recomputed bounds; it never silently extends an already certified inventory.

**CG-024 — Candidate-space expansion.** Define an ordered schedule before observing test answers: increases in retrieved menus, enabled expression depth/constructors, draws or pool cap are separately recorded. Report declared language coverage, actual sampled coverage and conditional useful coverage separately. Prefer nested language/pool expansions where feasible; preserve prior candidates, or explicitly identify a nonnested capped replacement. Expansion is not exhaustive search unless exhaustion is proved. Never respond to resource exhaustion by silently shrinking a language and calling the result a completed run of the original configuration.

**CG-025 — Reduced-support operation.** If only some family circuits complete, retain all deterministic controls and completed candidates. A predeclared fallback may solve that reduced inventory, but its generation status stays partial/failed as appropriate and its exactness claim is limited to that frozen inventory. Sampling with only completed families conditions on a different support and must be labelled accordingly; it is not an exact sample from the intended all-family distribution. Do not opportunistically report a likelihood or proposal-training loss with unknown omitted-family mass. Existing verified incumbents may survive a later expansion failure, bound to their own recorded inventory and proof.

## 7. Compiler engineering and resource supervision

**CG-026 — Native lifetime audit.** Audit `ref`/`deref` ownership of every retained accumulator, constraint, memoized comparison/bounds result and root. Reference a replacement before releasing the previous live root; do not collect an object still used by a Python container or local cache merely because Python retains its wrapper. Enable collection only after ownership is explicit, and cover cancellation, exceptions and cache eviction. Automatic collection/minimization is not a safe one-line fix. Start with controlled collection boundaries; evaluate bounded minimization separately so its cost is observable.

**CG-027 — Distinct measurements.** At construction phases, before/after collection, after minimization, serialization and restoration, record allocated manager nodes, native live/dead nodes, root-reachable nodes, SDD elements/edges, serialized bytes, peak worker RSS, and elapsed/CPU time. Retain phase and limit identifiers for every failure. Peak allocation/RSS and final reachable circuit size answer different questions. A safe hard memory cap must not be replaced by counting only reachable nodes; conversely, an allocated-node failure is not evidence that the final support needs that many live nodes. Unreferenced-but-needed nodes make native live counts misleading until CG-026 is satisfied.

**CG-028 — Vtree comparison.** Compare the historical right-linear layout with balanced and grammar-structure/grouped layouts, plus separately budgeted minimization if qualified. Group one-hot fields and related subtree/template decisions according to a documented variable map. Variable order alone does not specify the complete vtree. Preserve the same accepted encodings and reference probabilities across layouts; tune layouts on training/development only and freeze the selection rule for held-out runs. Include all attempted layouts in resource totals, not only the winning artifact.

**CG-029 — Hard boundaries.** Native apply/count/minimize/save operations and their model counting run in independently killable workers. Deadlines and supported RSS supervision include input transfer, startup, compilation, minimization, serialization, transport and restoration; otherwise report any excluded stage explicitly and enforce its separate limit. Keep cooperative checks for diagnostics, but do not rely on them to interrupt a blocking native operation. Cancellation must terminate descendants and prevent partial artifacts becoming cache hits. Expose memory supervision unavailable/unsupported rather than claiming a cap is enforced. Persistent caches must not turn an unbounded parent read or reconstruction into a deadline bypass.

## 8. Persistent immutable compilation cache

**CG-030 — Exact cache first.** Persist atomically written immutable SDD/vtree/evaluation-DAG artifacts across generation workers. Verify content hashes, versioned schema, variable/field binding, support/constraint identity, compiler and qualification versions, canonicalization, vtree and serialization format before reuse. Keep compilation limits/attempt identity and eligibility for the requested resource regime in the receipt: a cached artifact does not count as a successful cold compile under a smaller limit. Cache corruption is a miss or explicit failure, never permission to trust unverified bytes. Failed/partial compilations are not successful cache entries.

**CG-031 — Structural reuse second.** Only after exact-identity reuse passes conformance may an alpha-renamed schema cache be added. Its key must capture template/action and retained directions; vocabulary cardinalities and side membership; equality/shared-symbol patterns; fixed endpoint/expression structure; total order used by canonical conjunctions; bounds; contextual constraint/proof pattern; and compiler/vtree versions. Store and verify a bijective symbol/variable binding. Same vocabulary sizes do not prove isomorphism. Non-order-preserving renaming requires a correspondingly transformed canonicalization/order relation or recompilation. Reusing an abstract formula never transfers an ontology proof: bind each semantic support to the current immutable background and revalidate the proof identity.

**CG-032 — Cache fairness and capacity.** Neural values never enter structural cache keys, and new neural weights reevaluate the circuit. Record exact versus schema hits, lookup/load/reconstruction cost, cache capacity/eviction and bytes, failed lookup reasons and cold/warm totals. Set bounded disk/memory capacity. Share immutable read-only artifacts across workers; do not share unsafe native-manager handles. A warm-cache experiment must disclose how its cache was populated and charge/predeclare that cost for the intended deployment scenario.

## 9. Failure and learning contracts

**CG-033 — Status taxonomy.** The V3 top-level generation status follows [01](01-architecture-and-contracts.md): `COMPLETE_DECLARED_ENUMERATION` only for proved exhaustive completion of the declared finite language; `SAMPLED` for a completed scheduled sampling procedure without an exhaustion claim; `PARTIAL_RESOURCE_LIMIT` when resource limits leave requested generation unresolved; `INVALID_LANGUAGE` for invalid language/input declarations; and `ERROR` for operational or numerical failures. A logically empty but exhaustively resolved language may have `COMPLETE_DECLARED_ENUMERATION` with zero candidates; a conditional sampler with zero probability cannot report a successful draw. Preserve the original legacy status alongside its explicit adapter; never relabel a historical run as newly qualified.

Separate family/stage detail fields distinguish the following states; these detail labels do not replace the canonical V3 status:

| State | Required interpretation/action |
|---|---|
| `resolved` | Required family/stage work resolved; identify enumeration versus sampling explicitly |
| `partial` | Some declared families/coverage expansions unresolved; retain the cause and canonical top-level status |
| `not_applicable` | Template excluded by object kind, relation or eligibility; not a compiler success |
| `retrieval_empty` / `retrieval_truncated` | Measured menu availability/coverage limit; no semantic conclusion |
| `proof_unavailable` / `weakening_unproved` | Context claim unresolved; no invented ban or weakening claim |
| `empty_language` | Qualified proof/compilation establishes no valid encoding in the specified language |
| `zero_probability` / `numeric_failure` | Support/weights/arithmetic issue; never conflated with semantic inconsistency |
| `compile_node_limit` / `compile_memory_limit` / `compile_timeout` | Resource failure with phase and measured totals |
| `artifact_invalid` / `binding_invalid` | Invalid cached/transported structure; reject before use |
| `invalid_budget` / `input_invalid` | Contract violation, including a cap below mandatory entries |
| `cancelled` / `worker_error` | Operational outcome; not a logical verdict |

Generation status, proof-constraint coverage, verification status and optimization status are independent fields. Completion of a sampling procedure does not mean feasible, optimal, exhaustive, or semantically intended. Map invalid budgets to `INVALID_LANGUAGE`, resource exhaustion to `PARTIAL_RESOURCE_LIMIT`, and corrupt transport/numerical/worker failures to `ERROR`, while retaining per-family details and any independently completed pool artifacts. Cancellation records its operational cause rather than a logical verdict.

**CG-034 — Shared phase contract.** Preparation, training, development and inference must call one captured retrieval/template/endpoint/constraint-construction contract with identical declared semantics. Differences in draws, budgets or graphs must be explicitly configured and reported. Materialize newly retrieved typed endpoint alternatives on every path; passing them only to the graph or relying on pre-existing teacher candidates is insufficient. Never use teacher answers to repair a missing vocabulary menu in an ordinary arm.

**CG-035 — Proposal supervision coverage.** Record at object/case/family level whether a proposal loss was actually evaluated, whether a complete teacher target was representable, missing target mass and all compile failures. Do not report successful value training as successful proposal supervision. Suppressing repeat attempts for a failed compilation is an explicit bounded retry/cache policy, with its own identity and counts. Training/dev/inference must not silently use different proposal distributions or renormalize away unreachable teacher targets. A deliberately restricted-target loss is a new arm, with missing mass retained in evaluation.

**CG-036 — Omission controls.** A missing-candidate control must remove its designated bundle/activation identity from the final proposed inventory after all regeneration and deduplication routes, then assert absence before selection. This evaluator-only intervention is withheld from learned features and ordinary proposal conditioning. Its outcome diagnoses selection under known missing coverage; it is not a natural retrieval miss. Separate missing-symbol/constructor controls alter the declared observable menu/language and verify actual unreachability. Record which intervention was used, any equivalent aliases, and whether logical alternatives remain; absence of one syntax is not absence of all semantically equivalent repairs.

**CG-037 — Checkpoint evidence.** Report cached-inventory regret separately from generated-pool useful coverage, complete generation fraction, verified semantic quality, feasibility and calls/time to a verified repair and to matched quality. Use all scheduled cases in failure/completion denominators. Select checkpoints with a development-only predeclared rule containing a generation-coverage eligibility gate and verified generated-pool quality, with missing/unknown outcomes preserved. A model with successful cached decoding and pervasive proposal failure is not a completed end-to-end generator. The objective, semantic labels and no-leakage rules are specified in [07](07-corpus-and-training.md) and [13](13-semantic-fidelity-supervision.md).

## 10. Qualification and acceptance

**CG-038 — Finite conformance oracle.** On deliberately tiny languages, exhaustively enumerate assignments and canonical emitted candidates independently of the optimized compiler. Compare syntax acceptance, template roles, mirrors, retention, unused-field uniqueness, canonical bounds including attached endpoints, support/proof filtering, duplicate inverse maps, normalizers, branch/component posteriors and complete candidate likelihoods. Include zero-mass components, contradictory support constraints, fixed-bit elimination, free-variable restoration and identical bundles emitted by distinct templates. Test sums to one and zero outside support; sampled frequencies use stated statistical tolerances, not exact-frequency assertions. Check analytic/autograd gradients against finite differences and brute-force likelihood including `log Z`.

**CG-039 — Semantic regression fixtures.** Include CG-016 with immutable versus editable range/disjointness and surviving duplicate copies; a proved true weakening and an unproved generalisation premise; unsupported expressions and timed-out detectors; two locally admissible candidates with an unsafe joint combination; and the invited-paper necessary-condition counterexample from [06](06-first-experiment-implementation.md). A semantic decoder comparator must receive the same proof constraints. Show explicitly that detector silence and grammar acceptance never produce an acceptance certificate.

**CG-040 — Engineering regression fixtures.** Exercise aggressive controlled GC and exception paths, every vtree layout, cache eviction/corruption, cross-worker exact reuse, schema-binding permutations, non-isomorphic same-sized menus, changed background/activation/canonicalization, cancellation during native apply/save/restore, and enforced versus unavailable memory supervision. Endpoint alternatives must survive to the final pool; omission targets must not. Mandatory controls survive all generation-failure paths where input construction is valid, with invalid cap failures made explicit.

**CG-041 — Pilot repair gate.** Treat the 90 exposed development cases as engineering regression data. Before claiming that this pilot bottleneck is repaired, resolve all declared generation families on all 90 within the original recorded language and per-stage resource limits, including the formerly failed 79, or report the exact remaining failures and withhold that claim. Increasing a limit is a separately named resource experiment. Record cold and warm runs, every scheduled object/family, supported proof-filter scope and actual proposal-loss coverage. Confirmation of improvement requires fresh held-out structural parents/ontologies because these findings informed redesign. A completed pool is still not proof of exhaustive semantic optimality.

**CG-042 — Scientific acceptance.** Predeclare the primary end-to-end metric, generation eligibility threshold, resource limits and checkpoint rule before fresh evaluation; no unspecified threshold is a passed gate. Compare matched semantic benefit/cost outcomes, useful coverage conditional on retrieval, total verified feasibility, time/calls to first feasible repair and to matched semantic quality, and time to pool-optimality certification separately. Aggregate related siblings by parent and account for seed variability. Prefer circuits only when the selected benefit survives total-cost accounting and failure-inclusive matched comparisons; do not infer superiority from final SDD size or sampling throughput alone.

## 11. Fair ablations and conditional alternatives

**CG-043 — Matched arms.** The study must distinguish:

| Comparison | What remains matched / what is isolated |
|---|---|
| Historical monolithic vs repaired family SDD | Language, constraints and reference weights; memory/layout/factorization implementation changes |
| Grammar-only vs proof-filtered SDD | Language before semantic filtering, evidence and budgets; effect and cost of proved contextual conditions |
| Product vs mixture | Constraint support and resources; representational effect of correlated components |
| Uniform vs learned proposals | Same support and pool policy; define whether uniformity is over encodings or unique bundles |
| No cache vs exact cache vs qualified schema cache | Same outputs; cold/warm amortization and binding overhead |
| Bounded enumeration | Same finite language/constraints; no truncated prefix labelled exhaustive |
| Semantic masked/grammar decoder | Same semantic conditions, template/menu/bounds, observable evidence, elementary controls and total resources |
| Optional typed grammar circuit | Same support where possible; clearly distinguish a different probability model |

For a decoder claiming the same distribution, next-choice probabilities must incorporate the total valid completion mass under the reference model; local masks alone are not sufficient. A decoder promising only validity is a legitimate distinct model, not distribution-equivalent. Charge retrieval, proof generation, compilation/lookahead, failed attempts, transfer, model evaluation, normalization, sampling, verification and cache population under the predeclared scenario. Match candidate budgets and draws where meaningful, and also compare under equal total time/memory; report both rather than conflating accepted samples with work performed.

**CG-044 — Optional typed grammar inside/outside circuit.** If the repaired small-SDD implementation is outperformed or remains impractical, an alternative may build a finite acyclic derivation DAG with states for type, side, remaining depth/constructor budget, canonical-order information and fixed-endpoint activation bounds. Sum over productions and multiply independent child contributions; inside values provide partition functions, outside/reverse differentiation provides expected counts, and weighted traversal samples valid derivations. This follows semiring parsing, not a newly claimed general-purpose compiler. Canonical unambiguous derivations simplify candidate likelihood; otherwise aggregate every derivation yielding the same bundle. Any retained semantic constraints must be represented in states or equivalent qualified filtering with its explicitly changed distribution.

**CG-045 — Alternative limits.** Arbitrary cross-slot constraints, ordering, alias aggregation or ontology-wide conditions can cause state/circuit explosion. Do not promise polynomial compilation for arbitrary constraints, exact mixture MAP, or semantic completeness outside the declared fragment. A typed decoder with valid outputs need not realize CG-018. Such an alternative requires its own distribution version, equivalence/difference tests, supervision contract and total-cost comparison. It does not remove the global verifier or license omission of action families without a named ablation.

## 12. Implementation references

Use the established primary references in [references.md](references.md) for OWL semantics, SPLs, SDDs and knowledge compilation. Implementation-specific sources are the [PySDD manager API](https://pysdd.readthedocs.io/en/latest/classes/SddManager.html), [vtree API](https://pysdd.readthedocs.io/en/latest/classes/Vtree.html), and [reference-management examples](https://pysdd.readthedocs.io/en/latest/examples/build_formula.html). The optional derivation-circuit alternative follows Goodman (1999), [Semiring Parsing](https://aclanthology.org/J99-4004/). Qualify pinned library versions and the precise operations used; documentation does not substitute for conformance tests.
