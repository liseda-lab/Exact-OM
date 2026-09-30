# XR-2.1 exact selection, proof cuts and bounded verification

**Revision:** XR-2.1, 30 September 2026. **Status:** normative redesign; audited runtime b4c1ed0d5e12c45974bdcb4d230fb2ab6c6deb04. K-* identifiers belong to this document. [03](03-module-soundness.md) defines RE-* reasoning requirements. Planned v3 records follow [01](01-architecture-and-contracts.md); existing runtime/archive v2 records, frozen protocol JSON and the finite reference implementation remain unchanged.

The main selector remains weighted MaxSAT. No reinforcement-learning controller replaces exact selection in this design. Learned interaction values shape the frozen utility; a separate learned risk estimate may reorder a bounded shortlist without changing that utility or proving constraints.

## 1. Frozen optimization epoch

### K-01 — Validate and freeze all semantic and objective dependencies

An epoch contains resolved original snapshots/imports; original alignment; exact editable occurrences; fixed asserted remainder B; finite candidates A_i; complete emitted axiom bundles; active obligations; eligibility/locks; RE-03 monitored signature; exception evidence; query policy; structural constraints; model/context/proposal provenance; and the integer objective.

~~~text
X       = product_i A_i, restricted by declared hard edit/structural constraints
T_R     = B ∪ union_i Ax(a_i)
U(R)    = sum_i w_i(a_i) + sum_(i,j in Pairs) b_ij(a_i,a_j)
w_i(a)  = unary semantic benefit_i(a) minus explicit structural cost_i(a)
~~~

Pairwise terms express declared learned interactions. Record their semantic target, pair-selection policy, finite coefficients and model identity. Costs are explicit, nonnegative preferences and subtracted once. Model certainty is not logical evidence. The separate risk predictor is not silently added to U.

Candidates are complete replacements, including nonempty bundles and activation semantics. Keep emits originals, delete emits nothing only where permitted. Remove editable original occurrences from B; preserve all fixed/duplicate emitters. Candidate deduplication respects activation/restriction identity as well as canonical emitted axioms.

Run the same validation for preparation and externally loaded records. Reject missing signature coverage, invalid activations, malformed IDs/occurrences, nonfinite utilities, invalid objective shapes, unsupported schema or contradictory locks. No empty candidate row is allowed. An empty object inventory has one empty assignment and still requires verification.

A model, context, policy, preference, pair set, vocabulary or candidate-menu update starts a new epoch. Recompute objective/policy identities and bounds. Logical proofs may transfer after dependency validation; old utilities, scheduling exclusions and optimality claims cannot transfer automatically. A prior incumbent must be representable, rebound and revalidated for the new policy, then rescored. Candidate generation coverage remains a separate status; optimum in a sampled or resource-truncated pool is not optimum in the whole grammar.

Acquiring a sound cut or completed query changes the search-evidence ledger, not the frozen learned context, and does not itself restart an epoch. Refreshing learned encodings, candidate utilities or proposal context from that new evidence does. Deterministic cut applicability and frontier bookkeeping remain available throughout the current epoch.

### K-02 — Freeze and qualify exact integer optimization

For each object/candidate use selector x_ia and enforce exactly one selector per object. Preserve original relation semantics: a directional original does not gain a reverse direction merely because equivalence controls exist elsewhere.

For unary coefficients let M_i = max_a w_ia. A soft clause ¬x_ia with positive weight M_i minus w_ia gives total unary loss sum_i M_i minus selected unary utility.

For each signed pair coefficient b_ia,jb, encode y iff (x_ia and x_jb) with three hard clauses. Positive b uses soft clause y with weight b; negative b uses soft clause ¬y with weight minus b. The constant offset is:

~~~text
offset = sum_i M_i + sum_over_all_pair_entries max(0, b_ia,jb)
U(R) = offset minus encoded soft cost(R)
~~~

The pair sum is over individual entries, not just selected pairs. Recompute U from the returned assignment and check this identity. Ordered distinct object pairs, valid candidate indices and unique factor keys are required.

Quantize each combined unary benefit-minus-cost coefficient once with scale s and decimal half-even rounding; quantize pair coefficients once. Record unscaled values and s. If at most k coefficients contribute to any assignment, rounding error is at most k/(2s), and the quantized optimum is at most k/s below the unquantized optimum. Optimality claims concern the exported integer objective.

Use qualified public PySAT RC2 APIs initially. Record actual package/solver versions and test signed/large weights, exactness, interruption, incremental hard clauses and restoration. A feasible solver assignment is not an upper bound; interrupted or ambiguous None is not proof of emptiness.

The initial valid cap is:

~~~text
UB_cap = sum_i max_a w_ia + sum_over_all_pair_entries max(0, b_ia,jb)
~~~

It can be loose. A tighter bound is usable only when the adapter establishes it for the exact current work problem.

## 2. Sound evidence and reusable cuts

### K-03 — Whole-assignment exclusions are the universal safe fallback

A definite failure for complete assignment R licenses:

~~~text
OR_i ¬x_i,a_i
~~~

This removes exactly R and requires neither a minimal explanation nor monotonicity. Bind evidence to assignment, reconstructed theory, policy/activation and epoch. A failed required entailment must be a complete, qualified non-entailment result on the full selected theory; lack of a detector proof is insufficient.

Keep logical exclusions separate from temporary enumeration/scheduling exclusions even when their SAT clauses look identical. The evidence ledger records why a clause is logical, its exact scope and its validation status. Learned risk and timeout never license a logical exclusion.

### K-04 — Compile sufficient supports through semantic axiom presence

For each asserted axiom alpha needed by a support, define its presence:

~~~text
p_alpha = true                             if a fixed occurrence supplies alpha
p_alpha iff OR_(i,a: alpha in Ax(a)) x_ia  otherwise
~~~

Use every candidate emitter, whether mapping or ontology edit. If no emitter exists and alpha is not fixed, presence is false; reject a claimed proof from the selected theory if its purported support requires that absent axiom.

For support Γ that soundly entails an unconditional unwanted monotone consequence:

~~~text
OR_(alpha in Γ) ¬p_alpha
~~~

For a failure relevant only when frozen activation predicate g(x) is true:

~~~text
¬g(x) OR OR_(alpha in Γ) ¬p_alpha
~~~

Compile g with an exact equivalence, not a one-direction approximation. If one candidate activates the query, g = x_ia. If any of several candidates activates the same query, g is their disjunction; issue equivalent per-activator cuts or encode that disjunction exactly. A conjunction represents a genuinely joint condition only when the frozen policy declares it. Do not confuse these cases.

The proof establishes the semantic violation; activation establishes its policy relevance. Preserve both. A fixed-only unconditional failure produces a false clause and can prove no feasible plan. A fixed-only conditional failure yields ¬g, not global impossibility. Empty support is allowed only for a genuinely proved premise-free failure.

Example: two different editable candidates and a fixed imported occurrence all emit A ⊑ B. Deleting one candidate cannot falsify p_(A⊑B). If a violating support also includes an editable disjointness axiom, either its final emitter or another support axiom must disappear. An inactive impossible expression must not reject plans that do not activate it.

Canonical axiom identity determines semantic presence; occurrence provenance explains which objects can remove it. Private normalization/probe axioms are translated through their proven reduction before exporting a reusable support.

### K-05 — Validate cuts and constrain generalization

Accept a support cut only after RE-09/RE-10 validation: supported semantics, sufficient asserted premises, exact query, correct activation, current presence and trusted derivation/backend evidence. Learned or heuristic conflict predictions can trigger checks but cannot enter the hard master.

Missing required entailment is nonmonotone under axiom addition. Use its whole-plan exclusion unless a separate qualified generalization proves every excluded plan fails. An ontology patch may restore the consequence; ordinary positive presence clauses cannot encode that absence safely.

State-literal cuts require a proof that retained literals imply the same support/activation. Index zero, keep, off, weaker and delete are not semantic guarantees. Bounded support shrinking keeps the last proved support on unknown. Duplicate/subsumed cut elimination is optional and must preserve applicable evidence.

Across epochs, revalidate proof dependencies and compile against the new emitter/activation index. Reusing an old Boolean clause unchanged is forbidden when candidate identities or inventory positions change. Proof records are independent of a solver's variable numbering.

## 3. Learned risk, shortlists and complete frontier accounting

### K-06 — Rerank a bounded utility shortlist without changing U

Configure a finite shortlist size K, nonnegative integer utility window delta, construction budget, deterministic tie rule and risk-model identity. Obtain an exact work-master optimum v, then enumerate at most K distinct plans with U at least v minus delta, using temporary enumeration exclusions. Stop construction on its budget or on a proved next optimum below that floor.

Risk may use learned candidate interactions, predicted failure, verification cost and observed deployment context. It chooses order only within the declared shortlist. Tie-only ranking uses delta = 0. A broader delta may find an incumbent sooner but can defer higher-utility plans; those plans remain explicitly represented in the bound.

Do not resample or regenerate the inventory during this epoch. Do not hard-prune plans because the risk model dislikes them, and do not describe U minus a risk penalty as optimization of U. A genuinely different penalized objective requires its own frozen objective and experimental arm.

Enumeration exhaustion is not logical infeasibility: the shortlist may simply have hidden all remaining plans. A utility-floor constraint temporarily removes lower strata; it must never be used as an unrestricted global bound. Either retain a valid unrestricted residual bound, or record the hidden stratum's integer upper cap (floor minus one) and every other excluded set. The preferred initial implementation enumerates from the unrestricted work master and stops after observing the first residual optimum below the floor.

Risk can be ineffective or miscalibrated without compromising logic if these constraints hold. Evaluate it on saved verification/time outcomes, with unknowns represented explicitly; do not train it by declaring timeouts logically infeasible.

### K-07 — Maintain disjoint semantic/search ledgers

The parent-owned SearchLedgerV3 separates:

| Set/record | Meaning | Treatment |
|---|---|---|
| Logical cuts | Valid permanent constraints for this epoch | May eliminate only proved infeasible plans |
| Unenumerated work region W_region | Plans not individually retained elsewhere | Covered by current qualified work-master bound |
| Deferred D | Enumerated, untested shortlist plans | Store exact assignment and U; temporary exclusion only |
| Pending P | Checked plans with unresolved verification | Store U, completed events, missing obligations, cause and retries |
| Verified F/incumbent | Completely checked feasible plans | Their utilities are bounded by incumbent LB |
| Rejected records | Completed failures and their scope | Evidence-backed exclusion or applicable support cut |

Each nonexcluded plan must be covered by one region/set; never lose plans between transactions. A pending plan moving to retry remains covered until resolved. An untested higher-utility plan does not disappear when a lower-risk plan is chosen. Keep temporary exclusion identity separate from the evidence reason that may later make it permanent.

Completed sound support cuts may remove D or P entries only after evaluating their actual current axiom presence and activation. Retain a record explaining that resolution. Verified entries must never be removed by a purported sound cut; such a conflict is an integrity/qualification error.

### K-08 — Include all unresolved and deferred plans in the upper bound

Let W be a valid upper bound on the remaining unenumerated work region. Let d and p be the maximum exact U of D and P. Include a separate cap H for any temporarily hidden search stratum not already represented in W, D or P:

~~~text
candidate_UB = max(LB if present, W, d, p, H if present)
UB = min(previous_valid_UB, candidate_UB)
~~~

An empty set has maximum minus infinity; absent incumbent LB remains null in output. A proved empty work region contributes minus infinity. If an interrupted solve gives no qualified bound, use the last valid bound for that shrinking region or retain the previous global UB; never invent minus infinity.

Every W records the epoch, logical-cut revision and exact temporary exclusion/stratum state it bounds. Adding valid constraints shrinks the region, so an older valid upper bound stays conservative. Removing scheduling exclusions can enlarge a work region and invalidates its restricted W unless the restored plans are separately accounted for. The old valid global UB for the same epoch remains conservative.

Only a completely verified incumbent creates LB. Report gap = UB minus LB when both exist. If no incumbent exists, alignment/assignment output and gap are absent; diagnostic candidate records remain non-authorizing.

A feasible plan returned as the exact optimum of the full current logical master can certify optimality immediately, provided all higher-utility pending/deferred/hidden alternatives are also covered and cannot exceed it. The method does not need to verify every feasible plan or enumerate all optima. A work-master optimum that excluded a higher pending plan alone cannot certify the global optimum.

### K-09 — Bounded retries and exhaustion statuses

Inspect unprocessed plans fairly, with a finite per-plan retry allowance and explicit total/query/master budgets. A finite shortlist cannot permanently starve higher-utility deferred plans. Default scheduling processes a bounded shortlist before growing it; retries are reserved for unresolved plans that can improve LB or matter to certification.

Unknown is never relabeled infeasible because retries ended. Stop with INCUMBENT_WITH_GAP if a verified incumbent remains below UB; otherwise UNRESOLVED. OPTIMAL_IN_POOL requires verified LB = valid UB. NO_FEASIBLE_IN_POOL requires no incumbent, a proved empty logical search universe, and no unresolved/deferred/hidden feasible possibilities. Exhausted time or an empty temporary work master is insufficient.

Separate logical, search and generation/coverage status. COMPLETE_DECLARED_ENUMERATION, SAMPLED and PARTIAL_RESOURCE_LIMIT describe candidate generation, not successful reasoning. Preserve historical status interpretation rather than relabeling v2 records as completed v3 evidence.

## 4. End-to-end reference algorithm

### K-10 — Proof-guided MaxSAT with bounded risk ordering

~~~text
validate original assertions, occurrences, candidates and full public policy
obtain shared bounded baseline evidence under RE-05
freeze exception evidence, semantic objective, model/context and inventory epoch
incumbent = eligible completely verified warm start, if one exists
LB = its exact U, else null
UB = UB_cap
logical_cuts = independently validated applicable initial supports
D = empty; P = empty; F = incumbent if any
work_bound = UB_cap; ledgers persisted by parent

if original preservation is enabled and original input is fully verified:
    return original with its actual bound and declared optimization bypass

while end-to-end budget permits another bounded action:
    ingest completed proof events and validate before use
    add sound logical cuts; resolve affected D/P entries with evidence
    recompute conservative global UB over all covered regions
    if incumbent exists and LB == UB: return OPTIMAL_IN_POOL

    if D is empty and work region may improve LB:
        build at most K plans from the exact unrestricted work master
        after each completed solve:
            retain qualified residual/work bound
            atomically move selected plan into D with exact U
            add enumeration exclusion to work master only
            stop at K, construction budget, or utility-window boundary
        on interruption: retain prior valid global/work bounds

    if relevant D is nonempty:
        R = lowest declared predicted risk among eligible shortlist plans
    else if a relevant P entry has bounded retry allowance:
        R = selected retry; retain P coverage while it runs
    else:
        return certified emptiness, incumbent with gap, or unresolved as justified

    reconstruct T_R and active obligations
    run/reuse bounded sound detector; stream completed proof events
    if a definite applicable failure exists:
        add validated support cut, or whole-assignment fallback
        remove R from D/P with retained failure evidence
        continue

    qualify and run complete acceptance routes under RE-13
    retain streamed completed failures even if later work times out
    if a definite failure exists:
        add validated support cut or whole-assignment exclusion
        resolve R in D/P
    else if every obligation has complete passing coverage:
        move R to F; update incumbent/LB if U improves
    else:
        move R to P with exact U, unresolved obligations and retry state

return last completely verified incumbent and valid UB, or no verified repair
~~~

An incomplete baseline does not prevent useful rejection/search. Unresolved acceptance or exception obligations prevent promotion. No unbudgeted final reasoning begins after the deadline. Optional explanation minimization follows acceptance/rejection evidence and cannot delay preserving it.

All transitions that alter exclusions and frontier coverage are atomic in the parent ledger. Worker crashes leave the last acknowledged state and conservative bound intact. The scheduler can use a heuristic proposal source alongside the master only if proposed plans belong to this frozen inventory, are scored exactly, and do not invalidate master/frontier accounting.

## 5. Proof obligations

### K-11 — State what is guaranteed and under which assumptions

**Materialization.** Occurrence-aware replacement plus all-emitter presence reconstructs precisely B plus selected bundles. No removed original or stale inferred closure is retained merely because it existed before repair.

**Support soundness.** If Γ entails a monotone policy violation and g holds, every assignment retaining Γ and g fails. Therefore ¬g or some absent support axiom is necessary for feasibility. The presence encoding represents exactly that condition, including fixed/duplicate emitters.

**Whole-plan soundness.** A completed sound failure of a frozen obligation for R excludes exactly R; no monotonicity premise is required.

**Acceptance soundness.** RE-qualified evidence covering every active obligation on T_R authorizes only policy-feasible assignments. Learned scores, partial detections, syntax validity and solver optimality do not enter this proof.

**Bound preservation.** Logical cuts preserve every feasible plan. W_region, D, P, F and any hidden strata cover all remaining plans. Their maximum bounds the feasible optimum; intersecting with a previously valid epoch bound is safe. Risk changes order, not membership or utility, so it does not affect the argument.

**Optimality.** Verified LB attaining that global UB proves optimality for the frozen integer objective/inventory. The first checked feasible full-master optimum can satisfy this immediately. Candidate-relative optimality does not establish semantic fidelity.

**Conditional completeness.** With a finite inventory, fair enumeration, exact terminating master solves and complete terminating verification, at most product_i |A_i| distinct plans require first checks before finding a feasible optimum or proving none exists. Support cuts may remove many without verification. This is not a bounded-runtime promise or an enumeration of every feasible optimum. Operational limits and persistent unknowns weaken the conclusion to an incumbent/gap or unresolved.

**Complexity limit.** Deletion-only repair already encodes maximum-weight independent set using mappings A_v ⊑ B_v and fixed C_uv ⊑ A_u, C_uv ⊑ A_v, B_u ⊓ B_v ⊑ ⊥ for graph edges. Candidate products and expressive reasoning can be expensive. No polynomial-runtime or deadline-success claim follows from finite termination.

## 6. Supervision, persistence and replay

### K-12 — Supervise startup, serialization and full worker lifetimes

Budget preparation/loading, graph/retrieval, compilation/generation, scoring, diagnosis, master, detector, complete queries, explanation, serialization, transport, cleanup and persistence. Publish which API boundary starts the clock and memory scope. In-memory caller-owned preparation may be outside control only when explicitly excluded from the advertised bound.

The audited workers.py:70–104 starts a deadline but performs synchronous process.start() before its supervising loop; spawn argument serialization can overrun before interruption or RSS monitoring. Moving only the reasoner call into a worker is insufficient.

The redesign uses independently supervised preparation and operation workers. Pass bounded descriptors/handles to immutable, content-addressed snapshots/inventory records; perform large deserialization, normalization and payload creation inside a killable supervised domain. A startup watchdog or equivalent independently running supervisor must cover process creation and input transfer, not begin after they finish. Bound protocol frames and avoid executing unbounded user serialization in the coordinator.

The parent retains validated evidence, incumbent, global bound and frontier transactions. Use bounded event frames and acknowledged streaming; a stalled frame cannot block the deadline supervisor. Terminate the entire owned descendant process tree/group, with bounded cleanup. Uncertain surviving workers prohibit claiming successful cleanup; record that state. Never kill unrelated processes.

Define memory limits precisely. Sampled Linux worker-tree RSS can overshoot and double-count shared pages; it excludes caller-owned inputs and accelerator allocations unless separately measured. A required unsupported memory limit returns an explicit operational limitation, not silently disabled enforcement. OS allocation quotas and sampled monitoring are distinct guarantees.

On worker interruption, keep acknowledged failure events and previous valid bounds. Treat all uncompleted positive coverage, unexplained solver output and partial native updates as unresolved. Reserve a bounded cleanup/persistence allowance; do not begin a hidden unbudgeted acceptance pass after stop.

### K-13 — Persist a replayable search ledger

Planned SearchLedgerV3 records epoch/input/objective/policy hashes; inventory/generation coverage; immutable baseline references; variable/presence mapping; accepted proof cuts and trust manifest; temporary exclusions with reasons; D/P/F transitions; work-bound applicability; risk shortlist parameters/order; query events; resource failures; incumbent and final statuses.

Every master response binds its assignment, objective value, bound/completion status, solver version and current constraints/temporary-exclusion revision. Independently recompute assignment membership, restrictions and exact U before using it. Restarts rebuild solver state from the ledger; unsupported incremental updates fall back to reconstruction. Avoid generic infrastructure beyond what these correctness requirements need.

Safety replay reconstructs selected assertions/patches and full policy without neural scoring or optimization, verifies identity/coverage and rechecks or validates qualified evidence. Same-adapter replay establishes reproducibility, not independent verification; state that trust boundary.

Optimality replay additionally validates logical exclusions, removes temporary scheduling blocks or includes every excluded plan/region in the bound, and establishes a matching exact upper bound by qualified solver/proof or independent exhaustive search on tiny fixtures. A copied zero gap, artifact hash, missing pending list or empty work master is not a certificate.

Replay has its own finite budgets and can return unknown. Successful original acceptance is not retrospectively changed by an unavailable replay backend; replay status is separate. Corrupt or semantically invalid original evidence is a different integrity failure.

## 7. Required conformance tests and implementation boundary

### K-14 — Test invariants, not only happy-path outputs

| ID | Fixture/intervention | Required observation |
|---|---|---|
| K-T01 | Small exhaustive signed unary/pair objectives | RC2 result and offset equal direct enumeration |
| K-T02 | Empty object inventory, locked objects, nonempty first candidate | Correct assignment universe and unchanged policy verification |
| K-T03 | Duplicate editable emitters and fixed imported emitter | Presence/cuts match semantic occurrence materialization |
| K-T04 | Same expression activated by either candidate vs jointly | Exact OR vs AND activation cuts; inactive cases preserved |
| K-T05 | Fixed-only unconditional/conditional support | Global emptiness vs activation prohibition distinguished |
| K-T06 | Missing positive consequence restored by another candidate | Whole-plan fallback preserves restoring plan |
| K-T07 | Support shrinking unknown; malformed/absent premise | No unproved generalization |
| K-T08 | Higher-utility unknown, lower feasible plan | Pending U retains positive gap |
| K-T09 | Higher-utility untested shortlist plan, lower-risk feasible plan | Deferred U retains positive gap |
| K-T10 | Utility-window constraint hides lower stratum | Hidden bound retained; no false emptiness |
| K-T11 | Shortlist/master interrupted mid-enumeration | Prior valid bound and atomic frontier coverage survive |
| K-T12 | Temporary exclusions removed on retry/replay | Restricted work bound invalidated or restored plans separately bounded |
| K-T13 | Feasible full-master optimum with no higher unresolved alternative | Immediate optimality; no verification of every feasible plan |
| K-T14 | Same work optimum but higher pending plan | No premature optimum |
| K-T15 | New support cut applies to deferred/pending plan | Resolve only after actual presence/activation evaluation |
| K-T16 | Incorrect learned risk predictions | Ordering may worsen; objective/feasible set/bounds unchanged |
| K-T17 | Model/menu/context/policy update | New epoch; no inherited stale optimum or Boolean cut indexing |
| K-T18 | Startup serializer, import loader or frame receiver stalls | Deadline supervisor remains responsive; explicit operational result |
| K-T19 | Completed failure then timeout/crash | Acknowledged cut survives; no partial-positive acceptance |
| K-T20 | Exhausted retries/check/time budget | Unknown retained, incumbent/gap or unresolved returned |
| K-T21 | Exact and greedy consume shared baselines | Comparable verification budgets and separated preparation counts |
| K-T22 | Tampered theory, activation, objective or coverage record | No incumbent promotion or trusted replay |
| K-T23 | Restart during cut/frontier transaction | Recover last consistent ledger and conservative bound |
| K-T24 | Sampled/resource-truncated candidate generation | Pool optimum never labeled complete grammar optimum |
| K-T25 | Zero verification budget | No fresh acceptance call; only already validated eligible evidence can authorize |
| K-T26 | All-delete violates a required entailment | No assumed feasible fallback |
| K-T27 | Solver interrupted with ambiguous None | No emptiness certificate or invented bound |
| K-T28 | Source exception immutable across many plans | No repeated uncharged source classification |

The finite [reference model](reference/README.md) checks control-flow/encoding examples only. It does not implement OWL semantics, RC2, actual wall deadlines, native process supervision or the learned model. Existing reference tests/protocol JSON must remain interpretable as historical artifacts. New v3 execution/configuration and migration tests are separate implementation deliverables.

### K-15 — Audited baseline versus required implementation

At b4c1ed0 the runtime already has exact signed integer optimization, evidence-bound reports, unknown/pending bookkeeping, whole-assignment exclusions and independent safety/optimality rechecks. The presence encoder exists in maxsat.py:73–101 but kernel.py:337–342 does not pass proof supports. Unknown bounds are preserved, while shortlist/deferred accounting, proof streaming, shared baselines and end-to-end startup supervision specified here are new requirements.

The implementation acceptance gate is the conjunction of K-14 and RE-21 requirements plus recorded real-input capability/resource measurements. Specification completeness is not implementation completion; synthetic conformance is not biomedical-scale evidence.
