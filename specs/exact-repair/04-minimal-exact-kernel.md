# XR-2 finite selection and verification kernel

The core remains small: complete replacement records, an existing exact optimiser, a capability-checked verifier and a bounded orchestration loop. Neural proposals and values remain outside the logical kernel.

## 1. Frozen problem

Freeze B, the editable objects, finite candidate sets A_i, emitted axiom sets, policy/exception evidence, unary/pair benefit coefficients, profile costs and quantisation. B is the noneditable asserted remainder; eligible ontology axioms are removed from it and reintroduced only through selected replacements.

~~~text
X       = product_i A_i restricted by hard edit/structural constraints
T_R     = B ∪ union_i Ax(a_i)
U(R)    = sum_i w_i(a_i) + sum_(i,j in P) b_ij(a_i,a_j)
w_i(a)  = b_i(a) - u^T f_i(a)
~~~

All costs occur once. The benefit model need not be additive; the main comparison includes a bounded pairwise extension. Freeze P and its coefficients before solving. A later model, profile or proposal refresh starts a new problem.

Keep/delete states do not imply that deleting everything satisfies positive competency requirements. No mandatory all-off fallback or globally trusted mapping subset exists. A verified warm start is useful but not a prerequisite for search.

## 2. MaxSAT encoding

One Boolean x_ia selects each alternative. Enforce exactly one per object. Original-relation-aware weakening states are keep/delete for a subsumption and keep/delete/either direction for equivalence. Other revision templates are explicitly identified in 06.

For unary utilities let M_i = max_a w_ia. Add soft clause ¬x_ia with weight M_i − w_ia when positive. Exactly one selected state makes the violated unary weight sum_i M_i − sum_i w_i(a_i).

For pair coefficient b_ia,jb introduce y ↔ (x_ia ∧ x_jb) using three hard clauses. Positive b adds soft y of weight b; negative b adds soft ¬y of weight −b. This encodes the signed pair objective up to a constant.

Preferred first exact implementation is a qualified PySAT RC2 adapter using public interfaces. Pin actual versions and test interruption, incremental hard clauses, signed/large utility conversion and result completeness. A feasible assignment is not an upper bound. An interrupted None is not proven infeasibility.

Quantise the combined objective after recording unscaled benefit and cost. Use scale s and half-even rounding once per combined coefficient. If at most k coefficients contribute, assignment error ≤ k/(2s), and a quantised optimum can be at most k/s below the unquantised optimum. State optimality for the exported integer objective.

A universally valid initial upper bound is:
~~~text
UB_cap = sum_i max_a w_ia + sum_all_pair_coefficients max(0, b_ia,jb)
~~~
The pair contribution is deliberately loose; a tighter qualified bound is optional.

## 3. Sound logical exclusions

For any definitely infeasible complete assignment R, the full-assignment no-good OR_i ¬x_i,a_i excludes exactly it. It works without explanations, minimality or monotonicity.

For a supported unwanted entailment with axiom support Γ, create presence variables:
~~~text
p_alpha = true                                      if alpha ∈ B
p_alpha ↔ OR_(i,a: alpha ∈ Ax(a)) x_ia               otherwise
cut: OR_(alpha ∈ Γ) ¬p_alpha
~~~
Include editable ontology support and every duplicate emitter. Excluding only one current producer of an axiom is not an equivalent semantic support encoding.

For an expression obligation activated by candidate a, add:
~~~text
¬x_ia OR OR_(alpha ∈ Γ) ¬p_alpha
~~~
Retain all relevant activations for compound conditional policies. A support clause cannot reject an assignment merely because an unused expression is impossible.

Failure to entail a required positive consequence is not an entailed monotone violation. Use a full-assignment no-good after definite failure unless a separately proved stronger encoding exists. Shrinking is optional and bounded; an unknown shrinking check never licenses removal of support.

State-support cuts are permitted only when retaining their literals guarantees the same axiom support and active obligation. Index zero, “off”, or “keep” is not a sufficient criterion when ontology objects and nonempty first states exist.

## 4. Pending assignments and correct bounds

Unknown verification generates no logical cut. Store the assignment, theory/policy hash, utility, unresolved obligations, cause and remaining retry allowance in a pending set P_pending.

A work master may temporarily exclude pending assignments to inspect alternatives. These scheduling exclusions are separate from persistent logical clauses and are removed on replay. A later sound cut can eliminate a pending assignment only when that cut actually applies to its current axiom presence/activations.

For a completed work-master upper bound W and pending utilities:
~~~text
UB = min(previous_UB, max(LB if present, W, max_R_in_pending U(R)))
~~~
Treat an empty set's maximum as −infinity. Resolved feasible assignments are covered by LB; logical exclusions remove only infeasible assignments. If W is unavailable after interruption, retain the previous valid global UB. Never report the work-master bound alone after blocking unknowns.

Do not repeatedly solve to the same unresolved candidate. Inspect each unprocessed candidate at most once before any bounded retry phase. Retry only within the finite call/time allowance. No candidate reaching a timeout is relabelled infeasible.

## 5. Loop

~~~text
validate and freeze the problem; obtain bounded initial diagnosis
incumbent = none or a genuinely verified eligible warm start
LB = its utility, or null; UB = UB_cap
logical cuts = sound initial supports; pending = empty

while budget remains:
    solve the master with logical cuts and temporary scheduling exclusions
    if interrupted: retain previous UB and stop or use bounded supported retry
    update global UB including pending
    if verified LB reaches UB: return OPTIMAL_IN_POOL
    if no unprocessed candidate can improve LB:
        retry relevant pending assignments within their finite allowance
        otherwise return incumbent with gap, or UNRESOLVED
    verify the selected theory against every policy obligation
    if feasible: update incumbent and LB
    if definitely infeasible: add one sound progress exclusion
    if unknown: move to pending; add only a temporary scheduling exclusion
    remove pending entries contradicted by subsequently proved logical cuts

return the last verified incumbent and valid bound, or no verified repair
~~~

With an empty master, no incumbent and no pending assignments, report NO_FEASIBLE_IN_POOL only if emptiness is proved by valid constraints. An unresolved baseline need not stop useful detection/search. It can prevent full acceptance when policy obligations or exception semantics remain undecided.

## 6. Formal properties

**Soundness:** every authorised repair has a complete sound report for its exact T_R and policy. Scores, grammar validity and partial detection do not authorise acceptance.

**Exclusion preservation:** a full-assignment exclusion removes a proved failure. A monotone support clause removes only theories retaining a sufficient violating support and the relevant activations. Thus no feasible repair is removed.

**Completeness:** with finite A_i, exact terminating master solves and complete terminating verification, each rejected assignment is excluded. At most product_i |A_i| distinct candidate checks are needed to find a feasible optimum or establish that none exists. This is not enumeration of all optima.

**Optimality:** the logical master contains every feasible assignment. A verified incumbent attaining a valid bound that also accounts for unresolved assignments is optimal for the fixed integer problem. Exact solver optimality for an incompletely verified assignment is not optimal verified repair.

**Limits:** the candidate product and OWL reasoning can be expensive. Deletion-only repair already encodes maximum-weight independent set via mappings A_v ⊑ B_v and, for each graph edge {u,v}, fixed C_uv ⊑ A_u, C_uv ⊑ A_v, B_u ⊓ B_v ⊑ ⊥. No general polynomial runtime or successful completion by an arbitrary deadline is promised.

## 7. Bounded execution and replay

Budget loading, diagnosis, graph/retrieval, compilation, sampling, scoring, master, verification, optional explanation, serialization and cleanup. Run blocking calls in supervisor-controlled killable workers. The parent retains completed evidence and bounds. Finite combinatorial termination does not bound an uninterruptible external call.

Incremental verification is an accelerator; before incumbent promotion require a fresh reconstruction check until incremental correctness is established. Do not start unbudgeted final reasoning after a stop.

Safety replay reconstructs asserted inputs, ontology patches, selected axioms and policy, then rechecks. Optimality replay additionally validates cuts, removes scheduling exclusions, accounts for all pending alternatives, and establishes the bound with an exact solver or independently checked proof. A copied zero-gap field is insufficient.

The [finite reference model](reference/README.md) checks control-flow/encoding examples. It does not implement OWL semantics, RC2, wall deadlines, or the learned model.
