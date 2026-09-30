# XR-2.1 research contract

**Revision:** 30 September 2026. Target requirements following the first implementation review; results are reported separately in [11](11-benchmark-evidence.md).

## Research questions

- **RQ1 — Repair expressiveness.** At a fixed verification policy, do richer correspondence replacements and selected ontology edits retain more intended meaning than deletion or directional weakening? Which action families justify their generation and verification costs?
- **RQ2 — Learning proposals and interactions.** Can learning find useful candidates and combinations more efficiently than a symbolic system with access to the same action language? Does learned conflict risk reduce verification work to reach a matched repair quality, separately from the work needed to prove optimality?
- **RQ3 — Circuits with ontology context.** Does compiling bounded grammar constraints and proved local semantic conditions improve useful candidate coverage and amortised sampling cost over a decoder enforcing the same conditions? When does compilation cost outweigh that benefit?
- **RQ4 — Retained meaning.** Can weak LLM judgements grounded in definitions and verified consequences improve semantic-fidelity prediction beyond a symbolic consequence teacher? Does improvement hold under independent evaluation and ontology/matcher shift?
- **RQ5 — Reasoning and scale.** Which supported detection/verification routes, proof-support cuts and reuse strategies reduce total repair cost while preserving explicit guarantees on Conference and Bio-ML? How often must the method return unknown?

Ontology-edit caution is studied under RQ1/RQ4. Learning personal user preferences is a later optional extension, not the main purpose of LLM supervision. RL is not required by any primary question.

## Formal problem

Let O_s and O_t include resolved imports; M_0 is the complete provisional alignment and Ax(M_0) its OWL interpretation. Matcher scores and explanations are fallible evidence, not asserted axioms. Define

T_0 = O_s ∪ O_t ∪ Ax(M_0).

Eligible mapping or ontology-axiom occurrences form revision objects I. The immutable background B is the noneditable asserted remainder. For each object i, a declared generation procedure returns a finite set A_i of complete replacements. Each a carries emitted axioms Ax(a), any activated expressions and explicit structural edit features. A repair chooses exactly one a_i from each A_i:

T_R = B ∪ ⋃_{i∈I} Ax(a_i).

Keep emits the original axioms; delete emits none when permitted. Duplicate axiom occurrences and imports are tracked before taking the semantic set union. Editing one occurrence does not remove another emitter. A mapping repair and an ontology patch are different output artefacts.

The problem has two distinct computational layers. Generation chooses the bounded inventories A_i; exact selection optimises over their Cartesian product. Improving the first does not establish completeness over all expressible OWL repairs. Once an inventory changes, begin a new optimisation epoch.

## Feasibility

Consistency means T_R has a model. Sat_T(C) means C has a nonempty interpretation in some model of T; it does not mean a named instance has been asserted. Coherence requires each monitored class other than owl:Nothing to be satisfiable. Freeze an explicit policy containing:

1. consistency and a monitored public named-class signature;
2. source-only exception proofs, if the chosen policy permits pre-existing unsatisfiability;
3. required and prohibited hard consequences;
4. selected-candidate activated-expression satisfiability obligations;
5. edit eligibility, locks and structural restrictions.

The signature includes public classes introduced by admissible candidates. New public vocabulary requires a new policy/epoch; private query probes have separately scoped obligations. For S ⊓ E ⊑ T, the activated expression is S ⊓ E, not E alone. Under the open-world assumption, absence of assertions is not evidence of negation.

Compute separate O_s, O_t, O_s∪O_t and T_0 diagnostic reports. Only completed source-only unsatisfiability proofs from a source itself established consistent may create incoherence exceptions, frozen before selection. An inconsistent source would entail every class to be unsatisfiable and must not license blanket exemptions. Unknown source checks and contradictions first introduced by the union are not exceptions. An exception concerns the specific obligation; it does not remove consistency or other required checks.

If T_0 is completely verified feasible, return the alignment unchanged by default. If initial diagnosis is incomplete, repair may proceed under the frozen policy, but cannot claim the input already safe. A failed search produces no authorised patch unless a feasible incumbent exists.

## Objective and learning

For a frozen epoch maximise the quantised form of

U_θ(R) = Σ_i b_θ(i,a_i) + Σ_{(i,j)∈E} b_θ(i,a_i,j,a_j) − λᵀF(R)

over feasible R. E is a bounded, reproducible interaction set. F measures actual edits; λ is a declared nonnegative cost vector. Benefit estimates semantic fidelity under an explicitly versioned target basis. Symbolic consequences supply decidable supervision; LLM labels supply weak interpretation of intended meaning. Calibrate their scale before combining them with edit costs. Do not count the same cost twice.

The plan-risk function ρ_φ(G,R) estimates observed violation risk. It may reorder a bounded shortlist before expensive verification. It is not subtracted from U while claiming the same optimum, and it cannot create a hard clause. An independently specified objective with a risk penalty would be a different experiment.

Pair factors model complementarity or redundancy. They are not logical conflict constraints and cannot express all higher-order semantic effects. MaxSAT constructs the selected plan using these factors. The network does not directly output an authorised final repair. Teacher supervision and inference optimisation share the decomposition, not a guarantee that the learned benefit equals domain meaning.

## Three levels of guarantee

| Level | Meaning | Does not imply |
|---|---|---|
| Grammar validity | Correct typed bounded replacement encoding | Satisfiability or appropriateness |
| Encoded contextual conditions | Exact satisfaction of the specified proof-supported tests relative to their immutable context | Every relevant OWL condition was encoded |
| Global verified feasibility | All active policy obligations completed for the selected asserted theory | Semantic fidelity, conservativity or author intent |

Proved constraints that depend on editable axioms are conditional global constraints. They must not become permanent local bans, because another selected action may remove their support. A sound incomplete detector can reject a supported violation; silence cannot establish feasibility outside a complete qualified scope.

## Soundness, completeness and termination claims

| Claim | Required conditions |
|---|---|
| Correct circuit distribution | Equivalent encoded support, valid weighted model count, exact treatment of fixed bits and encoding multiplicity |
| Verified repair | Completed sound procedures for every active obligation in the declared supported theory/query scope |
| OPTIMAL_IN_POOL | Frozen inventory, policy and integer objective; valid exclusions; exact master bounds; verified incumbent with zero gap including all pending/deferred alternatives |
| NO_FEASIBLE_IN_POOL | Exhausted finite inventory with sound rejection evidence and no unresolved alternative |
| Complete finite search | Finite inventory and terminating complete selection/verification without resource interruption |
| Bounded operational return | All expensive stages supervised with finite cleanup; may return an incumbent with gap or unresolved |
| Semantic improvement | Independent evaluation supporting that claim, not just coherence or a higher learned score |

These are conditional algorithmic properties. A finite grammar does not guarantee affordable compilation; OWL reasoning does not have a useful worst-case wall-clock guarantee for these workloads. Resource limits ensure a return, not successful repair. Verification scope, generation coverage and optimisation status must always be reported independently.

The practical research claim is twofold: learning may explore richer repair actions efficiently, and may approximate semantic fidelity unavailable from hard logic alone. Reduced verification work is a third hypothesis to test. None is established by the existing pilot.
