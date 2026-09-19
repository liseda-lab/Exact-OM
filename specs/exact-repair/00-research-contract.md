# XR-2 research contract

**Revision:** 19 September 2026. Current normative research scope; no empirical claim.

## Research questions

- RQ1: Do complex correspondence replacements and axiom weakening preserve more intended consequences than deletion, at comparable logical validity?
- RQ2: Can an attention-based graph model learn useful proposals and values from generated cases and transfer to unseen ontology pairs and matchers?
- RQ3: Does constrained probabilistic generation improve useful candidate coverage per total computation compared with enumeration and grammar-constrained decoding?
- RQ4: When do selected ontology edits improve repair, and can provenance costs and user profiles make their tradeoff controllable?
- RQ5: How much verification and search completes on biomedical ontologies, and what guarantees remain when it does not?

Experiments may reject the usefulness of an action or model. A successful implementation is not evidence that a research hypothesis holds.

## Formal problem

Let O_s and O_t include resolved imports and let M_0 be a provisional alignment. Correspondences induce OWL axioms Ax(M_0). Scores and explanations belong to a separate evidence collection.

T_0 = O_s ∪ O_t ∪ Ax(M_0).

An editable object is a mapping or selected ontology axiom. Let B contain all fixed asserted axioms, with no stale closure facts. For each object i let A_i be its finite set of complete replacement candidates. A repair R chooses exactly one a_i ∈ A_i:

T_R = B ∪ ⋃_i Ax(a_i).

Keep emits the original axiom set; delete emits the empty set when allowed. B is the noneditable remainder, not the original ontologies plus a mandatory trusted mapping subset. Explicit locks are supported but not assumed.

Consistency means that T_R has a model. Sat_T(C) means that C is nonempty in some model of T. Coherence requires satisfiability of each monitored named class other than owl:Nothing. Coherence does not establish semantic fidelity or conservativity.

## Policy

Freeze the monitored signature, edit eligibility, optional proven source-only exceptions, hard required/prohibited consequences and active-expression obligations. A feasible repair satisfies:

1. consistency;
2. satisfiability of all monitored non-exempt classes;
3. all required and none of the prohibited hard consequences;
4. satisfiability of each expression activated by a selected specialisation;
5. explicit edit and structural restrictions.

For S ⊓ E ⊑ T, the active expression is S ⊓ E, not E alone. Missing assertions do not constitute negation under the open-world assumption. Query failure, unsupported input and timeout remain distinct.

The policy can permit pre-existing classes proved unsatisfiable in O_s or O_t alone. Freeze their evidence before selection. Unknown source checks and contradictions first found in O_s ∪ O_t do not automatically create exceptions. Strict coherence is a separately recorded policy.

## Objective

Maximise U_u(R) = B_hat_theta(R) − uᵀF(R), over feasible candidates. u is nonnegative and expresses edit preferences; B_hat estimates the symbolic teacher's semantic benefit. Unary and bounded pairwise value models are named comparisons. Costs are subtracted exactly once.

A higher cost for ontology edits, especially human-authored axioms, does not prove the mapping is wrong. An optimiser's certainty about its objective is not calibrated certainty about an ontology author's intent.

## Claims and non-claims

| Claim | Conditions |
|---|---|
| Grammar-valid proposals | Exact encoding of the finite constraint formula and nonzero normaliser |
| Verified repair | Every policy obligation decided by a sound complete procedure for the declared theory/query scope |
| Finite-pool optimum | Frozen inventory/objective/policy, valid exclusions, exact solver, verified incumbent and valid zero gap including pending alternatives |
| Complete finite search | Complete terminating verification and optimisation, finite inventory and no feasible assignment discarded |
| Bounded operational return | Supervised bounded calls, finite cleanup and retained verified evidence; no guaranteed successful repair by a deadline |
| Semantic improvement | Appropriate independent evaluation of consequences/intent, not coherence or solver optimality alone |

No claim concerns all OWL repairs unless the space is actually exhausted. A sampled inventory is not the whole grammar. An EL expression grammar does not make an expressive ontology pair EL. A replayable certificate trusts its stated reasoner and solver; it is not automatically a small independently checkable proof.

An initially fully verified input is returned unchanged by default. Exact-Repair repairs a stated policy violation; it is not an unsolicited enrichment pass.
