# XR-2 reasoning scope, diagnosis and modules

## Initial checks

Resolve imports and preserve asserted provenance before diagnosis. Record separate checks for O_s, O_t, O_s ∪ O_t and T_0 = O_s ∪ O_t ∪ Ax(M_0). This distinguishes source defects, shared-import/vocabulary interactions, and alignment-induced consequences.

Start useful diagnosis without requiring full initial HermiT or ELK classification. An incomplete baseline is recorded as unknown. Only proved source-only unsatisfiability can enter the frozen optional exception set. The absence of detected source defects is not proof that a later defect was caused by a mapping. Do not rebase exceptions after a candidate is selected.

The default policy checks the complete original monitored class signature, not just initial witnesses or mapped endpoints. Additional explicitly declared candidate symbols and expression obligations are included. An already completely verified input is returned unchanged.

## Backend roles

| Procedure | Can establish | Cannot establish by itself |
|---|---|---|
| Qualified sound Horn/LogMap-style detector | Supported violations with axiom provenance | Full-OWL feasibility from zero detections |
| Complete supported-fragment reasoner, e.g. an eligible ELK adapter | All supported obligations on the actual supported input | Acceptance after silently ignoring unsupported axioms |
| Complete expressive reasoner, e.g. an eligible HermiT adapter | Decided OWL/query obligations within its supported semantics | Completion on every ontology within a short budget |
| Neural model/circuit | Scores or encoded syntactic constraints | Logical acceptance of the selected theory |

Qualify the actual shared-snapshot API, imported constructs, generated axioms, query families, result completeness and failure behaviour. Package names, hierarchy access and successful method return are insufficient. Reuse the repository's Java-free shared stack; inability of an installed adapter to provide required checks remains a recorded limitation.

A ⊑ ∃r.B, A ⊑ ∀r.C, B ⊓ C ⊑ ⊥ makes A unsatisfiable. Dropping the universal restriction misses this. Materialising rules over existing individuals does not generally decide existential class satisfiability.

Unknown never means feasible or infeasible. The full selected theory is checked after joint selection; there is no mandatory OWL call per sampled expression.

## Expression satisfiability

Selecting S ⊓ E ⊑ T activates Sat(S ⊓ E). Sat(E) is insufficient. Check all selected antecedents in the resulting theory, including changes to ontology axioms.

Where qualified, fresh private definitions P_E ≡ E reduce expression queries to named-class checks. Prove the conservative-extension reduction, retain its dictionary and test against direct expression queries. Private probes are absent from public alignments, graph features and the original monitored signature. Only activated probe obligations reject a repair; an unused impossible expression does not.

## Modules

A bounded neural neighbourhood is not a logical module. Module acceleration is optional and disabled in the initial research settings.

Any subset of current asserted axioms can establish a monotone unwanted entailment soundly when its proof uses supported semantics. Absence of that entailment in a subset generally proves nothing about the full theory. Positive-entailment failure is not a monotone conflict.

For example:
~~~text
B = { C ⊑ ∃r.A, ∃r.B ⊑ ⊥ }
inventory adds A ⊑ B
extraction signature = {A,B}
~~~
A module omitting both original axioms can contain no unsatisfiable named class, while the full selected theory makes C unsatisfiable. Restricting the monitored signature to mapped endpoints silently changes the policy.

An accelerator that authorises acceptance must establish preservation of every enabled query for every allowed replacement. Its extraction signature includes query signatures, all candidate expressions, and all symbols affected by eligible ontology patches. The guarantee must cover removal/replacement of ontology axioms, not just extension of one immutable ontology. Otherwise rebuild from the selected theory and check globally.

No generic claim that every locality module contains every relevant justification is assumed here. State the extractor's precise theorem and preconditions before relying on it; sound full-assignment exclusions do not require minimal justifications.

## Cache invalidation and explanations

An explanation is sufficient support; a justification is subset-minimal support. Minimality is optional. Preserve axiom identities, every current emitter, fixed occurrences and conditional obligations. Editable ontology axioms participate in supports exactly as mappings do.

Do not retain inferred edges as fixed asserted facts after their support has changed. Cache keys include the full repaired theory or an explicitly proved dependency set, policy, exception evidence, query and backend version. Changing an inventory/menu or profile-dependent policy invalidates affected modules/caches.

## Acceptance cases

Require differential checks where complete backends overlap; tests of missing universal/cardinality/ABox semantics; imported/shared duplicate axiom edits; inactive/active expression obligations; unknown baseline without invented exceptions; stale inference after an ontology edit; and the out-of-signature C counterexample above. A detector's false-positive proof is a defect; its documented incompleteness is a coverage limitation.
