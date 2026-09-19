# XR-2 design audit and remaining research risks

This audit records the active resolution of issues found in the earlier proposal. It supersedes XR-1/XR-P1 assumptions where they differ. Formal definitions are in [00](00-research-contract.md), [04](04-minimal-exact-kernel.md) and [10](10-constrained-generation.md).

| Issue | Required resolution | Evidence to collect |
|---|---|---|
| Mandatory trusted mappings and fixed ontologies | All mappings provisional; explicit locks optional; eligible ontology axiom occurrences become revision objects | Mapping-only and ontology-edit comparisons |
| Coherence treated as semantic correctness | Separate hard feasibility, desired/unwanted consequences and edit preferences | Independent semantic probes, preserved named-class satisfiability |
| Full classification blocks initial search | Bounded initial diagnostics; sound partial conflicts can start search | Time to first conflict and first verified repair, pending fraction |
| Partial reasoner used to accept expressive inputs | Acceptance requires a complete supported verification scope | Capability checks and counterexamples outside the supported fragment |
| Circuit interpreted as OWL verification | Circuit guarantees only its encoded bounded grammar constraints | Grammar conformance plus independent final OWL verification |
| Guard/qualification vocabulary hides semantics | Use subclass-expression specialisation, necessary conditions and complex equivalence | Complete emitted bundles in every example |
| Context emptiness makes weakening vacuous | Check the whole selected subclass expression | Satisfiable E but unsatisfiable S intersection E fixture |
| Desired disjointness scored with an impossible antecedent test | Require both operands individually satisfiable | Typed teacher tests |
| Per-state best-completion labels counted repeatedly | Train benefit on complete repairs; compare unary/pairwise models | Redundant/complementary consequence fixture |
| Proposal and value heads conflated | Per-object generator produces expression assignments; completed-candidate head scores benefit | Node/readout/head ablations |
| Confidence treated as correctness probability | Treat matcher scores as fallible features unless calibrated | Score-shift and misleading-evidence tests |
| Generic graph pooling loses the repair target | Target-conditioned readouts over entities, evidence and conflicts | HGT versus R-GCN with matched readouts |
| Ontology deletion leaves stale inferred facts | Materialise from the revised asserted theory | Removal invalidates classification/explanation caches |
| Duplicate axiom emitters invalidate cuts | Axiom presence is the OR of all origins, including ontology objects | Duplicate-origin regression |
| Query failure treated as a monotone clash | Default to complete-assignment cuts for hard positive-query failures | Nonmonotone feasibility fixture |
| All-off assumed feasible | Initial incumbent is optional and must be verified | Required-entailment and background-conflict fixtures |
| Timeout treated as infeasibility | Pending alternatives stay in global upper bounds | Unknown high-value assignment fixture |
| Finite space interpreted as fast completion | Finite complete search is conditional; bounded execution can remain unresolved | Deadline and crash injection |
| Source-local modules assumed complete | Require the precise preservation condition; modules off by default | Missing cross-signature witness fixture |
| Generated-only labels mistaken for empirical meaning | Generated pretraining, real-structure adaptation, held-out real evaluation | Grouped splits and independent probes |
| Many mappings mistaken for many repair cases | Case/pair/ontology are distinct sampling units | Per-pair tables and cluster-level uncertainty |
| Optimal objective implies confident ontology correction | Higher edit costs encode preferences only; objective certainty is not author intent | Sensitivity, alternatives and provenance in explanations |

## Remaining risks to investigate

1. **Identifiability.** Different intentions can be compatible with identical observed evidence. The model must not be evaluated as if missing information were recoverable. Report ambiguity and useful alternatives.
2. **Expressiveness.** The bounded grammar may omit the required repair. Separate retrieval failure, grammar failure, sampling failure and optimisation failure.
3. **Teacher bias.** Generated corruption operators and query weights can teach shortcuts. Counterbalance evidence/provenance and hold out structural families.
4. **Approximation of semantic benefit.** Unary/pairwise factors cannot represent arbitrary higher-order consequences. Measure ranking/regret, rather than asserting faithful value decomposition.
5. **Circuit size.** Decomposability permits efficient inference after compilation; compilation can still be exponential. Record both construction and inference cost.
6. **Large-ontology verification.** Bio-ML may remain unresolved under the available complete backend. A fast detector improves diagnosis but cannot eliminate this limitation.
7. **Optional ontology edits.** They widen the solution space and introduce attribution risk. Freeze eligibility, compare with mapping-only repair, preserve imports and return exact patches.
8. **User preferences.** Sparse feedback may identify only some cost tradeoffs. Use regularisation and report held-out preference prediction when real feedback exists.
9. **Practical implementation size.** Reuse the shared OWL representation, reasoner interface, an existing circuit compiler and MaxSAT solver. Small orchestration code does not make the underlying problem polynomial.

These are research questions and engineering constraints, not reported experimental findings.
