# Exact-Repair specification suite

**Suite:** XR-2, 19 September 2026. **Status:** research design; implementation and benchmark results are not claimed.

This revision incorporates the decisions in *Exact-Repair: Learning to Repair Ontology Alignments* (19 September 2026). It replaces XR-1/XR-P1 as the current specification. The previous methodology report remains an implementation reference; older proposals are historical context. No trusted/provisional mapping partition is required.

Repair uses the [published native stack](../native-stack.md), including the shared core
and available optional reasoners, in both standalone and sequential execution. Matching
artifacts remain optional; available matching evidence follows the full input contract below.

## Scope

Input is two ontologies with resolved imports, a provisional alignment from any matcher, scores, and optional explanations, decompositions, alternative candidates, and other evidence. Output is a repaired alignment, an explicit patch for selected ontology axioms, and separate verification and optimisation records. The prototype studies mapping deletion, directional weakening, complex correspondences, endpoint revision, and selective ontology-axiom revision.

The main learned model is heterogeneous graph attention with shared proposal and candidate-value heads. A bounded probabilistic circuit generates expression/template choices. Weighted MaxSAT selects complete replacements jointly. A verifier accepts, rejects with evidence, or reports unknown. Generated cases provide symbolic supervision; controlled real-structure cases and held-out Conference/Bio-ML inputs test transfer and scale.

## Reading order and ownership

| Contract | Owns |
|---|---|
| [00 Research contract](00-research-contract.md) | Research questions, definitions and claim boundaries |
| [01 Architecture and records](01-architecture-and-contracts.md) | Inputs, outputs, graph/model interfaces and result fields |
| [06 Actions and implementation](06-first-experiment-implementation.md) | Complete replacement semantics, action registry and implementation sequence |
| [09 Graph and neural model](09-graph-and-neural-model.md) | Graph construction, attention, proposal/value heads and interactions |
| [10 Constrained generation](10-constrained-generation.md) | Grammar, Boolean encoding, circuit and proposal distribution |
| [04 Selection and verification](04-minimal-exact-kernel.md) | Fixed optimisation, cuts, pending assignments, bounds and proofs |
| [03 Reasoning scope](03-module-soundness.md) | Initial diagnosis, backend support, modules and cache validity |
| [07 Data and training](07-corpus-and-training.md) | Generated problems, typed symbolic teacher, losses and preferences |
| [02 Evaluation protocol](02-experimental-protocol.md) | Splits, baselines, metrics and failure accounting |
| [08 Experiment matrix](08-experiment-matrix-and-handoff.md) | XR-E studies, development defaults and handoff |
| [11 Benchmark evidence](11-benchmark-evidence.md) | Conference/Bio-ML editions, measured/published statistics and limitations |
| [05 Decision audit](05-design-audit.md) | Superseded assumptions and remaining research risks |

[Implementation work packages](implementation/XR-WP1-formal-kernel.md) and [experiment studies](experiments/README.md) refer to these contracts rather than defining conflicting alternatives. [Pilot settings](protocol/pilot.json) and [smoke settings](protocol/smoke.json) are exploratory starting values, not measured optima or a powered evaluation design. This is one methodology with controls, not a new production default.

## Invariants

1. A candidate emits a complete replacement axiom set. Remove the original object's axioms before adding the selected replacement; preserve duplicate origins.
2. Ontology axioms are editable only when explicitly eligible. Human authorship increases default edit cost; it is not an automatic hard lock or evidence of truth.
3. Soft evidence, logical feasibility, proposal probability, predicted benefit and edit preference are separate quantities.
4. Only complete supported verification of every active policy obligation authorises a verified repair. Zero conflicts from incomplete reasoning does not.
5. Unknown assignments remain pending and retain their contribution to the upper bound. Scheduling exclusions are not logical cuts.
6. Exactness is relative to a frozen finite inventory, integer objective and policy. Generation coverage, verification scope and search status are reported separately.
7. Typed probes prevent semantic credit from vacuous inclusions without incorrectly rejecting desired disjointness.
8. No reference answers, corruption traces, future explanations or test-derived fitting statistics enter deployment features.
9. Existing Exact-OM outputs remain unchanged when repair is disabled. Do not revise unrelated WP-* or matching E* specifications.
10. Reuse shared pyowl-core snapshots and the existing optional adapter architecture. Qualify actual input/result capabilities; neither a package name nor a second parser establishes correctness.

## Vocabulary migration

Use **axiom weakening by subclass-expression specialisation**, **adding a necessary condition**, **complex correspondence**, and **complex equivalence correspondence**. The old labels “guard” and “qualification” are historical aliases, not new public action names. Backend *capability qualification* remains a different, standard engineering term.

XR-2 uses version-2 records. Do not silently load XR-P1 settings or checkpoints: mapping-only objects, mandatory all-off fallback, old feature layouts, losses, action IDs, splits and result fields changed. The old settings are not converted by substituting words. The preserved pre-update snapshot and repository history support explicit migration.

## Verification and completion

Run the finite reference tests and static protocol checks described in [reference/README.md](reference/README.md). These validate narrow specification properties, not installed OWL/MaxSAT backends, trained models or throughput. Work-package acceptance requires the real component evidence listed there.

The research implementation includes all specified action families, HGT and its controls, circuit generation and its controls, symbolic supervision, joint search, and generated-to-real evaluation. Product UI, default enablement and a human-subject study are separate. Lack of licensed data or complete biomedical verification remains visible; it does not justify silently changing the research question.
