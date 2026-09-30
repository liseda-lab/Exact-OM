# Exact-Repair specification suite

**Current target:** XR-2.1, 30 September 2026. **Implementation reviewed:** `b4c1ed0d5e12c45974bdcb4d230fb2ab6c6deb04`. **Status:** revision requirements, not a claim that the revised system is implemented or evaluated.

The repository contains a first XR-2 implementation and generated-data pilot results. This revision defines the changes needed after reviewing that implementation and pilot. It supersedes conflicting XR-2 design prose; it does not relabel archived experiments, rewrite their configuration, or assert that the new method outperforms the old one. [12](12-implementation-migration.md) connects each required change to the implementation and a regression test. [11](11-benchmark-evidence.md) separates the preliminary experiment evidence from published benchmark statistics.

## Scope and decisions

Exact-Repair consumes shared ontology snapshots, a provisional alignment from any matcher, and whatever scores, explanations, decompositions and alternatives that matcher supplies. There is no mandatory trusted mapping subset. It returns a repaired alignment, explicit patches for eligible ontology-axiom occurrences, and separate verification and optimisation records.

The method keeps HGT, probabilistic circuits, weighted MaxSAT and symbolic verification. Elementary alternatives are constructed directly; learned circuits generate bounded complex replacements. The value model estimates retained semantic benefit and action interactions. A separate plan-risk model orders expensive checks. MaxSAT still constructs the repair plan. Only symbolic evidence can make an exclusion logically binding or authorise acceptance. Reinforcement learning is outside the main method.

The changes are: reliable small circuits with proof-supported ontology-context constraints; shared train/inference graph and interaction construction; supervised training on verified sampled repairs containing newly generated candidates; separate semantic-value and conflict-risk supervision; reusable proof-support cuts; and reasoning selected by supported constructs and queries before estimated cost. LLMs provide weak labels for preservation of intended meaning. User-personalisation remains an optional later study.

## Reading order and contract ownership

| Document | Owns |
|---|---|
| [00 Research contract](00-research-contract.md) | Problem, research questions, objective and claim boundaries |
| [01 Architecture and records](01-architecture-and-contracts.md) | End-to-end interfaces, identities, status and migration |
| [06 Actions](06-first-experiment-implementation.md) | Complete replacement semantics and concrete examples |
| [09 Graph and neural model](09-graph-and-neural-model.md) | Graph inputs, attention, proposal/value/risk outputs |
| [10 Constrained generation](10-constrained-generation.md) | Grammar, semantic constraints, circuit compilation and sampling |
| [03 Reasoning](03-module-soundness.md) | Baselines, detector, complete acceptance and cache validity |
| [04 Exact kernel](04-minimal-exact-kernel.md) | Frozen optimisation, cuts, scheduling, bounds and termination |
| [07 Training](07-corpus-and-training.md) | Generated data, symbolic teacher, sampled supervision and losses |
| [13 Semantic fidelity](13-semantic-fidelity-supervision.md) | LLM annotation, calibration and independent evaluation |
| [02 Evaluation](02-experimental-protocol.md) | Fair controls, grouped splits, metrics and accounting |
| [08 Studies and handoff](08-experiment-matrix-and-handoff.md) | Comparisons, dependency gates and run preparation |
| [11 Evidence](11-benchmark-evidence.md) | Pilot and benchmark evidence, denominators and limits |
| [05 Design audit](05-design-audit.md) | Decisions and remaining research risks |
| [12 Implementation migration](12-implementation-migration.md) | Audited code gaps and executable acceptance requirements |

[Work packages](implementation/XR-WP1-formal-kernel.md) and [study sheets](experiments/README.md) are implementation handoffs to these contracts. Where they disagree, the owning contract above governs; record and fix the discrepancy before running the affected study. All new experimental choices must be frozen in a new run manifest as described in [protocol/README.md](protocol/README.md).

## Invariants

1. Every selected candidate is a complete replacement bundle. Materialise from asserted axioms, removing the selected original occurrences while preserving any duplicate origins.
2. Ontology edits require explicit eligibility. Human authorship increases default cost; it is neither a hard lock nor proof of correctness.
3. Logical feasibility, semantic fidelity, proposal probability, predicted risk and edit cost are different quantities.
4. Circuit guarantees cover exactly the encoded constraints. Local admissibility and an incomplete detector's silence do not establish global feasibility.
5. Complete supported verification of all active obligations is required for a verified repair. Unknown is not false, negative training data or permission to accept.
6. All untested and unknown alternatives remain represented in the global upper bound. Scheduling blocks are separate from proved cuts.
7. Optimality is relative to a frozen finite inventory, integer objective and policy. Candidate coverage and verification scope remain explicit.
8. No clean parent, corruption trace, reference answer, future explanation or held-out feedback enters deployment features.
9. Reuse the [native shared stack](../native-stack.md); do not add a second OWL representation or a Java requirement to production. Repair remains opt-in.
10. A stronger ontology edit cost expresses caution. Neither learned confidence nor a solver proof establishes an author's intention.

## Versions and preserved evidence

The current runtime and archived results use version-2 records. XR-2.1 requires new record fields and changed feature/label identities; the migration in [01](01-architecture-and-contracts.md) uses an explicit version-3 boundary. It is not implemented by this specification update.

`protocol/pilot.json`, `smoke.json`, `schema.json` and `batches.json` are preserved XR-2 campaign artefacts. Their numerical settings are historical exploratory values, not an XR-2.1 configuration. Likewise, the existing supervisor instructions describe an operational workflow; they do not authorise jobs, notifications or external model calls merely because an agent reads them.

The old labels “guard” and “qualification” are historical aliases. Use **subclass-expression specialisation**, **necessary condition**, **complex correspondence**, and **axiom weakening** where the weakening condition is actually proved.

## Validation scope

The finite model and static legacy-protocol checks in [reference](reference/README.md) remain useful narrow checks. Passing them does not validate the new circuit implementation, risk model, detector or live OWL backends. XR-2.1 completion requires the concrete gates in [12](12-implementation-migration.md), with actual evidence. This commit changes specifications only.
