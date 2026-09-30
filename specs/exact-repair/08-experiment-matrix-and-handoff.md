# XR-2.1 experiment matrix and handoff

**Revision:** 30 September 2026. Studies test the questions in [00](00-research-contract.md) and obey [02](02-experimental-protocol.md). Study filenames remain stable for links; XR-E09 now concerns semantic fidelity, with personal preference fitting optional.

| Study | Main question and comparison | Required evidence |
|---|---|---|
| XR-E00 | Measurement and input correctness | Versioned records; materialisation, split, capability, budget and replay checks |
| XR-E01 | RQ5: reasoning and reusable exclusions | Detector plus complete verifier; assignment versus support cuts; routing and shared baselines |
| XR-E02 | RQ1: richer actions and ontology edits | Common-pool action ablations and full generation-language comparison; independent meaning measures |
| XR-E03 | RQ2/RQ5: interactions and verification order | Unary/pair values, counterfactual training, plan-risk shortlist scheduling, exact small-case truth |
| XR-E04 | RQ2: learned generation/value | Exact small teachers versus sampled plan training with new candidates; HGT/R-GCN controls |
| XR-E05 | RQ3: circuit benefit | Grammar-only circuits, ontology-informed circuits, matched semantic decoder; cold/warm cost |
| XR-E06 | RQ1/RQ2/RQ5: real transfer | Generated-only versus training-side adaptation; fresh Conference and separate Bio-ML cohorts |
| XR-E07 | RQ5: scale and bounded return | Candidate, circuit, graph, solver and reasoner budgets; first verified and matched-quality endpoints |
| XR-E08 | RQ2/RQ5: robustness | Missing candidate controls enforced end-to-end; evidence/import shift, process/backend failure |
| XR-E09 | RQ4: LLM weak semantic fidelity | Symbolic teacher versus grounded AI comparisons; independent held-out judgements and bias controls |

The [individual study sheets](experiments/README.md) specify the corresponding controls and interpretation. No full factorial over every optional idea is required; staged ablations isolate the cause of any gain before combining components.

## Dependency gates

1. **G0 — Reproducible baseline and correctness.** Preserve archived evidence; implement record/feature migration, shared interactions, missing-control enforcement, endpoint materialisation, public signature validation, fair baseline accounting and startup supervision.
2. **G1 — Reliable candidate language.** Repair SDD ownership/resource accounting; compile small families; pass exact support/probability tests; implement persistent cache and declared partial coverage. Test semantic constraints using immutable proofs. Evaluate the matched decoder before deciding the default.
3. **G2 — Reasoning and exact search.** Qualify detector rules and support provenance; connect presence cuts; stream completed failures; implement full-scope backend routing and shortlist ledger with valid bounds.
4. **G3 — Supervised generation and interactions.** Collect verified sampled repairs with new candidates, counterfactual contrasts and higher-order infeasibility examples. Train value/proposal/risk separately; select checkpoints on generated-pool development performance.
5. **G4 — Grounded semantic fidelity.** Freeze label rubric, evidence packets, anchors and independent evaluation. Add LLM weak labels only after feasible-plan generation and symbolic facts are reliable.
6. **G5 — Real-case evaluation.** Freeze fresh held-out inputs and settings; run Conference then distinct Bio-ML scale cohorts under declared support. Preserve unresolved cases and all costs.

G0–G2 are prerequisites for attributing fewer verifier calls to learning. G3 may initially use only symbolic semantics; G4 does not block correctness. Real training-side diagnosis can proceed before G5 to measure structural coverage without accessing test labels.

## New run manifest

Before launching XR-2.1, create a separately versioned executable configuration and strict loader as required in [protocol/README.md](protocol/README.md). Historical pilot/smoke/batch files are not overwritten. The manifest must resolve every stage budget, sampling/expansion limit, loss weight, shortlist size, feature/target identity, backend route and stopping criterion. Values are chosen on development data and logged, not presented as known optimal settings.

Each run exports asserted inputs, exact provisional alignment and evidence cutoff, editable occurrences and policy, four-part diagnosis, effective grammar/candidate inventory, circuit/model/objective identities, supports, pending/deferred ledger, teacher/LLM provenance where applicable, final verified patch and stage resources.

A simple alignment file cannot represent all complex correspondences. Publish normative OWL axiom bundles and an explicit ontology patch; label any simple-mapping projection as lossy. Do not substitute simple equivalence for a complex output.

## Acceptance and completion

[12](12-implementation-migration.md) lists code-level changes and tests. A substantive study starts only after its dependency gates pass using real adapters, not just the finite reference model. Training completion, process completion, verification success and a research result are different statuses.

This design update does not resume a paused campaign, authorise API expenditure or enable production repair. Implementation, run submission and LLM annotation are separate subsequent work. No missing resource justifies silently reducing the question or claiming a new model was tested.
