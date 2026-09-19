# XR-2 experiment matrix and implementation handoff

The primary research questions are defined in [00](00-research-contract.md). This file assigns concrete comparisons to them without prescribing a different methodology for each implementation stage.

| Study | Question addressed | Main comparison |
|---|---|---|
| XR-E00 | Measurement prerequisites | Snapshot, capability, split, budget and replay conformance |
| XR-E01 | RQ5: verification limits | Complete versus sound incomplete verification; cut and bound correctness |
| XR-E02 | RQ1/RQ4: actions and ontology edits | Deletion, directional weakening, complex correspondences, eligible ontology edits |
| XR-E03 | RQ1/RQ2/RQ5: joint choice | Greedy, exact unary, exact pairwise and exhaustive small-case reference |
| XR-E04 | RQ2: learned decisions | Symbolic/score controls, HGT, matched R-GCN; teacher and profile variants |
| XR-E05 | RQ3: constrained generation | Bounded enumeration, uniform constrained sampling, learned circuit proposals |
| XR-E06 | RQ2/RQ5: transfer and scale | Generated-only, training-side real adaptation, held-out Conference and Bio-ML |
| XR-E07 | RQ5: scale and deadlines | Candidate/circuit/solver/verification budgets and pending alternatives |
| XR-E08 | RQ2/RQ5: reliability | Evidence shifts, missing candidates, import changes and verifier failures |
| XR-E09 | RQ4: preferences and inspection | Default/simulated/learned profiles and interpretable patch alternatives |

See [experiments](experiments/README.md) for each study's dataset, controls, outputs and acceptance.

## Dependency order

1. Establish the shared-core revision representation, typed policy and replay records.
2. Qualify the verifier; implement the bounded finite-pool optimisation loop with no learned component.
3. Implement all eligible action families as complete bundles and reproduce the worked examples.
4. Generate controlled cases and typed teacher labels; make exhaustive small cases the reference.
5. Add the heterogeneous graph, shared heads and constrained circuit proposals. Compare unary and pairwise benefit.
6. Pretrain and adapt only on training/development groups; freeze artefacts before held-out evaluation.
7. Evaluate Conference and Bio-ML under their actual supported verification scopes. Record unresolved cases.

This order is a debugging dependency, not a claim that early versions implement the full research design.

## Deliverables

A run contains the asserted input/import hashes, exact provisional alignment and evidence, editable occurrence manifest, policy, capability report, four-part initial diagnosis, candidate bundles/activations and coverage, circuit and model hashes, objective coefficients, teacher provenance where applicable, selected patch, verification reports, logical cuts, pending assignments, bounds and stage resource use.

Return a repaired alignment and an ontology patch separately. A simple Alignment API file cannot express every complex correspondence; publish the normative OWL axiom bundle and mark any lossy projection. Do not silently replace complex output by a simple equivalence.

## Settings and migration

The versioned [pilot](protocol/pilot.json) is a bounded exploratory configuration covering the complete design. The [smoke](protocol/smoke.json) reduces counts and budgets for conformance; it cannot substantiate generalisation or model-quality claims.

XR-1/XR-P1 caches, action names and checkpoints are incompatible unless explicitly migrated with a preserved semantic mapping. Existing result files are not silently relabelled. Research runs are opt-in and do not enable production repair.

## Acceptance before a substantive experiment

- All action and typed teacher fixtures pass.
- The finite executable reference agrees with exhaustive enumeration, including unknown candidates, pair terms, nonmonotone policies and duplicate axiom origins.
- The actual solver and OWL backend pass their separate conformance suites.
- Circuit probabilities on tiny instances match enumerated normalised probabilities.
- All controls share candidate inventories, readout capacity, evaluation scope and budget where a matched comparison requires it.
- No held-out group supplies teacher labels, model selection, vocabulary answers or preference feedback.
- Every displayed result separates logical status, search status, verification scope and candidate coverage.

The reference checks included with these specs validate contracts only; they do not constitute an implementation of the research system.
