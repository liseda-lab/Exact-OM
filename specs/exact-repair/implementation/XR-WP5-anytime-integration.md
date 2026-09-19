# XR-WP5 — bounded evaluation and matcher integration

1. Implement the study matrix in [08](../08-experiment-matrix-and-handoff.md), generated/real-structure cohorts, Conference and versioned Bio-ML releases.

2. Integrate repair after any matcher's provisional alignment through an adapter; retain score decompositions/evidence without assuming they are calibrated probabilities.

3. Use supervised per-stage budgets, process cleanup and resumable records. Report logical status, verification scope, search status and inventory coverage separately.

4. Return the selected alignment and ontology patch with provenance, alternatives, cost/benefit decomposition and pending verification. Replaying a patch never mutates the upstream ontology files.

5. Acceptance: controlled timeout/crash tests; complete run manifests; paired case-level metrics including unresolved cases; complex-output fidelity; production repair stays opt-in and disabled by default.
