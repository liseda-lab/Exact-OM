# XR-E01 — Verification and replay

Research question: RQ5.

## Data

Exhaustive small generated cases; unsupported-construct, import, ontology-edit and unknown-verifier stress fixtures; held-out real outputs.

## Comparison

Compare complete verification, sound incomplete detection followed by verification, and module acceleration only after qualification. Compare assignment cuts with justified axiom-presence cuts.

## Measurements

Measure acceptance correctness, detected violations, supported scope, explanation coverage, proof/replay coverage, unresolved fraction and time to verified incumbent.

## Interpretation and acceptance

Require zero false verified-feasible results in conformance cases. Include conditional non-vacuity, duplicate origins, positive hard queries, pending bounds and no initial feasible repair. A fast zero-conflict result is not a coherence certificate.

Shared controls, split rules, resources and status reporting are defined in [02](../02-experimental-protocol.md).
