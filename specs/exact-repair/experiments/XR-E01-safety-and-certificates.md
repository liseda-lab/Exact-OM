# XR-E01 — Reasoning, proof supports and replay

**Target:** XR-2.1, 30 September 2026. **Question:** RQ5. Requirements for new runs; archived XR-2 results remain separate.

## Data

Small exhaustive OWL cases, restriction-essential witnesses, imports/duplicate emitters, source/union conflicts, expressive unsupported queries and held-out real outputs.

## Comparisons

Complete verifier alone; sound restriction-aware detector followed by qualified verification; assignment versus presence-support cuts. Compare capability-first routes on their actually supported scope. Test modules/incremental reuse only after preservation qualification.

## Measurements

False acceptance/rejection in fully decided fixtures; completed query coverage; support coverage/size; cut reuse; backend setup/query/wall costs; first verified repair and unresolved fraction.

## Acceptance and interpretation

Include whole active-expression satisfiability, ontology edits that remove a conflict premise, positive required non-entailment, inconsistent source exceptions and early failure followed by timeout. Full-OWL feasibility cannot follow from a detector finding zero conflicts. A sufficient support need not be minimal, but every cut must replay under the selected assignment.

Shared splits, resources, failure accounting and statistics follow [02](../02-experimental-protocol.md). Dependency gates are in [08](../08-experiment-matrix-and-handoff.md); code changes and tests are in [12](../12-implementation-migration.md). Resolve every numerical setting in a new [run manifest](../protocol/README.md) before execution.
