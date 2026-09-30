# XR-WP1 — revision semantics and reasoning

**Target:** XR-2.1, 30 September 2026. Requirements, not completed work.

1. Implement the explicit v3 boundary and shared-core input/policy records in [01](../01-architecture-and-contracts.md). Preserve v2 readers for archived evidence; reject invented migration proofs. All mappings are provisional unless explicitly locked.
2. Complete MIG-03, MIG-11, MIG-13–15 in [12](../12-implementation-migration.md): endpoint materialisation, fair four-baseline reports, streamed failure events, public signature validation and supervised startup.
3. Implement qualified restriction-aware detection and complete acceptance routing in [03](../03-module-soundness.md). Sufficient proof supports include imports, duplicate emitters and activation conditions. Unsupported axioms stay in the full theory.
4. Test shared imports, new public classes, source-only exceptions, union-only conflicts, existential class satisfiability, positive hard queries, stale cache and late timeouts. No parser or ontology model duplication.
5. Deliver capability/proof/replay evidence for G0/G2. A backend name or successful import is not qualification; unknown must remain explicit.
