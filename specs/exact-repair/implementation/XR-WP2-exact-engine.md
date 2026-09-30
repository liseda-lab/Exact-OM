# XR-WP2 — finite selection, cuts and verification order

**Target:** XR-2.1, 30 September 2026. Requirements, not completed work.

1. Keep PySAT weighted MaxSAT with one selected replacement per object and signed unary/pair coefficients. Follow [04](../04-minimal-exact-kernel.md); all coefficients, policies and candidate identities are frozen per epoch.
2. Complete MIG-10 and EXT-02 in [12](../12-implementation-migration.md). Connect proof-support presence cuts; keep whole-plan exclusions for unproved support or nonmonotone required-entailment failures.
3. Implement a bounded utility shortlist and optional learned risk ordering. Temporary enumeration blocks, unknown retries and durable logical cuts have separate ledgers. Every untested/deferred plan remains in the upper bound.
4. Preserve optional incumbent semantics. A verified master optimum may close the gap immediately; all-delete is never assumed feasible. A changed pool/objective starts a new epoch, with proof reuse only through valid dependencies.
5. Acceptance: exhaustive small-case agreement, duplicate emitters, activation, three-way conflicts, signed factors, interrupted solves, pending bound correctness and safety/optimality replay. RL and neural hard clauses are outside scope.
