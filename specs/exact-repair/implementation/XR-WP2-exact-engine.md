# XR-WP2 — finite-pool optimisation and verification loop

1. Implement [04](../04-minimal-exact-kernel.md) using an existing weighted MaxSAT solver: one-hot states, integer unary coefficients and optional bounded pair factors with conjunction auxiliaries.

2. Start with an optional verified incumbent. An infeasible all-off assignment neither proves the whole problem infeasible nor authorises a repair.

3. Use complete-assignment no-goods by default. Add axiom-presence conflict cuts only with proved support, every duplicate emitter and query activation. Required-query failure is not assumed monotone.

4. Keep unknown candidates in a pending ledger. Scheduling exclusions are temporary; global upper bounds include pending objective values. Bound solver, verifier, retries and cleanup.

5. Acceptance: exhaustive agreement on small cases; valid lower/upper bounds during interruption; conditional cuts; ontology objects; signed pairwise encoding; independent safety and optimum replay. Included reference code is a truth model, not the production solver.
