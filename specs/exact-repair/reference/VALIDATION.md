# XR-2 specification validation — 19 September 2026

The updated finite reference and static protocol suite passed **37 test methods**. This is specification conformance evidence, not a production repair, training or benchmark result.

Coverage includes:
- 1,026 exhaustive comparisons: 19 upward-closed unsafe families × 27 signed weight vectors × two cut strategies;
- 120 seeded multistate monotone cases and 120 seeded arbitrary-policy/pairwise cases against exhaustive optima;
- every Boolean valuation of small unary and signed-pair WCNF fixtures, including large integer coefficients;
- arbitrary nonempty first states, optional/failed/unknown initial assignments and no mandatory all-off solution;
- pending high-value candidates retained in global bounds and no logical cuts generated from unknown results;
- mapping/ontology duplicate axiom emitters, fixed copies and conditional activation;
- positive-query nonmonotonicity, support-probe validity, empty inventories, infeasible pools and forged optimum records;
- protocol inheritance, requested counts, strict schema fields and deliberately invalid safety, teacher, circuit, split and resource settings.

The finite reference uses exhaustive selection and a synthetic set-valued oracle. It does not invoke RC2, an OWL reasoner, a graph model or a circuit compiler. Its call budgets do not interrupt callbacks.

Not established by these tests: correctness/performance of the future production adapters, full OWL proofs, runtime scalability, learned semantic value, preference accuracy or superiority on Conference/Bio-ML. Such claims require the experiments and actual component checks described in the suite.
