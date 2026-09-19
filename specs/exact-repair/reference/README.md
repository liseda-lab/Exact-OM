# XR-2 executable specification checks

The standard-library finite model is an executable check of selected contracts, not the Exact-Repair implementation.

From the specifications directory:
~~~text
python3 -m unittest discover -s reference -v
python3 reference/validate_protocol.py
~~~

## What the model covers

[kernel_model.py](kernel_model.py) replaces OWL theories with finite axiom/tag sets and replaces MaxSAT with exhaustive selection. An oracle returns SAFE, UNSAFE or UNKNOWN. Arbitrary replacement alternatives are permitted; index zero need not be empty and an initial feasible assignment is optional.

The model checks unary/pairwise integer objectives, one-hot/conjunction WCNF-shaped encoding, assignment cuts, optional monotone support cuts, pending assignments, valid bounds and separate safety/optimality replay. Presence-cut helpers account for every emitter and fixed copies, including conditional activations.

Support lifting assumes UNSAFE is monotone in asserted axioms and active obligations. Disable it for nonmonotone requirements such as positive entailment. Governance is an independent assignment restriction. The fixed ontology remainder is part of the oracle; supplied axiom sets are the chosen replacements.

The result's status is the search status from the main contract. A non-null assignment was verified SAFE by the finite oracle. No production verification scope or OWL logical status is implied. A NO_FEASIBLE_IN_POOL result has no finite optimum/bound value; UNKNOWN alternatives prevent this conclusion.

Unknown candidates are scheduled once and retained in the pending ledger. The production spec permits bounded retries; this minimal model tests the no-retry case. Call-count budgets do not interrupt a blocked callback and must never be presented as wall-time enforcement.

## Conformance coverage

[test_kernel_model.py](test_kernel_model.py) includes exhaustive small-instance comparisons, all 19 monotone three-decision unsafe families with safe empty set, 27 signed weight vectors and two cut strategies, 120 multistate monotone cases, 120 arbitrary-policy pairwise cases, and targeted regression tests.

[test_protocol.py](test_protocol.py) checks pilot/smoke resolution, strict fields, version migration, numeric/cross-field invariants, invalid semantics, inheritance cycles and duplicate JSON fields. [validate_protocol.py](validate_protocol.py) implements only the schema vocabulary actually used by [schema.json](../protocol/schema.json), plus cross-field checks; it is not a general JSON Schema package.

See [VALIDATION.md](VALIDATION.md) for executed results. Real OWL/profile/import support, MaxSAT adapter behaviour, action compilation, graph/circuit/training correctness and supervised deadlines still require their work-package acceptance tests.
