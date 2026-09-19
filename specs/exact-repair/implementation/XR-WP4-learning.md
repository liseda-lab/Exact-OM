# XR-WP4 — graph model, symbolic teacher and preferences

1. Implement the graph, HGT, matched R-GCN control, per-object attention readouts, shared proposal head and completed-candidate value head in [09](../09-graph-and-neural-model.md). Conflicts are graph nodes; heads are shared functions, not separately trained networks per conflict.

2. Implement generated structural parents and typed teacher probes in [07](../07-corpus-and-training.md). Desired disjointness uses operand satisfiability; inconsistent and unknown theories cannot earn fabricated benefit.

3. Train full-repair benefit differences/rankings, comparing unary and bounded pairwise factors. Do not sum best-completion labels. Learn exact circuit likelihood/marginal targets only where a complete finite cache supports them.

4. Keep explicit costs outside benefit; begin with declared nonnegative profiles and add regularised preference fitting only with labelled choices. Separate factual clarification and hard prohibitions.

5. Acceptance: no latent corruption or held-out reference leakage; accurate partial-label masks; counterbalanced provenance; scale calibration; saved grouped splits and seeds; generated-only and real-adaptation controls.
