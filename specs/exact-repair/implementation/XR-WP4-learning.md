# XR-WP4 — supervised generation, interactions and semantic fidelity

**Target:** XR-2.1, 30 September 2026. Requirements, not completed work.

1. Implement [09](../09-graph-and-neural-model.md): shared HGT with object/candidate readouts, identical interaction selection on all paths, pair benefit and a separate plan-risk head. Complete MIG-01, MIG-08–09.
2. Implement [07](../07-corpus-and-training.md): exhaustive tiny teachers, verified sampled plans with newly generated candidates, fixed collection rounds, counterfactual quartets and higher-order conflict examples. Unknown is neither a feasibility negative nor a complete semantic label.
3. Separate value, proposal and risk objectives. Elementary teacher mass, unreachable candidates and sampled target approximation are explicit. Conditional proposal context is real input; independent object likelihoods do not magically learn a joint repair distribution.
4. Implement [13](../13-semantic-fidelity-supervision.md) as optional offline weak supervision after symbolic conformance: grounded meaning judgements, ties/abstentions, anchors and independent evaluation. Keep edit cost outside the judgement and personal profiles outside the primary study.
5. Deliver G3/G4 evidence: leakage tests, target/feature migration, generated-pool checkpoint rule, fixed-pool diagnostics, loss-mask tests, feasible quartet contrasts and three-way risk cases. No external LLM job is initiated by reading this work package.
