# XR-2.1 design audit and remaining risks

**Revision:** 30 September 2026. The first implementation was reviewed at `b4c1ed0d5e12c45974bdcb4d230fb2ab6c6deb04`. [12](12-implementation-migration.md) owns the code-level checklist; [11](11-benchmark-evidence.md) owns numerical evidence. This document records the methodological decisions.

| Issue | Resolution | What still needs evidence |
|---|---|---|
| Grammar-valid circuits presented as globally safe | Encode specific proved contextual conditions; retain complete global verification | Benefit beyond a decoder with the same constraints |
| Large monolithic circuit and misleading size cap | Small family/template circuits; correct native node lifetime and size telemetry | Coverage and cold/warm performance under matched budgets |
| Local bans invalidate joint ontology edits | Permanent bans depend only on immutable background; editable supports become conditional global cuts | Counterexample fixtures and cut replay |
| Low generation coverage hidden by fixed-pool scores | Select/report generated-pool development quality and all failures | New candidate coverage on fresh families and real inputs |
| Missing-candidate control regenerated the answer | Separate evaluator-only removal from the final generated pool from a named vocabulary/grammar omission; verify the intended omission after every producer | End-to-end omitted-term/bundle identity tests |
| Different pair construction during training and inference | One evidence-aware selector, including candidate-induced links | Identical frozen pair sets and transfer effects |
| Risk of misusing infeasible plans as semantic-value negatives | Separate conflict-risk learning; feasible labelled plans supervise meaning | Calibration and reduced calls to matched quality |
| Three-way conflict labelled as three pair conflicts | Plan-level risk plus proved support hyperedges | Higher-order fixtures with feasible constituent pairs |
| Independent proposal likelihood called joint reasoning | Explicit proposal context; supervised coordinated samples; frozen final objective | Whether conditional refinement improves coverage |
| Exact teachers too expensive for new candidates | Exact small universes plus verified sampled plan rankings | Coverage, selection bias and unknown-label rates |
| User proxy becomes the main LLM story | LLM weak labels for intended-meaning preservation; personal profiles optional | Independent judges, anchors, evidence and abstention |
| LLM consistency judgement overrides logic | Hard facts/verifier remain authoritative; AI rationale is soft evidence | False-label audit and independent held-out agreement |
| MaxSAT replaced by RL to reduce verification | Keep MaxSAT, prove reusable cuts and learn shortlist order first | Incremental benefit before considering RL complexity |
| Only complete-assignment cuts used | Connect sufficient axiom supports, duplicate origins and activation conditions | Cut generalisation without lost feasible plans |
| Backend chosen from ontology size alone | Qualify full constructs and query support first, then measured cost | Qualified native adapters and unknown-rate/cost tradeoff |
| Late timeout erases an earlier proved violation | Stream completed hash-bound events; positive acceptance still all-or-nothing | Timeout/crash injection and replay |
| Greedy and exact arms pay different baselines | Share or consistently charge four diagnostic theories | End-to-end and amortised timings |
| New public candidate classes omitted from coherence | Validate complete frozen signature on every loading/preparation path | New-class and direct-record regression cases |
| Worker timeout starts after expensive serialisation | Supervise startup/preparation and cleanup end-to-end | Slow pickle/startup and worker death fixtures |

## Decisions preserved from XR-2

There is no mandatory trusted mapping subset. Revision objects replace complete asserted bundles. Eligibility is explicit for ontology edits. Axiom provenance and all duplicate emitters remain part of patch semantics. Full active-expression satisfiability prevents vacuous “repairs”. Typed semantic probes treat disjointness differently from existential/subsumption consequences. Unknown alternatives remain in bounds; there is no assumed all-delete feasible fallback. Coherence, objective optimality and semantic correctness remain separate claims.

## Unresolved research risks

1. **Intent is not identifiable from logic alone.** Two coherent repairs can express different plausible meanings. Evidence-poor cases need abstention and alternatives, not an invented true label. AI labels remain a proxy without independent expert validation.
2. **Bounded coverage.** Retrieval, grammar, compilation and sampling can each miss the useful action. More candidates may also make selection and verification harder. Expose each source of loss.
3. **Compilation cost.** Efficient operations on a compiled circuit do not imply cheap compilation. Semantic constraints may simplify or enlarge it. No theoretical guarantee makes circuits universally preferable to decoding.
4. **Approximate utility.** Sparse unary/pair factors cannot capture arbitrary plan-level meaning. A flexible risk head addresses feasibility ordering, not this representation limit. Diagnose higher-order regret before adding more solver factors.
5. **Selection bias.** On-policy sampled repairs overrepresent existing proposals and easy-to-verify plans. Retain diversity strata, counterfactuals and unknown masks; evaluate on a separately frozen distribution.
6. **Reasoning limits.** Supported fast fragments may leave expressive biomedical cases unresolved. Detector completeness, accepted proof scope and final policy cannot be inferred from a backend name.
7. **Ontology-edit risk.** Increased cost encodes caution, not epistemic certainty. Return exact occurrence patches, provenance and alternatives; prevent a model from treating ontology deletion as a cheap shortcut.
8. **Semantic scale and cost confounding.** Unanchored pair preferences identify rankings only, while MaxSAT needs a scale relative to explicit costs. Fix calibration anchors and separate cost from meaning labels.
9. **Small real sample.** Shared ontologies and few independent pairs limit strong transfer claims. Model seeds and thousands of mappings do not create independent domains.
10. **Implementation size.** Prefer existing shared-core APIs, PySDD and PySAT; reuse the qualified restriction-aware detector if its code/proof contract is available. Avoid a new solver, generic agent framework, bespoke OWL parser or RL environment in this revision.

An unsuccessful comparison is a legitimate outcome. The circuit, risk or LLM component is retained as the default only if its declared experiment supports the relevant benefit without weakening the logical contract.
