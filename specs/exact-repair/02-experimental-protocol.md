# XR-2.1 common experimental protocol

**Revision:** 30 September 2026. Every XR-E study inherits these requirements. Conformance, exploratory measurement and confirmatory evaluation are separate stages.

## Cohorts and evidence

1. Small generated theories with complete finite inventories and fully decided teacher outcomes.
2. Larger generated theories with structured interacting conflicts, newly generated candidates and verified sampled plans.
3. Controlled corruptions on training-side Conference structure, then held-out actual matcher outputs.
4. Bio-ML transfer/scale with the 2024 selected-content and 2026 whole-ontology releases kept separate.

The first archived pilot is generated-only and has informed this redesign. It is now regression/exploratory evidence for XR-2.1; obtain fresh held-out parents for confirmatory tests. Conference/Bio-ML statistics in [11](11-benchmark-evidence.md) do not supply gold repair plans. References are partial correspondences, not exhaustive negatives or intended ontology patches.

Capture each original matcher alignment once, with relation interpretation, score provenance and optional evidence. Supply identical snapshots and captures across matched repair arms. Do not rerun matching differently for each repair method. Report raw matcher output and any upstream repair separately.

## Dataset construction and difficulty

Store clean parent, corruption operator and latent intent in the teacher-only partition. Deployment data contains only the presented ontologies, provisional alignment, available descriptions/evidence and bounded diagnostics. Generate structurally coherent controls as well as contradictions; random opaque names cannot support semantic interpretation by an LLM.

Explicitly vary: number of editable objects; support cardinality; distinct conflicts per mapping; overlap among supports; incidence-graph cycles; hub mappings; chain length; restriction type; ontology-edit eligibility; newly introduced candidate conflicts; pairwise and higher-order semantic complementarity. A graph cycle by itself is not an OWL contradiction. Every declared conflict family needs an actual symbolic witness and every coherent control needs completed verification at its stated scope.

First measure Conference's discovered support distribution under a frozen extraction budget. Use training-side measurements to shape generated difficulty, while keeping separately defined harder extrapolation cases. No fixed conflict-per-mapping number may be inferred from ontology sizes or unsatisfiable-class counts. Unextracted supports make measured incidence a lower bound; report truncation.

## Splits and leakage

- Split generated structural parents before names, scores, corruption variants or samples. All derived plans, LLM comparisons and teacher records inherit the parent split.
- Keep pair orientations, matcher outputs, corruptions and extracted neighbourhoods of each Conference pair together.
- Report pair holdout separately from whole-ontology holdout. The historical ekaw holdout (six incident test pairs, fifteen remaining pairs) is a named comparison, not six independent unseen domains.
- Keep ontology transfer and matcher transfer as distinct factors. If an ontology has appeared in training, do not describe a new pairing as an unseen ontology.
- Training-side real structure/reference labels may support a declared adaptation arm. Freeze development-only model selection, evidence budgets, calibration and costs before final test.
- Official Bio-ML training/validation and any licensed sources retain their specified split identities. Shared NCIT/SNOMED content prevents a claim of independent ontology transfer unless the design explicitly resolves it.
- Querying the presented test input at inference is allowed within the common budget. Future verification outcomes cannot be retrospective graph features for the decision that generated them.
- LLM annotation of held-out cases belongs to independent evaluation only. It cannot become model fitting, prompt tuning or checkpoint feedback.

## Required controls

| Comparison | Must be matched | Intended difference |
|---|---|---|
| No repair, score greedy, exact fixed-cost deletion | Input/policy, initial diagnostics and accounting | Selection and repair capacity |
| Symbolic rich-action system versus learned system | Action language, retrieval evidence, final verifier, proved cuts, budgets | Candidate prioritisation and/or calibrated semantic estimates |
| Fixed-pool unary versus pairwise value | Frozen candidates, costs, teacher outcomes, pair selector | Interaction benefit model |
| No risk versus plan-risk scheduling | Frozen objective, same master/cuts/shortlist sizes | Verification order; report first incumbent separately from optimum |
| Grammar-only versus ontology-informed circuits | Menus, language bounds and sampling budget | Encoded semantic constraints |
| Ontology-informed circuit versus semantic grammar decoder | Exactly the same supported semantic conditions and menus | Compilation/conditioning versus incremental decoding |
| Exhaustive versus sampled supervision | Training split and evaluation, declared labelling budget | Coverage/target approximation; not an implied exact distribution |
| Symbolic versus symbolic plus LLM fidelity | Hard facts, costs, comparison/evaluation protocol | Weak semantic supervision |
| HGT versus R-GCN/no-graph | Readouts, target definitions and comparable parameter/search budgets | Backbone and graph information |

Use both fixed-inventory and generated-pool evaluations. The former isolates value/selection, the latter tests the full candidate pipeline. A supplied finite inventory result cannot establish candidate-generation performance. The existing “symbolic” retained-axiom reward is a historical baseline, not the strongest symbolic semantics baseline.

Action ablations filter a common pool without regeneration, or explicitly run a separately named language-generation comparison. Report candidate identities in both. Non-complex alternatives remain guaranteed wherever eligible across all generators. If a shared symbolic candidate inventory is expensive to construct, count that construction cost rather than giving it free to one arm.

## Metrics and accounting

Report all scheduled cases, requested/started/completed attempts and unique reused case results. Deduplicate reused artefacts in scientific denominators while retaining operational attempts and their cost. A finished process does not mean logical success.

| Area | Required measurements |
|---|---|
| Logical | Four baseline reports; verified feasible/infeasible/unknown; complete scope and unsupported obligations; witnessed residual violations; source exception counts |
| Semantic | Independent typed consequence vector with masks; calibrated benefit and edit cost separately; held-out fidelity comparisons with ties/abstentions and evaluator identity; no self-score as ground truth |
| Generation | Retrieval coverage, grammar representability, requested/effective family coverage, compile failures, duplicate draws, useful verified candidates, full versus reduced pool |
| Search | Integer objective, incumbent, valid upper bound/gap, master solves, permanent cuts, deferred/unknown plans, completed proof of pool optimality |
| Verification work | Diagnostic passes, candidate passes, individual logical queries, reasoner initialisations, support extraction work, cache hits, wall time and memory |
| Whole pipeline | Input/preparation, retrieval/graph, circuit construction, sampling, neural scoring, solver, reasoning, serialisation/startup and cleanup time; peak memory; label/training cost separately |

Primary efficiency endpoints are time/queries to first verified repair and time/queries to a predeclared matched quality. Also report time to a proved optimum in the frozen objective. A method returning a low-quality feasible repair quickly is not automatically better. Risk scheduling can reduce work to an incumbent while increasing work to certify an optimum.

For circuit comparisons report cold compilation and warm reuse separately, with number of reuses required to amortise compilation. Node allocation, live/root-reachable nodes and SDD elements are different metrics. Include failed compilation cases in coverage and wall-time denominators.

Use identical accounting of O_s/O_t/union/T_0 diagnostics across controls. If the report shows shared amortised preprocessing, also show the end-to-end cost when it is not precomputed. Cache policy, hardware and concurrency are fixed or stratified. LLM token/call cost, detector cost and sampling retries are included in the relevant training/inference totals.

## Statistics and decisions

Treat clean structural parents and ontology pairs as grouping units; mappings, plan samples, seeds and repeated corruptions are nested observations. Pairs sharing ontologies are dependent: show per-pair outcomes and sensitivity to ontology-level grouping. Three biomedical pairs are three pair contexts, not thousands of independent repairs.

Report paired effects and grouped uncertainty intervals, failure/unknown coverage and per-family/per-pair results. Verified-subset quality must appear beside all-scheduled-case status tables. Never silently remove unknowns or code them as coherent. Exact regret requires a complete, applicable exhaustive teacher; otherwise name the best observed comparator and its limits.

Freeze hypotheses, split hashes, metrics, stopping objective, budget grid, selection rule and multiple-comparison policy in the new run manifest. Exploratory pilots may choose settings from development data. Later changes require a dated amendment and fresh confirmatory holdout where appropriate. Do not promise significance or a sample size before a grouped power/precision analysis is possible.

## Failures and reproducibility

All expensive calls have finite deadlines, enforceable memory limits where supported, retry ceilings and cleanup grace. Unknowns retain their bound and query masks. A compile failure may allow elementary repairs, but must remain a generation failure for that family. An interrupted teacher cannot invent a scalar label by renormalising the remaining answered probes.

Persist code/dirty-patch hashes, input/import/capture hashes, feature/schema/compiler/backend/model identities, menus and language bounds, frozen objective/policy/calibration, sampling context/seeds, split ancestors, evidence cutoff, support cuts, pending plans, LLM annotation provenance and stage resources. Published external statistics, archived pilot rows and newly measured results remain separately labelled.

Any falsely authorised feasible output invalidates that arm's logical-validity claim until fixed and rerun. Lack of complete biomedical verification is a result to report, not a reason to silently weaken policy. Tests demonstrate specific conformance properties, not universal reasoner soundness.
