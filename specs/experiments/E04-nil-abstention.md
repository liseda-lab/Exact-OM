# E04 — Benchmark NIL, pool misses, and abstention

**v2 scientific design revision, 2026-09-10: public Bio-LLM benchmark NIL.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Joint candidate/NIL probabilities, grouped fitting, source decisions, and NIL evaluation. See the current preparation ledger for tested scope and execution prerequisites.

**Required contract:** Explicit benchmark-pool labels, grouped fit, output/evaluator integration, distinct absence semantics, and optional listwise-none integration.

**Selected inputs:** Public Bio-LLM 2024 NCIT–DOID and SNOMED–FMA subsets, with original
ontology versions. Their explicit unmatched labels are benchmark-defined NIL, not verified
absence throughout an ontology. DISO public pools have no disclosed answers and remain
submission inputs; BioKG typed preferences are not negative/NIL certificates.

## Focus and dependencies

Primary focused case: **N0: legacy Bio-LLM NCIT–DOID research split; D0 pool-miss diagnostic**.
Resource envelope: **decisions** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, pool_freeze, E03**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

The user authorized choosing a public NIL task on 2026-09-10. Use the historical Bio-LLM
reference-defined unmatched examples under `benchmark_pool` semantics. Only the original
provided candidates receive benchmark positive/negative labels; an unlisted pair remains
unknown. Do not reinterpret these labels as verified ontology-wide NIL or apply them to the
expanded Bio-ML 2026 ontologies. Artificial gold removal remains a separate pool-miss diagnostic.

N0 uses a matched/unmatched-stratified 60/20/20 source split, seed 17: 60 training, 20 validation,
and 20 held-out research sources. Preserve every original pool, including the published
NCIT–DOID source with 70 candidates; the other 99 pools have 100. Fit only on N0 training,
select only on validation, and expose the internal-check labels only after freezing. N1 is the
separate SNOMED–FMA 100-source public historical reporting case. It is not training data.
Neither supports a blind official-test or nonbiomedical-generalization claim. Do not reuse
supervised D0/D1 heads whose training sources overlap these historical examples. A transferred
NIL head requires an explicit frozen application binding; otherwise that arm remains blocked.
Use non-NIL MRR as a guard. See [label provenance](LABELS-AND-SUBMISSIONS.md).

At most **5 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- nil_off: current acceptance control.
- nil_heuristic: joint-scale label-free abstention.
- nil_fitted: calibrated joint candidate/NIL head.
- pool_miss: remove gold from development pools, diagnostic only.
- listwise_none: conditional paired integration with the E07 winner.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Benchmark-pool NIL-aware F1; benchmark NIL and artificial pool misses reported separately.**
Use the family rule plus RUN-PLAN's frozen selection, practical-effect, reconstruction and cost
criteria. A screen chooses what to evaluate next; it does not establish a reporting-set claim.
Report all controls, negative results, corrections/harms where relevant, and inapplicable or
budget-deferred cells. Never suppress a difficult kind or source group from the denominator.

Broader validation happens on the designated development sentinel after a promising focused
screen, then only in E17's frozen final panel for the claims selected at G4. Do not run a full
OAEI confirmation for every treatment. A feature-specific claim needs its matching held-out
case; class-equivalence labels cannot substitute for property, instance, verified ontology-NIL
or typed-relation labels.
No individual experiment uses final outcomes to qualify its component for E17.

## Acceptance and recovery

- NIL and real candidates are on one declared probability scale.
- A true target outside the top-k is not relabeled ontology-wide NIL.
- Writers, source denominators, and evaluator counts agree on NIL cases.
- A model/threshold repair can rerun decisions without losing underlying evidence.

Durable boundaries: **NIL fit and source-level decisions/output.**
All changed inputs/semantics invalidate their consuming descendants; preserve valid upstream
artifacts. Store completed source/request/fold IDs and attempt lineage. Tests must demonstrate
this family's checkpoint/repair boundary, not merely mirror a formula. Mark screen-ready and
confirm-ready separately in the runtime readiness ledger, with evidence, once these checks pass.

## Deliverable

Produce a result record with actual treatment/configuration, case/role, supervision, artifact
IDs, source counts, metrics/cost, controls, uncertainty, decision and reason. A legitimate null,
removal, or inapplicability is a deliverable; an unimplemented arm is not an empirical null.
Resolve this family's question into numbered research questions and a primary endpoint in the
executable design before its screen; answer each as supported, not supported or inconclusive
with evidence. RUN-PLAN section 7 governs incomplete references and claim limitations.
