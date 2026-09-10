# Single-node run plan

**Normative v2 plan, 2026-09-09. All times below are execution caps, not measured predictions.**
The goal is useful, interpretable evidence in 14–21 days of continuous single-node operation.
Every applicable family gets a small, case-appropriate initial comparison. Broad search
means coverage of methods/features; only selected comparisons expand to larger samples, more
models, or a wider reporting panel. Large follow-on variants are not prerequisites.

## 1. Resource contract and feasibility gate

Target one RTX 5090 and 64 GB or 128 GB RAM. Record hostname/node identifier, actual VRAM,
usable RAM, CPU count, scratch capacity, OS, package versions and encoder/OpenRouter throughput.
Use one heavy GPU worker; CPU work and vector/graph caches must fit 64 GB in the normal profile.
The 128 GB variant may improve throughput or admit a larger feature case, with the same numerical
semantics. All generative LLM roles use OpenRouter; local encoders and non-generative heads use
the GPU. No local generative fallback is allowed. The user grants broad OpenRouter spending
discretion: forecast/log USD from current rates and retain request/token limits, without requiring
another monetary-cap approval.

Before opening development outcomes, time cold ontology loading/projection, warm embedding reuse,
64-source scoring, one small training job, and at most 20 LLM source groups if that capability is
enabled. Forecast each matrix from measured work units (unique sources, candidate pairs, tokens,
epochs), include a 1.5 safety multiplier, and write budget-plan.json. Warm-cache figures alone
cannot establish first-run feasibility. Include model/data materialization and disk growth.

The budget is cumulative across attempts, repaired directories, and reused campaigns' newly
performed work. Log both active node-hours and elapsed wall time; a pause does not reset spend.
Retained node allocation counts while idle. Human-requested pauses and external waiting are
reported separately from the continuous-runtime estimate.

| Budget envelope | Core 14-day cap | Additional allowance in 21-day profile |
| --- | ---: | ---: |
| E00 preflight, production replay, error inventory | 12 h | 0 h |
| Retrieval, including bounded E20 | 48 h | 24 h |
| E26/E24/channel/fusion screens | 48 h | 18 h |
| Selection, calibration, NIL, limited transfer | 36 h | 18 h |
| LLM viability, direct/comparative evidence, conditional E21 | 42 h | 24 h |
| Property/instance/representation/typed/graph capability branch | 18 h | 36 h |
| Broader development checks at gates | 18 h | 6 h |
| Frozen final study and reporting | 54 h | 18 h |
| Protected recovery/overrun reserve | 60 h | 24 h |
| **Total** | **336 h** | **168 h; total 504 h** |

Unused earlier allocation may move to later work under a recorded budget amendment, but final
study and repair reserves remain protected. Node-hours, request/token admission, and the
recorded OpenRouter spending authorization must all pass. No additional fixed USD cap is required. Do not launch a cell whose conservative completion estimate crosses its allowance.
Checkpoint and mark deferred_budget rather than silently shortening an epoch, source universe,
or final evaluation. Implement early-stop/patience only where the training recipe declares it.

The extended profile is chosen before final results are exposed. Extra time first funds the
priority-ordered expansions below; it is not permission for an exhaustive grid. If even the
minimum final design cannot fit, stop at the feasibility gate with the measured bottleneck and
revised resource/scope requirement. Do not promise a 21-day result on arbitrary hardware.

## 2. Data roles

| Role | Default population | Permitted use |
| --- | --- | --- |
| D0 training/development | Bio-ML NCIT–DOID train and valid | Most fitting and selection; 300 valid sources initially, nested expansion to min(1,000, all eligible) |
| D1 structural sentinel | SNOMED–FMA valid, 300 sources | Development robustness after retrieval and before final freeze; train labels only for the declared supervised control |
| D2 optional contrasting sentinel | Revision-pinned OMIM–ORDO valid, if available | Sparse-context/definition-heavy diagnostics in the extended profile; add its actual descriptor instead of guessing one |
| P0 property development | Capability-selected Conference/OAEI-KG/BioKG pair | Property-specific evidence and label semantics; distinct P1 final case |
| K0 instance/KG development | OAEI-KG v4 starwars-swg, or capability-selected BioKG instance pair | Instance/KG development; explicit research split policy below |
| T0 typed development | Available BioKG-Align pair with equivalence and both subsumption directions | E14 development; distinct T1 final case |
| N0 natural-NIL development | DISO stix-d3fend or capability-selected natural-NIL task | E04 development; distinct N1 task/split final |
| R0_case representation development | Same-information OWL/CSV view plus available BioKG Datalog case | E13 parity before enrichment; preserve information provenance |
| K1 extension final | OAEI-KG v4 starwars-swtor | Untouched extension reporting when that claim is selected; disclose the shared source ontology |
| H0 final in-pair | NCIT–DOID test | Frozen class-stack evaluation |
| H1 final structural | SNOMED–FMA test | Frozen class-stack evaluation |
| H2 final held-out pair | SNOMED–NCIT test | Pair-transfer check; no use of its outcomes for selecting the method |

H0/H1/H2 are the default final class panel and correspond to currently registered Bio-ML tasks.
H2 shares ontologies with development tasks; call it a held-out ontology-pair evaluation, not
unseen-ontology/domain generalization. In-pair supervised H2 artifacts may use its designated train
split with the already-frozen fitting recipe and internal training folds; the transfer/label-free
arms must not consume those labels. None may use its reporting labels or use H2 to select the recipe.

Final references stay inaccessible to screen workers, including logs/metrics APIs. Model inference
on a final ontology is allowed only with a declared transductive access policy; reading its gold
or adding gold mappings as graph edges is not. Create a read-access/exposure ledger.

The KG descriptor currently supplies a test reference, not a train/valid split. Do not pretend
otherwise. For K0 only, declare its entire reference a development resource, derive grouped
60/20/20 research train/valid/internal-check partitions with seed 17 before measuring outcomes,
and record original role=test, consumed role=research_development, transformation, and hashes.
It can no longer support an untouched official-test claim. K1 remains untouched. Preserve
completeness/unknown-negative semantics and grouped/component splits; if too few independent
groups or no valid supervision exists, run label-free evidence tests and defer supervised claims.
K1 sharing one ontology must be disclosed, with overlap/component sensitivity where feasible.

The user confirms OAEI and BioKG-Align are available. Resolve supplied/local/node paths and
real descriptors rather than propagating the old biokg_not_published or not_materialized flags.
Before model-quality outcomes, choose one development case per feature by entity kind, reference
semantics, evidence coverage, feasible size, then stable task ID. Prefer the named defaults when
these meet the feature need. Record exact IDs, hashes, split roles and the reason in case-lock.json.
A different case is appropriate when D0 lacks the required signal; do not choose the case where a
method already scored best. Choose distinct held-out feature cases at the same time. Generalize
the disclosed K0 research-split rule to P0/T0/N0 when no official development split exists;
preserve separate final data. A stale repository stub is an implementation/binding gap. If actual
labels lack a required capability, mark that case inapplicable and use the predeclared eligible
alternative before exposure, or report the precise remaining input gap. Do not fabricate labels.

Within D0, source sampling is deterministic and nested. Keep all candidates for a selected source.
Stratify declared diagnostics by observable difficulty and separately label development
error-enriched samples. Report selection probabilities/stratum weights; never report enriched
accuracy as the natural population accuracy. Source caps apply before expensive evidence work.
Keep labeled reference columns physically separate from the unlabeled candidate input.

## 3. Internal screen phases and gates

**G0 — runnable foundation.** E00 acceptance passes, required inputs are pinned, checkpoints
survive deliberate interruption, the cold/warm budget forecast fits, and the production baseline
has a full global/local replay. Emit baseline error attribution and the candidate-recall ceiling.
Resolve the WP-M/WP-N release-status discrepancy against actual evidence; use one supported,
pinned execution stack throughout. Broad unrelated engineering work is not a screening prerequisite.

**G1 — retrieval freeze.** E05 baseline/encoder/fusion/adaptive-k screens, then at most one
progressive combination. E20 gets one bounded encoder-training recipe and one cross-encoder
comparison if safe labels and budget permit. Optional LLM query rescue belongs here and cannot
depend on later E07/E21 winners. Compare recall at mean k within max(1 candidate, 2%). Use D1
only for selected retrieval survivors. Freeze one label-free pool policy and, if warranted, one
supervised policy. Publish a baseline choice when nothing improves. Downstream experiments
depend on these output policies, not on a successful treatment or every optional experiment.

Cheap E26/E24 diagnostics can run on baseline cached evidence before G1, but any proposed
downstream change must be replayed under the frozen G1 pool before selection.

**G2 — mechanism and decision selection.** Every applicable family gets its bounded initial
comparison on the feature-appropriate case; D0 remains the default, not a forced universal case.
E16 gets a small donor-transfer check and E22 a small three-budget curve, not full matrices.
E26 decomposition first; E24 missingness/contradiction;
bounded E06/E08/E09/E02; E10/E19 fitting; E03/E15 baseline decisions, then E18 reranking;
E01/E04 remaining decisions. Use D0,
cached evidence, one seed, and the per-experiment caps. Changes requiring different evidence
must materialize those artifacts, not silently reuse baseline evidence. Avoid full Cartesian
crosses; permit only named sequential comparisons. Preserve lexical-only, lexical-plus-attributes,
active-equal-weight, and current supervised baselines.

**G3 — useful LLM intervention.** E25 fixes controls and measures standalone judgments,
correction/harm, and practical headroom. E07 tests direct evidence and comparative source
judgments with a gate independent of pair confidence. E21 fits a router/student only after
helpful judgments exist. Optional evidence acquisition is compared under an explicit token/call
cap. Late pool-expansion ideas are diagnostic for a later design, not changes to the G1 pool.

**G4 — development robustness and composition freeze.** Build at most three composite candidates:
baseline, the core selected stack, and one justified optional LLM/extension profile. Run D0 plus
D1 (D2 only when predeclared), check high-risk interactions, repair incompatibilities, then freeze
one main stack and at most one separately budgeted optional profile. Individual experiments do
not inspect final data first. The selection/design record includes every final contrast and its
input/artifact hashes, endpoints, seeds, source universe, cost bounds, and multiplicity family.

**G5 — one final study.** E17 executes the frozen panel only. It cannot prune a harmful component
and rerun a nicer stack on the same exposed test data. Keep failures and underpowered outcomes.
Use the repair protocol for actual bugs; a method change after exposure needs fresh evidence.

The feature branches use P0/K0/T0/N0/R0_case and their own capability gates; they do not
inherit class supervision. The initial E11/E12/E13/E14/E23 screens are part of the broad search.
Only larger expansions are optional. An inapplicable/failed/budget-limited branch emits an
explicit terminal status and cannot be reported as a negative result. E17 consumes only the
eligible selected components and must not wait forever for all E01–E26 to win or be available.

## 4. Bounded matrices and selection

Default initial sample: 300 source groups, seed 17. Each E-file caps unique candidates; repeated
analytic replay over identical evidence is accounted separately from a fresh encoder/LLM pass.
Expand at most two non-control survivors per family to 1,000 sources before choosing one.
For stochastic confirmation use seeds 17, 29, 43. Deterministic artifact-identical runs are reused
once and reported as deterministic; three copies are not three independent replications.

The general development quality gate is delta F1 >= 0.003, with local MRR and candidate recall
no worse than -0.005 where applicable. Selection on a small screen is a development decision,
not a significance claim. Among eligible arms: primary quality, then lower cost, then simpler
configuration, then canonical arm ID. A cost candidate may instead use the predeclared
non-inferiority margin -0.005 and >=20% saving; it may not switch endpoints after results.
E05 uses recall gain >=0.005 and matched-pool constraints. Each E-file names exceptions.

Reconstruction tolerance is 1e-6 on deterministic CPU replay and 1e-5 on a pinned GPU backend,
with exact mapping/IRI/role identities. Record measured nondeterminism rather than loosening
tolerances after seeing results. Exactness and safe label use are hard guards.

Final primary endpoint: task-macro global F1 for class equivalence. Local MRR, P/R, retrieval
recall, coverage, calibration, and cost are mandatory secondary outcomes/guards. Primary gain
claims require a paired 95% CI excluding zero and the declared practical effect; cost-primary
claims require their frozen non-inferiority/cost bounds. Report per-task results and underpowered
cells as inconclusive. Never substitute a failed primary endpoint with a better secondary.

Use source-paired TP/FP/FN bootstrap (10,000 draws), paired across arms/seeds, with
candidate-graph/component clustering sensitivity for assignment/anchors/instances. State that
source bootstrap conditions on the sampled tasks and seeds. Three ontology pairs do not support
broad domain claims; report their individual deltas and worst case. Holm applies to multiple
confirmatory component contrasts. An oracle cannot enter selection or a deployed profile.

## 5. Final matrix and cost boundaries

The minimum class final matrix is baseline and frozen stack on H0/H1/H2, plus the frozen
label-free control wherever the main stack is supervised. At most four claimed component
leave-one-out contrasts and two predeclared 2x2 interactions run on H0; reuse identical cells.
A component attribution claim applies to H0 unless its contrast is explicitly expanded to other
tasks. Baseline-versus-stack generalization applies to the full three-task panel.

Core cap: 24 unique arm-by-task designs before stochastic seeds; extended cap: 36.
Selected property/instance/NIL/typed claims each require baseline and selected method on their
matching held-out case. Do not claim broad capability validation from the three class pairs. Deduplicate
overlapping controls/interactions by numerical identity. Choose which four components/two
boundaries to claim at G4, not by reporting performance. Components outside these contrasts
are part of the tested stack but have development-only individual evidence. If the minimum
scientifically necessary matrix exceeds the cap, narrow claims before G4 or defer the optional
profile. Never drop a required control to squeeze in another candidate.

Keep a cost-primary default profile and, only if budgeted, an optional quality profile. Record
wall, peak memory, one-time fitting, amortized reuse, and all LLM role costs. The default profile
uses <=1.2x baseline inference wall time; a more expensive profile needs its own explicit
predeclared cost ceiling and must not be described as meeting that default bound.

LLM planning ceilings: core 20,000 unique requests and 32 million total tokens; extended
40,000 requests and 64 million tokens. Count provider-billed reasoning/cached-token categories
and actual USD as well as input/output. The user grants broad spending discretion; these are
operational breadth limits, not an inferred monetary restriction. G0 may adjust them from the
unique-request inventory and current rates before outcomes, with an explicit forecast. They include preparation, retrieval/tool
judgment, decisions, rationales, teacher calls, retries, and uncertain-completion charges.
Freeze per-phase/per-arm allocations and reserve final calls before screen consumption.
Use at most five candidates per comparative call and at most 200 forced-screen sources initially.
A complete arm may require a lower frozen routing fraction to fit; choose it on development
and throughput, never truncate reporting calls after observing correctness.

## 6. What the 21-day expansion may add

Initial case-based screens span all applicable families. Additional time first expands the
most informative selected channel controls, property/instance/typed/representation cases,
E21 router/student, E22 label curve, and E23 structural-ablation result. A second pinned OpenRouter
model is one sensitivity check, not a model tournament. Large transductive graph training, full
donor-recipient matrices, exhaustive grids and participant recruitment remain follow-up work.
E14 uses the available real BioKG-Align data after typed capability binding. Fixtures validate
semantics only and cannot substitute for a real feature-specific result.
## 7. Baselines, reference validity, and claim boundaries

Preserve the shared controls: intended production behavior and the corrected v2 baseline with
parent R_0; lexical-only; lexical plus definitions/attributes; equal weighting of active channels;
and the existing supervised ranker/acceptance path. Reuse these artifacts across families.
Per-family caps describe that family's named treatments; the expanded planner must additionally
count every required shared control, diagnostic, task, fold, seed, and repeat in the budget.
The cap is never an excuse to omit a comparator or hide uncounted preparation work.

A competitive-matching claim additionally needs at least one strong published matcher under
compatible ontology versions, task modes, source universes, reference conventions and allowed
supervision. At G0 bind one available pinned matcher (for example an existing LogMap/AML/BERTMap
baseline represented in the project), chosen by capability and reproducibility before current
outcomes. Use verified same-task saved predictions where available; otherwise admit one bounded
run recipe and reserve its final cells inside the stated final cap. Do not launch a matcher
tournament. A local-ranking comparator uses the same benchmark pool; an end-to-end global
comparator may use its own retrieval, which must be disclosed. Paper tables with different
splits, supervision or no paired predictions are contextual only, not paired evidence. If this
comparison cannot be completed, report relative improvement and leave competitiveness unresolved.
E20's cross-encoder is also a cost/quality comparator for E07, with its supervision stated.

Before confirmation, freeze the number of independent groups, practical effect or
non-inferiority margin, assumptions, and powered/underpowered/descriptive status for each primary
task/kind/relation slice. A simulation is optional; the declaration is required. Freeze all
source-sampling and transductive graph-access policies, including target overlap, shared-ontology
and near-duplicate sensitivity. Prior paper/benchmark exposure belongs in the exposure ledger;
“untouched” refers to this campaign's selection protocol, not a previously unknown benchmark.
Do not micro-average unlike entity kinds or relations into one large-class-dominated number.
Report exact current-baseline and historical-baseline comparisons only where predictions,
populations and behavior are actually comparable; never invent a historical replay.

Known-incomplete or unknown references retain official raw metrics. “Absent from gold” is not
automatically wrong. For a precision-led claim, freeze a blinded disagreement audit before final
outcomes: union/deduplicate arm-disagreement mappings absent from gold; stratify by arm-only/both,
task/kind and score band; record inclusion probabilities and the sample-size/CI-width target
(at least 50 cases when available). Two domain-competent raters independently label
correct/incorrect/uncertain; a third adjudicates disagreement. Hide arm, score and hypothesis,
use identical evidence, and report agreement plus inverse-probability-weighted adjusted precision
with uncertainty beside raw scores. The audit cannot tune the matcher. Without authorized
available raters, prepare the blinded pack and mark this claim unresolved; do not replace them
with an LLM or delay all valid compute. Positive-unlabelled training controls remain mandatory
where ordinary negative labels are not justified. Human work has a separate schedule.

Quality claims use the frozen endpoint and effect gate. A default-eligible profile additionally
must have no task regression exceeding 0.005 F1 (or 0.005 MRR on a ranking-primary claim), preserve
exact declared explanations, and meet its frozen cost/coverage bounds within each claimed kind
or relation. Any inherent-capability cost override is declared before outcomes and both default
and override outcomes are reported. Scientific nulls and expensive gains remain reportable.
No execution automatically changes product defaults; release/promotion remains a separate task.
Any subsequent R_n is append-only, retaining its parent, flags, artifacts and evidence.

## 8. Dependency outputs and scheduling

Dependencies name completed output contracts, not a requirement that every earlier hypothesis
win. Materialize the following ports with task/kind/role and artifact IDs; a declared baseline
is always an allowed output of a completed null screen:

| Port | Producer and availability |
| --- | --- |
| baseline_evidence | E00 corrected production path; allows cheap E26/E24 diagnostics before G1 |
| E05_initial | E05 initial label-free retrieval selection, before E20 fitting |
| pool_freeze | G1 after E05 and admitted E20; one frozen policy per declared supervision regime, materialized separately for each task/kind/role |
| kind_pool_freeze | E00 property/NIL/instance baseline adapters, with E12's internal retrieval phase replacing the instance baseline before its evidence phase; no dependency on E12's later evidence result |
| typed_pool_freeze | E00/E13 compatible typed-case input and baseline pair pool before E14 typing, independent of the eventual typer |
| selected_heads | G2 frozen E03/E15/E18/E19 fitting recipe or current fitted control; does not wait for E16, E21 or E22 |
| E02_anchors | E02 audited same-kind anchor policy, or E00 anchor-free/current-safe control when the treatment is not selected |
| E25_initial | E25 initial fixed-judge/off/routing/forced-response phase; emitted before its late trust replays and before E07 |
| E07_judgment_evidence | E07 frozen judge format plus compatible actual training-only counterfactual responses; development responses select/assess the recipe and are not training labels |
| G4_frozen_selection | Frozen composite, required final controls/contrasts, fit recipes and case-role/budget locks after all admitted screens terminate |

E12 has an internal retrieval-then-evidence sequence; E04's optional listwise integration runs
after E07, while its non-LLM NIL comparison runs earlier. E25 first diagnoses the fixed baseline
judge; E07 may improve that judge; E21 then learns from training-only forced calls. The initial
E25 routing/viability result does not block E07 from testing a better evidence format. No late
LLM choice changes class retrieval. E25's late trust replays consume E07 responses and cannot
be required to emit E25_initial. Both probability-based replays use the same per-candidate
probability representation; a hard-choice-only provider marks those replays inapplicable while
retaining the source-choice comparison. E26's preliminary baseline screen must replay any selected
downstream change on G1 evidence; that replay is an expansion, not an unresolved dependency cycle.

The blueprint dependency list plus these ports must expand to an acyclic stage graph. Fail the
dry-run on unknown ports, cycles, unbounded branches, unallocated controls or missing inputs.
Use the numeric RUN-PLAN resource-envelope order for scheduling, with one heavy GPU worker.
Fit small CPU heads and prepare independent data/requests concurrently only within measured
memory and rate limits; count overlapping work as actual elapsed single-node time, not a sum
that double-bills the same wall-clock interval.

Training budgets are explicit and separate from evaluation source caps. Initially use at most
2,000 deterministic training source groups per fitted recipe (all if fewer), with source-group
OOF folds; E22 overrides this with its nested 25/100/400 budgets. Record actual positives,
confirmed negatives, unknowns and effective groups. Freeze epoch/patience/step limits before
the recipe runs. Expansion to more train groups must fit the phase forecast and is itself a
named development comparison. At G4 freeze each final recipe and training budget; do not
quietly refit on more labels, different features or a reporting pool.
