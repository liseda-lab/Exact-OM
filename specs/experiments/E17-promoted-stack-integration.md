# E17 — Fresh final stack integration

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Composition, leave-one-out/interaction generation and paired global-F1 reporting helpers exist.

**Agent must implement:** Development-only stack selection, new bounded final panel, explicit task mode across pool ablation, compatible refitting, fresh-role enforcement, and repair-aware result-set aggregation.

**Inputs/bindings to resolve:** Resolve final class and feature-specific data roles before screening; all are user-declared available.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **H0/H1/H2 plus selected feature-specific held-out cases**.
Resource envelope: **final** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, G4_frozen_selection**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Use the full eligible final populations locked before G4; the 300/1,000-source development caps do not truncate confirmation. If full final tasks do not fit, declare a sampled estimand and its sampling/coverage limits before any final exposure, or narrow claims at the feasibility gate. Use paired stochastic seeds 17, 29, 43 and reuse identical deterministic artifacts once.

## Question, treatments, and implementation contract

Do not require individual final-test wins before composition. Choose from development evidence, check D1/feature sentinels, freeze once, and open final references only for this matrix. No baseline/removal may silently change global versus local mode. Refit downstream artifacts when its pool/features change.

Use the final arm-by-task caps and frozen component/interaction limits in RUN-PLAN; never multiply the full development catalogue.

- baseline: frozen v2 production baseline.
- stack_all: one G4-selected stack.
- label_free: mandatory where the main stack uses target supervision.
- stack_minus_component: at most four H0 attribution contrasts.
- interaction: at most two predeclared high-risk 2x2 boundaries.
- optional_profile: only if separately selected and fully budgeted.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Task-macro class F1; separate feature-kind/relation claims with their own controls.**
Use the family rule plus RUN-PLAN's frozen selection, practical-effect, reconstruction and cost
criteria. A screen chooses what to evaluate next; it does not establish a reporting-set claim.
Report all controls, negative results, corrections/harms where relevant, and inapplicable or
budget-deferred cells. Never suppress a difficult kind or source group from the denominator.

Broader validation happens on the designated development sentinel after a promising focused
screen, then only in E17's frozen final panel for the claims selected at G4. Do not run a full
OAEI confirmation for every treatment. A feature-specific claim needs its matching held-out
case; NCIT–DOID cannot substitute for property, instance, natural-NIL or typed-relation labels.
No individual experiment uses final outcomes to qualify its component for E17.

## Acceptance and recovery

- Final references cannot be read by screen selection or candidate composition.
- A harmful final component is reported, not pruned and retested on the same outcomes.
- Required controls/denominators survive deduplication and repair lineage.
- Unavailable/budget-deferred optional branches do not deadlock the core final study.

Durable boundaries: **Frozen composition, predictions, evaluation and final result-set lineage.**
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
