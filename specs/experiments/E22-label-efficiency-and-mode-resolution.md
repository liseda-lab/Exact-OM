# E22 — Small label-efficiency and mode-resolution search

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Supervision modes/policy fields exist; grouped budget subsampling, active selection and fitted policy artifacts are missing.

**Agent must implement:** Nested deterministic label budgets, component-specific effective-unit counts, one active-versus-passive comparison, and frozen crossover/policy fitting.

**Inputs/bindings to resolve:** Recorded training-label budget by component/kind; no need for a new human annotation campaign.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **D0; feature-specific kind when labels differ**.
Resource envelope: **decisions** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, selected_heads**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

Apply this curve to one promising shared component recipe, not every model-stage combination. Budgets are nested and reference-safe. Publish inconclusive crossovers where sparse data cannot distinguish performance. Do not automatically fit an elaborate auto policy from three noisy points; compare with a fixed simple policy.

At most **5 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- budget_25: 25 effective training source groups.
- budget_100: 100 groups.
- budget_400: 400 groups or all available, explicitly recorded.
- label_free: selected zero-target-label control.
- active_100: optional active versus passive selection at the same 100-group budget.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Quality/cost versus effective labels; policy benefit over fixed resolution.**
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

- Raw mapping rows cannot masquerade as independent/effective source groups.
- Selecting queries or stopping active labeling never uses unseen reporting labels.
- The same group at a larger budget is not counted as a fresh independent sample.
- Policies carry kind/pool/features/count definitions and have a deterministic label-free fallback.

Durable boundaries: **Budget memberships, active-query history, per-budget fits and frozen policy.**
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
