# E21 — Exemplars, net-benefit routing, and a justified student

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Canonical controls/guards exist; exemplars, counterfactual router labels, student training and fitted trust are missing.

**Agent must implement:** Training-only exemplar retrieval, actual forced-call counterfactual data, net-benefit router fitting, gold-only/student comparison, and immutable teacher bindings.

**Inputs/bindings to resolve:** OpenRouter teacher identity and allocated spend; actual fitting code and compatible non-generative student.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **D0 or the E07-selected feature case**.
Resource envelope: **llm** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, E07_judgment_evidence, E25_initial**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

First establish that the judge can improve decisions. A null old uncertainty gate does not cancel this test; a demonstrably unhelpful judge can screen out supervised extensions. Initial broad screen uses one small recipe each, then only a survivor expands. Include teacher-data generation cost and preserve all raw responses.

At most **5 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- frozen_judge: selected E07 control.
- knn_exemplars: at most three training-source exemplars.
- benefit_router: predicted correction-minus-harm per token.
- student_gold: same small student trained on gold only.
- student_distilled: same student plus pinned teacher judgments.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Net final decision benefit per cost; teacher agreement alone is insufficient.**
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

- No exemplar retrieves the query or a development/final reference as training gold.
- Router targets use complete/adjudicated outcomes and distinguish correction, harm and no change.
- Student variants share architecture/features/data budgets except teacher information.
- Provider/model changes invalidate teacher/router compatibility rather than silently changing deployment.

Durable boundaries: **Teacher request ledger, counterfactual examples, exemplars, router/student training.**
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
