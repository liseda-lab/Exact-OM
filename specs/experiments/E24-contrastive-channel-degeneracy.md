# E24 — Missingness-aware contrastive evidence

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Normalized, absolute, asymmetric and off difference primitives with diagnostics exist.

**Agent must implement:** Supported/contradicted/unobserved semantics, zero contradiction authority for unjustified one-sided absence, controlled perturbations and valid typed asymmetry binding.

**Inputs/bindings to resolve:** Resolve a relation-bearing case and actual incompatibility semantics from the available data.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **D1 or feature-selected relation-bearing pair, with D0 control**.
Resource envelope: **channels** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, pool_freeze**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

Reproduce the one-sided-empty example and show why 0.50/0.90/0.00 are different semantics, not mere scaling. Test deletion, duplication, irrelevant insertion, equivalent serialization and side reversal. Measure corrected and harmed decisions; a more active negative channel alone is not success.

At most **5 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- normalised: current control.
- diff_off: removal control.
- missingness_aware: semantically justified contradiction only.
- absolute: raw absolute formulation diagnostic.
- asymmetric: optional typed-interpretation diagnostic, not automatic promotion.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **F1 and correction-minus-harm under natural and controlled missingness.**
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

- Unobserved relation on one side is not automatically contradictory.
- Explicit incompatible facts can contribute negatively with traceable semantics.
- Empty-both and empty-one cases have declared score/quality behavior.
- Changed contrastive semantics preserve reusable ontology/vectors but invalidate their score descendants.

Durable boundaries: **Difference raw components, perturbation inventories, scores.**
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
