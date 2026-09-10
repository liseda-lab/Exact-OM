# E12 — Instance matching and relational evidence

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Instance-kind plumbing exists; complete evidence bundles, shuffle and anchor crosses are incomplete.

**Agent must implement:** Type/literal/relation bundles, kind-specific multi-view retrieval, grouped research partitions, deterministic degree/predicate-preserving shuffle, and instance evaluation.

**Inputs/bindings to resolve:** Resolve the available KG split/reference completeness; the current official descriptor may require an explicitly disclosed research development partition.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **K0 OAEI-KG starwars-swg or a richer available BioKG instance pair**.
Resource envelope: **extensions** in RUN-PLAN.
Prerequisites/consumed outputs: **E00**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

Use a genuine instance/entity-alignment case; record linked records, duplicates, hubs and identifier-poor entities. Freeze the instance pool independently of the class pool before comparing evidence. An initial multi-view retrieval pass is not mixed with the relation-effect contrast. Run labels versus multi_view retrieval first, freeze the selected instance pool, then compare labels/types_literals/relations/shuffle on it. Baseline retrieval is a valid freeze output when multi_view loses.

At most **5 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- labels: lexical baseline.
- types_literals: typed attribute evidence.
- relations: bounded relation context.
- relations_shuffled: structural negative control.
- multi_view: one retrieval variant evaluated before its kind-pool freeze.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Instance F1, non-exact-label slice and component sensitivity.**
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

- Shuffle preserves declared degree/predicate statistics while destroying alignment signal.
- Training seed links never include validation/final mappings.
- Source/component split and duplicates are recorded; evaluation does not count candidate rows as independent.

Durable boundaries: **Instance pool, relational features, seeded/shuffled graph and heads.**
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
