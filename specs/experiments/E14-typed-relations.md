# E14 — Equivalence versus subsumption on typed BioKG data

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Heuristic/graph relation primitives and typed writers exist; trained typer, full bridge integration and coherence audit are incomplete.

**Agent must implement:** Replace stale BioKG stub with the available data descriptor; grouped typed fitting, leave-query-bridge-out semantics, oracle-pair/full-pipeline evaluation and supported-profile reasoning audit.

**Inputs/bindings to resolve:** The user states BioKG-Align is available; resolve its actual paths/revision/typed split. Do not retain the old biokg_not_published deferral.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **T0: available BioKG-Align pair with =,<,> labels; distinct T1 final**.
Resource envelope: **extensions** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, E13, typed_pool_freeze**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

Choose one actual typed pair for broad method screening. Run oracle-pair typing separately from end-to-end pair detection. No calibration or rule may use T1 outcomes. Exclude the queried mapping as its own bridge. Cycles/SCC collapse and logical unsatisfiability have different metrics.

At most **6 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- all_equivalent: required lower control.
- hierarchy_heuristic: current control.
- graph_entailment: relation paths conditional on declared anchors.
- learned_three_way: bounded multinomial relation head.
- semantic_then_learned: selected hybrid with explicit conflict abstention.
- bridge_parity: optional supported-OWL reasoner comparison.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Relation-macro F1 with oracle-pair and full-pipeline results separately.**
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

- < means source-subsumed-by-target consistently in every adapter/writer.
- A mapping cannot be independently justified solely by inserting itself.
- All relation labels have real counts/provenance; unresolved label completeness remains explicit.
- Reasoner timeout/unsupported profile is unknown, not safe or proven equivalent.

Durable boundaries: **Typed train features/heads, bridge queries, typed predictions.**
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
