# E23 — Graph structure with controlled information loss

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Graph-channel schema/guards exist; fitted graph-head production and controlled TBox input orchestration are missing.

**Agent must implement:** One inductive structural head, degree/predicate-preserving control, controlled hierarchy removal, graph/profile provenance, and supported explanation integration.

**Inputs/bindings to resolve:** Available OAEI/BioKG graph data, safe training seeds, and measured feature memory; resolve capabilities rather than carrying stale unavailable flags.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **K0 graph-rich/TBox-poor case plus one controlled TBox-rich development case**.
Resource envelope: **extensions** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, kind_pool_freeze, E12, E13, E02_anchors**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

Use one small inductive recipe first. Hold labels, entities, retrieval and seed mappings fixed during TBox ablation. Report graph-only and label-only evidence, non-exact-label and degree slices. Three ablation levels support a coarse structural effect, not a precisely estimated universal crossover. Large transductive training is follow-up unless separately budgeted. Prefer D1 as the TBox-rich case when the class-graph path is eligible; otherwise bind a richer same-kind case before outcomes. Cross-kind results remain separate. No universal TBox crossover is inferred by comparing unrelated cases.

At most **10 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- rich_case_ablation: graph_off versus inductive at 0%, 50%, 100% hierarchy removal, six cells on one TBox-rich case.
- natural_kg: graph_off versus the same declared inductive recipe, two cells on K0.
- graph_only: natural-KG diagnostic with lexical evidence disabled.
- graph_shuffled: natural-KG diagnostic matched to its inductive arm in degrees/predicates.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **F1 net of shuffle, qualitative structural profile effects, memory/training cost.**
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

- No development/final alignment is inserted as a graph seed.
- Graph gains must be compared with the matched shuffle and lexical controls.
- Changing graph/profile schema invalidates consuming fusion heads.
- 64 GB execution chunks graph features; 128 GB changes throughput, not graph semantics.

Durable boundaries: **Normalized/ablated/shuffled graphs, graph features, fit state and scores.**
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
