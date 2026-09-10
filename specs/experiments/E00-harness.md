# E00 — Runnable baseline, stage artifacts, and recovery

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Strict screen/confirm schemas, task/arm expansion, provenance, source-level metrics, same-fingerprint resume, inference/additional-model checkpoints, atomic explanation storage, and E17 composition helpers exist.

**Agent must implement:** Explicit task-mode routing; v2 per-stage identities and readiness; cross-directory import; selective repair/reuse; budget and broad-search planning; reporting access ledger; per-role OpenRouter cost admission; relocated and interrupted end-to-end proof.

**Inputs/bindings to resolve:** Locate the user-available task/model files and node access; resolve OpenRouter identities and budget. Current local review environment lacked installed Exact-OM distribution metadata.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **D0**.
Resource envelope: **foundation** in RUN-PLAN.
Prerequisites/consumed outputs: **none; E00 supplies the foundation**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Use the 64-source G0 throughput probe and a bounded 300-source development vertical replay; fixture interruption tests precede long jobs. These are operational acceptance checks, with no treatment winner.

## Question, treatments, and implementation contract

Materialize stage artifacts and train/dev/report pools separately. Fit one current selector on actual train features and apply it on development. Verify every control executes the intended path. Produce source-level error attribution and distinct oracle ceilings for retrieval, ranking, acceptance/NIL, exact anchors, collisions, and typing. Preserve shared lexical, lexical-plus-definition/attribute, active-equal-weight and current supervised controls. Resolve the bounded published-matcher comparison in RUN-PLAN section 7; relative improvement alone does not establish competitiveness.

At most **2 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- production: intended current global and local paths, with experimental switches off.
- replay: identical numerical configuration after interruption and relocation.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Exact replay and operational acceptance; descriptive, no gain selection.**
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

- All ten CHECKPOINT-RECOVERY acceptance scenarios pass.
- Candidate files cannot silently turn global runs into local runs; global selector/extraction counters prove execution.
- A no-op comparison reproduces mappings and metrics; changed fusion and evaluator repairs demonstrate zero unnecessary encoder calls.
- Replace legacy experiment-wide ready with verified per-arm/per-stage capability readiness.

Durable boundaries: **All stages and attempt lineage.**
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
