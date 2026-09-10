# E25 — LLM viability with valid off and transferable routing

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Analytic/quantile/forced/oracle primitives and gate instrumentation exist; off control and development-ID transfer are defective.

**Agent must implement:** Actual decision-off wiring, source-top-fraction and transferred-threshold policies, stratified forced judgments, decoupled trust, and budget-matched real-response oracle diagnostics.

**Inputs/bindings to resolve:** OpenRouter primary identity/logprobs and budget; complete or justified development labels for counterfactual outcomes.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **D0 plus one feature-appropriate confounder/NIL development case**.
Resource envelope: **llm** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, pool_freeze**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

Do not replay development pair identities at inference. Compare gate error detection with its ability to find errors this judge can fix. Include confident errors, close candidates, natural NIL, pool misses, collisions and clear controls. Force the same source sample when comparing judge formats. Standalone accuracy, correction, harm and post-selector effect are separate outputs. The three trust replays are a separate sequential diagnostic after E07 supplies a compatible judge, using the same observed responses and no new calls; do not cross every routing fraction with every trust setting. Keep pre-LLM and post-LLM decisions, and freeze acceptance on permitted train/development groups. A fitted trust weight is a later E21 variant.

At most **11 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- decision_off: disabled decision/brief/rationale roles with shared upstream evidence.
- analytic: shipped pair gate control.
- source_top_001: top 1% eligible sources.
- source_top_005: top 5%.
- source_top_010: top 10%.
- forced_sources: up to 200 stratified development sources, diagnostic.
- oracle_perfect: perfect-answer ceiling with fixed interventions, development only, no live calls.
- oracle_observed: budget-matched routing ceiling using already-recorded real forced responses.
- trust_shipped: cached forced-response replay with the current beta*U weight.
- trust_constant: same cached per-candidate probabilities with constant weight 0.5.
- trust_source: use the frozen comparative choice as the routed within-source ranking, with separately frozen acceptance/cardinality.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Net final F1/cost and useful-call headroom; no blanket conclusion about every LLM role.**
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

- Every disabled role has zero OpenRouter requests; experiment.enabled=false is not accepted as off.
- A previously unseen high-uncertainty source can route under the frozen source policy.
- An oracle cannot make live calls, select a product or read final gold during screening.
- Constant or fitted trust can be tested independently of beta*U; malformed outputs and zero denominators follow a frozen fallback.

Durable boundaries: **Source statistics, selection policy, request/response ledger, judgment/decision traces.**
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
