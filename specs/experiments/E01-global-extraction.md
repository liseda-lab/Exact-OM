# E01 — Threshold-aware global extraction

**v2 specification, amended with user approval 2026-09-25.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Implemented contract:** Frozen-pool global execution, unrestricted threshold extraction,
greedy, mutual-best, stable marriage, accepted-utility assignment, and legacy assignment.
The amended screen gives all methods the same explicit lexical-anchor conflict policy.

**Inputs/bindings to resolve:** Pin the primary class reference and output entity eligibility.
One-to-one matching is an experimental restriction, not a universal ontology-alignment
correctness requirement; the user confirms the broader OAEI tasks are available.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **D0**.
Resource envelope: **decisions** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, pool_freeze**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

Hold candidate pools, source population, pair scores, threshold **0.7**, label-free
supervision, and disabled selector/LLM decision gate fixed. Compare unrestricted extraction
with one-to-one extraction. Use a selector on/off interaction only for the selected strategy
at G4; do not multiply the initial grid. Record changed sources, target collisions, dummy
matches, cap fallbacks, ambiguous lexical anchors, and mappings suppressed by cardinality.

At most **6 distinct treatment configurations/cells as specified below** before any explicitly
declared source expansion. This is a bounded sequential design, not a Cartesian product.
Shared deterministic scoring inputs are computed once.

- threshold_unrestricted: baseline; keep eligible edges at or above threshold, with no source or
  target cardinality limit (`matching.cardinality=null`, `target_cardinality=null`).
  As in every arm, collision-free protected anchors are retained even below threshold;
  ambiguous anchors lose protection and must meet the threshold.
- greedy: threshold-first one-to-one control, eligible for selection.
- mutual_best: reciprocal top candidate among eligible edges, one-to-one.
- stable_marriage: source-proposing, canonical ties, one-to-one.
- assignment_accepted_utility: eligible edges with utility score-threshold and zero unmatched
  utility, one-to-one.
- assignment_legacy: raw-score optimization followed by threshold, one-to-one diagnostic only;
  it cannot be selected for deployment.

All six use `matching.extraction.anchor_conflict_policy=compete`. Collision-free lexical
anchors remain protected; lexical anchors sharing a source or target compete at their existing
scores. Their ambiguity is identified identically across all arms, independently of the arm's
cardinality. No lexical match is promoted to verified equivalence. This is an extraction-only
rule: E02 continues to own rescoring and alternative anchor trust policies.

The threshold baseline is retained unless a selectable constrained method satisfies the frozen
quality, recall, and cost criteria. Publish both cardinalities with the extraction policy so
later composition cannot silently restore one-to-one matching. Compare precision gain and
recall loss from imposing cardinality; one-to-one compliance alone is not a correctness result.
Output entity eligibility follows the existing declared task scope. Imports may supply context;
there is no new namespace filter and no exclusion chosen to remove a collision.

All generative roles use OpenRouter. Local non-generative encoders/heads use the allocated single GPU.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Global F1; local pair scores must be unchanged.**

- RQ1: Does a one-to-one method improve global F1 over unrestricted threshold extraction,
  and what precision/recall tradeoff does that restriction introduce?
- RQ2: Among the constrained methods, does threshold-aware assignment improve over greedy,
  mutual-best or stable marriage? Legacy assignment remains diagnostic.

The screen is descriptive, with the existing 0.003 F1 practical-effect threshold for selecting
a constrained method over the unrestricted baseline. A failure to pass retains unrestricted
extraction; it does not establish that all true ontology alignments have unrestricted multiplicity.
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

- The 0.90/0.69/0.69 graph at threshold 0.70 retains 0.90 under primary assignment.
- All strategies enforce declared cardinality and deterministic ties.
- Non-greedy global arms work with frozen candidate files and reject local-ranking semantics.
- The shared `compete` rule resolves ambiguous lexical protection without changing scores;
  strict hard-anchor controls outside this amendment still reject conflicting constraints.
- All six arms, including the historical greedy control, are rerun for extraction/evaluation.
  The prior five-arm attempt is retained as superseded evidence. Reuse upstream scores only
  when the scorer, pool, source population and ontology scope are unchanged and validated.
  Create a new immutable declaration; never rewrite the failed attempt or its signatures.
- Selection uses public development references only. Private/reporting references cannot
  guide cardinality, anchor handling, output eligibility, or threshold choices.

Durable boundaries: **Decisions, per-component extraction, evaluation.**
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
