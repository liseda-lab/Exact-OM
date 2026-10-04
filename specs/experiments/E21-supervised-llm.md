# E21 — Exemplars, net-benefit routing, and a justified student

**v2 specification; approved binary-judge amendment, 2026-10-04.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

The original 655f599 audit is historical. Shared exemplar retrieval, forced-call teacher
records, grouped linear router/student fitting and teacher identity checks now exist. The
previous admission required a comparative judge with `source_first` integration and rejected
the actual E07 winner, `facts_binary`, before any E21 scientific cell or teacher data existed.

The user approved support for that selected binary judge, retaining all five treatments and
the existing train/development roles. This is an explicit method amendment, not permission to
substitute the losing E07 listwise judge or alter completed E07 results. The bounded protocol
and current evidence state are recorded in
[E21-binary-judge-amendment.yaml](E21-binary-judge-amendment.yaml). The combined E21/E04 fixture
suite passed 382 tests in 94.31 seconds, including the actual-data anchor and effective-rationale
regressions. Metadata preflight against the actual completed
E07/E25 outputs passed for all five arms in 46.98 seconds, with no training, inference or
provider calls. Source freeze, publication and measured results remain separate gates. The
immutable launch must bind the final focused tests and preflight receipt.

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

### Approved binary-judge continuation

Bind the completed E07 `facts_binary` selection and resolved teacher profile, prompt,
structured evidence, categorical probability semantics and request seed. Preserve its
`beta_u` rule: `w = clip(beta * U, 0, 1)` and
`S_final = (1 - w) * S_base + w * p_teacher` on invoked candidate pairs. Other pairs retain
their original scores. Do not replace binary decisions with an invented source-choice or
NONE response. Rationales remain disabled.

The current bound D0_E03 screen uses 200 development sources, seed 17, at most 2,000 disjoint
training sources, and at most 200 teacher sources; these are the existing smaller hosted
limits, not a new expansion. For each selected teacher source, force the same deterministic
top-five candidates (descending base score, target IRI tie break) used by the selected
judge. Retain the entire source candidate pool when forming counterfactual outcomes:
uninvoked candidates remain eligible and unchanged. Persist every completed request before
continuing and retain actual token/cost usage. An incomplete or invalid teacher source blocks
fitting while retaining completed requests/shards for recovery; do not silently reduce the
training denominator or fabricate labels. A recovered completed request must not be paid again.

Fit only from the public training reference and explicitly labeled training candidates.
BioML benchmark distractors support negatives only within their supplied training pool;
unlisted pairs and incomplete global-reference omissions remain unknown. Router fitting
requires all candidates of each included training source to have safe outcomes. Development
labels may establish the predeclared helpfulness prerequisite but must not become exemplar,
router or student targets. Reject training/development source overlap and private test access.

The binary router's source-group target records corrections minus harms per 1,000 actual
teacher tokens under the frozen scoring/acceptance rule: compare the accepted top target
(or no accepted target) before and after the intervention over the full source candidate pool.
This is a source-local proxy before global extraction, not the measured final mapping benefit;
cross-source target competition is evaluated by the actual downstream extraction. The experiment measures the
actual extraction, F1, corrections/harms and total cost on the complete development population.
The current selected run's effective pair threshold is **0.7**, with `beta=0.8`; use the actual
effective acceptance policy, not an unrelated configured/calibrated threshold field.

`student_gold` and `student_distilled` use identical pair features, training population,
architecture, optimizer, source-group folds and inference integration. Only their targets
differ: gold-only versus the existing equal gold/teacher mixture for valid teacher-covered
pairs, with gold retained for other safely labeled pairs. Report teacher coverage and all
generation/fitting costs. Fitting artifacts bind the probability/prompt/provider identity,
population, seed, effective threshold and fusion rule; incompatible artifacts are rejected.

Before spending on E21, require the selected judge's development F1 gain over E25
`decision_off` to meet the existing 0.003 threshold on matching sources, candidate pairs,
reference population and no-call scores/acceptance. The saved scores are 0.823 and 0.779,
respectively, but this is not sufficient alone: their sampled-pool fingerprints differ due
to missing cached encoder/retrieval metadata. Saved full pool manifests and all 4,000 actual
candidate pairs/base scores agree, including 106 protected exact anchors with absent scorer
features (105 emitted, one removed by extraction in both runs). The verified compatibility
proof binds these states, effective acceptance and actual reference bytes; it does not waive
a mismatched fingerprint. Admission rechecks this proof. Failure of this prerequisite
means blocked or screened out, not a completed negative E21 experiment.

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
- Binary continuation tests must cover exact fusion/threshold behavior, an uninvoked candidate
  remaining the best candidate, invalid/partial teacher outcomes, train/development isolation,
  matched students, teacher request/shard recovery, and helpfulness population verification.
- Queue publication requires a new immutable recipe and verified frozen code, preservation of
  cumulative accounting and paid-request history, and the existing single-worker/storage
  guards. Neither this amendment nor fixture tests alone authorize a scientific-completion claim.

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
