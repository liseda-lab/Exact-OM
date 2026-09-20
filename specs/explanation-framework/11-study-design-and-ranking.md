# Ranking study design and analysis — B0/B3/B5/F1

**Accepted amendment, 2026-09-20.** The primary task is to rank up to five plausible equivalent candidates, starting from the system's displayed order. Some cases have an acceptable candidate; some have none among the displayed five. This replaces the earlier accept/reject/defer-first human-study design. It does not change automatic matching or enable new NIL inference.

## Question, conditions and scope

Primary question: does the complete explanation interface improve human reranking of a fixed five-candidate set over candidates/scores plus access to ontology inspection? A second question is whether it helps participants avoid endorsing a candidate when none is equivalent.

- `explanation`: same source/candidates/scores/order and external resources as baseline, plus integrated original context, independent entity profiles, hierarchy, evidence and prepared comparison.
- `ontology_baseline`: candidates/scores, minimal source/candidate identity and the same ranking controls; permitted ontology files and separately installed Protégé remain available, but integrated explanation endpoints are denied.

Participants are asked to install Protégé and obtain both pinned ontologies before the scored study. Setup confirmation and an unscored navigation check precede randomization into scored cases. Actual use is optional per case and recorded in both conditions. The baseline already contains system advice; it is not an unassisted Protégé condition. Optional use cannot establish a causal effect of choosing Protégé. The primary condition comparison must not adjust away consultation behavior that the condition itself may affect; report such associations separately.

All generative content is frozen before participant exposure. Both conditions use the same allowed information universe; the comparison estimates whole-tool benefit, not the isolated contribution of one graph or prose component. Later ablations are optional extensions, not launch blockers.

## Case definition and adjudication

Each case is one source class and exactly five unique candidate identities in a frozen initial order, preserving the system ordering of the selected candidates. Natural cases retain the untouched production top five. Public `display_position` runs contiguously from 1 to 5; private provenance also retains original production ranks. If constructed negatives are used, do not expose rank gaps or rank-encoded IDs that reveal them. Use the neutral label Initial order consistently across study cases, while retaining actual system scores and their meaning. The task asks for equivalent disease/concept meaning, not just relatedness or source-broader/narrower compatibility. Retain typed reference relations when building the key.

Researcher-only `CaseKey` contains `case_kind` (`answer_present`, `answer_absent`, or unresolved/non-primary), independently adjudicated acceptable candidate IDs, scope/criterion, supporting evidence and adjudication version. Verify all five candidates, blind to interface/condition, including whether several are defensible equivalents. A single benchmark gold row does not prove that every alternative is false.

`answer_absent` means no acceptable equivalent among these five. Distinguish natural retrieval failure, deliberately constructed candidate exclusion and ontology-wide absence; the last requires separate evidence and is not required here. Prefer real retrieved sets. If negative cases must be constructed, retain the original candidate set, precise replacement/removal rule and original scores/ranks; label them in researcher provenance and analyze separately from natural failures. Contiguous public list positions are presentation indices, never a claim that original production ranks were contiguous. Keep original production ranks privately recoverable without leaking case kind. Do not fabricate scores, change evidence to make candidates wrong, or present a constructed list as the untouched production top five.

Do not infer a negative case merely from missing benchmark references or weak source context. Unresolved adjudication is a separate challenge set excluded from primary scoring by a rule frozen before results. In positive cases with several accepted candidates, RR uses the first accepted candidate; report benchmark-single-reference results separately if they differ. Clean single-answer cases are preferred for the initial position-balanced panel.

Tell participants that some cases may have no equivalent candidate, without revealing quotas. Avoid filenames, IDs, ordering, API fields or loading behavior that disclose case kind. Each source/case is seen once per participant; group near-duplicate sources to avoid cross-condition answer transfer.

## Assignment and workload

Case count and negative prevalence are configurable, frozen at study publication after a duration pilot. The recommended design example is 24 cases: 20 answer-present plus four answer-absent, split into two blocks of 12. Each condition block has two positives at each original system accepted rank 1–5 and two negatives. This is a provisional workload recommendation, not a completed power calculation or a requirement to collect 24 cases under every study.

A supported shorter preset is 20 cases: 16 positives and four negatives, ten per condition. Eight positives per condition cannot balance five original ranks exactly. Use complementary forms that alternate which condition receives an extra case within each original-rank stratum; document the fixed case-bank prevalence and analyze by case/rank. A near-balanced 16-positive bank can contain 4/3/3/3/3 cases at ranks 1–5. Neither preset should reshuffle system order to create an artificial rank distribution. Exact balance is a sampling/assignment property; do not alter the matcher to attain it.

For either preset, publish complementary case-to-condition forms, crossed with explanation-first versus baseline-first block order. In the 24-case design this yields four basic schedules; randomize order within blocks using a persisted seed. Balance assignment across participants and, where recruitment permits, predeclared broad experience strata without creating many sparse cells. Each case appears in each condition across participants, never twice to one participant. Test allocation by simulation before use. Reserve allocation transactionally after setup/questionnaire completion; invited but unstarted links do not consume completed-participant quotas. Preserve started allocations and report dropouts rather than silently reassigning them.

Use identical unscored tutorials/practice exposure for both tools, progressing from simple to complex and including ranking, partial ranking and explicit none. Do not systematically order scored cases from easy to hard: that confounds difficulty with learning. Installation, tutorial and questionnaire time are separate from case-task time. Participant number/power is determined from a worthwhile improvement and pilot participant/case variability, not from cases-per-person alone.

## Response semantics

Workflow state (`not_started`, `draft`, `submitted`) and semantic response are separate. Submitted response types are mutually exclusive:

| Response | Required payload | Meaning |
|---|---|---|
| `ranked_candidates` | Ordered list of 1–5 unique presented candidate IDs | Candidates the participant considers plausible equivalents, most plausible first |
| `none_of_these` | Empty list, explicit selection | No displayed candidate appears equivalent |
| `insufficient_evidence` | Empty list, explicit selection | Cannot make a justified ranking from the information available |

An untouched/empty draft is unanswered, never an implicit none response. Omissions are `not_ranked`, not logical proof of non-equivalence. Draft selection can be saved; switching to none/insufficient evidence clears the submitted list, with undo/recovery before submission. No ties or rank gaps in v1. Never append omitted candidates automatically.

The initial order remains visible separately from the editable answer. Keep candidate identity and initial list position stable while displaying the participant's rank distinctly. For natural top-five cases, initial position equals original system rank. Start the answer empty; offer an explicit 'Keep initial order' action. Picking the next candidate, move-up/down and remove controls are primary accessible interactions; optional conventional drag-and-drop has identical semantics. Require intentional submit and preserve acknowledgement. First submission is primary by default; submitted cases are read-only. Any enabled later revision is append-only, labeled and analyzed under a predeclared rule.

After committing each ranking, ask whether external ontology resources were consulted and which tools/files were used (13). Stop the ranking timer at ranking submission, before that questionnaire. Progress to the next case after its consultation response is saved. Both steps resume independently.

## Metrics and missingness

Primary positive-case metric: RR = 1 / rank of the first adjudicated acceptable candidate; zero for a submitted list omitting all acceptable candidates, explicit none or insufficient evidence. MRR averages this over answer-present cases only. Report original-system MRR and participant-minus-system RR on the same cases, Hit@1, acceptable-candidate omission, response depth and empty/uncertain rates. MRR assesses the first accepted result, not the entire order below it. In the clean 24-case position-balanced example, original-system positive MRR is approximately 0.457 by construction; this is not estimated natural production performance.

Negative cases have no relevant candidate and no meaningful target reciprocal rank. Report correct explicit-none rate, false-endorsement rate (any submitted nonempty ranked list), and insufficient-evidence rate separately, using all submitted negative cases for each denominator. Do not invent RR=1 for selecting none or silently mix negatives into positive MRR. An optional overall decision-success rate (positive accepted top-1 or negative explicit none) must be named separately with its prevalence and denominator.

Unsubmitted/abandoned/technical-failure cases are missing, not none or zero by default. Freeze incomplete-session inclusion, exclusions and sensitivity analyses before collection, report assigned/submitted denominators and attrition by condition, and retain missingness reasons. Report condition effects with participant and case clustering, effect sizes/intervals, original-rank strata and negative-case origin. Correct-to-wrong and wrong-to-correct top-1 changes compare the submitted top candidate with the original system top candidate on natural answer-present cases. These quantify deterioration/improvement relative to system advice, not a change from an independent initial human judgment, which this design does not collect. Do not infer subgroup benefit from an underpowered pooled result.

Time is reported alongside quality, not substituted for it. Browser invisibility may mean active Protégé work; see 12. Component ratings/preferences and event counts are descriptive, not evidence that a component caused a gain.

## Readiness

B5 requires synthetic mixed-case schedules, scoring oracles and durable response recovery, not participant recruitment. Final publication additionally requires adjudicated real case keys, frozen instructions/questionnaires, a duration pilot, analysis plan and research approval/consent arrangements applicable to the study. No participant contact or live launch is implied by writing these specs.
