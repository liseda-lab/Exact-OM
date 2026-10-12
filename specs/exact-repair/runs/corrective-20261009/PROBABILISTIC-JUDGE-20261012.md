# Matched probabilistic evaluator pilot — 12 October 2026

Neither tested interface is qualified for broader study evaluation. The native
Decisions route did not improve this semantic-fidelity task over Chat Completions.
This is a small diagnostic result, not a general comparison of model intelligence.

The frozen design used eight authored controls and two previously observed DEV
diagnostics, both presentation orders, with a second repetition of both DEV
orders. Every presentation had a Chat response, an independent Decisions evidence
gate, and a conditional native score/preference response. SF-010 criteria, equal
weights, anchors, 0.05 overall tie tolerance and 0.25 criterion-variation threshold
were fixed before calls. Unknown required criteria produced null totals.

| Check | Luna Chat | Luna Decisions |
|---|---:|---:|
| Structurally valid judgment presentations | 24/24 | 24/24 |
| Correct full control profiles | 14/16 | 2/16 |
| Correct direct control preferences | 16/16 | 13/16 |
| Broader evaluation qualified | No | No |

A full control profile requires every criterion to satisfy its frozen gold
constraints and both the score-derived and direct preference to be correct.
Native distributions must put a strict majority of probability on allowed gold
levels; this is not a claim that the probabilities are calibrated. The Decisions
full-profile result therefore must not be described as 2/16 pairwise accuracy.

Decisions assigned identical complete theories overall scores 0.771667 and
0.698333, beyond the frozen 0.05 tie tolerance, despite a direct `tie` decision.
It also withheld complete DEV scores through its evidence gate. Chat handled
seven of the eight control cases correctly in both orders. On the partial-evidence
control, it extended assertion assessment into opaque fixed context despite the
declared focal scope. Its overall abstention was correct, but its criterion profile
was not. This exposes the need to express criterion scope separately from domain
evidence; the original gold has not been changed retrospectively.

There was also a pilot implementation defect. For the two DEV cases, swapping
plan objects did not swap `context.plan_a_id` and `context.plan_b_id`. Four
presentations, accounting for 12 requests, are therefore invalid for scientific
interpretation. Their apparent order effects are not attributed to the models.
The authored controls have no conflicting context fields and are unaffected.
Chat's winner change on valid **unswapped** domain repetitions remains a separate
observed problem: the same focal entailment receives different assertion and
collateral damage ratings (0.75 versus 0.25).

The version-2 adapter aligns all plan identifiers and rejects inconsistent input
context. Regression tests cover both orders through Chat, gate and native-score
construction, source immutability and invalid context rejection. Unknown-target
scoring also retains the actual missing/contradictory probability mass; its
stricter check changes no observed control outcome. The focused suite passes
35 tests, and mypy passes. Old source, requests and responses remain frozen.
No further paid calls were needed to establish nonqualification from the controls.

The workload completed as detached Slurm step `14451.191` inside allocation
14451, using two CPUs, 8 GiB reserved RAM and no GPU. Slurm recorded 197 seconds,
7.920 CPU seconds and about 120 MiB peak RSS. The three original offline-check
steps and the correction-check step are also retained in cumulative accounting,
including the initial failed native-control preparation. The existing supervisor
code is reused with updated experiment instructions.

All 72 requests returned, with reported total cost **$0.197053855**. Their
conservative **$1.262658** reservation remains in the ledger, bringing cumulative
reserved exposure to **$21.51803635** under the **$35** hard cap. All 622 historical
reservations and 154 future TEST slots are preserved. No trained weights, study
labels, selection results or TEST data were changed.

Authoritative artifacts are under
`data/exact-repair-corrective-20261009/qualification/probabilistic-judge-20261012-001/`:
`plan.json`, `preflight.json`, `prepared.json`, `review.json`,
`accounting-final.json`, `phase-after-pilot.json`, and reproduction scripts.
Raw responses and the complete machine-readable score report are under
`work/qualify-probabilistic-judge-20261012-001/`; completion and immutable launch
receipts are under `attempts/qualify-probabilistic-judge-20261012-001/001/`.

The next design should start from the stronger Chat result, make criterion scope
explicit, and use the corrected plan identity projection. A new Decisions design
would need fresh qualification and an explicit grounding contract. Neither route
may silently substitute these diagnostic outputs for existing study supervision.
