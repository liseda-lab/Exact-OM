# Exact-Repair: single-node experiment plan for the October deadlines

**Date:** 9 October 2026. **Depends on:** [15 Corrective requirements](15-preliminary-corrections.md). **Machine-readable planning limits:** [liseda05-20261009-plan.json](protocol/liseda05-20261009-plan.json). These are future execution requirements, not evidence that corrections or experiments have completed.

The user-supplied abstract deadline is **18 October 2026** and the full-paper deadline is **25 October 2026**. Aim to finish primary measurements on 16 October and freeze the checked result snapshot on 17 October. All dates below use Europe/London. The actual submission cutoff was not supplied: this plan deliberately finishes ahead of either deadline and does not assume an end-of-day submission time.

The guaranteed machine is `liseda-05`: 16 CPU resources, 128 GB advertised RAM, one RTX 5090 and one RTX 2080 Ti. A possible larger CPU node and six A100 GPUs are not on the critical path. The objective is a defensible, mostly completed primary study by the 18th, not the maximum number of scheduled experiments.

## 1. Resource admission

Before launching, record the allocation's actual CPUs, physical/allocated memory, GPU UUIDs, visible VRAM, driver/CUDA/PyTorch versions and native-library versions. Advertised CPU/RAM is a planning ceiling, not proof of an available allocation. Validate each GPU with a forward/backward/optimizer/checkpoint cycle. The RTX 5090 must be qualified early; a missing compatible runtime is a P0 environment issue.

Use at most **min(14, measured allocated CPUs minus 2) worker CPU tokens**, leaving two for the operating system, supervision and I/O; a nonpositive limit cannot admit this profile. Set worker RAM admission to **min(100 GiB, 80% of measured available allocation RAM)**. Reserve each job's whole process tree, including native calls and BLAS/OpenMP threads; a child's reservation cannot be added outside its parent's declaration. No two jobs own the same GPU. GPUs run independent jobs; do not make heterogeneous distributed training a dependency.

| Profile | CPU tokens | Host RAM reservation | Role |
|---|---:|---:|---|
| Reasoning/label worker, up to three | 3 each | 20 GiB each | Qualified full-theory checks, teacher acquisition, evaluation |
| RTX 5090 training worker, one | 4 | 24 GiB | Primary model fitting; dynamic measured VRAM batch cap |
| Optional RTX 2080 Ti worker, one | 1 | 8 GiB | Compatible inference, small ablation or independent training only after measurement |
| Default with three reasoners + 5090 | 13 | 84 GiB | Primary concurrent profile |
| Default plus optional 2080 Ti | 14 | 92 GiB | Admit only if measured host/VRAM budgets fit |

These are starting reservations, not predictions. If a backend needs more memory/threads, replace smaller reservations instead of adding work. Nested operations share the reserved CPUs; limit Torch/OpenMP/BLAS and reasoner thread counts explicitly. The optional GPU does not justify additional CPU oversubscription. An out-of-memory correction changes the run's resource identity; retries retain their cost.

CPU evaluation workers must not independently load models onto an occupied GPU. Use one admitted batching/inference worker or serialised exclusive device acquisition, with the corresponding CPU/RAM reservation; a measured CPU inference route is also permitted. Queueing and inference remain in the caller's end-to-end allowance. Do not exempt acquisition or DEV decoding from GPU ownership merely because they are called by a CPU job.

Keep code/configuration in the normal code location and use verified node storage paths for active datasets, compiled caches, results and checkpoints. Do not invent cluster mount paths. Cache immutable inputs and compiled languages locally where permitted. Prefer small per-worker append/commit records and batched exports over repeatedly hashing or copying entire checkouts. Move completed artifacts only after checksum verification and dependency checks.

## 2. Dates, cumulative limits and automatic prioritisation

| Latest planned period | Mandatory outcome | Bound after corrections are available |
|---|---|---|
| 9–11 October | Integrate fixes; pass native/schema/action/lifecycle checks; qualify hardware; calibrate concurrent throughput | Calibration at most 4 node-elapsed hours, separate from implementation work |
| 12–13 October | Generate/qualify new TRAIN/DEV cases; collect useful verified labels; start fitting on committed compatible shards | Primary acquisition at most 30 elapsed hours with at most three reasoning workers |
| 12–14 October | Train HGT unary/pair models; perform DEV selection; finish optimizer/resume evidence | At most 18 total RTX 5090 occupancy hours for six primary training jobs; at most 3 hours each |
| 12–14 October | Full-schedule DEV decoding and checkpoint selection | Separate cap: 12 elapsed hours and 36 reasoning-worker service hours; every validation pass is charged |
| 14 October, 18:00 | Freeze primary models, all test case IDs, targets, resource limits and comparison schedules | No model/target/threshold selection using subsequent test outcomes |
| 15–16 October | Run primary generated-data and Conference comparisons | At most 48 elapsed hours with the calibrated CPU/RAM admission; default is three reasoning workers |
| 17 October | Finish bounded technical recovery, aggregate all rows, inspect examples, produce paper tables and a hashed evidence bundle | Protected day; no optional sweep may consume it |
| 18 October | Use the already checked result snapshot for the abstract | Do not depend on a late last-minute run |
| 19–24 October | Additional replications, optional scope and paper preparation | Separate manifests/results; primary results remain preserved |
| 25 October | Full-paper deadline | No assumed submission hour |

The primary execution envelopes are **4 + 30 + 18 + 12 + 48 = 112 hours** if run serially; fitting, DEV and acquisition may overlap after shard identities are fixed. This arithmetic is a reservation, not a completion-time guarantee: acquisition and evaluation also consume the worker CPU/RAM envelope, while fitting is bounded in GPU occupancy. Additionally cap the combined acquisition/fitting/DEV stage at 66 elapsed hours and each admitted model job at six elapsed hours, always clipped by the 14 October freeze. Queueing after admission counts. Initial native qualification is part of 9–11 October engineering; do not hide it in reported learning cost. The protected 17 October day and unused time absorb failures and analysis.

Every call receives the minimum of its own remaining allowance, stage remaining allowance and absolute deadline, after reserving cleanup. Resume never replenishes budgets. At most one technical retry for the same failure is permitted after a specific fix, under the same cumulative allowance; deterministic logical unknowns are not technical retries. Do not replay completed rows to improve a table.

If progress slips, remove optional encoder expansion, large robustness grids, depth-three generation, full Bio-ML and additional LLM volume first. Preserve the primary comparisons and real Conference evaluation. Reduce future unstarted data quotas symmetrically within declared strata before test opening; do not remove slow corrupted test cases or give a preferred model extra time. If even the minimum study cannot fit, finish and report a smaller exploratory study with the actual denominators. No safety/semantic requirement is relaxed to meet a date.

## 3. Four-hour calibration and workload freezing

Calibration uses exposed or new training/development data only. Include clean and corrupted cases, every enabled action family, at least two training-side Conference pairs where available, and at least one overlapping/cyclic support case. If a specified real pair is unavailable, record it and choose a development replacement by the frozen availability rule, not by repair quality.

Measure cold and warm runs, isolated and concurrently admitted runs, including grammar/proof preparation, compiler supervision, sampling, native acceptance, soft query scoring and cleanup. Run a real optimizer segment and save/resume. Record timeout-inclusive service times, useful labels per hour, memory after optimizer state exists and GPU occupancy. A failed stage still consumes its measured cost. Never extrapolate only from successful cheap cases.

Choose final case counts, assignment quotas and resource reservations before held-out opening. For stage elapsed allowance T and w qualified worker slots, a conservative admission estimate is:

`sum_over_strata(case_count * arm_or_attempt_count * timeout_inclusive_cost) <= 0.70 * w * T`.

Use an observed high percentile with timeout costs included; also inspect the worst supported cases. Enforce CPU-seconds and RAM feasibility separately. The 30% margin covers variability, cleanup, receipts and bounded recovery. Fit repeated qualification calls in the same ledger. If calibration cannot establish a finite useful rate, fix the affected bottleneck or exclude the affected optional scope explicitly; do not launch thousands of doomed rows.

Starting case budget: **300 seconds end-to-end** for small generated/Conference development cases; at most **600 CPU-seconds** within the declared worker reservation. A full check gets at most **30 seconds**, compiler calls at most **20 seconds**, and cleanup at least **2 seconds** reserved, all clipped by remaining budgets. These are starting caps, not runtime targets. Initial diagnosis is capped at 20 seconds and total rich generation at 60 seconds; leave at least one final-check allowance before beginning any expansion epoch. Target startup/transport to be a small measured fraction through reuse, not by removing supervision.

For staged search, reserve up to 30 seconds for the elementary selection/verification attempt, then up to 60 seconds for rich generation and the remainder for selection/verification. Costs already incurred by initial checking are included in the same 300 seconds. If no check can complete within its remaining allowance, return the current verified incumbent or an explicit unknown. A different Bio-ML resource profile is a separate frozen experiment.

## 4. Minimum data and supervision programme

Use the existing named generator mechanisms and explicit structural ancestry. Start from eight repair-relevant strata covering hierarchy/disjointness, domain/range effects, existential restrictions, semantic complementarity/redundancy, overlapping conflicts, cycles and eligible ontology/composite edits. Map actual generator families to strata in the release manifest; no invented family name counts as implemented coverage. Every enabled action type must additionally pass a concrete native regression case, even if it is rare in the statistical corpus.

The initial quota is **8 TRAIN, 2 DEV and 4 TEST independent parents per stratum**, each with a clean control and one corrupted sibling: **128 TRAIN, 32 DEV and 64 TEST cases** over eight strata. These are conditional workload targets, not a guaranteed minimum or a power calculation. E1/E2 alone can approach 69 worker-hours if every generated case reaches its 300-second cap; calibration must also admit E3–E5, Conference and all DEV passes before freezing these counts. Prefer an already frozen, genuinely unopened compatible test release; never reclassify previously exposed parents as new held-out data. New case/parent IDs must be checked against historical archives. Changes to counts are frozen after calibration, before inspecting test outcomes.

Keep all siblings, renamings and derived variants of a parent in one split. An input graph cycle is not evidence of a cyclic conflict: record the actual violation/support structure, conflict cardinality, support overlap and number of conflicts per mapping. Use complete small-case enumeration to verify selected construction targets where affordable. Corpus labels distinguish construction intent from proved outcomes.

Begin with up to **16 unique planned assignment checks per TRAIN case**; acquire a second, targeted round of up to 16 for cases where coverage or interactions remain weak and budget permits. Parent qualification and all failed/duplicate attempts also count. Tiny exhaustive products may replace sampling when the complete cost fits; larger products must not delay the entire release. Preserve known masks and study denominators.

Targets per qualified parent are several distinct feasible semantic alternatives and decided infeasible examples where they exist. For interaction strata, target at least one nonzero feasible quartet on a selected pair; across each sign/mechanism aim for at least four independent TRAIN parents, while retaining zero-effect controls. These are coverage targets, not demands to fabricate examples where no such repair exists. Stop repeated sampling when it produces only duplicates or unchanged target values; spend the remaining declared stratum allowance on other parents.

The semantic basis contains desired consequences, unwanted consequences and appropriate non-vacuity checks. Freeze family weights and edit costs before comparisons. Do not derive queries from the model's selected repair or reward a specialisation merely because it is the proposed action. Include clean-input preservation and examples where deletion, directional weakening, specialisation, a complex correspondence or an eligible ontology edit is preferable under the declared target.

Store committed TRAIN shards with input/policy/query/candidate/support identities and masks. A checkpoint binds the exact shards and acquisition-model snapshot used. Append-only acquisition can overlap training only at declared round/epoch boundaries; a running minibatch never reads mutable labels. DEV collection remains excluded from fitting. Record the useful independent-parent count and target diversity, not just label volume.

## 5. Primary model and comparison matrix

Primary fitting consists of **HGT-unary and HGT-pair**, each at seeds **13, 37 and 73**: six independent training jobs. Retain the existing width/layer/head and loss settings as starting values; fit their full-schema memory in calibration. Use gradient accumulation if needed to preserve an explicitly declared effective batch size. Do not silently alter architecture to fit one GPU.

Set at most 50 epochs and the cumulative three-hour per-model occupancy bound. Default to **at most two full DEV passes per model**, one at a predeclared completed epoch (starting choice: epoch five) and one after the final complete epoch; identical checkpoints are evaluated once. There is no patience-based early stopping in this two-pass default. Calibration may admit more full passes and a separately frozen patience rule only within the DEV service/elapsed envelope and model-freeze date. It may instead admit only the final pass, in which case checkpoint choice is fixed by the predeclared training endpoint. Every begun pass retains all scheduled DEV cases, including unfinished outcomes; reserve enough time for the final pass before launching further optimizer work. Record actual completed updates/epochs; timeout is not convergence. If acquisition/DEV checks run inside the fitting job, their elapsed time and CPU cost count and GPU idle time is reported; prefer separate committed caches and release GPU reservations while awaiting long CPU acquisition. Selection uses the full scheduled DEV denominator and the frozen coverage-aware rule. Never choose a seed using test results; report all scheduled seeds.

| Comparison | Changes | Fixed quantities | Primary interpretation |
|---|---|---|---|
| E1 End-to-end | Full learned pair system; strong observable symbolic rich-action control; native deletion; observable score-greedy baseline when scores exist | Cases, edit eligibility, policy, target evaluation and total case budget | Verified coverage, retained meaning, time/calls to useful repair |
| E2 Scoring | Observable symbolic score; learned unary; learned pair | One independently generated/frozen inventory per case, verifier, costs and search budget | Selection quality and interaction contribution; exact regret only on fully labelled small pools |
| E3 Generation | E3a: learned versus uniform weights within circuits; E3b: circuit versus grammar decoding with identical fixed proposal weights/distribution | Same declared typed language, contextual tests, unique-candidate/draw/time caps, common observable selector | Useful candidate coverage/quality per total generation cost; separate learning and decoder effects |
| E4 Risk scheduling | Risk ordering on/off | Same checkpoint, pool, integer objective, shortlist/window, solver, initial proof set and cut-generation rules | First verified repair and matched external-quality verification effort; subsequently discovered cuts may differ |
| E5 Staged execution | Elementary-first versus one-stage generation | Final declared language, model/selector and total resource budget | Early incumbents, final verified utility, partial-generation robustness |

E1 and the full versus unary part of E2 are primary seed-replicated comparisons. E3–E5 initially use a fixed, preselected development-independent seed (13) and a matched, stratified subset; expand only after the core schedule fits. Shared identical E1/E4 full-system rows are reused by identity, not rerun or counted as independent replications. Report generator acquisition and common-pool construction costs even when amortised across scoring arms.

The historical original-axiom-retention control remains a named diagnostic, not the strongest symbolic baseline. Implement an observable symbolic semantic control as specified in COR-11. A clean-parent oracle is an upper reference restricted to controlled data. Query-evaluation work is charged; missing semantic labels do not become a default good score. Keep scoring error, pool coverage and finite-search gap separate.

The grammar-constrained decoder comparison must implement the same probability/alias convention and local proof constraints, or be labelled a different proposal distribution. Do not compare full enumeration of a large language against bounded circuit sampling and attribute the difference solely to decoding. Compare cost to obtain the same number of unique admissible candidates, then evaluate their usefulness. Exact circuit likelihood is still tested separately on small enumerable supports.

A compact, frozen robustness subset covers output-vocabulary omission, candidate removal, evidence degradation and increased conflict overlap. Log whether each intervention changed the actual inventory/evidence. Use the same source revision across paired arms; mixed historical revisions are excluded from causal comparisons.

R-GCN and no-graph/MLP, each unary/pair, remain planned secondary arms. Add one diagnostic seed first, then all three only if the resource projection fits without displacing E1–E5 or Conference. No HGT-superiority claim is made without those comparisons. Broad grammar-depth/candidate-cap grids and a full robustness factorial are optional.

## 6. Conference first; bounded biomedical transfer separately

Reserve at least one third of the primary evaluation worker-time envelope for Conference before admitting optional synthetic breadth. Use actual ontologies with resolved imports and licensed/versioned matcher outputs. The main test follows the suite's whole-ontology holdout (including the existing `ekaw` rule when still uncontaminated); freeze actual pair membership and record earlier exposure. Do not infer the number of usable pairs from the track name. Test all scheduled pairs, including initially coherent and unsupported/unknown cases.

Use training-side Conference data for any real-data adaptation, in a separately named condition. At minimum compare generated-only transfer against symbolic/deletion controls. Controlled corruptions of Conference alignments can test specific actions but remain a separate benchmark from actual matcher errors. Existing reference alignments do not automatically supply ground truth for complex repairs or ontology edits. Evaluate independently declared consequences and, if available, separate semantic judgements.

Prepare Bio-ML inputs, imports, capability summaries and a bounded scale diagnostic in parallel when CPU capacity remains. Whole-ontology biomedical repair is optional before the 18th and cannot consume Conference's reserve. A projection is named as a projection; it is not evidence of full-source coherence or full Bio-ML repair. Learned routing or a new large-scale reasoner implementation is not on this deadline's critical path.

LLM weak-semantic supervision remains a separate planned condition using the existing OpenRouter path. First qualify new-label attachment, evidence masking and independent evaluation on a small development sample. It must not block the symbolic-teacher primary study. Any paid run uses an explicit existing applicable authorisation and a resolved finite spending cap; unspecified cap means disabled, not unlimited. Do not carry a matcher campaign's budget into repair by inference. Freeze prompts/model/rubric and held-out judgement protocol; no LLM calls during repair inference. Add a bounded annotation subset only after the primary schedule is secure.

## 7. Admission, stopping and deliverables

Before primary fitting, require all of the following measured conditions:

1. Source integration, native action fixtures, schema compatibility, candidate/cost identity, logical certificates and worker lifecycle tests pass for every enabled primary feature.
2. On a small fixed development qualification suite, each enabled repair action is exercised on at least one corrupted input with a verified feasible output under the intended policy, plus a coherent pass-through control. The suite includes the proof-supported interaction and composite cases. This is engineering acceptance, not a general performance threshold.
3. TRAIN acquisition contains usable parent/target diversity across the included strata, with missing quotas explicit. An arm cannot claim to learn an interaction or proposal family with no qualifying supervision. Partial data are allowed with masks; a hidden switch to only easy overlap cases is not.
4. Full-schema training performs actual optimizer updates and exact resume; complete DEV scheduling and coverage accounting work.
5. A conservative timeout-inclusive projection fits the remaining CPU/RAM/GPU/time envelope. Freeze final counts and budgets in resolved manifests before test opening.

These are automatic checks with retained evidence. Once experiment execution has been authorised, routine passing checks do not require repeated human confirmation. A failed correctness check blocks dependent work; independent preparation and already qualified comparisons continue. An optional branch failure never stalls the primary campaign.

Publish by 17 October: all scheduled outcome rows; parent/split/source manifest; selected checkpoints and training coverage; native verification receipts; candidate/proposal coverage; actual per-stage and total resource cost; clean/corrupt and family/pair tables; paired parent-level intervals where support permits; and a concise explanation of uncompleted comparisons. Preserve both failure rows and technical retry lineage. Archive a checksum-verified bundle that includes enough configs/code identities to replay, and states which large inputs/checkpoints remain in node storage.

Primary reported quantities are verified repair coverage, semantic utility among independently evaluable verified outputs, unwanted consequences, clean-input changes, time/calls to the first verified repair, time/calls to matched external utility, and unresolved optimisation/generation scope. Report semantic-unavailable but logically verified repairs separately. Missing outputs remain missing/failed coverage, not semantic zero unless an explicit deployment utility defines that penalty.

For the manuscript, distinguish measured results from remaining hypotheses and specification-only features. Freeze the abstract snapshot before optional later experiments; later full-paper additions identify their own manifests and dates. The larger cluster, if it becomes available, may accelerate independent frozen jobs with hardware recorded, but primary timing comparisons remain on `liseda-05` and never mix device timings as if identical.
