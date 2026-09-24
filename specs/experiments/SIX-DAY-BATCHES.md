# Remaining experiment batches: six-day target

**Planning amendment, 2026-09-24. Soft target: 144 hours of continuous execution.**
The user explicitly permits longer execution. Missing the six-day target does not block an
otherwise correct batch that fits the cumulative resource budget.
This changes dispatch priorities, not scientific scope. It is not an executable campaign
lock, a measured completion promise, a reset of accumulated costs, or a job time limit.
The current single-node assumption is liseda-03: one RTX 4090, six allocated CPUs and
125 GiB host RAM. The user confirmed that only this node is available.

The full unoptimized declaration cannot currently be admitted within six days. The useful
next step is to prepare independent work while E09 runs, remove repeated numerical work
with verified reuse, and update the full-scope completion forecast within the first 12 hours.
Implementation gaps are recorded below rather than counted as runnable experiments.

## Inventory and batches

Completed scientific cells: E05 4, E06 4, E26 8 and E02 6: **22 total**, plus passed G0/E00.
Import their verified results; do not rerun them because the scheduler or source tree changed.

The prepared campaign declares **116 remaining development cells**, **8 G4 cells**, and
**63 class-final cells**. A cell is an arm × case × execution mode × seed, before valid reuse.
These counts include conditional rows, exclude completed families, and do not include the
additional scope listed after the table. A declaration is not an admitted job.

| Batch | Work | Declared cells | Target dispatch window |
| --- | --- | ---: | --- |
| B0 | Current qualification and E09 hierarchy comparison | 4 | First 12 h; already running |
| B1 | E01 extraction, E03 calibration, E08 attributes, E10 analytic fusion, E20 retrieval training, E24 missingness, E25 initial LLM controls | 35 | Hours 0–30 |
| B2 | E15 acceptance, E19 fitted fusion, E07 judge formats, E18 reranking, conditional E21 router/student, E25 forced responses | 24 | Hours 12–48 |
| B3 | E16 transfer, E22 label budgets and count policy, E25 oracle/trust replays | 16 | Hours 36–60 |
| B4 | E04 NIL/pool miss/listwise; E11 properties; E12 instance retrieval/evidence; E13 representation/enrichment; E14 typing; E23 graphs | 37 | Interleave during hours 0–60 |
| B5 | G4 baseline/core composition on D0 and D1, global and local, up to 1,000 sources | 8 | Hours 60–72 |
| B6 | Frozen E17 panel and published comparator; complete inference populations and validated submission files | 63 | Hours 72–132 target; later if needed and budgeted |
| Reserve | Recovery, output validation and final accounting | — | Hours 132–144 |

Windows are scheduling targets, **not runtime estimates or parallel GPU reservations**.
Dispatch an eligible step as soon as its actual input artifacts exist; do not wait for every
family in its batch. The machine-readable [batch inventory](six-day-batches.yaml) assigns
every uncompleted declared step exactly once. The detailed source-bound inventory and audits
are in [six-day-plan-01](../../data/experiments-v2/six-day-plan-01/).

The 187 declarations are not the entire completion claim. Explicitly budget required shared
controls/perturbations absent from the lock, any justified 1,000-source expansions, selected
component contrasts, matching held-out feature panels, and the separate OAEI-KG/BioKG/DISO
submission jobs. Their cell counts remain unknown until bound. Human precision-audit work
and organizer-side private scoring have a separate schedule; prepare the audit pack where
required and do not present those external outcomes as completed compute.

## Dependencies that determine the order

- Keep the completed E05 label-free retrieval policy. E20 has a separate, conditional
  supervised policy; its budget disposition must publish the permitted pool contract rather
  than deadlock all downstream families. A changed pool requires a separately identified run.
- Within E10: six analytic settings → frozen selected constants → two acceptance fits.
  Merely running its current eight flat declarations does not enforce that method.
- E03 → E15; E10/E26 → E19; E03/E15/E19 → E18 selected heads → E16/E22 → E22 count policy.
- E25 initial → E07 formats → actual forced responses → no-call oracle/trust replays.
  E21 additionally needs a compatible, demonstrably useful judge and a **training-only**
  teacher cache; E07 development judgments cannot become its training labels.
- B4 is an independent preparation lane, not wholly independent inference: E04 follows E03;
  optional E04 listwise follows E07; E12 retrieval freezes before instance evidence;
  E13 parity precedes typing/enrichment; E23 needs its correct case and supervision bindings.
- G4 freezes all final methods, recipes, controls and claim scope before reporting inference.
  A family that retains its baseline still supplies a valid dependency output. A required
  unresolved implementation or input gap remains incomplete, not a negative result.

## Work needed before the batches can be admitted

The core runner, native ontology path, checkpoints and most method implementations exist.
Required correctness work is concentrated in staged selection, case/label bindings,
feature-specific diagnostics and final population/export handling. E16/E22 and later
policy comparisons also need genuine upstream fitted artifacts; placeholder artifacts
cannot make those steps ready. Measure pending fitting recipes before admission.

Cross-cell caching, evidence replay and larger scoring batches are efficiency work.
They should be prioritized where they save substantial repeated computation, but a
scientifically ready, budgeted small comparison need not wait for every optimization.

| Preparation task | Required result |
| --- | --- |
| Shared numerical work | Verified reuse for common training features, E01 extraction replay and E10 channel-evidence replay; immutable identities include actual evidence, population, model, numerical settings and role. |
| E10 staged selection | Runtime consumer of the analytic winner, then materialize both acceptance recipes from the frozen constants. |
| E08 identifiers | Descriptor-backed namespaces/normalization/exclusivity and cross-property fact deduplication; the current empty allowlist is not executable evidence. |
| E24 diagnostics | Bind D0 control and declared perturbation inventories; separately bind or explicitly dispose of the optional typed asymmetry diagnostic. |
| E20 and other fitting | Use existing G0 fit64 evidence, then measure the actual pending recipes; admit full frozen training budgets instead of assuming inference timings cover fitting. |
| E21 and E04 listwise | Bind the compatible selected E07 judge; verify benefit and safe teacher labels for the conditional learning branch. |
| E11/E13 parity and E04 | Small prepared P0 and historical N0 inputs can be checked early; keep their limited statistical scope and historical ontology versions. |
| E13 enrichment | Provenance-bound native Datalog materialization and term conversion. Rule-containing input is not accepted by the current facts-only adapter. |
| E14 | Verify relation-macro/oracle-pair/full-pipeline metrics; the optional reasoner bridge remains a separate implementation gap. |
| E23 | Bind the TBox-rich rows to an appropriate case; K0 positive-unlabelled references cannot train the currently implemented negative-requiring graph head. |
| Final populations and exports | Bind full eligible ontology signatures for global inference. Preserve original local query membership rather than source-unioned pools; reconcile the strict exporter with those public inputs. Bind gold-free track-specific outputs. |

Capability checks are separate from resource admission. Old generic “G0 missing” strings in
prepared-campaign10 do not invalidate the G0 that passed, and passing fixtures do not establish
that these real-data prerequisites are complete. Optional or inapplicable treatments receive
an explicit evidence-backed disposition; required unfinished work is not silently dropped to
make the calendar fit.

## Efficiency and the six-day decision

The new D1 cold300 run on the RTX 4090 completed 5,998 pairs in **60.2 minutes**:
about **31.4 minutes ontology loading**, **9.4 minutes candidate preparation**, and
**16.8 minutes inference**. Peak host RSS was about 9.0 GiB and CUDA reserved memory 2.5 GiB.
Most cold wall time is fixed preparation; multiplying the entire cold run by source count
would exaggerate scoring cost. Conversely, warm inference alone would omit first-use costs.

The final public local pools already contain **1,568,650 distinct source–candidate pairs**
across H0/H1/H2. Original query membership is also needed: the 15,384 unique sources came
from 16,144 original query rows. Current preparation unions candidates by source; those
merged pools fail the strict local exporter. Preserve the original gold-free query rows and
fix the preparation/export contract before final inference, especially for pool-dependent
ranking and listwise judgments. Query-independent pair evidence may still be shared.

For global submissions, existing native inventories contain **211,958 NCIT classes** and
**385,973 SNOMED classes**. Applying the retained k20 to the full source census for the three
pairs gives about **19.68 million raw global candidates**, or **21.25 million pairs including
local inference, per profile/seed**. Final eligibility and exact-match filtering must still
be bound, so this is a sizing scenario, not a verified final workload.

A direct extrapolation of the observed 0.08–0.168 seconds per pair gives **472–992 hours for
one such pass**, before fitting, setup or safety allowance. Full-scale caches and evidence
density may change this substantially; it is not a reliable completion ETA. It does establish
that the current measurements do not support six-day admission. The prepared final matrix
also declares three profiles and three seeds; proven reuse must replace repeated work before
any completion promise.

Even one shared 21.25-million-pair pass would need about **148 pairs/second** to fit a 60-hour
final window with the 1.5 safety factor, before allowing for setup or distinct heads/models.
The measured rates are about 6–12.5 pairs/second. This makes batching and cached/vectorized
scoring a substantial performance task, not just a scheduling change.

Prioritize these output-preserving reductions:

1. Reuse completed scientific cells, verified prepared inputs, native projections where a
   persistent compatible artifact actually exists, and pinned embedding caches. The current
   per-cell training-score cache does **not** yet share work between cells.
2. Score identical train/evaluation evidence once. Replay extraction and analytic fusion,
   then fit each distinct head/fold on the correctly scoped cached features. Recompute
   downstream probabilities, uncertainty, decisions and traces for each treatment.
3. Reuse deterministic artifact-identical controls/seeds with an explicit equivalence record.
   Preserve stochastic training and prompt-order replications. Current seed-sensitive cache
   keys do not automatically establish this optimization.
4. Reuse actual forced LLM responses for the already implemented E25 policy replays. Different
   prompts still require different calls; training teachers remain separate from development.
5. Measure larger embedding/scoring batches on fixed development evidence and verify numerical,
   decision and trace parity before changing execution. The cold D1 run encoded 45,723 texts
   in 10,122 encoder batches; this suggests avoidable overhead but does not prove a speedup.
6. Interleave CPU preparation, small CPU fits and hosted calls with one heavy GPU worker.
   Keep the existing two CPU-worker/four hosted-request concurrency unless a measured
   operational amendment supports more. Do not run several large fits on a 24 GiB GPU blindly.

By hour 12, create a measured unique-work forecast, including cold materialization, fitting,
reporting, retries, memory/disk and all hosted roles. Apply the existing **1.5 safety factor**.
Report how the remaining critical path compares with the target calendar and recovery
reserve. Admission must pass the cumulative budget envelopes; the six-day calendar is advisory. Do not count proposed cache savings before parity and
throughput verification. If the target is missed, record the realistic full-scope ETA and
continue eligible budgeted work. A resource-budget failure still requires a valid amendment;
unfinished rows remain pending rather than being called completed.

Only the current node is available. If the measured, optimized full scope still exceeds
144 hours, give the longer single-node ETA and exact unfinished work. Do not resolve this
by assuming extra GPUs or silently reducing the scientific scope. The six-day target is
aspirational and is not an additional runtime or admission limit.

## Execution handoff

Only B0 currently has an active detached job: Slurm step **14372.2**, within the retained
interactive allocation, tmux `exact-d1-production-14372-replayfix`. Cold and warm scoring
completed. A post-scoring check incorrectly required evaluation output from these inference-only
jobs; the repair retained both completed runs and resumed at the completed-cache replay.
The queue proceeds to the four E09 arms after the remaining qualification and admission checks.
Its numerical source snapshot and scientific settings remain unchanged.

Prepare each following batch using the existing campaign runner, verified output ports and
measured admission. Keep one heavy GPU lane, durable checkpoints and one cumulative budget;
use detached **srun steps inside the retained interactive allocation**. A new plan is not
permission to cancel that allocation. There is no six-day runtime kill timer.

Rationales remain off. Ontology loading/projection remain native. Training/development and
reporting inputs remain separated, with no private test references used for optimization.
Final execution follows [LABELS-AND-SUBMISSIONS.md](LABELS-AND-SUBMISSIONS.md), with evaluation
disabled and validated mappings/rankings exported for submission. Do not launch the legacy
confirm path to bypass missing private references, fabricate scores or use partial populations.
