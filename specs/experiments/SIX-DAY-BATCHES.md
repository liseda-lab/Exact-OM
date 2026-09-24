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
The implementation amendment below prepares the remaining features; source qualification, real-data artifacts and measured admission still precede execution.

## Inventory and batches

Completed scientific cells: E05 4, E06 4, E26 8 and E02 6: **22 total**, plus passed G0/E00.
Import their verified results; do not rerun them because the scheduler or source tree changed.

The prospective [prepared-campaign-12](../../data/experiments-v2/prepared-campaign-12/) declares
**121 remaining development cells**, **8 G4 cells**, and **63 class-final cells** (**192 total**).
The historical campaign10 planning inventory was **187** (116 + 8 + 63). The amendment adds
one repeated E08 deduplication control and four E24 D0 controls. Step splits for E10, E14 and
E23 preserve the number of unique settings. A cell is an arm × case × execution mode × seed, before valid reuse.
These counts include conditional rows, exclude completed families, and do not include the
additional scope listed after the table. A declaration is not an admitted job.

| Batch | Work | Declared cells | Target dispatch window |
| --- | --- | ---: | --- |
| B0 | Current qualification and E09 hierarchy comparison | 4 | First 12 h; already running |
| B1 | E01 extraction, E03 calibration, E08 attributes/identifiers, E10 analytic fusion, E20 retrieval training, E24 D1/D0 and optional asymmetry, E25 initial LLM controls | 38 | Hours 0–30 |
| B2 | E10 acceptance, E15 acceptance, E19 fitted fusion, E07 judge formats, E18 reranking, conditional E21 router/student, E25 forced responses | 26 | Hours 12–48 |
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

The amended 192 declarations are not the entire completion claim. Explicitly budget required shared
controls/perturbations beyond the amended lock, any justified 1,000-source expansions, selected
component contrasts, matching held-out feature panels, and the separate OAEI-KG/BioKG/DISO
submission jobs. Their cell counts remain unknown until bound. Human precision-audit work
and organizer-side private scoring have a separate schedule; prepare the audit pack where
required and do not present those external outcomes as completed compute.

## Dependencies that determine the order

- Keep the completed E05 label-free retrieval policy. E20 has a separate, conditional
  supervised policy; its budget disposition must publish the permitted pool contract rather
  than deadlock all downstream families. A changed pool requires a separately identified run.
- Within E10: six analytic settings → frozen selected constants → two acceptance fits.
  The implementation now declares E10-analytic before the two E10 acceptance arms.
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

## Implementation and admission

The [implementation handoff](IMPLEMENTATION-HANDOFF.md) records the production changes,
configuration bindings, native preparation commands and recovery boundaries. The prospective
lock uses current recipes with the original case/model/baseline bindings and E13 parity views;
it starts no work and grants no admission. The existing cumulative budget is unchanged.

| Area | Implemented behavior and remaining execution gate |
| --- | --- |
| Shared numerical work | Checksummed pair/channel and training-table reuse. Measure actual parity, throughput and disk before counting savings in a forecast. |
| Staged decisions | E01 changes extraction only; E10 analytic selection precedes two acceptance recipes; E21 requires selected judge and training-only teacher artifacts. |
| E08 | Descriptor-backed identifiers, explicit exclusivity, and cross-property provenance deduplication before truncation. No curated exclusive namespace is currently bound: E08-identifiers is separately inapplicable, not an empirical null. |
| E24 | D0 controls, missingness semantics and pinned controlled perturbations. Optional typed asymmetry is separately dispositioned; controlled decision replay is limited to the validated label-free path. |
| E20/E16/E22 and other fitting | Actual training recipes and upstream selected-head/donor artifacts must be measured and produced before admission. |
| E11/E13 parity and E04 | Existing small P0 and historical N0 bindings retain their statistical scope and historical versions; run their admitted comparisons. |
| E13 enrichment | Native Souffle preparation/closure/recovery implemented. Materialize and bind both public raw and enriched views; the declared auxiliary-free structural fragment imports no alignment anchors and makes no full-OWL claim. |
| E14 | Relation-macro full-pipeline metrics and separate oracle diagnostics; optional OWL bridge isolated from required CSV typing arms. |
| E23 | Three D1 paired hierarchy ablations; K0 remains a separate natural comparison. Positive-unlabelled data do not train a negative-requiring head. |
| Final populations/exports | Native complete global census and original-query local sharding/export implemented. Materialize final recipes only after G4, with permitted public training inputs retained and development/test answers stripped. |

Historical generic G0-missing strings do not undo passed G0. Fixtures support implementation
review; genuine fitted inputs, materialized consequences and realistic memory/time measurements
remain execution prerequisites. Optional/inapplicable treatments keep explicit declarations and
reasons. No missing implementation or input becomes a scientific negative result.

## Efficiency and the six-day decision

The new D1 cold300 run on the RTX 4090 completed 5,998 pairs in **60.2 minutes**:
about **31.4 minutes ontology loading**, **9.4 minutes candidate preparation**, and
**16.8 minutes inference**. Peak host RSS was about 9.0 GiB and CUDA reserved memory 2.5 GiB.
Most cold wall time is fixed preparation; multiplying the entire cold run by source count
would exaggerate scoring cost. Conversely, warm inference alone would omit first-use costs.

The final public local pools already contain **1,568,650 distinct source–candidate pairs**
across H0/H1/H2. Original query membership is also needed: the 15,384 unique sources came
from 16,144 original query rows. Legacy preparation unions candidates by source; those merged pools are not valid submission
inventories. The new preparation/export path preserves original query rows and isolates
repeated-source queries for pool-dependent ranking and listwise judgments. Materialize those
public inventories for the frozen final recipe before inference. Query-independent pair evidence may still be shared.

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
   persistent compatible artifact actually exists, and pinned embedding caches. The new shared numerical cache is implemented; its safe compatible scope does not imply
   every method/seed can reuse the same table.
2. Score identical train/evaluation evidence once. Replay extraction and analytic fusion,
   then fit each distinct head/fold on the correctly scoped cached features. Recompute
   downstream probabilities, uncertainty, decisions and traces for each treatment.
3. Reuse deterministic artifact-identical controls/seeds with an explicit equivalence record.
   Preserve stochastic training and prompt-order replications. Seed-sensitive cache keys do not automatically establish deterministic equivalence.
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

Only B0 currently has an active detached job: Slurm step **14372.3**, within the retained
interactive allocation, tmux `exact-d1-production-14372-prefixfix`. It retains completed
cold300, warm300, completed-replay, ancestor-prefix and sibling-prefix work and is running the D0 control.
The ancestor receipt check had compared an interrupted checkpoint with later timing files;
the [repair receipt](../../data/experiments-v2/d1-prefix-repair-01/repair.json) records the fix.
Durable checkpoint receipts now bind verified content-addressed blobs; final operational
measurements remain separate. Historical charges are retained and no numerical scoring was
repeated for this repair. The numerical source snapshot `75c4d7a` and scientific settings are
unchanged. Follow the live [status file](../../data/experiments-v2/d1-production-probe-01/status.json);
the queue proceeds to four E09 arms only after qualification and admission checks pass.

Prepare each following batch using the existing campaign runner, verified output ports and
measured admission. Keep one heavy GPU lane, durable checkpoints and one cumulative budget;
use detached **srun steps inside the retained interactive allocation**. A new plan is not
permission to cancel that allocation. There is no six-day runtime kill timer.

Rationales remain off. Ontology loading/projection remain native. Training/development and
reporting inputs remain separated, with no private test references used for optimization.
Final execution follows [LABELS-AND-SUBMISSIONS.md](LABELS-AND-SUBMISSIONS.md), with evaluation
disabled and validated mappings/rankings exported for submission. Do not launch the legacy
confirm path to bypass missing private references, fabricate scores or use partial populations.
