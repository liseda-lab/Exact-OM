# E02 mechanism screen

The channel screen completed at **00:39:58 UTC on 2026-09-20**, Slurm step
**14220.4**, with exit zero. All eight E26 and four E06 configurations are complete;
three cells were reused and nine ran in that continuation. Neither family selected a
replacement. E05's top-k20 pool and both channel controls remain the frozen outputs.
These are 300-source development screens, not final benchmark results.

## Next admitted comparison

`data/experiments-v2/prepared-campaign-10/` prepares the complete six-arm E02 screen
on D0 NCIT–DOID, seed17, 300 development sources, global alignment. It imports the
original E05/E06/E26 decisions without rescoring them. The six treatments, their
supervision labels and selection rules remain unchanged.

Trusted anchors are selected deterministically from the public training positives:
3,845 distinct pairs, 576 excluded because either endpoint has competing mappings,
3,269 eligible pairs, and a fixed 2,000-anchor sample. All anchors are one-to-one and
disjoint from the entire 619-source development universe. The predicted-anchor rule
uses the existing threshold0.95 and margin0.1, declared before execution; it is not
a fitted or outcome-optimized rule. Both noise diagnostics use the same trusted
population. Input hashes and the deterministic builder are retained with the lock.

All six resolved configurations passed admission, alongside 141 campaign/scoring
regression tests. A further regression covers predicted-anchor preparation, disabled
LLM calls during the base pass, fresh-trainer checkpoint reuse, and invalidation
when diagnostic corruption changes. Before any experiment cell, the detached
launcher must verify all anchor endpoints against native ontology class signatures.
A failed native check stops the queue.

Hosted matching retains the baseline analytic gate; **rationale generation stays
off**. Structure supervision does not trigger selector/head fitting in these arms.
No private test reference is used. Native package fingerprints stay fixed.

## Resources and recovery

The queue uses one heavy GPU worker, two numerical CPU threads, and an `srun` step
inside the existing interactive allocation **14220**. The launcher has no wall-time
limit. Its 7.75-hour conservative admission forecast includes cold preparation,
structural rescoring and a1.5 safety factor; it is not a measured E02 duration.

Under the user's instruction to provide sufficient hosted allowances, this wave
records an explicit request planning cap amendment from20,000 to50,000. The32M
token cap,5,000-request/8M-token final reserves,336node-hour cap and every envelope
remain unchanged. E02 reserves19,200 requests and12.288M tokens from a conservative
extrapolation of G0's hosted probe, without assuming cache savings. Historical
spend and failed attempts remain in the cumulative ledger.

The latest channel-screen-02 prepared data and embedding cache are copied and
verified. Its request cache contains28 earlier rejected attempts; the new wave
also imports G0's3,883 verified completed responses, preserving both histories.
Only exact matching request identities can reuse these responses; changed prompts
remain separate. Cache import itself incurs no new hosted usage.

The immutable lock, input provenance, admission, forecast and bootstrap/launch
receipts are under `prepared-campaign-10/` and `mechanism-screen-01/`. The live
launch-status file is authoritative for whether native validation, screening,
completion or failure has occurred.

```bash
cat /home/pgcotovio/Exact-OM/data/experiments-v2/mechanism-screen-01/launch-status.json
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/mechanism-screen-01/launcher.log
tmux attach -t exact-mechanism-01
```

Detach tmux with Ctrl-b then d. Preserve allocation14220 and shell step14220.0.
The launcher receipt records the batch's own Slurm step. Compatible checkpoints
resume within this wave; immutable prior campaign selections remain external
history, and their progress files are not copied across campaign identities.

## Remaining blockers

E01 needs a common frozen scorer/selector/cardinality binding and extraction-only
replay; the current runner would repeat scoring and mix cardinality policies.
E10 needs its internal analytic selection phase before the two acceptance-training
arms; six analytic arms alone cannot complete the eight-arm family. E08 still
needs verified signed-identifier semantics and broader provenance deduplication.
E09 requires native runtime/resource admission on its specified D1 case; D0
cannot replace it. These families remain blocked, not empirically screened out.

## Reboot and exact-policy repair, 2026-09-20

Step **14220.5** ended when the node rebooted at **04:56:14 UTC**; Slurm closed
the step/allocation at04:57:03. No application traceback was recorded. The host
reboot's hardware/OS cause remains undetermined. The user has a new unlimited
interactive allocation, **14234**, and authorized restarting the experiment.

Native anchor validation passed before screening: all2,000 endpoints were present
on each side, peak RSS11.68GiB, elapsed267.47s. Two cell executions completed, but
inspection found that the soft treatment still used hard exact-match filtering.
The alignment action assigned its policy to `configs.dataset_params`, which returns
a temporary compatibility object. The assignment was discarded.

The fix updates `configs.dataset.filter_exact_matches` before configuration output
and timing fingerprints. Six propagation cases check hard, soft and unspecified
policies against both original flag values; the two incorrect override branches
failed before the fix. All18 focused tests passed with the native environment.
This repair implements the already-declared treatment; it does not change its
selection rule, source population, supervision or parameter choices.

The hard baseline already usedTrue and is compatible. Its saved scientific output
bytes and original measured cost are verified before migration. The old soft result
is invalid and remains historical evidence, not an empirical result for selection.
The trusted arm stopped during ontology loading without an inference checkpoint;
the remaining three arms had not started. Therefore **one cell is reused and five
need execution**. Native validation and exact-key hosted/embedding caches are retained.

`mechanism-repair-01/` records the source hashes, regression proof, invalidation,
interruption accounting and hard-only artifact migration. `mechanism-screen-02/`
is the continuation; the original wave is preserved. Its closed attempt accounts
for3 new hosted requests,1,097 tokens and$0.0001659, with zero unknown deliveries.
The charged interval runs from completed native validation to Slurm cancellation
and includes possible controller delay; it is not an exact active-compute measure.

The continuation uses a Slurm step inside allocation 14234, one GPU worker, two
numerical threads, disabled rationales and no wall-time limit. Its status and log
are authoritative once submitted:

```bash
cat /home/pgcotovio/Exact-OM/data/experiments-v2/mechanism-screen-02/launch-status.json
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/mechanism-screen-02/launcher.log
tmux attach -t exact-mechanism-02
```

Keep allocation 14234 and shell step 14234.0; detach tmux with Ctrl-b then d.


## E02 reporting repair, 2026-09-20

All six E02 cells completed successfully in continuation step **14234.1**, which
ran from 05:41:59 to 07:04:29 UTC. Final reporting then rejected unequal candidate-pool
fingerprints. These fingerprints include post-retrieval hard-anchor filtering and
cache metadata, so equality was inappropriate for the declared hard/soft contrast.
The original failed launch status remains historical evidence.

The E02-only guard now verifies the raw retrieval manifest, canonical retrieved
source/target pairs and entity kinds, and the frozen source population, including
sources with no candidates. Sampled manifests remain checked against their recorded
SHA-256. Missing/corrupt manifests, changed pairs or populations still fail; other
experiment families retain their existing checks. All six saved datasets have the
same 6,000 candidate pairs and 300 sources. The fix passed 116 focused tests.

`mechanism-report-repair-01/` finalizes the original six extraction/evaluation
artifacts without rescoring, new hosted calls or prediction-key migration. It
archives the original reports, verifies scientific output identities before and
after reporting, and charges finalization time to the cumulative reserve ledger.
Its separate status file records Slurm step 14234.2; the original launch failure is
not overwritten. This repair changes reporting validation only, not the treatments,
population, selection rule or scientific outputs.

Finalization completed successfully at **11:09:16 UTC**, after 137.81 seconds.
E02 is **screened_out**: no candidate met the predeclared promotion rule, so the
hard-anchor baseline remains selected. The final selection and paired reports are
under `mechanism-screen-02/runtime/exact-om-focused-v2/`; the repair receipt records
six reused cells, zero new scored pairs and zero hosted requests. This is a
300-source development decision, not a confirmatory finding on the final benchmarks.


## D1 native preparation measurement

With E02 finalized, a model-free D1 probe started at **11:12:19 UTC on 2026-09-20**
as Slurm step **14234.3**, detached in tmux `exact-d1-native-01`. It processes the
full SNOMED ontology and then FMA in separate native workers, retaining phase times,
peak RSS, projection provenance and cold/cached feature records. Source features
use the frozen seed17 sample of 300 public development entities. Target features
use 300 deterministic eligible entities; these are not an E09 retrieved candidate pool.

The probe uses the existing native benchmark and a 16-thread cap, with no wall-time
limit, model calls, generated rationales or private references. It reserves 2 hours
from the cumulative repair/resource envelope as a planning allowance, not a timeout,
and records actual elapsed time. A 56GiB worker RSS threshold requests cooperative
interruption to leave host/shell headroom; the actual D1 memory requirement is still
being measured. Per-ontology peaks do not bound full alignment memory use.

`data/experiments-v2/d1-native-probe-01/` contains the immutable input/config/code
bindings, launcher, status, per-side logs and phase reports. This is preparation
measurement, not the four-arm E09 screen. E09 still needs actual hierarchy coverage,
semantic/recovery checks and candidate runtime admission, including ancestor-IC
traversal cost; baseline preprocessing alone cannot establish those properties.

```bash
cat /home/pgcotovio/Exact-OM/data/experiments-v2/d1-native-probe-01/status.json
tail -F /home/pgcotovio/Exact-OM/data/experiments-v2/d1-native-probe-01/{source,target}.log
```

Keep allocation 14234 and interactive shell step 14234.0 alive. The probe is already
detached; ending the allocation terminates its step. No further experiment is queued
automatically after this measurement.
