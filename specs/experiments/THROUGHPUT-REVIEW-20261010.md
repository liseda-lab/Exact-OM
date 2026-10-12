# Implementation review and post-G4 verification

The implementation is **partially accepted, not scientifically qualified or
deployed**. On 2026-10-10 the user authorized review and supervisor-managed checks
on liseda-03 after G4. This authorizes the verification queue below, not the full
corrected scientific campaign. The previous
[implementation evidence](THROUGHPUT-IMPLEMENTATION-20261009.md) remains historical.

## 2026-10-12: fitted G4 performance investigation

The user requested a checkpoint stop and aggressive optimization review before
continuing scientific work. The stop marker is registered with the supervisor;
repairs and dispatch are paused while monitoring and allocation 14372 remain.
The earlier post-G4 completion dependencies below are historical, not permission
to restart G4. No GPU benchmark or scientific run was launched by this review.

**The dominant observed defect is repeated hashing of a fitted model.** Frozen G4
uses `42b27c06`; the inspected development tree is `7e14916a`. With hosted scoring
enabled, the frozen numerical wrapper skips its batch identity context. Each
channel lookup then follows:

`_scope → dataset.cache_fingerprint → _configured_retrieval_artifacts → assert_unchanged → _artifact_sha256`.

The fitted cross-encoder weight file is 2,239,614,572 bytes. A single real artifact
validation read 2,261,814,284 logical bytes in **1.3566 seconds**, with no physical
disk reads. G4 accumulated approximately **294 TB of logical reads** in about
50 hours. A separate 10.05-second observation recorded 16.43 GB of logical reads,
and the weight file was open in 39/40 samples. CPU usage averaged 98.8% of one
core; sampled GPU utilization was 0%. Approximately sixteen validations per pair
would cost 21.7 seconds, closely matching the observed 22 seconds/pair. This is a
measured mechanism and a strong attribution, not a full statistical stack profile:
debugger attachment was denied by the host's ptrace policy. There is no evidence
here that database lock waits or LLM latency explain the dominant delay.

An isolated CPU reproduction made two hosted forwards, each with 16 pairs and 16
channel lookups per pair. Frozen code performed **512** dataset/model identity
checks; prepared T02 code performed **2**, with identical outputs and 256 genuine
channel computations plus 256 cache hits in each case. It did not simulate GPU
throughput or read the large model 512 times. T02 addresses the repeated-hash
multiplication, but it has not been deployed to G4.

Earlier baseline cells on this same 4090 completed inference at **3.62–3.94
pairs/s** on October 8–9. The resumed core checkpoint records 9,216 pairs and
203,549 cumulative inference seconds, with an EMA of 0.04525 pairs/s. These are
different recipes; the evidence supports a fitted-path defect, not a demonstrated
progressive slowdown caused by growing caches.

### Required implementation and verification order

1. **Finish T02 for real fitted artifacts.** Validate mandatory model inputs
   explicitly, then reuse the verified identity across a scoring batch. Add a
   tiny real-artifact regression which counts full-file reads in hosted forwards,
   cache hits, cache-disabled operation and repeated batches. Full bytes must be
   checked at admission/resume; optional verified inventory/stat reuse must detect
   replacement, same-size edits, restored mtime, added/missing files and symlinks.
   No initial trust based only on a filename or timestamp.
2. **Keep provenance failures fatal.** Current development's optional-cache
   exception handler catches `ValueError` from a changed retrieval artifact and
   falls back to scoring. Both old and new handlers can also swallow `OSError`
   for missing mandatory artifacts. Move mandatory validation outside this
   fallback path, including disabled-cache and cache-hit execution. Only failure
   of disposable cache storage may become a cache miss.
3. **Deploy the reviewed cache and diagnostic improvements together.** Prepared
   T03 provides process-owned encoder connections and bounded bulk I/O; T07
   removes ten synthetic diagnostic replays from ordinary inference. Preserve
   natural decision evidence and the existing dedicated diagnostic results.
   Use node-local, bounded scratch for disposable SQLite caches; retain durable
   checkpoints and paid-call accounting. `/tmp` is local NVMe with approximately
   633 GiB available at inspection; both home and the dataset mount are NFS.
   This is available space, not a new cache allowance or permission to copy the
   entire historical cache. SQLite documents that WAL is unsupported over
   [network filesystems](https://www.sqlite.org/wal.html).
4. **Repair batching in separable stages.** Probe channel caches before preparing
   structural blocks: the rejected implementation still computed 726 matrix
   groups and 334 combined-context groups during replay. Batch cache/planning
   work and compact transfers first, retaining existing encoder call shapes,
   GPU arithmetic, evidence ordering and precision. Then isolate encoder
   rebatching, `mm` versus `bmm`, and CPU/GPU reductions in separate GPU checks.
   Saved pair 8 changes selected facts after a near tie becomes an exact tie;
   pair 6 changes prompt text from `0.867` to `0.868` without changing those facts.
   Neither a stable sorter nor checking top-k membership alone establishes parity.
   Test exact prompt bytes, citations, routing, fitted selectors and mappings.
5. **Amortize whole-run work.** Use T05/C04 to fit once per identical pair, recipe
   and seed, then load verified fits in deployment. Share native ontology features,
   label vectors and exact retrieval primitives under their dependency contracts.
   Measure first-stage search and cross-encoder reranking independently: selected
   global recipes can add 4.13–7.72 million reranker evaluations per case, outside
   the natural-scorer throughput target. Hosted requests already have concurrency
   four; any further overlap must preserve query/gate ordering, reservations and
   bounded queues, with no database transaction held through a network wait.
6. **Replace baseline-only confidence with actual-recipe evidence.** Use a small
   unchanged G4 fitted workload first, retaining complete query groups and saved
   compatible responses. Compare against stored scientific evidence where valid;
   do not spend hours repeating the known pathological reference for accounting.
   Then measure cold, warm and partial-cache operation on this 4090, including
   artifact-read counts, cache waits, throughput by batch and physical storage.
   The larger H0/H1/H2 checks remain necessary after the narrow failures are fixed;
   fixture success alone cannot release scientific execution.

**The one-day objective is an end-to-end performance target, not a kill timer.**
Conditional full global-plus-local work is 4.31/8.06/8.81 million pair occurrences
per recipe across H0/H1/H2. At a genuinely measured 200 pairs/s, numerical scoring
alone would take 6.0/11.2/12.2 hours; retrieval, hosted work, fitting, extraction and
durability still count. Baseline and selected H2 recipes together require
approximately 24.5 numerical hours without reuse. Forecast actual unique work and every phase;
do not promise a day from the numerical rate alone or reduce scientific coverage.

Focused offline verification on `7e14916a`: **52 tests passed in 9.40 seconds**
(`scoring_cache_throughput_test.py`, `numerical_cache_test.py`,
`retrieval_experiments_test.py`), plus **5 throughput-contract tests in 4.07
seconds**. These do not yet cover the new real-artifact
read-count and swallowed-provenance-error regressions. Audit artifacts and the
reproduction are under `data/experiments-v2/g4-performance-audit-20261012-01/`.
Current G4 stop/restart state belongs to the live supervisor receipts, not this
dated review. Scientific resumption remains held; 200 pairs/s and one-day
completion are unverified.

## Review findings and corrections

1. **Frozen-recipe admission:** a deployment could change its threshold, retrieval
   or evidence configuration while keeping the selected-config binding and a
   self-consistent inference manifest. Admission now reconstructs the expected
   deployment from the selected recipe and verified runtime fits. Only the
   declared case, mode, seed, public populations and manifest substitutions are
   admitted. Regression tests cover three changed recipes and legitimate fitted
   global/local deployments. Commit `6b33909a`.
2. **False successful parity command:** the comparator wrote a failed receipt but
   exited zero. It now exits nonzero on mismatch and rejects empty comparisons.
   The supervised worker also inspects the receipt, independently of exit status.
3. **Qualification isolation:** an accidental hosted code path could modify a
   request ledger before the socket-level network block fired. The candidate
   harness now denies hosted generation before key, ledger or transport access.
   Tests inject both client-owned and inherited ledger paths. Workers additionally
   isolate inherited experiment state, caches, temporary files and fixture ledgers.
   These safeguards and the queue tool are in commit `d205cdee`.

Independent review found no additional default-path scoring/cache/native evidence
regression in its inspected scope; 167 focused CPU tests passed. The deployment
review suite passed 40 tests, and the combined supervisor/queue/parity/guard/
throughput suite passed 106 tests. These suites overlap; their counts are not a
unique test total. No GPU measurement or paid request was made during this review.

## Finite GPU checks

Artifacts are under
`data/experiments-v2/throughput-review-20261010-01/`. The candidate is the detached
worktree `candidate-source/` at `d205cdeed08505ea21422b96a626bd691df2c358`.
`gpu-checks-01/queue-proposal.json` contains three prepared launch descriptors:

| Queue ID | Pair | Work |
| --- | --- | --- |
| `gpu-checks-01-H0` | NCIT–DOID | Reference prefix parity; complete bounded cold/warm/replay measurement |
| `gpu-checks-01-H1` | SNOMED–FMA | Same checks on its independently frozen public workload |
| `gpu-checks-01-H2` | SNOMED–NCIT | Same checks; fixture templates only, not a frozen scientific recipe |

Each check waits for successful `G4-run-once-followup` completion, following its
registered recovery lineage. The single GPU and six-CPU resource envelope
serializes execution; one case's failure does not erase the other queued cases.
G4 and its current source are unchanged. The workers use detached numeric Slurm
steps inside allocation 14372, six CPUs and 112 GiB RAM, with no wall-time kill
timer. They require an RTX 4090 receipt, matching source/input/precision/step
identities, at least 5,000 unique computed pairs, three chunks, 30 seconds, and
nonempty evidence parity. A measured throughput miss is reported separately from
failed parity; neither grants scientific promotion. Existing storage guards and
intervention handling remain active. A duplicate invocation preserves old terminal
receipts instead of replacing a successful exit with a failed duplicate's exit.

The reference is the default unbatched scorer from `5cd906b2`, whose recorded 2060
prefix matched the original `0eb61e8f` scorer. Its new isolated copy adds only an
observer field reporting actual precision. File inventories bind this overlay;
`reference-observer-review.json` identifies it explicitly. No old source snapshot
was modified. The new comparisons run reference and candidate on the **same
4090**; old-GPU values cannot substitute for them.

The existing workloads contain 6,028 / 6,092 / 6,026 unique candidate pairs before
exact prefiltering. Original complete queries and source groups remain intact;
private reference labels are absent. H2 fixture templates are a recorded union of
already generated H0/H1 templates, with H1 precedence for the one collision.
Both compared paths use identical templates. This is explicitly qualification
evidence and cannot be promoted into the actual fitted H2 scientific recipe.

At review, no implementation-check worker remained active to adopt. Allocation
14452 had already been released; other work on liseda-05 remains untouched.
Supervisor 14372.269 and G4 14372.270 were alive. Four baseline cells had completed,
two core D0 cells had failed, and core D1 global was active. The existing supervisor
retains responsibility for G4 recovery; this is not an eight-cell completion.

## Required supervisor continuation

`T09-fitted-checks-and-admission-prepare-20261010` is a preparation task after G4
and the three GPU checks. It must prepare real follow-up workers and update their
dependency links, rather than mark scientific acceptance from binary fixtures:

- Reconcile G4's failed cells using the reviewed fusion replay correction and
  supplied-local-pool retrieval fix; preserve completed work and paid accounting.
  Obtain fair deployment costs before the corrected mechanical selection/fitting
  freeze. Do not open H2 validation before that freeze.
- Extend verification to actual frozen fitted recipes, including grouped or
  exemplar decisions where selected, fitted selectors and complete mapping
  semantics. The current binary fixture harness cannot establish these.
- Complete the counted whole-program forecast implementation and measurements.
  `forecast_scoring_successor.py` currently produces conditional top-20
  sensitivities and unknown whole-program totals; supplying more GPU receipts
  alone does not turn it into final admission. Include retrieval, fitting, hosted
  work, assignment, all physical storage roots and protected spending limits.
- T04 batching already failed evidence parity and remains disabled. It needs an
  implementation correction plus independent proof before scientific use. Do not
  repeatedly rerun the same rejected batching to claim progress toward 200.
- Prepare corrected launch artifacts only after the remaining gates pass. The
  historical 63-cell E17 schedule must not be dispatched. The two legacy pending
  E17 rows are held for replacement by the corrected design and explicit rollout
  admission; the current review authorizes checks only.

The supervisor can handle correctness fixes, checked recovery and the remaining
verification preparation within existing policy. It must request intervention
for an actual scientific decision, allowance change or unresolved repeated error.
It may not waive parity, change precision/coverage, raise caps or invent missing
receipts. Keep rationales off and all G4/E17 usage in the same spending family.

`publication.json` in the review artifact directory records whether the queue was
published and binds the before/after registry and unchanged supervisor policy.
Current queue/health files, not this dated snapshot, determine live status.
