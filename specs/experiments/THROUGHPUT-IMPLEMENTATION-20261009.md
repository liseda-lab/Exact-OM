# Scoring implementation and corrected-study handoff

**2026-10-10 review continuation:**
[THROUGHPUT-REVIEW-20261010.md](THROUGHPUT-REVIEW-20261010.md) records subsequent
correctness fixes and the authorized post-G4 supervisor verification queue.
The no-queue-change statements below describe the original implementation task.
Full corrected campaign rollout remains separately gated.

Recorded 2026-10-09; verification continued into 2026-10-10 UTC. **Implementation is
partially accepted; deployment is blocked.** The experimental batched path failed
real-GPU evidence parity and missed 200 cold unique pairs/s. CPU fixture success
is not scientific acceptance. No corrected campaign, queue replacement or
supervisor restart was performed. This record supplements, without weakening,
[the throughput contract](SCORING-THROUGHPUT-200.md), [the corrected design](FINAL-STUDY-CORRECTION-20261009.md)
and [machine-readable status](throughput-final-plan-20261009.yaml).

All paths below are relative to `/home/pgcotovio/Exact-OM`. Full local evidence is
under `data/experiments-v2/throughput-20261009-01/` (called `E` below). Small tracked
bindings and results are in `throughput-implementation-evidence-20261009.json`.
Qualification outputs are not scientific checkpoints and must not be promoted.

## Work delivered and acceptance boundaries

| Package | Implementation and evidence | Remaining acceptance |
| --- | --- | --- |
| T01 | Fixed public, label-free H0/H1/H2 workloads, complete original query grouping, counted input inventory, instrumented cold/warm/replay tool, conditional resource forecast. | Actual frozen retrieval union, unique fits/texts/requests and whole-program forecast depend on corrected selection. |
| T02 | Batch-scoped numerical identities also when hosted inference is enabled; bounded buffered writes flushed before hosted calls; exception-safe restoration. | Final exact source/per-recipe GPU qualification. |
| T03 | Persistent process-owned SQLite connections, bulk reads/writes, exact-text deduplication/scatter, bounded device/CPU caches, actual weight/tokenizer identity, physical database cap. | Final exact source/per-recipe GPU qualification; batch-sensitive GPU encoding remains a scientific concern. |
| T04 | Staged pair-specific context encoding and shape-bucketed matrix/reduction batches, bounded tiles and OOM splitting. | **GPU parity failed.** Default off; scientific use is rejected before cache lookup. Only isolated `throughput_fixture` / `qualification-*` use is admitted. |
| T05 | Dependency-scoped primitives and discoverable immutable references to completed fitting aggregates; actual payload checksums, disjointness checks on hits, no duplicate durable aggregate. | Legacy aggregates lacking sufficient identity proof are conservative misses; no speculative migration. |
| T06 | One traversal over native-produced edges; argument-correct example cache; original iteration order retained. | Native packages unchanged; CPU fixtures passed. |
| T07 | Ordinary inference excludes controlled synthetic perturbation diagnostics; required natural evidence remains. | Existing completed diagnostic evidence retained; future missing component analyses remain separate. |
| T08 | Checksummed durable frames, bounded frame readers/compaction, read-only inspection, physical multi-root guards, compact resident evidence after durable write, original query reconstruction and whole-case SQLite extraction. | Full-scale RAM/storage admission pending. Assignment still uses whole-case semantics and memory; legacy unindexed stores retain their compatibility reader. |
| T09 | Strict nested evidence/decision comparator, real-GPU natural stress sample, measured cold/warm/replay, recovery and operational regression tests, source/receipt bindings. | **Acceptance not met:** T04 mismatch, missing H1/H2 and selected-recipe GPU receipts, hosted and full end-to-end measurements. |
| C01 | Corrected logical compiler: 12 primary, up to 2 required H0 controls, 3 deterministic LogMap, up to 54 bounded cells; only missing predeclared component work. | Actual freeze and execution bindings. |
| C02 | H2 validation binding enforces corrected selection freeze first, fixed label-free source selection, original local membership, report-only use. | Actual H2 validation cohort deliberately not opened/prepared before freeze. |
| C03 | G4 reuse requires actual scientific parity and fair matched deployment costs; qualification receipts cannot substitute. | Live G4 completion, corrected cost evidence and selection decision. |
| C04 | Freeze actual fitted runtime artifacts, including dynamically written heads; bind deployment recipes to the selected/fitted receipt. Final inference loads these artifacts with zero refit/raw reference reads. | Actual seed-specific fitting/freeze, artifact checksums and complete deployment manifests. |
| C05 | Corrected worker/queue compilers, actual recovery identities, guarded write roots, terminal receipts and 71 disabled proposed rows. | Actual scientific bindings, source/policy admission and explicit rollout authorization; live queue unchanged. |
| C06 | Coverage checks preserve empty/original queries, global source/target populations and submission contracts. | Real complete mappings, local exports and organizer evaluation remain pending. |

Preparatory C01–C06 fixtures do not establish any future scientific result.
The programme is not marked complete merely because implementation tests pass.

## Reconciliation and commits

The checkout was not reset to specification commits `3d55bb6d` / `bfd2f7c0`.
Commit `9044d1b4` reconciles the reviewed frozen-G4 fixes (`f4fd5b26`,
`035a1a56`) into development, preserving later work. These include native input
checks, reference-free deployment freeze and complete final populations.
Concurrent repair/frontend changes and their merge were retained. No push was
performed by this task; the unrelated `exports/exact-repair-first-tests-20260929.tar.gz`
was left untouched.

Principal implementation commits (full identities are in the tracked evidence):

- `26079586`, `50e3f81b`: numerical/encoder scopes, persistent bulk cache,
  dependency-correct reuse and physical limits.
- `858cf84f`, `a7c38c1a`: staged numerical batching/native one-pass examples/
  ordinary diagnostics, then rejection of unqualified scientific batching.
- `3f702001`, `ce5658fd`, `0ea59624`, `e5c611b0`, `d05a89b3`: complete groups,
  global extraction, compact durable evidence and recovery/read-only readers.
- `e56ebb52`, `87eb8d90`, `06b41079`, `9ef275e5`: actual fitted artifacts,
  mixed-scale design, scientific reuse guard and freeze-linked recipes.
- `a22e211e`, `5cd906b2`, `f4dfbdf8`, `f4bb9e77`, `9a7495e4`: counted
  verification, strict parity, stress coverage and conditional forecasting.
- `7dedbbc5`, `b90ecf8b`, `9fc44447`: source audit indexing and fixture updates.
- `4a23c799`, `8c8753ef`: full supplied local pools and actual G4 fusion replay regression.
- `9fd61fe7`, `db6a7da2`: physical SQLite/WAL reservation and measured forecast denominators.
- `e0fad852`, `c8f15fc3`: corrected workers/cohorts, offline queue export, recovery identities, terminal receipts and storage/spending admission.
- `09e8722d`: close concrete Java/Python temporary and model/Matplotlib/extraction write-root gaps.

## Verification actually performed

The final clean source `09e8722db5f88e9d9b0ea5ac6e1895f34d54605b` passed
**748 tests in 112.00 seconds across 48 files**. The exact command and log
location are recorded in `E/cpu-regression-source03.json`; no GPU job or
paid call was made. Coverage includes cache identity/fork/mutation behavior,
training aggregate discovery, native relations, fitted artifact loading, original
queries/global extraction, durable recovery, spending and dispatch boundaries,
and all corrected worker/queue paths. Eleven warnings concern filesystem mtime,
fixture entity-kind fallback and the deliberately exercised fork path.

The first combined source-02 run retained in `E/cpu-regression-handoff.json`
reported 736 passes and one fixture failure: the new Java storage guard correctly
rejected an altered executor before the fixture's expected scientific-descriptor
error. The fixture now checks the earlier rejection without weakening either
guard. Independent review also closed concrete temporary/cache write-root gaps.
Both failed and passing receipts are preserved; no GPU result is attributed to
these later changes. `E/independent-review-handoff.json` records review scope.

Earlier `E/cpu-regression-final.json` records **486 tests passed in 48.11 seconds**, 30
files, source `a7c38c1a3d2ab32354dd14f69115b52e60f958ba`. This covers cache
identity and mutation/fork handling, native relation examples, runtime fitted
artifacts, final public inference, query boundaries, extraction, checksummed
recovery, compaction, artifact inspection, spending and supervisor behavior.
Additional focused tests exercised staged scoring, numerical cache boundaries
with hosted fixtures, physical eviction and deployment admission. Exact commands
and the complete output are retained; these were offline tests with no paid calls.

GPU verification used a separate allocation **14452**, one **RTX 2060 SUPER
8 GiB**, six CPUs and 56 GiB RAM on **liseda-t2**, through detached numeric Slurm
steps. Active G4 allocation 14372 on liseda-03 was preserved. liseda-05 belongs to
other work and was not used after the user's reservation. An earlier completed
198-pair smoke there is retained as historical evidence, not target acceptance.

The measured source was immutable `5cd906b210696b06dc204cb126861450aec087f2`
(`E/candidate-source-01`). Later identity/storage/admission fixes have CPU
verification but were not silently attributed to this GPU run. Model precision
was unchanged. Native processing used the installed native ontology/reasoner
packages. Networking was blocked in the numerical worker; hosted decisions were
fixtures, so hosted-quality and end-to-end scientific time are explicitly unknown.

| Path on RTX 2060 SUPER | Cold unique pairs | Cold seconds | Cold unique pairs/s | Warm-encoder numerical rows/s | Numerical replay rows/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default unbatched, bounded prefix | 198 | 103.250 | 1.918 | 12.157 | 36.902 |
| Experimental batched H0 | 5,997 | 495.492 | **12.103** | 42.634 | 39.084 |

The second row spans 13 completed chunks and 6,069 reporting rows. Its numerator
excludes 72 duplicate rows and all exact-prefiltered pairs/cache hits. Median
chunk cold rate was 13.146; worst was 4.392 pairs/s. Cold time includes evidence
selection, required numerical work, identity/cache operations and 12.975 seconds
of durable evidence writes. Native preparation was separately 557.122 seconds,
model loading 0.846 seconds (local filesystem caches may be warm). Peak process
RSS was 12,404,568 KiB; peak allocated GPU memory was 2,963,368,960 bytes. Inclusive
phase timers overlap and must not be summed. Warm/replay do not count as new cold
work. Replay still performs staged preparation and is not faster than warm here.
The old receipt's non-cold `distinct_computed_pairs=0` is a conservative numerator
convention; warm uses a separate numerical cache while retaining encoder vectors.

**Default-path parity passed:** `E/H0-default-parity-01.json` compares the original
`0eb61e8f` scorer with the default unbatched `5cd906b2` path on the same GPU,
workload and precision. All 198 rows / 60,826 floating fields match exactly:
maximum error zero, zero numerical or discrete mismatches, identical query
boundaries. Fitted selector/final mapping parity and the final source's complete
recipes remain pending. The reference tool predates cache-complete measurement
and must not be used as a throughput claim.

The completed verification allocation **14452 was released** after both workers
exited zero and only its extern step remained. `E/verification-allocation-release.json`
records this; allocations 14372 and 14451 were preserved.

**Parity failed**, independently of the throughput miss. The same-device
198-pair prefix compared 60,826 floating fields and found 7,441 numerical and
1,174 discrete path mismatches at GPU absolute tolerance 1e-5. Selected evidence,
its order/provenance and hosted packet text differ. The maximum final-score
absolute difference was 0.0001054406; a reported 0.5 maximum over all numeric
fields is a reordered evidence-specificity value, not a final-score error.
Both paths routed the same 45 pairs and had the same prediction flags in this
prefix; this does not excuse the evidence mismatch. Half-precision batch shape
can change near ties (one observed support pair changed from .71142578125 /
.7109375 to equal .71142578125). Precision and tie rules were not relaxed.

`E/H0-anchor-inventory-review.json` confirms that differing exact-anchor inventory
sizes in the old prefix tool cannot cause this baseline mismatch: the only
consumer is disabled by `hier.enabled=false` and `mode=labels`. The current tool
nevertheless binds the complete anchor inventory independently of `max_chunks`
for future active-overlap configurations. `E/H0-legacy-stress-coverage.json`
records 99 long contexts, 198 uneven contexts, eight missing-channel pairs and
two complete 100-candidate queries in the fixed prefix. These are natural stress
cases, not synthetic speed replays; separate stress-stratum timings are pending.
H1/H2 inventories exist, but real-GPU parity/performance for them and the selected
stack is pending. A 4090 result cannot be inferred from a 2060 measurement.

## Counted work and remaining resources

`E/workloads/work-counts.json` records the public populations. Per arm, the three
local cases contain 1,819 / 3,414 / 10,911 original queries, 1,614,400 candidate
occurrences and 1,568,650 unique pairs in total. Full global source counts are
206,379 / 385,973 / 385,973; target counts are 17,028 / 104,721 / 206,379.
Global retrieval uses full target ontologies, never the local ranking pools.

`E/forecast-05.json` gives counted **conditional scenarios**, not an approved
whole-program ETA. With the inspected top-20, nonadaptive, no-anchor-rescoring
recipes, primary scoring has at most 42,361,800 occurrences before exact
prefiltering and reuse, or 46,671,280 including both H0 controls. This excludes
fitting, bounded/component work and LogMap. The actual union of numerical
identities is unknown until the corrected recipes are frozen.

Applying the measured rates to those occurrence caps gives 255.7–281.7 scoring
days for the small unbatched 2060 sample, or 40.5–44.6 days for the **scientifically
rejected** batched sample. These are rate sensitivities, not a reliable range or
an estimate for the 4090. Native preparation, retrieval, fitting, hosted waits,
global assignment, packaging and three published runs are additional. The former
7–10 day aspiration is unestablished. Whole-program wall/GPU time remains unknown
rather than treating missing phases as zero or assuming 200 pairs/s.

The same samples project 177–266 GB of full evidence alone, before candidate
indexes, fitted artifacts, exports, caches/WAL, bounded/component/published
outputs and atomic replacement space. Initial physical programme usage was
833,468,020,838 bytes, including 157,906,882,560 bytes of shared numerical/encoder
databases. The unchanged 1 TiB programme cap and 16 GiB reserve left about 249 GB
before further live growth. The later read-only scan recorded **850,843,879,066
bytes**, leaving **231,487,879,526 bytes** (215.6 GiB) after the unchanged reserve.
The forecast includes allocated directory bytes, hard-link deduplication and
database/WAL overhead. The default-path evidence scenario alone requires
241.888–266.495 GB, already above the 231.488 GB headroom before the excluded
outputs. The lower 177 GB projection belongs to the rejected batched path and
cannot justify admission. These are conditional size scenarios, not a claim
that the eventual frozen programme must have exactly these sizes.
Filesystem free space is not an extra quota.
`/mnt/datasets/pgcotovio` is NFS, not NVMe scratch. Any additional root must be
explicitly guarded. Final placement/retention and the whole-program physical
forecast have not passed admission. No live checkpoint/cache cleanup was done.

Historical admitted usage at the recorded snapshot was **101,946,432 tokens and
$15.4178172 known cost**, with **31 unpriced historical attempts** retained.
This leaves 98,053,568 campaign tokens and 20,393,643 E17 tokens before existing
caps; unknown-send reservations remain in the authoritative ledger. This task
made **zero new paid calls**. An inherited request ledger is not counted again
on top of the cumulative admission ledger.

At historical decision averages, even a hypothetical 0.1% routing fraction over
the primary occurrence cap implies 101.75M decision tokens / $15.28; 1% implies
1.0175B / $152.82. These exclude other roles and are sensitivities, not predicted
routes or permission to spend. Current-recipe route fractions, reusable requests,
fits, rate limits and unknown sends must be resolved before admission. Preserve
10M notifications/warnings, 25M family and 200M campaign pauses, prompt limits,
all historical charges and intervention notifications. No allowance was raised.
Rationales remain off.

## Existing G4 failures and successor corrections

The live worker remains frozen at `035a1a56`. Its four baseline cells completed;
core D0 global and local cells recorded errors before it continued to core D1
global. This is not eight-cell G4 completion or selection readiness.

- Global D0 failed the neutral fitted-fusion replay guard. The subsequent
  development correction `fd8a6b11` was already present and has been preserved.
  A bounded streaming audit of **all 202,212 saved public-development rows**
  passes the declared mixed-precision replay at unchanged tolerance (maximum
  absolute error **2.384185791015625e-7**); old float64 replay fails 84,066 rows.
  No additional arithmetic/tolerance change was needed. `8c8753ef` adds a
  numerical-only regression from an actual failing row; 21 focused tests passed.
  Receipt/source: `E/G4-fusion-replay-audit.json` and `.py`. JSON parsing passes
  over aggregate fields but retains only numerical fusion channels/S_base;
  no reference files were opened, no labels were used and no fitting occurred.
  This proves the replay correction, not every dependency needed to migrate that
  aggregate into a different scientific execution.
- Local D0 applied global cross-encoder `top_k=20` to supplied local pools.
  Its 619 original queries contain 62,468 unique pairs and up to 158 candidates
  per query; all exceed 20. `4a23c799` removes this check only for supplied pools,
  retaining every pair/original query and leaving generated global retrieval
  bounds unchanged. All 21 retrieval tests passed.
- `9fd61fe7` tightens physical encoder-cache reservations to include SQLite page
  packing and WAL copies. The unchanged quota was tested with 1,024 real
  2,048-byte vectors, existing WAL and declined optional persistence; 30
  cache/identity tests passed. Fresh outputs remain complete when caching stops.
- `db6a7da2` counts full physical evidence storage and uses only observed token,
  cost and service-time denominators; missing observations remain unknown.
  Five forecast/parity tests passed.

None of these findings caused a live G4 patch/restart or paid-request retry.
The next agent must reconcile any failed cells/checkpoint compatibility before
claiming G4 completion. RTX 4090 checks may wait until that live worker finishes,
as the user explicitly requested.

## Migration, recovery and rollback

Keep old source, locks, completed rows, request attempts and checkpoints immutable.
New computation has a separate source/primitive namespace and storage roots.
Migrate only primitives/aggregates whose actual dependency identity and checksum
prove compatibility; unsupported legacy identities are misses. Label-free
qualification outputs cannot serve as fitted artifacts or scientific decisions.
Incomplete/corrupt frames cannot count complete. Paid requests in uncertain-send
state require ledger reconciliation, never blind resubmission.

New frame indexes permit a bounded reader. Huge old unindexed stores retain the
legacy reader until compaction in an isolated copy is admitted. Compaction writes
replacement payloads and commits the new index before removing old referenced
files; interruption tests cover both sides. Cleanup only removes provably dead,
local-owner temporary compactions; uncertain/remote ownership is preserved.
Whole-case assignment and some candidate/overlay collections remain O(pair count)
in memory; do not claim a universal fixed-memory pipeline.

Rollback means retaining the old pending-descriptor snapshot and frozen source,
stopping only a newly admitted successor step if necessary, recording completed
chunks/paid exposure, and restoring only still-pending rows under the registry
lock. Never cancel the interactive allocation or rewrite a completed/running G4
row. Do not delete source/cache versions needed for reconstruction.

## Successor preparation and rollout gates

The final implementation source is the clean detached Git worktree
`E/successor-source-03/` at **`09e8722db5f88e9d9b0ea5ac6e1895f34d54605b`**.
The worker checks this exact source identity and its own module location. The
older source/bundle versions remain historical, including source-02 at
`c8f15fc3` and its failed combined fixture receipt;
an archive without its own Git identity is not a deployable worker checkout.

`E/successor-03/successor-bundle.json` contains 71 logical cells: 12 primary,
2 provisionally required H0 controls, 3 published and 54 bounded. Controls can
be removed only with the declared supervision/equivalence receipt. Missing
predeclared component work remains conditional. Actual fits/selection are absent,
so there are zero bound executable workers. `E/queue-03/queue-proposal.json`
contains 71 disabled, `needs_user=true` rows with no live publication. Both have
`launchable=false` and bind registry/dispatch snapshots. Fixtures exercise valid
worker compilation, cache/checkpoint restoration and failure reconciliation;
fixture bindings cannot be substituted for actual scientific artifacts.

`E/successor-environment-proposal-02.json` is an **unapplied** placement proposal:
batching/prefetch off, rationales off, compact resident evidence on, isolated
primitive namespace, node-local hot caches and unchanged durable programme cap.
Its proposed 32 GiB physical numerical cache, 8 GiB encoder cache and 16 GiB
prepared cache all count within a proposed 64 GiB additional-root guard.
HF/Torch/XDG/Matplotlib caches and Java temporary files are explicitly inside
that root;
create and verify the reviewed `TMPDIR` before launch export so Python cannot
fall back to an uncovered directory. The pinned model/tokenizer files still need
copying and checksum verification
for offline loading, with every physical copy counted. This
is not new quota authorization. Check actual liseda-03 free space/server quota,
retention, Java/temp paths, every run directory and all growing roots before
reviewing a policy/environment binding. The worker rejects uncovered write roots.

The 2026-10-10 00:37:50 UTC read-only handoff snapshot records G4 **14372.270**
and supervisor **14372.269**, both running on liseda-03, G4 nonce
`0d708e3527ac4cb6949ea9261e5a5d0a`. Four baseline cells finished; core D0 global
and local failed as described above; core D1 global is still active after its
3,737-second dataset stage. Allocation **14452 is released**. The user's
**14451 / liseda-05** allocation is preserved. Registry SHA-256 remains
`fd7e1190355ff383a6042409e7e39187c99e24ece53b1fa80762e2faebaa6e17` and policy
SHA-256 remains `ffe02f319196b4227ebfa9e874d574248741f7a26df3600b2d4b78f699701ae0`.
See `E/live-state-handoff.json`; these observations must be refreshed at rollout.

The following commands reproduce **offline preparation only**, using fresh output
directories because published preparation artifacts are immutable:

```bash
REPO=/home/pgcotovio/Exact-OM
EVIDENCE="$REPO/data/experiments-v2/throughput-20261009-01"
CODE="$EVIDENCE/successor-source-03"
PYTHON="$REPO/.venv/bin/python"
cd "$CODE"
"$PYTHON" -m tools.prepare_mixed_scale_successor \
  --registry "$REPO/data/experiments-v2/hourly-supervisor-01/registry.json" \
  --public-inputs "$REPO/data/experiments-v2/g4-comparison-20261008-03/public-input-verification.json" \
  --source-revision 09e8722db5f88e9d9b0ea5ac6e1895f34d54605b \
  --output "$EVIDENCE/successor-04"
"$PYTHON" -m tools.prepare_corrected_queue \
  --bundle "$EVIDENCE/successor-04/successor-bundle.json" \
  --output "$EVIDENCE/queue-04"
```

The next agent should proceed in this order:

1. Let live G4 finish; reconcile failed cells, scientific reuse and corrected
   matched deployment costs. Freeze selected configuration and actual fitted
   artifacts first. The current G4 receipt is not eight-cell completion.
2. Only then bind the public H2 validation cohort, report-only, and prepare
   seed-specific bounded/final fitted manifests. The bounded preparer is
   `python -m tools.prepare_corrected_cohort_inference --config SELECTED_CONFIG
   --output NEW_OUTPUT --cohort BOUND_COHORT --selection-freeze ACTUAL_FREEZE
   --source PUBLIC_SOURCE --target PUBLIC_TARGET --mode MODE
   --fitted-artifacts ACTUAL_FITTED_RECEIPT` (omit the last argument only for a
   recipe proven to require no fitted artifact). Preserve all original queries
   and full target universes. Final workers do not refit or read raw references.
3. After G4 releases the 4090, verify the final source and every actual fitted
   recipe on that GPU. The current measurement harness is a binary fixture and
   rejects grouped/exemplar paths; extend the harness to replay the actual
   frozen recipes before claiming their parity or throughput. Test fitted
   selectors, full mappings and stress strata, and record hosted/end-to-end
   costs and extraction/output checks. T04 needs an exact-parity implementation
   correction, not merely another GPU run. Keep it off in scientific workers.
   A throughput miss alone does not prohibit otherwise feasible authorized work.
4. Resolve the counted remaining-work and all-root storage/spending forecast.
   Re-run the compiler with `--selection-freeze ACTUAL_FREEZE --deployments
   CELL_TO_MANIFEST_BINDINGS_JSON`, using fresh output paths. Review exact source,
   environment, native/model dependencies and supervisor/prompt policy hashes.
   The historical final runner enforces three seeds and must not receive this
   corrected matrix unchanged. A repair worker cannot approve its own policy.
5. **After explicit rollout authorization**, supply `--admissions
   CELL_TO_ADMISSION_BINDINGS_JSON` to `tools.prepare_corrected_queue`. Each
   admission must have kind `corrected_worker_rollout_admission`, authorized
   rollout, the cell's descriptor binding, exact commit, E17 family, reviewed
   supervisor/policy/environment/spending bindings, code/worker roots, Python,
   Slurm resources/GRES and measured seconds/requests/tokens/USD forecast.
   Preserve the base-campaign binding, cumulative ledger and unknown requests.
   This creates guarded launch descriptors but still leaves every row disabled.
6. Under **`registry.json.lock`**, re-read and compare both registry and dispatch
   snapshots, abort on drift, and enable only reviewed rows while replacing only
   the still-unstarted `E17-run-once-followup` and
   `E17-published-run-once-followup` entries. Use their fresh nonces, retain all
   running/completed rows and historical accounting. Do not replay an uncertain
   launch. The existing dispatcher polls every 15 seconds and launches numeric
   Slurm steps inside allocation 14372; **no supervisor restart is needed**.
   If that allocation changes, review the current retained-allocation ownership
   helper before admission. Preserve the interactive shell and extern step.
7. Verify actual complete mapping/query coverage and submission checksums; leave
   organizer results pending until returned. Respect unchanged intervention
   notifications, pause limits and rationale prohibition throughout.

No action in this handoff authorizes steps 5–6. Exact future launch bindings
cannot be fabricated before selection, fits, resource admission and authorization.
