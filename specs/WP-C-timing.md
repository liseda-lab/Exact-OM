# WP-C — Timing Ledger (accurate time under caches & resumes)

**Depends on**: WP-A. **Blocks**: WP-D (trainer refactor rebases on the hooks added here).
**Size**: M (1 agent, single PR).
**Behavior**: output *files* change (new `timings.json`; `times.txt` becomes derived); alignment
results unchanged.
**Status**: Done (2026-07-16).

## Context — how timing lies today

`times.txt` is a flat `"<Step>: <value> minutes"` file (`exact/utils/timing.py`, 43 lines;
step names in `exact/core/values.py:TIMING_STEP_ORDER`). `update_recorded_timings` merges new
values over old ones, skipping only `None`. `AlignmentAction.run` measures stages with
`time.time()` deltas; the trainer reports `last_stage_timings` from `perf_counter` spans.
Audited failure modes:

1. **Fully-resumed run zeroes real cost.** When all examples are restored from checkpoint,
   `predict` returns `{"Alignment.Inference": 0.0, ...}`; `update_recorded_timings` overwrites
   the genuine inference minutes with `0.0`. The measurement of an expensive run is destroyed
   by the cheap re-run that follows it.
2. **Partial resume undercounts.** The inference span counts only the remaining batches; nothing
   accumulates the restored portion.
3. **`Total` is per-invocation**, not cumulative across the sessions that actually produced the
   artifacts.
4. **Cache-hit double bookkeeping.** After a cache hit, `Dataset.CacheLoad` is written while the
   stale `Dataset` line survives the merge; `summarize_progress_estimates` (`exact/utils/logs.py`)
   then prefers `Dataset.CacheLoad`, flipping next-run ETAs to the cache-load cost.
5. **Feedback loop**: corrupted values seed `RunProgressLogger` estimates (inference ETA ≈ 0).
6. Different configs run in the same dir blend into one merged file — estimates and reported
   times mix experiments.

## Design

Implement `TimingLedger` exactly as specified in `02-shared-contracts.md` §7 (JSON schema, API,
`CacheStatus`, append-only sessions, atomic writes + advisory lock, `stage_totals`, `estimates`).
Key semantics, restated as invariants:

- **I1**: a session never modifies another session's records.
- **I2**: `compute_seconds(stage) = Σ seconds over records with status FRESH|RESUMED` — the honest
  "what did it cost to produce the current artifacts" number; CACHE_HIT/SKIPPED seconds are
  tracked as `overhead_seconds`, never conflated.
- **I3**: a stage that did no work this session records `SKIPPED` (seconds≈0) — it can never
  erase history (fixes failure 1).
- **I4**: sessions carry `config_fingerprint`; `estimates()` aggregates only matching-fingerprint
  sessions when one is given, falling back to all (fixes 6, and 4 via per-status separation:
  estimates for `Dataset` use FRESH records only).
- **I5**: monotonic clock (`perf_counter`) for spans; wall timestamps only for session metadata.
- **I6**: crash-tolerant — the session file is flushed after every `record()`; a session with
  `ended_at: null` is a crashed run, still countable.

`config_fingerprint`: sha1 over `ConfigModel` canonical dump (sorted-keys JSON of
`model_dump()`, excluding volatile fields: paths outside the run dir, device, logging).
Implement as `ConfigModel.fingerprint()`. Coordination: WP-J later migrates configs to schema
v2 and requires that a v1 file and its v2 migration produce the **same** fingerprint (it
fingerprints the resolved model, not the raw YAML) — keep the implementation on the resolved
model so that holds for free.

## Tasks

1. **Rewrite `exact/utils/timing.py`**: `TimingLedger`, `RunSession`, `StageSpan`, `CacheStatus`,
   `StageTotal`. Keep module-level `load_recorded_timings`/`write_recorded_timings` as thin
   wrappers over the ledger for the derived `times.txt` (deprecation note in docstring).
   `times.txt` is rewritten at session end from `stage_totals(config_fingerprint=current)`,
   with a `# derived from timings.json — do not edit` header line (parser tolerates it).
2. **Re-instrument `AlignmentAction.run`** (`exact/core/actions/alignment.py`): replace the ~dozen
   manual `time.time()` deltas, the four near-identical `stage_timings["Postprocess"] = sum(...)`
   blocks, and the final `timmings` re-projection loop with `ledger.session(...)` +
   `session.stage(...)` context managers. Stage names stay exactly `TIMING_STEP_ORDER`
   (`values.py:8-26`). Dataset cache hit records `Dataset.CacheLoad` as `CACHE_HIT` and
   `Dataset.Process` etc. as `SKIPPED`.
3. **Trainer hooks** (narrow edits to `impl/trainer/semantic_runner.py`; WP-D will relocate them
   intact):
   - Persist `timing.inference_seconds_cumulative` (+ `examples_per_second_ema`) in the inference
     checkpoint manifest (contracts §8); seed the accumulator from it on restore. The session
     records this session's span with `cache_status=RESUMED` and `work_done/work_total` from
     `processed_examples`; cumulative truth is reconstructible from either view (fixes 2).
   - The fully-restored fast path reports `SKIPPED`, not `0.0`-fresh (fixes 1).
   - Replace the `_last_stage_timings` dict handoff with a typed
     `list[StageRecord(stage, seconds, cache_status, work_done, work_total)]` the action feeds
     into the session verbatim.
4. **Progress estimates**: `summarize_progress_estimates` (`exact/utils/logs.py`) reads
   `ledger.estimates(config_fingerprint=...)`; prefer per-unit rates (`seconds/work_done` ×
   `work_total`) when work counts exist, so partial sessions still yield good ETAs (fixes 5).
5. **Eval timing**: standalone `EvaluationAction.run` opens its own session (`command="eval"`);
   inline post-alignment eval records `Postprocess.Evaluation` in the align session. (WP-E keeps
   this wiring when it restructures evaluators — coordinate via the contracts file.)
6. **Effective-threshold notice (audit F4)**: while re-instrumenting `AlignmentAction.run`,
   add the one-line log stating the effective decision threshold and its origin (configured
   value vs. selector-median override from `semantic_runner.py:2845-2857`) — pure logging.
7. **Report at end of run** (log + `run_stats.json` addition): `this_session_seconds` per stage,
   `cumulative_compute_seconds` per stage, and a one-line
   `total compute across N sessions: Xm (this session: Ym)` summary — this is the number to
   report in OAEI submissions.

## Tests (`tests/timing_ledger_test.py` + updates)

Reproduce each audited failure as a regression test:
1. Session A fresh (600 s inference) → Session B fully resumed → `stage_totals` inference
   `compute_seconds == 600`, `times.txt` still shows it; B's record is `SKIPPED`.
2. A processes 40% (400 s) then "crashes" (session not ended) → B resumes the rest (500 s) →
   cumulative inference 900 s; checkpoint round-trips `inference_seconds_cumulative`.
3. Cache-hit dataset: totals keep `Dataset` FRESH cost; estimates for `Dataset` use FRESH only;
   `Dataset.CacheLoad` tracked separately.
4. Two configs in one dir → totals/estimates separate by fingerprint.
5. Concurrent sessions (two processes, `multiprocessing`) append without loss (lock test).
6. Corrupt/absent `timings.json` → ledger recovers (backs up corrupt file, starts fresh, warns).
7. Legacy `times.txt`-only dir: first ledger session imports it as a synthetic
   `legacy` session (best effort, minutes→seconds) so old runs keep their history.

## Out of scope

Relocating trainer code (WP-D), evaluator restructuring (WP-E), removing `times.txt` (2.1).

## Acceptance criteria

1. All regression tests above green; existing checkpoint tests
   (`tests/semantic_runner_checkpoint_test.py`) still green.
2. Manual: run align twice in one dir (2nd fully cached) on fixtures — `timings.json` has 2
   sessions; cumulative inference equals run 1; `times.txt` values don't regress to 0.
3. `AlignmentAction.run` contains no raw `time.time()` stage bookkeeping (grep gate).
4. `run_stats.json` carries the session/cumulative split; log prints the OAEI-reportable line.

## Deviations

None.
