# E00 — Experiment Harness & Frozen Baselines

**Blocks all other experiments. No result-changing intent** — this is the instrument that makes
the others measurable. Size: S.

## Deliverables

1. **`tools/run_experiment.py`**: takes an experiment YAML (`exp/experiments/EXX/exp.yaml`)
   declaring arms (config overlays over a shared base), tasks, seeds; executes the full matrix
   (locally or emitting sbatch, reusing `tools/run_exact_job.py`); collects per-run
   `evaluation_results.json`, ledger timing, LLM usage counters into one
   `results.parquet` + `results.md` summary table per experiment.
2. **Paired bootstrap** module (`exact/analysis/significance.py`): resamples per-source
   decision outcomes (global task) / per-anchor ranks (local task) 10k×; reports Δ, 95% CI,
   p-value for any two arms. Unit-tested against known synthetic cases.
3. **Frozen baseline**: run the post-overhaul system (defaults, 3 seeds) on the pinned
   reporting matrix (Bio-ML tasks, Anatomy, Conference, DISO ranking) and commit
   `exp/experiments/baseline/results.parquet` + summary. Every experiment compares to this
   artifact. Verify it matches the WP-B parity numbers (sanity: the overhaul didn't drift).
4. **Determinism guard**: the deterministic tie-break from audit F6 must be merged first
   (WP-D); the harness runs a same-seed repeat on one task and asserts metric deltas < 1e-4 on
   CPU (GPU jitter documented, not asserted).
5. **Leakage guards active**: audit F1/F2 fixes asserted at harness startup — the harness
   refuses to run an arm whose selector calibration or LLM calibration would see
   full-reference labels, and stamps reference-file hashes into every results row.
6. **Cost columns**: wall-time (ledger `cumulative compute`), LLM calls, LLM tokens
   (from the existing model-usage counters), per arm×task×seed.

## Acceptance

- Rerunning the baseline matrix from scratch reproduces the committed parquet within seed
  noise (documented tolerance).
- `results.md` for the baseline renders per-task and macro tables with CIs.
- A dry-run mode prints the full run matrix (arms × tasks × seeds) without executing.
