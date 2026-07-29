# E00 — Experiment Harness & Baseline Lineage

**Blocks all other experiments. No result-changing intent** — this is the instrument that makes
the others measurable. Size: M.

## Research questions

- **RQ00.1**: Can repeated runs reproduce decisions and metrics closely enough that the effect
  sizes targeted by E01–E23 are distinguishable from execution noise?
- **RQ00.2**: Does the harness prevent reference leakage and preserve enough dataset/model/config
  provenance to audit every supervision claim?
- **RQ00.3**: Can one results schema compare quality, coverage, significance, runtime, memory,
  and LLM cost across entity kinds, relation types, formats, and supervision regimes?
- **RQ00.4**: Does the frozen post-overhaul class baseline reproduce WP-B parity, and are the
  shipped property, instance, pure-KG, relation-typing, and no-label capability baselines
  reproducible on every eligible reporting slice?
- **RQ00.5**: What effect size is detectable for every task×kind×relation slice under the
  planned bootstrap, seed count, and multiplicity policy?
- **RQ00.6**: Can rolling baseline and hosted-model provenance detect a candidate-pool, default,
  fitted-artifact, or provider-model change before it confounds an experiment?

These are measurement-system questions, not hypotheses about a better matcher. Acceptance
below answers each with a pass/fail result; any failure blocks downstream claims.

## Deliverables

1. **`tools/run_experiment.py`**: takes an experiment YAML (`exp/experiments/EXX/exp.yaml`)
   declaring arms (config overlays over a shared base), tasks, seeds; executes the full matrix
   (locally or emitting sbatch, reusing `tools/run_exact_job.py`); collects per-run
   `evaluation_results.json`, ledger timing, LLM usage counters into one
   `results.parquet` + `results.md` summary table per experiment.
2. **Paired bootstrap** module (`exact/analysis/significance.py`): resamples per-source
   decision outcomes (global task) / per-anchor ranks (local task) 10k×; reports Δ, 95% CI,
   p-value for any two arms. Unit-tested against known synthetic cases.
3. **Immutable baseline lineage**: run the post-overhaul system (3 seeds) and commit
   `exp/experiments/baselines/B0/{results.parquet,results.md,baseline_manifest.json}`; it
   contains (a) the default class-equivalence baseline on Bio-ML, Anatomy, Conference, and
   DISO ranking; and (b) current
   shipped capability baselines for every eligible property/instance/OAEI-KG slice, each input
   representation, `hierarchy_heuristic` relation typing, and the no-training fallback. The
   class subset must match WP-B parity (sanity: the overhaul did not drift); additive
   capability subsets establish their own pre-experiment reference rather than borrowing class
   numbers. A not-yet-published task is recorded as unavailable, never replaced by a mini
   fixture performance number. Each baseline manifest stores baseline ID/parent, commit and
   resolved-config hashes, dataset locks, candidate-pool hashes, enabled flags, fitted-artifact
   hashes, and results hashes. The harness can append (never overwrite) a rolling `R_n` snapshot
   after an E17/single-promotion gate under `baselines/R_n/` and can reconstruct its complete
   ancestry to `B0`.
4. **Determinism guard**: the deterministic tie-break from audit F6 must be merged first
   (WP-D); the harness runs a same-seed repeat on one task and asserts metric deltas < 1e-4 on
   CPU (GPU jitter documented, not asserted).
5. **Leakage guards active**: audit F1/F2 fixes asserted at harness startup — the harness
   refuses to run an arm whose selector calibration or LLM calibration would see
   full-reference labels, and stamps reference-file hashes into every results row.
6. **Cost columns**: wall-time (ledger `cumulative compute`), LLM calls, LLM tokens
   (from the existing model-usage counters), per arm×task×seed.
7. **Capability inventory**: materialize the `dataset_inventory.parquet` required by the plan,
   plus per-run columns for entity kind, relation vocabulary, input representation, and
   supervision label. Add peak-memory and load/index-time fields for E12/E13 and the descriptor's
   reference-completeness declaration. Add E23's **structural profile** columns — hierarchy depth
   distribution, ancestor coverage, class-to-instance ratio, axiom density, triples per entity,
   distinct predicates, relational entropy — so the TBox-richness axis is measurable from the
   inventory before any experiment reads a result. Record the **resolved supervision mode per
   component** (README §"Supervision as a configured mode"), auto-policy artifact hash, observed
   policy features, effective-unit definition/count, and resolution reason in the run manifest and
   results schema. Fail the run when a row's supervision label disagrees with its resolved config
   — the E18–E23 arms are distinguished by that resolution, so it cannot be reconstructed later.
8. **Power and adjudication planning**: write `power_analysis.parquet` per
   task×kind×relation×metric with independent-unit count, match prevalence, smallest attainable
   metric step, baseline seed/cluster-bootstrap variance, and MDE at 80%/90% power. Estimate MDE
   by injecting controlled paired decision improvements into baseline per-source outcomes and
   rerunning the exact planned bootstrap/multiplicity test; use connected-component clusters for
   the E12 sensitivity calculation. Also emit the sample size required for the shared
   adjusted-precision adjudication protocol at declared CI widths. The calculation uses no
   experiment-arm result.
9. **LLM identity/drift stamps**: for every request record provider, requested and response
   model IDs, immutable revision/deployment/API version when available, endpoint identity
   (without secrets), router route, tokenizer, prompt/template hash, decoding/logprob parameters,
   seed, cache key, and request time. Paired arms must resolve to the same model fingerprint;
   an alias/version change mid-matrix aborts rather than mixing responses. A provider that
   exposes only a mutable alias and no controlled deployment/revision is exploratory-only for
   confirmatory LLM comparisons; use an immutable hosted deployment or self-hosted snapshot for
   a promotion claim.

## Acceptance

- Rerunning `B0` from scratch reproduces the committed parquet within seed
  noise (documented tolerance).
- Appending a synthetic `R_1` leaves `B0` byte-identical, records `B0` as parent, and detects a
  changed config, candidate pool, or fitted artifact in a lineage comparison.
- `results.md` for the baseline renders per-task and macro tables with CIs.
- `power_analysis.parquet` covers every eligible inventory slice and the experiment dry-run
  refuses a confirmatory cell with no pre-registered power status.
- The run manifest stamps the spec's pre-run power-declaration hash and rejects a changed
  declaration once a reporting result exists.
- A dry-run mode prints the full run matrix (arms × tasks × seeds) without executing.
- A deliberately mislabelled target-supervised arm is rejected, and result aggregation refuses
  to silently pool entity kinds or relation types when their per-slice rows are absent.
- A run configured `supervision.mode: label_free` with a training reference on disk resolves
  every component label-free and records that resolution; a run configured `supervised` with no
  resolvable training reference fails loudly rather than falling back silently.
- A fake `profile_rule` fixture resolves differently across an effective-unit/structural-profile
  boundary, records the policy hash and reason, and never reads a test reference. With no training
  reference it produces the same resolved label-free config and decisions as the explicit
  `label_free` arm.
- A fitted head or policy (selector, reranker, fusion weights, encoder, LLM gate, structural
  head, or auto-resolution policy) whose recorded feature/policy schema, candidate-pool
  fingerprint, graph hash, or resolved LLM identity does not match the current run is refused
  rather than loaded.
- A fake hosted-model response whose resolved identity changes between paired arms triggers the
  LLM drift guard.
