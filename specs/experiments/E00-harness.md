# E00 — Lean Paper Experiment Runner & Frozen Baseline

**Blocks all other experiments. No result-changing intent.** This is a small, paper-grade
execution and measurement layer, not a general workflow platform. Size: S–M.

## Research questions

- **RQ00.1**: Can the same runner screen many arms cheaply and then confirm a frozen survivor
  without changing code paths, metrics, or data semantics?
- **RQ00.2**: Does every result carry enough split, supervision, configuration, model, dataset,
  candidate-pool, and source-code provenance to audit leakage and reproduce the comparison?
- **RQ00.3**: Can paired arms produce per-source evidence and paper-ready quality, coverage, and
  cost summaries with deterministic aggregation?
- **RQ00.4**: Does the frozen post-migration baseline reproduce current production behavior, and
  do disabled experiment flags leave its decisions unchanged?
- **RQ00.5**: Can independent runs execute concurrently, resume safely, and reuse only artifacts
  whose fingerprints match?

These are measurement-system questions. E00 does not test a better matcher and does not need a
complete benchmark or product-release service before screening can begin.

## Single-suite execution model

`tools/run_experiment.py` implements one suite with `screen` and `confirm` stages:

- `screen` runs the full development arm sweep, normally with one seed and an optional
  deterministic source cap. It may use only development labels or a spec-named frozen diagnostic
  subset. The experiment config declares the selection rule before execution.
- `confirm` accepts a frozen selection record from `screen` and runs only the selected candidate,
  the current baseline, and required controls on untouched reporting data with at least three
  paired seeds. It refuses to run if the selection record, design declaration, base configuration,
  or reporting matrix has changed.

There is no separate pilot runner. Both stages resolve the same base config plus arm overlay, use
the same metrics, and write the same schema. If no candidate passes the development selection
rule, the experiment ends as `screened_out` without opening reporting data.

## Required deliverables

1. **Lean runner**: read one experiment YAML under `exp/experiments/EXX/`, resolve its base config,
   arms, tasks, stage, seeds, and optional source cap, then execute locally by reusing
   `tools/run_exact_job.py`. Support `--dry-run`, `--resume`, and bounded `--jobs`. CPU-independent
   runs may run concurrently; GPU and LLM runs are serialized per device/profile unless an
   explicit safe concurrency limit is configured.
2. **Isolated artifacts**: one directory per experiment×stage×arm×task×seed. A completed run is
   reused only when its manifest fingerprint matches; partial or failed runs retain a reason and
   can resume without being mistaken for results.
3. **Run manifest**: record stage, experiment/arm, seed, commit, Exact-OM version, pyowlcore
   version, resolved-config hash, dataset and reference hashes, split role, supervision label,
   candidate-pool fingerprint, fitted-artifact hashes, model revisions, selection-record hash,
   design-declaration hash, start/end time, and status. Do not record secrets.
4. **Frozen paper baseline**: after the Exact-OM 2.1.0 / pyowlcore 0.2.0 migration and production
   tests pass, freeze the current default as the paper's `R_n` baseline manifest. Preserve
   historical `B0` as an optional longitudinal comparator; E00 does not need a complete rolling
   lineage service. A retrieval change creates a new candidate-pool fingerprint before any
   downstream confirmation.
5. **Results**: retain per-source decisions/ranks and aggregate per task×entity kind×relation.
   Emit machine-readable CSV or JSON for P/R/F1, MRR/Hits@1, candidate recall, coverage,
   abstention, wall time, peak memory when relevant, and LLM calls/tokens when used. Also report
   macro summaries; never silently pool entity kinds or relation types.
6. **Statistics**: provide a small paired-bootstrap utility over per-source decisions or ranks,
   with 10,000 resamples by default, delta and 95% CI. Cluster instance sensitivity analyses by
   connected component only where the experiment requires it. Unit-test the utility on synthetic
   identical, positive-effect, and deterministic cases.
7. **Dataset inventory**: emit CSV or JSON only for tasks declared in the paper matrix, including
   split availability, entity/reference counts, candidate coverage, representation, relation and
   entity-kind support, reference completeness, and any experiment-specific capability field.
   Building an inventory for unused tracks is not an E00 prerequisite.
8. **Leakage and supervision guards**: reporting references cannot be used by screening,
   calibration, training, threshold selection, routing, early stopping, or fallback selection.
   `label_free` ignores available training labels; `supervised` fails when usable training data is
   absent. Every result's declared supervision must equal its resolved runtime mode.
9. **LLM provenance**: when an LLM arm is confirmed, record provider, requested and resolved model
   IDs/revisions, endpoint identity without credentials, tokenizer, prompt hash, decoding
   parameters, seed, cache key, and request time. A model-identity change within paired arms aborts
   the comparison. Mutable hosted aliases are exploratory-only.

## Frozen design declaration

Before `confirm`, write and hash a design record alongside the experiment config containing:

- selected arm and development selection rule/result;
- reporting tasks and exclusions;
- primary comparison and endpoint;
- required slices and controls;
- paired seeds;
- independent-unit definition;
- hypothesized effect or non-inferiority margin;
- `powered|underpowered|descriptive` status and its assumptions;
- multiplicity rule when more than one confirmatory comparison remains.

This record is immutable once any reporting result exists. A full automated power simulator is
optional; an explicit design declaration is mandatory. The runner never writes into `specs/`.

## Acceptance

- A dry run prints the exact stage×arm×task×seed matrix and resolved output paths.
- A one-seed development screen produces a selection record without reading reporting references.
- Confirmation rejects an unfrozen, missing, or mismatched selection/design record.
- Repeating a deterministic same-seed run reproduces decisions and metrics within a documented
  CPU tolerance; GPU nondeterminism is measured and recorded.
- With every experiment flag disabled, runner output matches the frozen production baseline.
- A changed config, dataset, candidate pool, fitted artifact, split, or model identity prevents
  unsafe resume/reuse.
- A deliberately mislabelled supervised or label-free run fails before inference.
- Two independent CPU runs can execute concurrently without sharing mutable output files; device
  limits prevent GPU/LLM oversubscription.
- Aggregation reports missing/failed cells rather than silently dropping them and emits the
  per-source data required for paired inference.

## Explicit non-goals

E00 does not require Parquet, automatic sbatch generation, a dashboard, a workflow database,
automatic prose/table insertion into specs, a complete B0/R_n ancestry manager, a full inventory
of every available benchmark, automated expert adjudication, or exhaustive power simulation.
Those may be added only when a selected experiment or the paper submission actually requires
them. Simplifying orchestration never relaxes split isolation, provenance, paired statistics, or
the frozen confirmatory design.
