# Tuning and Slurm

Use YAML job wrappers for repeatable local and cluster commands. Always inspect the resolved
command before submitting it:

```console
poetry run python tools/run_exact_job.py --run-config run.yaml --dry-run
poetry run python tools/run_user_study_job.py --run-config study.yaml --dry-run
```

`tools/hparam_tuner.py` creates one configuration, trial metadata file, and output directory per
trial. It supports explicit grids and a lightweight low-discrepancy sampler with local
exploitation.

```console
poetry run python tools/hparam_tuner.py \
  --tuner-config tuner.yaml \
  --strategy grid \
  --dry-run
```

Tuner paths use config-v2 names such as `matching.threshold` and `candidates.top_k`. Legacy
trial templates are migrated through the same declarative map as `exact config migrate`.

Cluster wrappers live under `deploy/sbatch/`. Keep run directories unique per trial, request a
GPU only for jobs configured to use one, and materialize/verify datasets before scheduling
compute. A resumed job keeps its original configuration fingerprint; changing a tuned field
starts fresh rather than accepting an incompatible checkpoint.

After completion, `tools/aggregate_results.py` reads evaluation artifacts through `RunReader`
and joins them with each `trial.json`:

```console
poetry run python tools/aggregate_results.py --exp-dir EXPERIMENT
```
