# Inspecting alignments

Install the viewer extra and open a run from any working directory:

```console
pip install "exact-om[viz]"
exact-inspect open RUN_DIR
```

Open mode reads mappings, explanations, statistics, and provenance through `RunReader`. Layout
v2 performs source-local indexed reads; historical run directories fall back to their legacy
mapping and explanation artifacts. Ontology panels use configured ontology paths when
available and continue to accept a precomputed `ontology_cache.json`.

## Service mode

Serve a fixed, read-only bundle for an iframe deployment:

```console
EXACT_INSPECT_RUN_DIR=/srv/run exact-inspect serve --host 0.0.0.0 --port 8000
```

The query `/?source=<IRI>` is preserved for LimeSurvey embedding. New environment variables use
the `EXACT_INSPECT_` prefix. Legacy viewer names remain temporary compatibility shims and emit
a deprecation warning.

## Build a study bundle

```console
exact-inspect bundle RUN_DIR BUNDLE_DIR
```

If `RUN_DIR/analysis/user_study` contains a curated selection, bundle mode uses it. Otherwise it
derives a view from the run's mappings and explanations. `BUNDLE_DIR` receives the selected
mapping table, explanation records, optional ontology cache, copied resolved configuration,
and `study_bundle.json`; the run's indexed explanation store remains the source of truth.

For repeatable or scheduled exports, bundle mode also accepts the historical YAML job shape:

```console
exact-inspect bundle --job-config bundle-job.yaml --dry-run
exact-inspect bundle \
  --job-config bundle-job.yaml \
  --sbatch-script deploy/sbatch/exact_inspect_bundle.sh
```

Release wheels with the `viz` extra contain the static frontend. A source checkout uses the
development export under `explanations_visualizer/out`; run `make build-frontend` to rebuild
and copy it. If neither static directory exists, the API starts in an explicit API-only mode
instead of crashing.
