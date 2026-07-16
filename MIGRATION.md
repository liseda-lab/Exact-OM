# Migrating from Exact-OM 1.x to 2.0

Exact-OM 2.0 modernizes configuration, ontology loading, dataset acquisition, run artifacts,
and the alignment viewer. Existing v1 configuration and run directories remain readable for
the 2.0 compatibility window unless this guide explicitly says otherwise.

## Before upgrading

1. Keep a copy of any v1 YAML files and unfinished run directories.
2. Upgrade Exact-OM and install the extras used by your workflow.
3. Migrate configuration files with `exact config migrate`.
4. Start a new output directory for the first 2.0 run; use `exact run info` to inspect old
   and new runs.

## Runtime and ontology backend

The Java/mowl runtime was removed. Do not initialize a JVM or pass heap-size settings. The
`exact.init_jvm` symbol remains temporarily as a stub that raises a migration-focused error,
and legacy heap flags are accepted but ignored with a deprecation warning. Ontology access now
goes through `KnowledgeSource`; OWL inputs use the native `py-horned-owl` backend.

Reasoner settings now select an Exact reasoner plugin. The removed
`reasoner_timeout_secs` and `reasoner_force_hermit` keys are reported and dropped by the
config migrator.

## Configuration schema v2

Every new file starts with:

```yaml
config_version: 2
```

Files without `config_version` are treated as v1, migrated in memory, and emit one
deprecation warning. To create a reviewed v2 file:

```console
exact config migrate old.yaml -o config.yaml
exact config default -o default.yaml
```

Unknown keys are now errors and include a near-match suggestion. The resolved v2 model—not
the spelling of the input YAML—is fingerprinted, so a v1 file and its migrated equivalent
share cache and timing fingerprints.

The declarative migration table in `exact.core.entities.configs.migration.V1_TO_V2` is the
authoritative field-by-field map. The main section moves are:

| v1 location | v2 location |
| --- | --- |
| `seed`, `logging_level`, `use_file_cache` | `run.*` |
| `dataset_track.*` | `data.*` |
| ontology and evidence fields in `dataset_params` | `dataset.*` |
| legacy context fields in `dataset_params` | `dataset.legacy.*` |
| verbaliser fields in `dataset_params` | `llm.verbaliser.*` |
| `candidates_params.*` | `candidates.*` |
| `alignment_params` decision fields | `matching.*` |
| `model`, `second_model`, `model_chain`, `second_pass_params` | ordered `pipeline` entries |
| `inference_params.*` | `inference.*` |
| `llm_profiles`, `llm_routing` | `llm.profiles`, `llm.routing` |
| top-level `k` | `evaluation.k` |
| alignment save flags, `plot_params`, `sanity_check_params` | `output.*` |

Tuning and job-runner YAML should address v2 paths. Their compatibility loaders use the same
migration table as the main command.

## Commands and Python imports

| 1.x | 2.0 | Compatibility |
| --- | --- | --- |
| `bioml-eval` | `exact-eval` | Both console scripts remain installed. |
| `EvalutionRunner` | `EvaluationRunner` | Misspelled alias warns and remains through 2.0. |
| `exact.utils.paths` graph helpers | `exact.utils.graph_search` | Old module warns and re-exports through 2.0. |
| `exact.utils.llm_routing` | `exact.llm.routing` | Old module warns and re-exports through 2.0. |
| `data/get_data.py` | `exact data pull`, `verify`, `status` | Old script is removed. |
| `exact-study-viz` | `exact-inspect` | Old command warns and delegates through 2.0. |
| `study_visualizer_runtime` | `exact_inspect` | Old package is a deprecation shim through 2.0. |
| `exact.analysis.study_visualizer` | `exact_inspect.bundles` and `exact_inspect.app` | Old imports warn and re-export through 2.0. |
| `tools/prepare_study_visualizer_bundle.py` and its job wrapper | `exact-inspect bundle` and `--job-config` | Old scripts warn and delegate through 2.0. |
| `EXACT_STUDY_*` | `EXACT_INSPECT_*` | Legacy environment variables warn; new names win. |

The primary programmatic functions are `run_alignment(...)` and `run_evaluation(...)`.
`AlignmentAction.run` and `EvaluationAction.run` remain deprecated aliases during the same
window.

## Dataset acquisition

Dataset tracks are revision-pinned and verified by provider:

```console
exact data pull bioml_hf/ncit-doid
exact data verify bioml_hf/ncit-doid
exact data status bioml_hf/ncit-doid
```

Materializations write `datasets.lock.json`. A status is one of `ok`, `local-drift`,
`upstream-moved`, or `not-materialized`; do not silently reuse a drifted directory.

## Run layout and explanations

New runs use layout v2:

```text
run_manifest.json
alignment/  evaluation/  explanations/  stats/  plots/  checkpoints/
timings.json  exact.log  config.yaml
```

The old `model/alignment/...` layout remains readable through `RunReader` and
`exact-inspect`. New explanation records are stored once in compressed, source-indexed shards.
The monolithic `full_explanations.json` is now a derived compatibility export:

```console
exact run export RUN_DIR --what explanations --format json
exact run export RUN_DIR --what explanations --src SOURCE_IRI --format json
```

Set `output.save.full_explanations_json: true` only for a consumer that still requires the old
file. It is deprecated and scheduled for removal in 2.1.

`times.txt` is replaced by the append-safe `timings.json` session ledger. During the
compatibility window, Exact can still read historical files and render a derived `times.txt`;
new timing state is recorded in `timings.json`.

Checkpoint schema v2 and full-payload checkpoints remain readable. New checkpoints point at
the explanation store, discard an uncheckpointed shard suffix safely on resume, and are pruned
at successful finalization according to `output.retention.checkpoints`. Layout-v1 run folders
are never rewritten merely by opening them.

Useful maintenance commands are:

```console
exact run info RUN_DIR
exact run clean RUN_DIR --dry-run
exact run clean RUN_DIR
```

Cleanup removes only manifest-owned or recognized resume files and preserves foreign files.

## Optional extras

| Install | Enables |
| --- | --- |
| `exact-om[viz]` | `exact-inspect` service and CLI (FastAPI/Uvicorn). |
| `exact-om[hf]` | Hugging Face dataset-track providers. |
| `exact-om[bioml-eval]` | Upstream OAEI Bio-ML evaluation backend. |
| `exact-om[docs]` | Documentation build toolchain. |

Without an optional extra, the corresponding integration fails with an installation hint;
the core matcher does not import optional service dependencies at startup.
