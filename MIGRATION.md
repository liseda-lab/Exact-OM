# Migrating to Exact-OM 2.1

Exact-OM 2.1 completes the shared Java-free OWL stack migration. The 2.0 configuration,
dataset, artifact, and viewer migrations remain documented below; existing v1 configuration
and run directories stay readable unless this guide explicitly says otherwise.

## From 2.0 to 2.1: shared OWL snapshots

OWL inputs now load once into an immutable `pyowl-core` snapshot. Exact's source facade,
structural indexes, projector, and selected hierarchy reasoner all consume that exact object;
there is no Exact-owned parsed ontology or second projector graph. RDFLib remains confined to
generic RDF/OAEI input handling and is not an OWL fallback.

The base package requires the compatible `pyowl-core` and `pyowl2vec-star-projector` 0.1
release lines. Asserted hierarchy queries need no extra. Install the optional Java-free
reasoners only when selected:

```console
pip install "exact-om[reasoning]"
```

`dataset.reasoner` accepts `asserted`, `elk`, or `hermit`; `dataset.projector.backend` accepts
`auto`, `python`, or `native`. Native accelerators remain optional upstream wheels. The Exact
wheel itself never invokes Java, Cargo, or a native build.

Pre-2.1 dataset/projection cache metadata is deliberately incompatible. Exact logs an
actionable warning and rebuilds from the original source bytes; it never unpickles a legacy
ontology object. Start a fresh cache if disk policy requires explicit cleanup.

New OWL runs add `ontology_stack.source` and `ontology_stack.target` to `run_manifest.json`
and `stats/run_stats.json`. The records contain core/projector/reasoner versions and backend
selection, structural/logical/signature fingerprints, import and resolver digests,
source-document hashes, diagnostics, options/schema identities, and verified-wire state. They
contain no local source path, temporary path, Python object ID, or credential.

The compatibility names `ParsedOntology`, `ParsedEntity`, and `parse(...)` now only alias or
delegate to the shared core contracts and are scheduled for removal after 2.1. `init_jvm`
remains an error-only migration shim; no supported workflow initializes a JVM.

The `bioml-eval` extra now targets OAEI-Bio-ML-eval 0.2 on Python 3.10 and newer. When an
inline run requests the `bioml` backend, official coherence receives the already-loaded Exact
snapshot providers; it no longer falls back to the old structural-proxy seam or reparses their
origin paths. Standalone `exact-eval` path inputs are still supported and are loaded once by
the shared core.

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
goes through `KnowledgeSource`; in 2.1, OWL inputs use the shared `pyowl-core` snapshot.

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
| `exact-om[bioml-eval]` | OAEI Bio-ML 0.2 metrics and Java-free official coherence. |
| `exact-om[reasoning]` | Optional Java-free pyELK and pyHermiT hierarchy reasoners. |
| `exact-om[docs]` | Documentation build toolchain. |

Without an optional extra, the corresponding integration fails with an installation hint;
the core matcher does not import optional service dependencies at startup.
