# Migrating to Exact-OM 2.1

Exact-OM 2.1 completes the shared Java-free OWL stack migration on the released pyOWL 0.2
package family. The 2.0 configuration, dataset, artifact, and viewer migrations remain
documented below; existing v1 configuration and completed run directories stay readable unless
this guide explicitly says otherwise.

## From 2.0 to 2.1: pyOWL 0.2 shared views

The base package requires `pyowl-core>=0.2,<0.3` and
`pyowl2vec-star-projector>=0.2,<0.3`, including released
`pyowl-core==0.2.0`. Asserted hierarchy queries need no extra. Optional integrations use:

```text
pyelk-reasoner>=0.2,<0.3
pyhermit>=0.2,<0.3
oaei-bioml-eval>=0.2.1,<0.3
```

Install the Java-free reasoners only when selected:

```console
pip install "exact-om[reasoning]"
```

OWL inputs load once into an immutable public core view. Exact's source facade, structural
indexes, projector, and selected hierarchy reasoner all consume that exact owner; there is no
Exact-owned parsed ontology or second projector graph. RDFLib remains confined to generic
RDF/OAEI input handling and is not an OWL fallback.

Core 0.2 uses model schema 2, including corrected anonymous-individual and repeated
isomorphic-component scoping. Exact negotiates the current wire writer, encoded schema 2, and
`pyowl_core.EncodedStructuralView.DESCRIPTOR_SHA256` through public attributes. It does not
request or decode encoded buffers, import implementation modules, or retry a failed consumer
from an OWL path.

`dataset.reasoner` accepts `asserted`, `elk`, or `hermit`;
`dataset.projector.backend` accepts `auto`, `python`, or `native`. Native accelerators
remain optional published upstream wheels. The Exact wheel itself never invokes Java, Cargo,
or a native build.

### Required cache rebuild

Every schema-1 ontology-derived cache is incompatible. Exact rejects parsed-ontology,
projection, compiler, dataset, and resumable-workflow cache metadata before unsafe unpickling
or identity interpretation, then rebuilds it from the original source through core 0.2. A
matching path, mtime, or source digest does not make the cache reusable.

Never convert or reinterpret schema-1 caches. Completed immutable run results remain readable,
but an unfinished run cannot resume through schema-1 ontology cache state.

New OWL runs add `ontology_stack.source` and `ontology_stack.target` to
`run_manifest.json` and `stats/run_stats.json`. The records contain public
core/projector/reasoner distribution and schema values, structural/logical/signature
fingerprints, import and resolver digests, source-document hashes, options, cache state,
bounded handoff diagnostics, and verified-wire state. They contain no local source path,
temporary path, Python object ID, pointer, or credential.

The compatibility names `ParsedOntology`, `ParsedEntity`, and `parse(...)` only alias or
delegate to public core contracts and are scheduled for removal after 2.1. `init_jvm` remains
an error-only migration shim; no supported workflow initializes a JVM.

The `bioml-eval` extra targets OAEI-Bio-ML-eval 0.2.1 or newer in the 0.2 line on Python
3.10 and newer. Inline official coherence receives the already-loaded Exact snapshot
providers; it does not reparse origin paths. Standalone `exact-eval` path inputs remain
supported and are each loaded once by core.

## Before upgrading

1. Keep a copy of v1 YAML files, unfinished run directories, and their original ontology
   sources.
2. Upgrade Exact-OM and install the extras used by your workflow.
3. Let Exact rebuild ontology-derived caches; do not copy or convert schema-1 cache state.
4. Migrate configuration files with `exact config migrate`.
5. Start a new output directory for the first 2.1 run; use `exact run info` to inspect old
   and new runs.

## Runtime and ontology backend

The Java/mOWL runtime was removed. Do not initialize a JVM or pass heap-size settings. The
`exact.init_jvm` symbol remains temporarily as a stub that raises a migration-focused error,
and legacy heap flags are accepted but ignored with a deprecation warning. Ontology access now
goes through `KnowledgeSource`; in 2.1, OWL inputs use one retained public
`pyowl-core` 0.2 view.

Reasoner settings select an Exact reasoner plugin. The removed `reasoner_timeout_secs` and
`reasoner_force_hermit` keys are reported and dropped by the config migrator.

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

| 1.x | 2.0+ | Compatibility |
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

Checkpoint schema v2 and full-payload checkpoints remain readable when independent of an
incompatible ontology cache. New checkpoints point at the explanation store, discard an
uncheckpointed shard suffix safely on resume, and are pruned at successful finalization
according to `output.retention.checkpoints`. Layout-v1 run folders are never rewritten merely
by opening them.

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
| `exact-om[bioml-eval]` | OAEI Bio-ML >=0.2.1,<0.3 metrics and Java-free official coherence. |
| `exact-om[reasoning]` | Optional Java-free pyELK and pyHermiT >=0.2,<0.3 hierarchy reasoners. |
| `exact-om[docs]` | Documentation build toolchain. |

Without an optional extra, the corresponding integration fails with an installation hint;
the core matcher does not import optional service dependencies at startup.
