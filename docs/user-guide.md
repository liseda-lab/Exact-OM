# User guide

This page preserves the practical material from Exact-OM's former hand-written site while the
documentation is reorganized into focused guides. Generated references are available for the
[configuration](reference/configuration/index.md) and
[command line](reference/cli/index.md).

## Install

Install from the repository root with Poetry:

```console
poetry install
poetry run exact --help
```

Large biomedical runs benefit from a CUDA-capable GPU, although the command can run on CPU.
Node and npm are only needed when rebuilding the explanation visualizer frontend.

## Inputs

Every alignment run needs a source ontology, a target ontology, an output directory, and a YAML
configuration. Training references, full references, and candidate files are optional.

| Input | Required | Format | Used for |
| --- | --- | --- | --- |
| Source ontology | Yes | OWL | Source labels, annotations, hierarchy, and graph evidence. |
| Target ontology | Yes | OWL | Target labels, annotations, hierarchy, and graph evidence. |
| Training reference | No | TSV with `SrcEntity`, `TgtEntity`, `Score` | Selector calibration. |
| Full reference | No | TSV with `SrcEntity`, `TgtEntity`, `Score` | Evaluation and analysis. |
| Candidate file | No | TSV with `SrcEntity`, `TgtEntity`, `TgtCandidates` | Local ranking mode. |

## Run an alignment

### Global alignment

Omit `-c` to let Exact-OM build the candidate set and write a filtered global alignment.

```console
poetry run exact \
  -s data/ncit-doid/ncit.owl \
  -t data/ncit-doid/doid.owl \
  -o exp/runs/ncit_doid/global \
  -y exact/default_config.yaml \
  -r data/ncit-doid/train.tsv \
  -f data/ncit-doid/test.tsv \
  -l -e -d 0
```

### Local candidate ranking

Pass `-c` to score and rank an existing candidate set. All candidates remain in the ranking
file.

```console
poetry run exact \
  -s data/ncit-doid/ncit.owl \
  -t data/ncit-doid/doid.owl \
  -o exp/runs/ncit_doid/local \
  -y exact/default_config.yaml \
  -f data/ncit-doid/test.tsv \
  -c data/ncit-doid/test.cands.tsv \
  -l -e -d 0
```

See the generated [command-line reference](reference/cli/index.md) for the options supported by every
entry point.

### YAML runner

For repeatable jobs, use a run-config YAML with the job runner:

```console
poetry run python tools/run_exact_job.py \
  --run-config exp/runs/ncit_doid/run.yaml \
  --dry-run
```

The same helper can submit through Slurm with
`--sbatch-script deploy/sbatch/exact_single_run.sh`.

## Configuration

The default configuration is `exact/default_config.yaml`. Copy it into a run folder and edit
only the blocks needed for that run; partial overrides are merged with defaults.

### Alignment decisions

```yaml
alignment_params:
  threshold: 0.7
  cardinality: 1
  target_cardinality: 1
  save_json: true
```

`threshold` filters global alignments. In local mode it labels rationales as positive or
negative while preserving the ranking.

### Candidate generation

```yaml
candidates_params:
  retrieval_strategy: hybrid
  top_k: 20
  lexical_encoder_name: sentence-transformers/all-MiniLM-L6-v2
```

Global mode uses these settings to build source-local target candidates before scoring.

### Disable LLM use

```yaml
model:
  params:
    use_llm: false
    generate_llm_rationales: false
```

This keeps lexical and structural scoring active while avoiding hosted or local LLM calls.
The most common first-pass tuning knobs are `candidates_params.top_k`,
`alignment_params.threshold`, `dataset_params.n_hops`,
`dataset_params.hierarchy_max_depth`, and `model.params.tau_LLM`.

### Use an OpenRouter profile

```console
export OPENROUTER_API_KEY=...
```

```yaml
llm_routing:
  decision_profile: openrouter_gpt4o_mini
  rationale_profile: openrouter_gpt4o_mini
```

Hosted decision scoring is capability-gated. If a selected backend cannot provide the required
decision signal, the runtime uses the configured fallback profile.

The generated [configuration reference](reference/configuration/index.md) lists every schema field,
its type, its value in the default YAML, and its Pydantic description.

## Evidence channels

| Channel | Signal | What to inspect |
| --- | --- | --- |
| Lexical | Best label or synonym similarity. | `s_label`, `I_label`, selected labels. |
| Hierarchy | Aligned parents or configured hierarchy families. | `s_hier`, `I_hier`, selected hierarchy triples. |
| Similarity | Supported non-hierarchical object-property triples. | `s_sim`, `I_sim`, selected similarity triples. |
| Difference | Informative triples on one side without support on the other. | `s_diff`, `I_diff`, unsupported evidence. |
| Attribute | Definitions, synonyms, xrefs, and projected literals. | `s_attr`, `I_attr`, selected attributes. |
| LLM | Decision probability on ambiguous or disagreeing evidence. | `p_llm`, `I_llm`, generated rationale. |

## Outputs

A run writes mapping, metric, cache, checkpoint, and explanation artifacts beneath the output
directory. The precise layout is versioned by Exact-OM; inspect the run manifest and summary
files rather than assuming an artifact is always present.

Typical mapping and analysis artifacts include:

| Artifact | Purpose |
| --- | --- |
| `src2tgt.maps_global.tsv` | Filtered global mappings with source, target, and score. |
| `src2tgt.maps_local.tsv` | Per-source ranked target candidates. |
| `summary_metrics.csv` | Flattened scores, weights, selector fields, and labels. |
| `full_explanations.json` | Full candidate-level evidence and rationale records when legacy JSON export is enabled. |
| `run_stats.json` | Run-level aggregates, LLM usage, and score distributions. |
| `timings.json` | Session and stage timing ledger. |

## Evaluation

Use `-e` during an alignment run, or evaluate an existing mapping with `exact-eval`:

```console
poetry run exact-eval \
  --alignment-file exp/runs/ncit_doid/global/model/alignment/src2tgt.maps_global.tsv \
  --output-dir exp/runs/ncit_doid/global \
  --full-reference-file data/ncit-doid/test.tsv \
  --source-ontology-file data/ncit-doid/ncit.owl \
  --target-ontology-file data/ncit-doid/doid.owl \
  --save-logs
```

Global evaluation reports precision, recall, and F1 against the reference. Local candidate
evaluation reports ranking metrics such as MRR and Hits@K when candidates are supplied.

## User-study analysis

`exact-user-study` builds reusable artifacts from an existing local-ranking run:

```console
poetry run exact-user-study \
  --run-dir exp/runs/omim_ordo/local \
  --top-k 5 \
  --per-rank 4 \
  --shortlist-per-rank 8 \
  --generate-rationales
```

Outputs include pair and panel metrics, balanced shortlists, selected records with rationales,
a compact visualizer payload, a failure taxonomy, and an analysis notebook.

## Study visualizer

The visualizer serves a fixed study run through FastAPI and a static React/Cytoscape frontend.
It is intended for read-only inspection and LimeSurvey iframe embedding.

```console
cd explanations_visualizer
npm install
npm run build
```

The service commands and deployment names are being consolidated as part of the 2.0 overhaul;
consult `--help` on the installed visualizer entry point before launching a run.

## Python API

Use the API wrappers when integrating Exact-OM into a script or notebook:

```python
from exact import AlignmentRunner

runner = AlignmentRunner(
    source_ontology_file="data/ncit-doid/ncit.owl",
    target_ontology_file="data/ncit-doid/doid.owl",
    output_dir="exp/runs/ncit_doid/api",
    training_reference_file="data/ncit-doid/train.tsv",
    full_reference_file="data/ncit-doid/test.tsv",
    config_file="exact/default_config.yaml",
    save_logs=True,
    run_eval=True,
    device=0,
)
runner.run()
```

## Operations

- **Caching:** `use_file_cache: true` reuses compatible dataset and model caches. Inference
  checkpoints resume compatible runs.
- **Devices:** pass `-d 0` for GPU 0. If CUDA is unavailable, the alignment action falls back
  to CPU.
- **Slurm:** combine the scripts under `deploy/sbatch/` with the YAML runner for reproducible
  cluster jobs.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Hosted LLM falls back locally | Missing hosted key or an incompatible hosted decision profile. | Configure the selected profile and key, or disable LLM use. |
| No legacy explanation JSON | JSON export is disabled. | Enable the corresponding output option before the run. |
| User-study analysis cannot start | The run is not a local ranking or lacks explanations. | Run with candidates and enable explanation output. |
| Very slow preprocessing | Large inputs, high candidate `top_k`, or wide evidence pools. | Reuse caches, lower `top_k`, or reduce structural caps for exploratory runs. |
