# EXACT-OM

EXACT-OM is a tool for explainable, context-aware ontology matching.
It combines lexical similarity, ontology structure, and optional language-model
reasoning to predict mappings between entities while keeping the decision process
inspectable.

The repository currently defaults to a pair-adaptive scorer: lexical matching is
still the main signal, structural evidence is split into separate channels, and
the LLM sees a compact pair brief built from pair-specific evidence. The earlier
single-context scorer is still available as a legacy mode.

## Citation

If you use this repository in academic work, please cite:

Cotovio, P. G., Nunes, S., Jiménez-Ruiz, E., & Pesquita, C. (2026).
*Interpretable Context-Aware Models Improve Expert Validation in Ontology
Matching*. In *The Semantic Web: 23rd European Semantic Web Conference*.
Springer.

## How EXACT-OM Works

At a high level, EXACT-OM follows four steps:

1. **Lexical matching**
   Label variants are embedded with SapBERT, and the strongest pairwise
   similarity is used as the lexical signal.
2. **Pair-adaptive structural evidence**
   For each candidate pair, the model gathers multiple structural views instead
   of relying on a single pooled context. The current scorer separates:
   hierarchy-based evidence, non-hierarchical similarity triples, distinctive
   difference triples, and auxiliary literal or annotation attributes.
3. **Confidence-weighted fusion**
   Structural channels are first combined into one structural score. That score
   is then fused with lexical similarity using a confidence-aware function so
   that stronger evidence has more influence on the final prediction.
4. **Optional LLM arbitration**
   When a pair remains ambiguous, the system can build a short pair brief and
   ask an LLM for a binary decision. This can be done with local Hugging Face
   models or hosted OpenRouter models, depending on configuration.

One of the main goals of the project is interpretability. For each evaluated
mapping, EXACT-OM can export:

- the lexical, structural, and LLM contributions to the final score
- the per-channel structural breakdown
- natural-language rationales aligned with the final outcome
- structured JSON outputs for downstream inspection and analysis

## Installation

The project is managed with Poetry.

### Requirements

- Python 3.10
- Java available on the system path for JVM-based preprocessing
- Poetry
- CUDA-capable GPU if you want GPU execution; CPU-only runs are also supported

### Setup

```bash
poetry install
poetry run exact --help
```

The package exposes these main entry points:

- `exact` for ontology matching runs
- `bioml-eval` for standalone evaluation
- `exact-user-study` for post-run user-study analysis

## Running EXACT-OM

The matching CLI lives in `exact/delivery/cli/align.py`.

The core inputs are:

- `-s` / `--source_ontology_file`: source ontology in OWL format
- `-t` / `--target_ontology_file`: target ontology in OWL format
- `-o` / `--output_dir`: directory for outputs, logs, and artifacts
- `-y` / `--config_file`: YAML configuration file

Useful optional inputs:

- `-r` / `--training_reference_file`: training mappings for supervised settings
- `-f` / `--full_reference_file`: reference mappings for evaluation and some
  context-building steps
- `-c` / `--candidates_file`: candidate restriction file, useful for validation
  slices
- `-e` / `--run_eval`: run evaluation after inference
- `-d` / `--device`: CUDA device id; omit for CPU
- `-m` / `--jvm_heap_size`: heap size passed to the JVM, for example `32G`
- `-l` / `--save_logs`: write `exact.log` inside the output directory

Example:

```bash
poetry run exact \
  -s data/ncit-doid/ncit.owl \
  -t data/ncit-doid/doid.owl \
  -o exp/runs/ncit_doid/manual \
  -y exp/debug_new_approach/full_ncit2doid_local_small/config.yaml \
  -f data/ncit-doid/test.tsv \
  -c data/ncit-doid/test.cands.val.tsv \
  -l -e -m 60G -d 0
```

A typical workflow is:

1. Prepare the ontologies, reference mappings, and optional candidate file.
2. Copy and edit a YAML config, for example
   `exp/debug_new_approach/full_ncit2doid_local_small/config.yaml`.
3. Run `exact` and inspect the output directory for alignments, logs, plots, and
   explanations.

## Configuration Notes

### Threshold semantics

`alignment_params.threshold` is the shared final decision threshold.

- In global mode, it filters saved alignments and also determines positive vs.
  negative rationale polarity together with `cardinality`.
- In local mode, all candidates remain in the ranking output, but the same
  threshold is still used to label rationales as positive or negative.

### LLM routing

LLM behavior is split into two configuration blocks:

- `llm_profiles` defines named local or hosted backends
- `llm_routing` selects which profile is used for each task:
  `verbaliser`, `summary`, `decision`, and `rationale`

Hosted profiles use OpenRouter. The API key is resolved in this order:

1. `OPENROUTER_API_KEY`
2. the profile-specific `api_key_path`
3. `~/.config/openrouter/api_key`
4. an interactive prompt for a key-file path
5. local fallback with a warning

For decision scoring, the hosted path is stricter than the others. EXACT-OM uses
a constrained binary decision head and relies on usable chat logprobs for both
labels. If the selected hosted route does not support that path, the runtime
falls back to the configured local decision model.

## Outputs

Depending on the run mode, a typical output directory includes:

- alignment TSV files
- logs and runtime artifacts
- evaluation summaries
- `full_explanations.json` with detailed per-pair explanations

User-study analysis expects a run directory containing
`src2tgt.maps_local.tsv` and `full_explanations.json`.

## Evaluation And Analysis

### Standalone evaluation

`bioml-eval` mirrors the evaluation API and can be used on an existing alignment
file.

```bash
poetry run bioml-eval \
  --alignment_file exp/runs/ncit_doid/manual/model/alignment.tsv \
  --output_dir exp/runs/ncit_doid/manual \
  --full_reference_file data/ncit-doid/test.tsv \
  --source_ontology_file data/ncit-doid/ncit.owl \
  --target_ontology_file data/ncit-doid/doid.owl \
  --save_logs -m 32G
```

### Python API

For programmatic use, `exact.delivery.api` exposes:

- `AlignmentRunner`, which mirrors the main matching CLI
- `EvalutionRunner`, which wraps the standalone evaluation flow

### User-study analysis

`exact-user-study` is the post-run analysis CLI used to build balanced study
selections, export mappings, generate failure taxonomies, and produce the
notebook used for expert validation analysis.

The helper `tools/run_user_study_job.py` wraps this flow from a YAML run config.

## YAML Runners And Slurm Helpers

For repeatable runs, the repository includes small wrappers around the main CLI:

- `tools/run_exact_job.py` builds and runs an `exact` command from a YAML file
- `tools/run_user_study_job.py` does the same for `exact-user-study`

These can be executed locally with `--dry-run` to inspect the resolved command,
or submitted through Slurm by passing the corresponding wrapper script:

- `deploy/sbatch/exact_tune_run.sh`
- `deploy/sbatch/exact_user_study_run.sh`
- `deploy/sbatch/exact_single_run.sh`

## Hyperparameter Tuning

`tools/hparam_tuner.py` generates per-trial experiment folders, trial configs,
and `sbatch` commands from a tuner YAML file such as
`exp/tuning/ncit_doid_val/tuner.yaml`.

Two common modes are supported:

- grid search over explicit parameter values
- a lightweight smart sampler for low-discrepancy exploration with local
  exploitation

Example:

```bash
poetry run python tools/hparam_tuner.py \
  --tuner-config exp/tuning/ncit_doid_val/tuner.yaml \
  --strategy grid --dry-run
```
