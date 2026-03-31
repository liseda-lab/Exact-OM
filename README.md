# **EXACT-OM: An Explainable Context-Aware Matching Model for Ontology Alignment**

**EXACT-OM** is a hybrid ontology matching model that integrates lexical, contextual, and language model-based signals within a unified, interpretable architecture.
It combines the strengths of **language models** and **ontology semantics** to align entities across heterogeneous knowledge bases while preserving explainability.

### **Core Principles**

1. **Lexical Understanding**
   Entities are represented through all label variants and embedded using **SapBERT**, capturing synonymy and abbreviation patterns. The highest pairwise similarity defines the lexical correspondence.

2. **Contextual Semantics**
   Each entity’s local neighbourhood is extracted and **verbalised into natural language**, allowing a contextual encoder (e.g., **BGE-Large**) to model the relational meaning of concepts beyond surface labels.

3. **Adaptive Fusion**
   Lexical and contextual similarities are combined through a **confidence-weighted function**, granting higher influence to the most reliable modality for each pair.

4. **Uncertainty-Driven Language Model Inference**
   When ambiguity remains high, configurable LLM backends summarise both entities’ contexts and issue a **binary decision** whose probability is incorporated into the final score. The verbaliser, summary model, decision model, and rationale model can now be routed independently to either **local Hugging Face models** or **hosted OpenRouter models**.

### **Explainability**

For every evaluated mapping, EXACT-OM provides:

* Label and context contributions with their confidence and weight.
* Relative importance of lexical, contextual, and generative LM components.
* Natural-language rationales aligned with the final model outcome.
  All results are exported as structured JSON explanations, ensuring transparent and traceable alignment decisions.

### **Outcome**

By integrating structured semantics, adaptive weighting, and interpretable reasoning, EXACT-OM delivers high-quality ontology alignments with explicit, human-readable evidence of how and why each mapping was made.


## Running Exact-OM

### CLI (`exact`)

`exact` (defined in `exact/delivery/cli/align.py`) exposes the following key flags:

- `-s/--source_ontology_file`, `-t/--target_ontology_file`: required OWL inputs.
- `-o/--output_dir`: target directory (created automatically) for checkpoints, plots, and logs.
- `-y/--config_file`: path to a YAML configuration (defaults to built-in settings if omitted).
- `-r/--training_reference_file`, `-f/--full_reference_file`: TSV ground-truth mappings. `-r` is used for supervised tasks; `-f` unlocks evaluation and certain context caches.
- `-c/--candidates_file`: optional `test.cands*.tsv` restricting the search space.
- `-l/--save_logs`: write `exact.log` inside the output dir; stdout only otherwise.
- `-e/--run_eval`: run the evaluation stage after inference (requires `-f` when no candidates file is used).
- `-m/--jvm_heap_size`: heap passed to the JVM-based preprocessing (accepts sizes like `32G`).
- `-d/--device`: CUDA device id; omit to run entirely on CPU.

Typical workflow:

1. Prepare ontologies (`*.owl`), references (`train.tsv`, `test.tsv`), and candidates (`test.cands.val.tsv`).
2. Adjust a config (copy `exp/debug_new_approach/full_ncit2doid_local_small/config.yaml` and edit thresholds, model names, etc.).
3. Invoke `exact` with the paths above; include `-c` when running on validation slices, drop it for full-test inference.

### Hosted LLM configuration

The default config now supports both local and hosted LLM backends.

- `llm_profiles` defines named backend profiles.
- `llm_routing` decides which profile is used for each task:
  - verbaliser
  - summary
  - decision
  - rationale

Hosted profiles use OpenRouter and resolve the API key in this order:

1. `OPENROUTER_API_KEY`
2. the profile-specific `api_key_path`
3. `~/.config/openrouter/api_key`
4. an interactive prompt for a key-file path
5. local fallback with a warning

The decision path is stricter than the other LLM tasks: if the selected hosted
model cannot support the configured binary-head chat-logprob scoring path,
EXACT-OM falls back to the configured local decision model.

Hosted decision scoring now uses OpenRouter `/chat/completions` with a
constrained binary head. The runtime asks the model to emit exactly one label
(default: `A` or `B`), applies equal positive `logit_bias` to both label
tokens, reads first-token `top_logprobs`, and normalizes the two label scores
into the positive-class probability used by the scorer. A runtime probe checks
whether the selected model/provider route exposes usable chat logprobs for both
labels before the hosted decision path is used.

Hosted summary, rationale, and verbaliser calls still send one prompt per API
request, but the runtime now executes them with bounded concurrency using the
corresponding batch-size settings as the concurrency cap.

Example:

```bash
exact \
  -s data/ncit-doid/ncit.owl \
  -t data/ncit-doid/doid.owl \
  -o exp/runs/ncit_doid/manual \
  -y exp/debug_new_approach/full_ncit2doid_local_small/config.yaml \
  -f data/ncit-doid/test.tsv \
  -c data/ncit-doid/test.cands.val.tsv \
  -l -e -m 60G -d 0
```

### Threshold and rationale semantics

`alignment_params.threshold` is the shared final decision threshold.

- In **global mode**, it filters saved alignments and also defines positive vs
  negative rationale polarity together with `cardinality`.
- In **local mode**, all candidates remain in the ranking output, but the same
  threshold is still used to label rationales as positive or negative.

### Python API (`exact.delivery.api`)

For programmatic control:

- `AlignmentRunner` mirrors the CLI parameters. Constructor arguments map 1:1 to the flags listed above (e.g., `source_ontology_file`, `target_ontology_file`, `output_dir`, `config_file`, `training_reference_file`, `full_reference_file`, `candidates_file`, `save_logs`, `run_eval`, `device`, `jvm_heap_size`). Call `runner.run()` to validate, boot the JVM, and execute the alignment.
- `EvalutionRunner` (note spelling) encapsulates the standalone evaluation tool. Key args are `alignment_file`, `output_dir`, `full_reference_file`, optional ontologies/references for contextual metrics, `K` (list of cutoffs), `log_level`, `save_logs`, and `jvm_heap_size`. Call `run()` to produce precision/recall scores and CSV summaries.

### Evaluation CLI (`bioml-eval`)

`bioml-eval` (defined in `exact/delivery/cli/eval.py`) accepts the same fields as `EvalutionRunner`:

- `--alignment_file/-a`: TSV with predictions.
- `--output_dir/-o`: destination for metrics and logs.
- Optional: `--source_ontology_file`, `--target_ontology_file`, `--train_reference_file`, `--full_reference_file`, `--reference_candidates`, `--K`, `--log_level`, `--save_logs`, `--jvm_heap_size`.
- `--error_on_fail/-e`: make evaluation raise on missing references instead of logging warnings.

Example:

```bash
bioml-eval \
  --alignment_file exp/runs/ncit_doid/manual/model/alignment.tsv \
  --output_dir exp/runs/ncit_doid/manual \
  --full_reference_file data/ncit-doid/test.tsv \
  --source_ontology_file data/ncit-doid/ncit.owl \
  --target_ontology_file data/ncit-doid/doid.owl \
  --save_logs -m 32G
```

### YAML-driven runner

`tools/run_exact_job.py` lets you express runs declaratively. A config consists of:

- `dataset`: `data_dir`, `source`, `target`, `train_reference`, `full_reference`, `candidates`, plus optional runtime knobs forwarded to the runner (`memory`, `device`, `run_eval`, `save_logs`).
- `job`: `name`, `output_dir`, `config_file`, `memory`, `device`, `save_logs`, `run_eval`.

Usage:

- `python3 tools/run_exact_job.py --run-config exp/run_configs/ncit_doid_val.yaml --dry-run` shows the resolved `exact` command.
- Drop `--dry-run` to execute locally.
- Pass `--sbatch-script deploy/sbatch/exact_tune_run.sh` to submit the same run through Slurm.

### Slurm helper scripts

- `deploy/sbatch/exact_single_run.sh`: minimal template with hard-coded paths for quick manual edits (e.g., update the variables at the top and call `sbatch …`).
- `deploy/sbatch/exact_tune_run.sh`: argument-driven script used by the tuner and YAML runner. Supports overrides via `--data-dir`, `--source`, `--target`, `--candidates`, `--config-file`, `--run-eval`, `--save-logs`, etc., or the equivalent environment variables when invoking `sbatch`.

## Hyperparameter tuning helper

The `tools/hparam_tuner.py` utility creates per-trial experiment folders, configs, and
corresponding `sbatch` commands. Describe your dataset, base config, and search space
in a tuner file (see `exp/tuning/ncit_doid_val/tuner.yaml` for an example), then pick a
strategy:

Tuner YAML layout:

- `base_config`: path to the reference config that every trial will clone and mutate.
- `experiment_root`: directory where per-trial folders, manifests, and submit scripts are written.
- `job_name_prefix`: label applied to each generated job (suffixed with trial index + params).
- `dataset`: values forwarded to the Slurm runner (`data_dir`, ontology filenames, references, candidates, memory, device, run_eval, save_logs).
- `slurm`: `script` (defaults to `deploy/sbatch/exact_tune_run.sh`) plus optional `sbatch_args` (partition, nodes, etc.).
- `search_space`: keys reference dotted config paths (e.g., `model.params.gamma`). For each parameter:
  - `type`: `float`, `int`, or `categorical`.
  - `values`: explicit list used during grid search (and as anchor points for smart sampling).
  - `bounds`: `[min, max]` interval for the smart sampler; combine with `scale: log` for log-space sampling and `quantize`
    to snap results (e.g., batch sizes) to sensible increments.
- `smart`: knobs for the low-discrepancy sampler: `num_samples`, `exploit_fraction` (portion of configs perturbed around
  anchor configs), `exploit_noise` (relative perturbation magnitude), `random_seed`, and explicit `anchor_configs`
  (each providing `values` per parameter). When omitted, the sampler centers exploitation around the base config.

Common command-line options:

```bash
# Dense, exhaustive combinations over the provided discrete values.
python3 tools/hparam_tuner.py \
  --tuner-config exp/tuning/ncit_doid_val/tuner.yaml \
  --strategy grid --dry-run

# Lightweight low-discrepancy sampling with local exploitation.
python3 tools/hparam_tuner.py \
  --tuner-config exp/tuning/ncit_doid_val/tuner.yaml \
  --strategy smart --num-samples 24
```

Running without `--dry-run` writes trial configs under `experiment_root`, generates a
`submit_all.sh` helper with ready-to-run `sbatch` lines, and produces a manifest of the
sampled hyperparameters. Use `--submit` if you want the script to enqueue every job
immediately after generation.

The tuning Slurm wrapper (`deploy/sbatch/exact_tune_run.sh`) honors environment overrides
for the dataset, config path, candidates file, and memory budget, so every generated job
can stay self-contained without duplicating shell scripts.
