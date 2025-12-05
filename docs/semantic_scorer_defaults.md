# SemanticScorer Default Parameters

This document summarises the default values used in `exact/default_config.yaml`
and provides a short explanation for each knob. Parameters are grouped by
section (top-level, dataset, inference, model). Use this document as the source
of truth when crafting custom configs.

## Top-level toggles

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `logging_level` | `INFO` | Global logger verbosity (`DEBUG/INFO/WARNING/ERROR`). |
| `seed` | `42` | RNG seed for reproducibility (`null` disables seeding). |
| `use_file_cache` | `True` | Reuse previously saved datasets/models when available. |

## Model chain

- `model` — primary scorer (default `SemanticScorer`).
- `second_model` — optional chained model (default `SecondPassReranker`, disabled).
- `model_chain` — advanced list override to run an arbitrary sequence; if present it supersedes `model`/`second_model`.

## Alignment params (`alignment_params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `threshold` | `0.7` | Filter alignments below this score (`null` = keep all). |
| `cardinality` | `1` | Max targets per source (`null` = unbounded). |
| `save_json/save_csv/save_stats_csv/append_stats_to_summary_csv` | `True` | Control which artefacts are written after inference. |
| `review_low/review_high` | `0.5 / 0.8` | Define the “review band” flag in explanations. |

## Dataset params (`dataset_params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `num_workers` | `null` | DataLoader workers (null = auto). |
| `filter_exact_matches` | `True` | Remove known perfect matches before inference. |
| `drop_exact_match_sources` | `True` | Drop every candidate for sources with an exact match (set `False` for ranking). |
| `n_hops` | `1` | Context graph depth. |
| `context_method` | `greedy` | Context extraction strategy (`bfs`/`greedy`). |
| `best_path_method` | `dp` | Path scoring method for greedy extraction. |
| `context_hop_penalty` | `0.1` | Hop penalty inside the path finder. |
| `context_token_ratio/context_safety/max_input_tokens_context` | `1.3 / 0.8 / 256` | Token budget and safety margins for context. |
| `only_taxonomy` | `False` | Use only taxonomy edges when `True`. |
| `all_labels` | `True` | Supply all available labels per entity. |
| `add_connectivity_bridges` | `True` | Add explanation-only connector triples so contexts stay connected (disable to save time). |
| `bridge_max_hops` | `null` | Optional hop cap for connector search (null = unbounded). |
| `verbaliser_name` | `Qwen/Qwen2.5-3B-Instruct` | LLM used to verbalise triples. |
| `gen_max_new_tokens` | `64` | Max new tokens from the verbaliser. |
| `do_sample/temperature/top_k/top_p` | `False / 0.1 / null / 0.9` | Standard generation knobs. |
| `max_verb_gen_retries` | `1` | Number of retries in case template generation fails. |
| `exclude_missing_dr` | `False` | Skip entities missing domain/range info. |
| `batch_size_verbaliser` | `4` | Verbaliser batch size. |
| `delimiter` | `"\n"` | Delimiter when concatenating verbalised triples. |
| `which` | … | Metrics plotted for dataset inspection (see YAML). |
| `candidate_share_k` | `2` | Number of top MiniLM candidates used in share computation. |

## Candidates (`candidates_params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `lexical_encoder_name` | `sentence-transformers/all-MiniLM-L6-v2` | Encoder for candidate generation. |
| `encode_batch_size` | `512` | Batch size for embedding labels. |
| `search_batch_size` | `4096` | Batch size for cosine similarity search. |
| `top_k` | `50` | Number of candidates retained per source. |
| `use_amp` | `True` | Use mixed precision to speed up search. |

## Sanity checks / plots

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `sanity_check` | `True` | Emit dataset sanity logs. |
| `n` | `3` | Number of examples shown. |
| `max_ctx_show/max_label_show` | `3 / 3` | Limit for printed contexts/labels. |
| `plot_params` (`bins`, `figsize`, `dpi`, `kde`, `alpha`) | `30`, `[7,5]`, `300`, `False`, `0.6` | Default matplotlib/seaborn plot settings. |

## Inference params (`inference_params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `batch_size` | `64` | Semantic scorer batch size. |
| `num_workers` | `5` | DataLoader workers at inference time. |
| `log_every` | `10` | Batch interval for progress logs. |
| `mixed_precision` | `True` | Use autocast on GPU. |
| `which` | … | Metrics plotted after inference (see YAML). |
| `checkpoint_every` | `10` | Batch interval for checkpoint writes. |
| `resume_from_checkpoint` | `True` | Pick up from the latest checkpoint if present. |
| `enable_checkpoints` | `True` | Toggle checkpointing altogether. |


## Optional second model (`second_model.params`)

Configured via the top-level `second_model` block; the reranker runs after the
primary scorer on ambiguous sources only.

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `enabled` | `False` | Enable the optional post-inference reranker (runs as a second model). |
| `top_k` | `5` | Number of top candidates per source examined during reranking. |
| `epsilon` | `0.03` | Score gap used to deem a source “ambiguous”. |
| `min_ties` | `3` | Minimum candidates within `epsilon` of the best score required to trigger reranking. |
| `ce_model_name` | `null` | Optional Hugging Face reranker used in the second pass (`null` disables CE). |
| `ce_weight` | `0.5` | Mix weight applied when CE scores are available. |
| `use_symbolic` | `True` | Whether to apply the symbolic tie-breaker (string similarity). |
| `symbolic_weight` | `0.05` | Contribution of the symbolic similarity to the score. |
| `use_llm` | `False` | Allow the LLM fallback when other tie-breakers fail. |
| `llm_trigger_epsilon` | `0.01` | Tighter threshold before invoking the LLM fallback. |
| `max_llm_prompts` | `0` | Maximum number of LLM reranking prompts (0 = disabled). |
| `max_prompt_candidates` | `5` | Max candidates included in a single LLM prompt. |


## Model Core Behaviour

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `use_lexical` | `True` | Enable lexical (SapBERT) similarity. |
| `use_context` | `True` | Enable context encoder (BGE) similarity. |
| `use_llm` | `True` | Enable the LLM decision head. |

## Adaptive Context

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `tau` | `0.5` | Mid-point that defines the “neutral” similarity when computing adaptive weights. |
| `gamma` | `1.0` | Controls the adaptive fusion between `s_label*` and `s_ctx`. |

## LLM Gating

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `beta` | `0.8` | Scales LLM contribution: `w_i = β · U`. |
| `tau_LLM` | `0.8` | LLM fires when `U ≥ tau_LLM`. |
| `force_llm_summaries` | `False` | Generate summaries for every pair even if `U < tau_LLM` (only the Yes/No head stays gated). |

## Miscellaneous

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `max_input_tokens_*`, `max_total_tokens_*`, `pooling_method`, etc. | See `exact/default_config.yaml` | Legacy model sizing and pooling knobs unchanged from the base SemanticScorer implementation. |

All dataset-level defaults (e.g., candidate share `k`) also live in
`default_config.yaml`. Use that file for the authoritative list whenever you
need to override parameters in a custom experiment.
