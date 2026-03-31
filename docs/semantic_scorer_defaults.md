# SemanticScorer Default Parameters

This document summarises the defaults in
[`exact/default_config.yaml`](/home/pgcotovio/Exact-OM/exact/default_config.yaml)
and explains how the main configuration blocks interact.

The current runtime supports both:

- local Hugging Face backends (`backend: local_hf`)
- hosted OpenRouter backends (`backend: openrouter`)

The four LLM-backed tasks can now be routed independently:

- verbaliser
- summary generator
- decision scorer
- rationale generator

Hosted decision scoring is stricter than the other LLM tasks. The runtime now
uses OpenRouter `/chat/completions` with a constrained two-label head
(default: `A/B`). A hosted decision profile must therefore expose usable chat
`logprobs`/`top_logprobs`, accept `logit_bias`, include a tokenizer mapping for
the biased decision labels, and pass a runtime probe; otherwise the runtime
warns and falls back to the configured local decision profile.

## Top-level toggles

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `logging_level` | `INFO` | Global logger verbosity (`DEBUG/INFO/WARNING/ERROR`). |
| `seed` | `42` | RNG seed for reproducibility. |
| `use_file_cache` | `True` | Reuse previously saved datasets and caches when available. |
| `k` | `[1, 5, 10]` | Cutoffs reported during evaluation. |

## Model chain

- `model` — primary scorer (default `SemanticScorer`)
- `second_model` — optional chained reranker (default `SecondPassReranker`, disabled)
- `model_chain` — advanced override for arbitrary model pipelines

## Alignment params (`alignment_params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `threshold` | `0.7` | Shared final decision threshold. In global mode it filters saved alignments; in local mode it still defines positive vs negative rationale polarity. |
| `cardinality` | `1` | Maximum targets per source in the final global alignment (`null` = unbounded). |
| `save_json/save_csv/save_stats_csv/append_stats_to_summary_csv` | `True` | Control which output artefacts are written after inference. |
| `review_low/review_high` | `0.5 / 0.8` | Review band used in explanations and reports. |

## Dataset params (`dataset_params`)

### Dataset construction and context extraction

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `num_workers` | `null` | Dataset worker count (`null` = let the runtime decide). |
| `filter_exact_matches` | `True` | Remove known exact lexical matches before semantic scoring. |
| `drop_exact_match_sources` | `False` | If `True`, remove every candidate for sources with an exact match. |
| `reasoner_timeout_secs` | `120` | Timeout for ontology reasoner construction. |
| `reasoner_force_hermit` | `False` | Force a HermiT retry after lighter reasoners. |
| `n_hops` | `1` | Context graph depth. |
| `context_method` | `greedy` | Context extraction strategy (`bfs` or `greedy`). |
| `best_path_method` | `dp` | Path selection strategy inside context extraction. |
| `context_hop_penalty` | `0.1` | Penalty applied per graph hop. |
| `context_token_ratio/context_safety/max_input_tokens_context` | `1.3 / 0.8 / 256` | Context token budgeting controls. |
| `only_taxonomy` | `False` | Use taxonomy-only extraction and fixed subclass templates. |
| `all_labels` | `True` | Use all label variants instead of a single best label. |
| `add_connectivity_bridges` | `True` | Add explanation-only bridge triples so contexts remain connected. |
| `bridge_max_hops` | `null` | Optional cap on bridge path length (`null` = unbounded). |
| `delimiter` | `"\n"` | Delimiter used when joining verbalised context triples. |
| `which` | … | Dataset metrics plotted after preprocessing (see YAML). |
| `candidate_share_k` | `2` | Number of top lexical candidates reused by candidate-share utilities. |

### Verbaliser generation controls

The verbaliser now generates structured relation templates directly and expects
each template to contain the literal placeholders `$SRC` and `$TGT`.

| Parameter | Default | Scope | Explanation |
|-----------|---------|-------|-------------|
| `verbaliser_name` | `Qwen/Qwen2.5-3B-Instruct` | local-only fallback | Local fallback model used when the verbaliser runs on Hugging Face. |
| `gen_max_new_tokens` | `64` | shared | Max generated tokens for relation template creation. |
| `temperature` | `0.1` | shared | Generation temperature for local and hosted verbaliser calls. |
| `top_p` | `0.9` | shared | Nucleus sampling cutoff for local and hosted verbaliser calls. |
| `do_sample` | `False` | local-only | Enable sampling instead of greedy decoding for local verbaliser runs. |
| `top_k` | `null` | local-only | Top-k sampling cutoff for local verbaliser runs. |
| `max_verb_gen_retries` | `1` | shared | Retry count when a generated template is invalid. |
| `exclude_missing_dr` | `False` | shared | Skip relations lacking domain/range examples when generating templates. |
| `batch_size_verbaliser` | `4` | shared | Local batch size for verbaliser inference and hosted concurrency cap for OpenRouter verbaliser requests. |

## Candidates (`candidates_params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `lexical_encoder_name` | `sentence-transformers/all-MiniLM-L6-v2` | Encoder used during lexical candidate generation. |
| `encode_batch_size` | `512` | Batch size for embedding labels. |
| `search_batch_size` | `4096` | Batch size for cosine similarity search. |
| `top_k` | `50` | Number of candidates retained per source. |
| `use_amp` | `True` | Use mixed precision during lexical retrieval when supported. |

## Sanity checks and plotting

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `sanity_check` | `True` | Emit dataset sanity logs. |
| `n` | `3` | Number of sample examples shown. |
| `max_ctx_show/max_label_show` | `3 / 3` | Limits for printed context sentences and label variants. |
| `plot_params` (`bins`, `figsize`, `dpi`, `kde`, `alpha`) | `30`, `[7,5]`, `300`, `False`, `0.6` | Default plotting settings for dataset and inference reports. |

## Inference params (`inference_params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `batch_size` | `64` | Candidate-pair batch size for the semantic scorer. |
| `num_workers` | `5` | DataLoader workers during inference. |
| `log_every` | `10` | Batch interval for progress logging. |
| `mixed_precision` | `True` | Use autocast on GPU. |
| `which` | … | Metrics plotted after inference (see YAML). |
| `checkpoint_every` | `10` | Batch interval for checkpoint writes. |
| `resume_from_checkpoint` | `True` | Resume from the latest checkpoint if present. |
| `enable_checkpoints` | `True` | Toggle checkpointing on or off. |

## LLM profiles (`llm_profiles`)

`llm_profiles` defines named backend profiles. Each profile points either to:

- `backend: openrouter` for hosted inference through OpenRouter
- `backend: local_hf` for local Hugging Face inference

### Hosted profile fields

| Parameter | Explanation |
|-----------|-------------|
| `model` | OpenRouter model id, for example `openai/gpt-4o-mini`. |
| `tokenizer` | Optional Hugging Face tokenizer id used when the hosted decision path needs token ids for `logit_bias`. |
| `api_base` | Usually `https://openrouter.ai/api/v1`. |
| `api_key_env` | Environment variable checked first for the API key. |
| `api_key_path` | Profile-specific key file path checked after the environment variable. |
| `timeout_secs` | Per-request timeout. |
| `provider` | Optional OpenRouter provider-routing object passed through to requests. Useful keys include `require_parameters`, `order`, `only`, `ignore`, and `sort`. |

### OpenRouter key resolution order

For hosted profiles, the runtime checks for an API key in this order:

1. the configured environment variable, usually `OPENROUTER_API_KEY`
2. the profile-specific `api_key_path`
3. the default path `~/.config/openrouter/api_key`
4. an interactive prompt asking for a key-file path
5. local fallback, with a warning

## LLM routing (`llm_routing`)

`llm_routing` decides which named profile each task uses.

| Parameter | Explanation |
|-----------|-------------|
| `default_profile` | Shared default profile when no task-specific profile is set. |
| `verbaliser_profile` | Model profile for relation template generation. |
| `summary_profile` | Model profile for entity summarisation. |
| `decision_profile` | Model profile for hosted binary decision scoring. Should point to a chat-logprob + `logit_bias` capable model if not local. |
| `rationale_profile` | Model profile for natural-language rationale generation. |
| `verbaliser_fallback_profile` | Fallback used only for verbaliser calls. |
| `fallback_profile` | Shared fallback for summary and rationale calls. |
| `decision_fallback_profile` | Fallback used when hosted decision scoring cannot provide usable logprobs. |

### Recommended routing usage

- Use smaller or cheaper models for verbalisation when instruction-following is good enough.
- Use stronger models for summaries if ontology contexts are complex.
- Use only models that support chat `logprobs`, `top_logprobs`, and `logit_bias` for hosted decision scoring.
- Use a separate rationale model if explanation quality matters more than raw speed.

### Suggested profile split

- Generation-oriented hosted profiles: `openrouter_claude_sonnet_46`, `openrouter_gpt5`, `openrouter_qwen3_235b`, `openrouter_gpt_oss_120b`
- Decision-oriented hosted profiles: `openrouter_qwen35_122b`, `openrouter_qwen3_235b`, `openrouter_llama33_70b`, `openrouter_deepseek_chat_v31`

Decision-oriented profiles are still probe-gated at runtime; the defaults above
are recommendations, not a hard guarantee that every OpenRouter provider route
will expose usable chat-side token logprobs for both decision labels.

## Model core behaviour (`model.params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `lexical_model_name` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | Encoder for label similarity. |
| `context_model_name` | `BAAI/bge-large-en-v1.5` | Encoder for verbalised context similarity. |
| `llm_model_name` | `Qwen/Qwen2.5-7B-Instruct` | Local-only fallback model for summary, decision, and rationale tasks. |
| `fp16_inference` | `True` | Use fp16 where supported during local model inference. |
| `pooling_method` | `mean` | Token pooling for encoder outputs (`mean`, `max`, `sum`). |
| `label_pair_pooling` | `max` | Pooling over the label-pair similarity matrix (`mean` or `max`). |
| `use_lexical` | `True` | Enable lexical similarity. |
| `use_context` | `True` | Enable context similarity. |
| `use_llm` | `True` | Enable LLM-backed summaries, decision scoring, and rationales. |

## Model token limits and adaptive weighting

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `max_input_tokens_lexical` | `32` | Max tokens per label passed to the lexical encoder. |
| `max_input_tokens_context` | `256` | Max tokens per joined context passed to the context encoder. |
| `max_total_tokens_llm_summary` | `512` | Local-only prompt cap for summary generation. |
| `max_total_tokens_llm_decision` | `384` | Local-only prompt cap for decision scoring. |
| `max_total_tokens_llm_rationale` | `512` | Local-only prompt cap for rationale generation. |
| `max_new_tokens_llm` | `64` | Shared generation cap for summaries and rationales. |
| `max_new_tokens_llm_rationale` | `128` | Rationale-specific generation cap. Use this when explanations need to be longer than summaries. |
| `tau` | `0.5` | Neutral midpoint used when estimating ambiguity and confidence. |
| `gamma` | `1.0` | Controls how sharply lexical/context confidence changes the fusion weight. |
| `beta` | `0.8` | Maximum contribution allowed from the LLM branch. |
| `tau_LLM` | `0.8` | LLM branch activates when ambiguity `U >= tau_LLM`. |
| `force_llm_summaries` | `False` | Generate summaries for every pair even when the decision LLM is gated off. |

## Model LLM generation controls

These knobs have different scope depending on whether the backend is local or
hosted.

| Parameter | Default | Scope | Explanation |
|-----------|---------|-------|-------------|
| `llm_temperature` | `0.1` | shared for summary/rationale | Generation temperature for local and hosted summary/rationale calls. |
| `llm_top_p` | `0.9` | shared for summary/rationale | Nucleus sampling setting for local and hosted summary/rationale calls. |
| `llm_do_sample` | `False` | local-only | Enable sampling for local summary/rationale generation. |
| `llm_summary_batch_size` | `8` | shared for summary batching | Batch size for local summary generation and hosted concurrency cap for OpenRouter summary requests. |
| `llm_decision_batch_size` | `8` | shared for decision batching | Batch size for local decision scoring and hosted concurrency cap for OpenRouter decision requests. |
| `llm_rationale_batch_size` | `8` | shared for rationale batching | Batch size for local rationale generation and hosted concurrency cap for OpenRouter rationale requests. |
| `hosted_decision_labels` | `["A","B"]` | hosted-only | Ordered pair of hosted decision labels. The first label is treated as the positive class. |
| `hosted_decision_logit_bias` | `20.0` | hosted-only | Equal positive `logit_bias` added to both hosted decision labels to pull them into the top-logprob slice. |

### Hosted decision scoring

Hosted decision scoring does **not** use the summary/rationale sampling knobs.
Instead, it uses OpenRouter `/chat/completions` with a constrained binary head:

- the prompt instructs the model to emit exactly one token, for example `A` or `B`
- the runtime applies equal positive `logit_bias` to both decision labels
- the runtime reads first-token `top_logprobs` and aggregates the logprob mass
  for each configured label
- a two-way softmax recovers `p_positive` and `p_negative`, then keeps the
  positive-class probability as `p_llm`

Hosted decision calls use fixed deterministic settings internally:

- `temperature = 0`
- `top_p = 1`
- `max_tokens = 1`
- `logprobs = true`
- `top_logprobs = 20`
- equal positive `logit_bias` on both configured decision labels
- `provider.require_parameters = true`

Before first hosted decision scoring, the runtime performs a one-time probe for
the selected profile. If the model/provider route does not expose usable first-
token logprobs for both configured decision labels, the runtime warns and falls
back to the configured local decision profile.

Hosted decision scoring is concurrency-limited rather than true API-batched on
OpenRouter chat: the runtime issues one request per candidate pair, up to
`llm_decision_batch_size` concurrent requests at a time.

Hosted summary, rationale, and verbaliser calls also use bounded concurrency on
OpenRouter. They are still one prompt per request at the API level, but the
runtime now issues multiple requests in parallel up to the configured
`llm_summary_batch_size`, `llm_rationale_batch_size`, or `batch_size_verbaliser`
limit.

## Explanations, calibration, and caches

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `return_explanations` | `True` | Include candidate-level explanations in model output. |
| `generate_llm_rationales` | `False` | Generate final positive/negative rationales. |
| `use_llm_calibration` | `False` | Enable post-hoc calibration on the decision probability. |
| `llm_calibration_a/llm_calibration_b/llm_calibration_info` | `null` | Optional preconfigured calibration metadata. |
| `cache_dir` | `null` | Cache root (`null` uses the default cache directory). |
| `cache_namespace` | `null` | Optional namespace to isolate cache files. |
| `dataset_signature` | `null` | Optional cache fingerprint override. |
| `persist_cache_to_disk` | `True` | Persist summary/rationale/embedding caches between runs. |
| `max_cached_labels/max_cached_contexts/max_cached_summaries/max_cached_rationales` | `null` | Optional in-memory cache caps. |

## Final rationale semantics

Rationales are conditioned on the final outcome rather than only on the raw LLM
subdecision:

- **Global mode:** positive rationale means the pair survived both `threshold`
  and `cardinality` and is part of the saved alignment.
- **Local mode:** positive rationale means `S_final >= threshold`, even though
  all candidates remain in the ranking output.

## Optional second model (`second_model.params`)

Configured via the top-level `second_model` block; the reranker runs after the
primary scorer on ambiguous sources only.

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `enabled` | `False` | Enable the optional second-stage reranker. |
| `top_k` | `5` | Number of top candidates per source examined during reranking. |
| `epsilon` | `0.03` | Score-gap threshold used to deem a source ambiguous. |
| `min_ties` | `3` | Minimum number of near-tied candidates required before reranking fires. |
| `ce_model_name` | `null` | Optional cross-encoder used during reranking. |
| `ce_weight` | `0.5` | Weight assigned to the cross-encoder contribution. |
| `use_symbolic` | `True` | Apply symbolic reranking features. |
| `symbolic_weight` | `0.05` | Weight assigned to symbolic reranking features. |
| `use_llm` | `False` | Enable LLM-assisted reranking inside the second pass. |
| `llm_trigger_epsilon` | `0.01` | Ambiguity threshold used to trigger LLM reranking. |
| `max_llm_prompts` | `0` | Hard cap on LLM reranking prompts. |
| `max_prompt_candidates` | `5` | Maximum number of candidates packed into one reranking prompt. |

Use
[`exact/default_config.yaml`](/home/pgcotovio/Exact-OM/exact/default_config.yaml)
as the authoritative default whenever you build a custom experiment config.
