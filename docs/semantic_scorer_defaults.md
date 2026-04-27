# EXACT Default Parameters

This document summarises the defaults in
[`exact/default_config.yaml`](/home/pgcotovio/Exact-OM/exact/default_config.yaml)
and explains how the main configuration blocks interact.

The filename is historical, but the current default scorer is
`PairAdaptiveSemanticScorer`, not the legacy `SemanticScorer`.

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

- `model` — primary scorer (default `PairAdaptiveSemanticScorer`)
- `second_model` — optional chained global candidate-set selector (default
  `CandidateSetSelector`, enabled in the default config)
- `model_chain` — advanced override for arbitrary model pipelines

### Current default scoring flow

The default scorer combines signals in four stages:

1. lexical matching over all label variants
2. pair-adaptive structural evidence split into hierarchy, non-hierarchical
   similarity, difference, and auxiliary attribute channels
3. lexical-versus-structural fusion with confidence-aware weighting
4. optional LLM scoring on ambiguous pairs using one pair brief

The legacy `SemanticScorer` still exists and instead uses one cached context
subgraph per entity plus optional per-entity LLM summaries.

## Alignment params (`alignment_params`)

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `threshold` | `0.7` | Shared final decision threshold. In global mode it filters saved alignments; in local mode it still defines positive vs negative rationale polarity. |
| `cardinality` | `1` | Maximum targets per source in the final global alignment (`null` = unbounded). |
| `save_json/save_csv/save_stats_csv/append_stats_to_summary_csv` | `True` | Control which output artefacts are written after inference. |
| `review_low/review_high` | `0.5 / 0.8` | Review band used in explanations and reports. |

## Dataset params (`dataset_params`)

### Common dataset construction

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `num_workers` | `null` | Dataset worker count (`null` = let the runtime decide). |
| `filter_exact_matches` | `True` | Remove known exact lexical matches before semantic scoring. |
| `drop_exact_match_sources` | `False` | If `True`, remove every candidate for sources with an exact match. |
| `reasoner_timeout_secs` | `120` | Timeout for ontology reasoner construction. |
| `reasoner_force_hermit` | `False` | Force a HermiT retry after lighter reasoners. |
| `all_labels` | `True` | Use all label variants instead of a single best label. |
| `delimiter` | `"\n"` | Delimiter used when joining verbalised context triples. |
| `max_input_tokens_context` | `256` | Shared context-text ceiling. Legacy mode uses it directly; pair-adaptive mode treats it as a global encoder cap above per-channel limits. |
| `which` | … | Dataset metrics plotted after preprocessing (see YAML). |
| `candidate_share_k` | `2` | Number of top lexical candidates reused by candidate-share utilities. |
| `n_hops` | `1` | Shared raw graph neighborhood radius. In pair-adaptive mode this feeds the projected object-triple and literal pools used by `sim`, `diff`, and projected-literal attributes. In legacy mode it is the single-context graph depth. |

### Default pair-adaptive dataset controls

These are the main dataset knobs for the default
`PairAdaptiveSemanticScorer` + `PairAdaptiveContextDataset` path.

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `projection_include_literals` | `True` | If `True`, projected literal/annotation edges are also available to the pair-adaptive dataset. Ontology-native hierarchy extraction remains separate. |
| `hierarchy_max_depth` | `2` | Maximum ontology-native hierarchy depth collected per configured relation family. |
| `max_hierarchy_triples_per_family` | `6` | Maximum hierarchy triples retained per family and entity before pair-specific reranking. |
| `max_object_triples` | `48` | Maximum non-hierarchical object-property triples retained per entity for the similarity channel. |
| `max_diff_triples` | `24` | Maximum distinctive non-hierarchical triples retained per entity before pair-specific reranking for the difference channel. |
| `max_attr_items` | `12` | Maximum annotation/literal snippets retained per entity for the auxiliary attribute channel. |
| `hierarchical_relation_families` | see YAML | Named hierarchy families used by the default scorer. Each family declares ontology relation IRIs and lexical seeds so ontology-native edges can be routed into the hierarchy channel. |

### Legacy single-context extraction controls

These parameters mainly matter only if you switch back to
`model.name: SemanticScorer`, which uses one context subgraph per entity.

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `context_method` | `greedy` | Legacy context extraction strategy (`bfs` or `greedy`). |
| `best_path_method` | `dp` | Legacy path selection strategy inside context extraction. |
| `context_hop_penalty` | `0.1` | Penalty applied per graph hop in the legacy extractor. |
| `context_token_ratio/context_safety` | `1.3 / 0.8` | Token-budget heuristics for the legacy context extractor. |
| `only_taxonomy` | `False` | Use taxonomy-only extraction and fixed subclass templates in legacy mode. |
| `add_connectivity_bridges` | `True` | Add explanation-only bridge triples so legacy contexts remain connected. |
| `bridge_max_hops` | `null` | Optional cap on legacy bridge path length (`null` = unbounded). |

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
| `retrieval_strategy` | `hybrid` | Candidate retrieval mode. `hybrid` unions label embedding retrieval with lexical token/char retrieval over labels plus short annotation aliases; `primary_label` keeps the old primary-label embedding search. |
| `lexical_encoder_name` | `sentence-transformers/all-MiniLM-L6-v2` | Encoder used during lexical candidate generation. |
| `encode_batch_size` | `512` | Batch size for embedding labels. |
| `search_batch_size` | `4096` | Batch size for cosine similarity search. |
| `top_k` | `20` | Number of candidates retained per source. |
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
| `batch_size` | `128` | Candidate-pair batch size for the semantic scorer. |
| `num_workers` | `0` | DataLoader workers during inference. The default avoids fork stalls after JVM and Hugging Face tokenizer initialization. |
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
| `summary_profile` | Model profile for the summary task. In the default scorer this generates pair briefs; in legacy mode it generates one summary per entity. |
| `decision_profile` | Model profile for hosted binary decision scoring. Should point to a chat-logprob + `logit_bias` capable model if not local. |
| `rationale_profile` | Model profile for natural-language rationale generation. |
| `verbaliser_fallback_profile` | Fallback used only for verbaliser calls. |
| `fallback_profile` | Shared fallback for the summary task and rationale calls. |
| `decision_fallback_profile` | Fallback used when hosted decision scoring cannot provide usable logprobs. |

### Recommended routing usage

- Use smaller or cheaper models for verbalisation when instruction-following is good enough.
- Use stronger models for the summary task if ontology evidence is complex. In the default scorer this means better pair briefs.
- Use only models that support chat `logprobs`, `top_logprobs`, and `logit_bias` for hosted decision scoring.
- Use a separate rationale model if explanation quality matters more than raw speed.

### Suggested profile split

- Generation-oriented hosted profiles: `openrouter_claude_sonnet_46`, `openrouter_gpt5`, `openrouter_qwen3_235b`, `openrouter_gpt_oss_120b`
- Decision-oriented hosted profiles: `openrouter_qwen35_122b`, `openrouter_qwen3_235b`, `openrouter_llama33_70b`, `openrouter_deepseek_chat_v31`

Decision-oriented profiles are still probe-gated at runtime; the defaults above
are recommendations, not a hard guarantee that every OpenRouter provider route
will expose usable chat-side token logprobs for both decision labels.

## Model core behaviour (`model.params`)

The defaults below describe the current `PairAdaptiveSemanticScorer`. If you
switch back to `SemanticScorer`, the lexical encoder and most LLM controls
still apply, but the structure path collapses back to one joined context per
entity.

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `lexical_model_name` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | Encoder for label similarity. |
| `context_model_name` | `BAAI/bge-large-en-v1.5` | Encoder for structural text blocks. In pair-adaptive mode this means hierarchy, similarity, difference, and attribute evidence. |
| `llm_model_name` | `Qwen/Qwen2.5-7B-Instruct` | Local-only fallback model for the summary task, decision scoring, and rationales. |
| `fp16_inference` | `True` | Use fp16 where supported during local model inference. |
| `pooling_method` | `mean` | Token pooling for encoder outputs (`mean`, `max`, `sum`). |
| `label_pair_pooling` | `max` | Pooling over the label-pair similarity matrix (`mean` or `max`). |
| `use_lexical` | `True` | Enable lexical similarity. |
| `use_context` | `True` | Enable structural/context evidence. In pair-adaptive mode this means the hierarchy, similarity, difference, and attribute channels. |
| `use_llm` | `True` | Enable the LLM summary task, decision scoring, and rationales. |

## Model token limits and adaptive weighting

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `max_input_tokens_lexical` | `32` | Max tokens per label passed to the lexical encoder. |
| `max_input_tokens_hier` | `128` | Pair-adaptive: max tokens per joined hierarchy-family block. |
| `max_input_tokens_sim` | `256` | Pair-adaptive: max tokens per joined non-hierarchical similarity block. |
| `max_input_tokens_diff` | `256` | Pair-adaptive: max tokens per joined distinctive-evidence block. |
| `max_input_tokens_attr_item` | `96` | Pair-adaptive: max tokens per embedded attribute snippet. |
| `max_input_tokens_context` | `256` | Shared joined-context ceiling. Legacy mode uses it directly; pair-adaptive mode also enforces the channel-specific limits above. |
| `max_total_tokens_llm_summary` | `768` | Local-only prompt cap for the summary task. In the default scorer this is the pair-brief prompt. |
| `max_total_tokens_llm_decision` | `640` | Local-only prompt cap for decision scoring. In the default scorer the prompt includes the pair brief. |
| `max_total_tokens_llm_rationale` | `896` | Local-only prompt cap for rationale generation. In the default scorer the rationale sees the pair brief and final decision context. |
| `max_new_tokens_llm` | `64` | Shared generation cap for pair briefs, legacy summaries, and short rationale helpers. |
| `max_new_tokens_llm_rationale` | `256` | Rationale-specific generation cap. Use this when explanations need to be longer than briefs or summaries. |
| `tau` | `0.5` | Neutral midpoint used when estimating ambiguity and confidence. |
| `gamma` | `0.73` | Controls how sharply lexical-versus-structural confidence changes the fusion weight. |
| `beta` | `0.8` | Maximum contribution allowed from the LLM branch. |
| `tau_LLM` | `0.6` | LLM branch activates when ambiguity `U >= tau_LLM`. |
| `force_llm_summaries` | `False` | Always run the summary task even when the decision LLM is gated off. In the default scorer this means pair briefs. |

## Model LLM generation controls

These knobs have different scope depending on whether the backend is local or
hosted.

| Parameter | Default | Scope | Explanation |
|-----------|---------|-------|-------------|
| `llm_temperature` | `0.1` | shared for summary/rationale | Generation temperature for local and hosted summary-task/rationale calls. |
| `llm_top_p` | `0.9` | shared for summary/rationale | Nucleus sampling setting for local and hosted summary-task/rationale calls. |
| `llm_do_sample` | `False` | local-only | Enable sampling for local summary-task/rationale generation. |
| `llm_summary_batch_size` | `8` | shared for summary batching | Batch size for local summary-task generation and hosted concurrency cap for OpenRouter summary requests. |
| `llm_decision_batch_size` | `8` | shared for decision batching | Batch size for local decision scoring and hosted concurrency cap for OpenRouter decision requests. |
| `llm_rationale_batch_size` | `8` | shared for rationale batching | Batch size for local rationale generation and hosted concurrency cap for OpenRouter rationale requests. |
| `hosted_decision_labels` | `["A","B"]` | hosted-only | Ordered pair of hosted decision labels. The first label is treated as the positive class. |
| `hosted_decision_logit_bias` | `20.0` | hosted-only | Equal positive `logit_bias` added to both hosted decision labels to pull them into the top-logprob slice. |

### Hosted decision scoring

Hosted decision scoring does **not** use the summary-task/rationale sampling knobs.
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

Hosted summary-task, rationale, and verbaliser calls also use bounded concurrency on
OpenRouter. They are still one prompt per request at the API level, but the
runtime now issues multiple requests in parallel up to the configured
`llm_summary_batch_size`, `llm_rationale_batch_size`, or `batch_size_verbaliser`
limit.

## Explanations, calibration, and caches

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `return_explanations` | `True` | Include candidate-level explanations in model output. |
| `generate_llm_rationales` | `True` | Generate final positive/negative rationales. |
| `use_llm_calibration` | `False` | Enable post-hoc calibration on the decision probability. |
| `llm_calibration_a/llm_calibration_b/llm_calibration_info` | `null` | Optional preconfigured calibration metadata. |
| `cache_dir` | `null` | Cache root (`null` uses the default cache directory). |
| `cache_namespace` | `null` | Optional namespace to isolate cache files. |
| `dataset_signature` | `null` | Optional cache fingerprint override. |
| `persist_cache_to_disk` | `True` | Persist summary-task, rationale, and embedding caches between runs. |
| `max_cached_labels/max_cached_contexts/max_cached_summaries/max_cached_rationales` | `null` | Optional in-memory cache caps. In the default scorer, `max_cached_summaries` governs pair-brief cache entries. |

### Explanation shape in the default scorer

The default pair-adaptive scorer exposes explanations at two levels:

- top level: lexical, structural, and LLM contributions
- structural level: hierarchy, similarity, difference, and attribute channels
- top-level confidence fields include `s_label`, `S_struct`, `S_base`, `S_final`,
  `Q_struct`, `p_llm`, `U`, `U_ind`, and `U_dis`
- top-level importances include `I_label`, `I_struct`, and `I_llm`
- structural importances include `I_hier`, `I_sim`, `I_diff`, and `I_attr`
- structural evidence is also broken down into selected hierarchy triples,
  selected similarity triples, selected difference triples, and selected
  attributes with per-item support/importance metadata

Legacy `SemanticScorer` explanations keep the older lexical-versus-context
layout:

- top level: lexical, joined context, and LLM
- one joined context block per side rather than four structural subchannels
- no `S_struct`, `U_ind`, or `U_dis` split because ambiguity is not decomposed
  into indecision versus lexical-structural disagreement

## Final rationale semantics

Rationales are conditioned on the final outcome rather than only on the raw LLM
subdecision:

- **Global mode:** positive rationale means the pair survived both `threshold`
  and `cardinality` and is part of the saved alignment.
- **Local mode:** positive rationale means `S_final >= threshold`, even though
  all candidates remain in the ranking output.

## Optional second model (`second_model.params`)

Configured via the top-level `second_model` block; the selector runs after the
primary scorer in global mode and compares each source's candidate set jointly.
`SecondPassReranker` remains available as a backward-compatible alias.

| Parameter | Default | Explanation |
|-----------|---------|-------------|
| `enabled` | `True` | Enable the optional second-stage selector. |
| `strategy` | `calibrated_rank_accept` | Fit a lightweight supervised ranker and accept/reject calibrator from the CLI training reference (`-r/--training_reference_file`) when available; otherwise fall back to the heuristic selector. |
| `global_only` | `True` | Skip local candidate-file ranking tasks. |
| `replace_final_score` | `True` | Replace `S_final` with the selector score for threshold/cardinality. |
| `use_no_match` | `True` | Add an explicit source-level abstention risk. |
| `temperature` | `0.75` | Temperature for source-local competition probabilities and gap scaling. |
| `support_weight` | `0.60` | Weight on the original pairwise score; the remaining mass goes to the unweighted average of candidate competition, distinctive evidence, and equivalence safety. |
| `no_match_threshold` | `0.55` | Abstain when the unweighted average of ambiguity, weak support, generic evidence, and close competition reaches this value. |
| `calibration.enabled` | `auto` | Use supervised calibration only when a training reference path is wired into the model. |
| `calibration.min_positive_sources` | `50` | Minimum train sources whose gold target appears in the candidate set before fitting supervised weights. |
| `calibration.background_negative_weight` | `0.02` | Small pseudo-negative weight for non-training source groups, used only by the accept/reject calibrator. |
| `calibration.l2` | `0.001` | L2 regularisation for the rank and accept models. |
| `calibration.learning_rate` | `0.05` | Adam learning rate for the small torch models. |
| `calibration.max_epochs` | `200` | Maximum calibration training epochs. |
| `calibration.threshold_grid_step` | `0.005` | Grid resolution for choosing the accept/reject threshold under the configured acceptance objective. |
| `llm.ambiguity_margin` | `0.08` | Heuristic fallback: trigger arbitration near candidate ties or near the `NO_MATCH` threshold. |
| `llm.max_candidates` | `5` | Maximum top candidates shown in each LLM arbitration prompt. |
| `llm.trigger_acceptance_margin` | `0.025` | Calibrated mode: trigger LLM only near the learned accept/reject boundary. |
| `llm.trigger_rank_margin` | `0.03` | Calibrated mode: trigger LLM only when the top two rank probabilities are close. |
| `llm.min_confidence` | `0.75` | Minimum confidence required before applying a calibrated-mode LLM decision. |

In calibrated mode, the same LLM arbitration path is also used for model
disagreement: if the accept/reject calibrator says `NO_MATCH` but the top
candidate has broad pairwise, label, structural, and rank agreement, the
selector asks the LLM to verify the source instead of adding another
task-specific threshold knob. The evidence floor is derived from the existing
alignment threshold, `calibration.min_precision`, and the learned accept
threshold.

Use
[`exact/default_config.yaml`](/home/pgcotovio/Exact-OM/exact/default_config.yaml)
as the authoritative default whenever you build a custom experiment config.
