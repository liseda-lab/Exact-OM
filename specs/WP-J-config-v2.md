# WP-J — Config Schema v2 (+ migrator)

**Depends on**: WP-A, and lands **early in wave 2** — after the wave-1 WPs (B/C/E/I) have added
their keys under v1 names, so this WP folds them into the v2 shape once. WP-F/WP-G (wave 2)
write v2-native keys. **Size**: M.
**Behavior**: v1 configs keep working for one release via auto-migration; resolved settings are
identical (exhaustively tested).

## Context

`exact/default_config.yaml` (408 lines) has grown organically:

- `dataset_params` conflates three concerns: pair-adaptive structure extraction, **legacy-only**
  single-context controls (`context_method`, `best_path_method`, `context_hop_penalty`, …,
  explicitly labeled "Legacy" in comments), and **LLM verbaliser generation** params
  (`verbaliser_name`, `temperature`, `top_p`, …) that belong with the LLM config.
- Eval knobs live at top level (`k: [1,5,10]`).
- Three overlapping ways to declare the model stack: `model`, `second_model`, `model_chain`,
  plus a `second_pass_params` legacy shim (`ConfigModel._merge_registry_entry`).
- Output-artifact flags are scattered (`alignment_params.save_json/save_csv/...`,
  `plot_params`, `sanity_check_params`).
- **Every default exists twice**: in the YAML and in pydantic `Field(config[...])` defaults
  reading a module-global `config` dict loaded at import time (`exact/__init__.py`) — untestable
  with alternate defaults and a standing drift hazard.

## Design

### J1. v2 layout

```yaml
config_version: 2
run:        {seed, logging_level, use_file_cache}
data:       {track, task, root, revision, source, target, refs, candidates}   # WP-I
io:         {input_format, source_options, target_options, output_formats}    # WP-G
matching:   {threshold, cardinality, target_cardinality, review_low, review_high,
             entity_kinds,            # WP-F
             relation_prediction}     # WP-G
dataset:    {reasoner, num_workers, filter_exact_matches, drop_exact_match_sources,
             filter_ignored_alignment_classes, projection_include_literals,
             hierarchy_*, max_*_triples*, max_attr_items, n_hops, all_labels,
             hierarchical_relation_families: {...},
             legacy: {context_method, best_path_method, context_hop_penalty,
                      context_token_ratio, context_safety, only_taxonomy,
                      add_connectivity_bridges, bridge_max_hops}}
candidates: {...}                      # ex candidates_params
pipeline:                              # replaces model/second_model/model_chain
  - {name: PairAdaptiveSemanticScorer, params: {...}}
  - {name: CandidateSetSelector,       params: {...}}
inference:  {...}                      # ex inference_params
llm:
  profiles: {...}                      # ex llm_profiles
  routing:  {...}                      # ex llm_routing
  verbaliser: {model, max_new_tokens, temperature, top_p, top_k, do_sample, batch_size}
evaluation: {backends, k, bioml: {...}}   # WP-E + ex top-level k
output:     {save: {json, csv, stats_csv, append_stats_to_summary},
             plots: {...}, sanity_checks: {...}}
```

Principles: one section per lifecycle stage; anything marked "Legacy:" in current comments goes
under `dataset.legacy`; a key's section = the component that reads it. Keys removed rather than
moved: `reasoner_timeout_secs`, `reasoner_force_hermit` (WP-B deleted the machinery; migrator
drops them with a notice).

### J2. Single source of defaults

- Defaults and descriptions live **only** on the pydantic models
  (`Field(default=..., description=...)`). Delete the import-time global `config` dict from
  `exact/__init__.py` and every `Field(config[...])` back-reference.
- `exact/default_config.yaml` becomes a **generated artifact**: `exact config default
  [--format yaml]` renders the full commented default config from the models (descriptions →
  comments). Commit the generated file for discoverability with a `# GENERATED — edit the
  pydantic models` header; CI check asserts it's in sync (regenerate-and-diff).
- WP-H's generated config-reference docs read the same `description=` metadata — one source,
  three views (models, YAML template, docs site).

### J3. Versioning & migration

- `config_version: 2` required in v2 files. Files without it are treated as v1.
- **Loader**: v1 detected → migrate in-memory via a declarative key-map, log a one-line
  deprecation pointing at `exact config migrate`. Unknown keys (v1 or v2) are **errors**
  listing near-miss suggestions — today they're silently ignored, a known footgun.
- **Key-map**: a single table `V1_TO_V2: dict[str, str | Drop | Transform]` covering every v1
  key (the section moves above, plus transforms: `model`+`second_model`+`model_chain`+
  `second_pass_params` → `pipeline` list; verbaliser split out of `dataset_params`). The table
  doubles as documentation (WP-H migration page renders it).
- **CLI**: `exact config migrate old.yaml [-o new.yaml]` writes the v2 file with comments
  preserved where mechanical (ruamel.yaml round-trip; add as dep of the `cli` — it's small) and
  prints a report: moved / transformed / dropped-with-reason keys. The `exact config` group
  plugs into the subcommand scaffolding WP-I added to the `exact` entry point.
- **Fingerprint compat** (WP-C): `ConfigModel.fingerprint()` is computed over the **resolved v2
  model dump**, so a v1 file and its migration produce the same fingerprint — timing ledgers
  and cache fingerprints survive migration. Add a regression test.

### J4. Consumers to update

`ConfigModel.load_config` and all nested params models (`core/entities/configs/config.py`);
`resolve_dependencies`/`get_model_sequence` (now over `pipeline`); `tools/hparam_tuner.py` and
`tools/run_exact_job.py` (tuner YAMLs address keys by path — support v2 paths, migrate v1 trial
configs through the same map); `exp/` example configs regenerated; `tests/config_loading_test.py`
expanded as below.

### J5. Expose buried scoring/selection constants (audit F5)

Promote the hard-coded constants inventoried in `04-methodology-audit.md` F5 (channel blend
weights, per-relation caps, stability factor, retrieval fusion weights/DF ceiling, attribute
property weights, `U` constants, `s_attr` floor, alias caps) to config fields with the current
values as defaults — bit-identical by construction, verified by the full suite. This is a
declared prerequisite for the `specs/experiments/` sweeps. Place them under the owning v2
sections (`matching.channels.*`, `candidates.*`) with `description=` metadata.

## Tests

1. **Exhaustive round-trip**: build a v1 config exercising *every* v1 key (generate from the old
   model schema), migrate, assert the resolved `ConfigModel` equals the one from the equivalent
   hand-written v2 file — and that `fingerprint()` matches.
2. Loader: v1 auto-migration warning fires once; unknown key → error with suggestion;
   `config_version: 3` → clear "too new" error.
3. `exact config default | exact` (self-feed) resolves identical to no-config defaults.
4. Migrator CLI: report lists the `second_pass_params` transform and the dropped reasoner keys.
5. Generated-file sync check (CI target from J2).
6. Full hermetic suite green with defaults — proves resolved defaults unchanged.

## Out of scope

New behavior behind any key; changing key *semantics*; renaming registry component names;
touching the LLM profile schema internals (moved wholesale under `llm.profiles`).

## Acceptance criteria

1. Every v1 example config in the repo (`exp/` templates referenced by README, test fixtures)
   migrates cleanly; resolved settings identical (test 1).
2. `exact -y old_v1.yaml ...` still runs, warns once, produces identical outputs on fixtures.
3. Global import-time `config` dict is gone (`git grep "config\[" exact/core/entities/configs`
   clean; importing `exact` no longer reads the YAML).
4. Tuner + job-runner tools work with v2 paths (their tests updated).
5. WP-F/WP-G land their keys under v2 names without touching the migrator (their keys have no
   v1 ancestry).
