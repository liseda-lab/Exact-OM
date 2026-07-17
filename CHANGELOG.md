# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release completes engineering work packages A–L. Methodology experiments remain
separate and are not included in these changes.

### Added

- Python 3.10–3.12 CPU packaging, hermetic test markers, import-boundary checks, docstring
  coverage, wheel smoke tests, and fixture performance budgets (WP-A).
- A native OWL parser and indexed `KnowledgeSource` implementation, asserted hierarchy
  reasoner/plugin seam, conformance fixtures, and backend parity tests (WP-B).
- An append-safe `timings.json` session ledger with cache/resume status, atomic writes,
  cumulative and per-session summaries, and a derived legacy timing view (WP-C).
- Registered `builtin` and optional `bioml` evaluation backends, namespaced JSON reports,
  evaluator provenance, typed BioKG metrics, and reference/candidate hash guards (WP-E).
- Kind-aware candidate generation, evidence, selection, explanations, and evaluation for
  classes, object properties, data properties, and individuals (WP-F).
- OWL, RDF, and descriptor-driven CSV-KG source adapters; alignment writer plugins for global
  TSV, local TSV, OAEI RDF, typed TSV, and JSON; hierarchy-based relation typing; a BioKG
  profile and submission validator (WP-G).
- A strict MkDocs Material site with generated Pydantic configuration and argparse references,
  API reference, migration/contribution guides, and focused workflow documentation (WP-H).
- Declarative HTTP and optional Hugging Face dataset tracks, safe archive transforms,
  `datasets.lock.json`, licensed-input guidance, custom descriptors/plugins, and
  `exact data list|pull|verify|status` (WP-I).
- Strict version-2 configuration models, generated defaults, a declarative v1 migrator with
  reports and suggestions, stable resolved fingerprints, and `exact config default|migrate`
  (WP-J).
- The optional `exact_inspect` package and `exact-inspect open|serve|bundle` CLI, packaged
  static frontend resolution, API-only fallback, portable bundles, and renamed Render assets
  (WP-K).
- Layout-v2 runs with `RunLayout`, `RunReader`, checksummed `run_manifest.json`, compressed
  source-indexed explanations, crash-safe overlays, checkpoint retention, and
  `exact run info|export|clean` (WP-L).
- One immutable `pyowl-core` snapshot per OWL source, shared structural views, a delegated
  OWL2Vec* projector, asserted/pyELK/pyHermiT hierarchy adapters, verified-wire worker
  isolation, and path-free `ontology_stack` run provenance (WP-M).

### Changed

- Split trainer orchestration, checkpointing, audit I/O, overlays, and rationales; split the
  selector into grouping, feature, calibration, and acceptance modules; decomposed user-study
  analysis; and centralized shared scorer helpers while preserving registry names (WP-D).
- Replaced protocol-shaped actions with `run_alignment(...)` and `run_evaluation(...)`, and
  consolidated CLI/API validation and invocation assembly under `exact.delivery.common`
  (WP-D).
- Made mapping tie-breaking deterministic by source/target IRI and moved LLM routing to
  `exact.llm.routing` (WP-D).
- Replaced the OWL-API/mOWL runtime with `py-horned-owl` plus an RDFLib fallback. Legacy heap
  arguments remain accepted, warn, and are ignored (WP-B).
- Replaced the transitional `py-horned-owl` parser with the compatible `pyowl-core` and
  `pyowl2vec-star-projector` 0.1 lines. Optional pyELK/pyHermiT distributions moved behind the
  independent `reasoning` extra; RDFLib now serves only generic RDF/OAEI formats (WP-M).
- Moved FastAPI, Uvicorn, and pydantic-settings behind the `viz` extra; Hugging Face tracks,
  upstream Bio-ML evaluation, and documentation tooling remain independent extras (WP-E/I/K).
- Exposed pair-adaptive channel, retrieval-fusion, alias, and uncertainty constants in the v2
  configuration with behavior-preserving defaults (WP-J).
- New runs write resolved configuration, alignments, evaluation, statistics, plots,
  explanations, caches, and checkpoints through a single version-aware artifact layout (WP-L).

### Deprecated

- `EvalutionRunner` in favor of `EvaluationRunner`, and `exact.utils.paths` in favor of
  `exact.utils.graph_search`; both compatibility imports remain through 2.0 (WP-A).
- `AlignmentAction.run` and `EvaluationAction.run` in favor of the plain action functions,
  and `exact.utils.llm_routing` in favor of `exact.llm.routing` (WP-D).
- `bioml-eval` as the historical command name; `exact-eval` is preferred and both remain
  installed (WP-A/E).
- `exact.init_jvm`, heap flags, and `EXACT_STUDY_JVM_HEAP_SIZE`; compatibility surfaces only
  report how to migrate (WP-B).
- Unversioned v1 YAML. It is migrated in memory for 2.0; use `exact config migrate` to write a
  reviewed v2 file (WP-J).
- `study_visualizer_runtime`, `exact-study-viz`, and `EXACT_STUDY_*` in favor of
  `exact_inspect`, `exact-inspect`, and `EXACT_INSPECT_*` (WP-K).
- Written `full_explanations.json` and `times.txt` as sources of truth. Indexed explanation
  shards and `timings.json` replace them; derived compatibility exports remain available
  through 2.0 (WP-C/L).

### Fixed

- Restored complete metric registration and the `[1, 5, 10]` evaluation default.
- Replaced unsafe candidate-file `eval` calls with `ast.literal_eval`.
- Repaired pytest collection, package inclusion, console-script installation, and stale
  Matcha-DL deployment/test artifacts (WP-A).
- Restricted learned LLM calibration samples to training-reference sources and rejected
  mismatched reference/candidate provenance (WP-A/E).
- Prevented cached, resumed, or skipped stages from being counted as fresh compute and retained
  cumulative timing across repeated runs (WP-C).
- Preserved source groups across entity kinds during selector calibration and prevented
  cross-kind candidates from entering retrieval or evaluation (WP-F).

### Removed

- Runtime dependencies and imports for mOWL, JPype, OWL-API, Gensim, and process-based
  reasoner management (WP-B).
- The direct `py-horned-owl` dependency, Exact-owned parsed ontology graph, local projection
  engine, and path/pickle reasoner handoff (WP-M).
- The unverified `data/get_data.py` downloader in favor of pinned track providers (WP-I).

[Unreleased]: https://github.com/liseda-lab/Exact-OM/compare/v1.0.0...HEAD
