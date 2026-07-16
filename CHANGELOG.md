# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CPU-compatible packaging and CI quality gates for Python 3.10 and 3.12.
- The `exact-eval` console-script alias.
- Packaging for the existing study visualizer runtime.
- Registered `builtin` and optional `bioml` evaluation backends with canonical JSON reports.
- Reference/candidate SHA-256 provenance and selector-calibration hash guards.
- A Java-free OWL parser, indexed knowledge-source backend, asserted reasoner plugin seam,
  parity fixtures, and performance regression harness.

### Changed

- Renamed `EvalutionRunner` to `EvaluationRunner` and `exact.utils.paths` to
  `exact.utils.graph_search`, retaining deprecation shims.
- Consolidated shared formatting and numeric helpers.
- Restricted learned LLM calibration samples to training-reference sources.
- Kept historical builtin evaluation CSVs while namespacing multi-backend results.
- Replaced the OWL-API/mOWL runtime with `py-horned-owl` plus an RDFLib parser fallback;
  legacy heap arguments remain accepted with deprecation warnings and are ignored.

### Fixed

- Restored complete metric registration and the `[1, 5, 10]` evaluation default.
- Replaced unsafe candidate-file `eval` calls with `ast.literal_eval`.
- Repaired pytest collection and removed stale Matcha-DL deployment/test artifacts.

[Unreleased]: https://github.com/liseda-lab/Exact-OM/compare/v1.0.0...HEAD
