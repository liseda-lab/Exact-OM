# XR-E06 — Transfer to real ontology alignments

**Target:** XR-2.1, 30 September 2026. **Question:** RQ1, RQ2, RQ5. Requirements for new runs; archived XR-2 results remain separate.

## Data

Fresh generated test parents, Conference 2025 captured matcher outputs, Bio-ML 2024 subsets and Bio-ML 2026 whole pairs in separate cohorts. Complex-track examples are optional expression evidence, not preferred-repair truth.

## Comparisons

Generated-only versus training-side real-structure adaptation; structural-family, pair, ontology and matcher holdouts as different settings. Measure real support distributions before deciding that generated training covers complex conflicts.

## Measurements

Case/parent/pair/ontology counts; discovered support cardinalities and conflicts per mapping with extraction limits; semantic metrics supported by actual labels; verification coverage, time, unknowns and per-pair results.

## Acceptance and interpretation

All pair orientations, matcher outputs and derived samples stay in one fold. Reference absence is unspecified. Preserve named historical ekaw holdout as a comparison but disclose shared ontologies and new holdout after redesign. Published public counts are context, not measured repair outcomes. Do not combine biomedical editions or claim thousands of independent repair cases.

Shared splits, resources, failure accounting and statistics follow [02](../02-experimental-protocol.md). Dependency gates are in [08](../08-experiment-matrix-and-handoff.md); code changes and tests are in [12](../12-implementation-migration.md). Resolve every numerical setting in a new [run manifest](../protocol/README.md) before execution.
