# Proposed OAEI-Bio-ML-eval participant API

Exact integrates the organizer library through one capability-discovery adapter. This document
records the small, pure-function surface that would let participant tools avoid depending on
the library's CLI or filesystem side effects.

Upstream inspected revision: `900ba7f242f254d08b0e01e6826975c61d0438b2`.

## Proposed stable surface

```python
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

PathLike = str | Path

def evaluate_equivalence(
    predictions: PathLike | Sequence[tuple[str, str]],
    reference: PathLike | Sequence[tuple[str, str]],
    *,
    excluded_entities: set[str] | None = None,
) -> dict[str, float]: ...

def evaluate_ranking(
    rankings: PathLike | Mapping[Hashable, Sequence[str]],
    gold: PathLike | Mapping[Hashable, str],
    *,
    hits_ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]: ...

def evaluate_typed(
    predictions: PathLike | Sequence[Mapping[str, Any]],
    answers: PathLike | Sequence[Mapping[str, Any]],
    **options: Any,
) -> dict[str, float]: ...

def structural_coherence(
    predictions: PathLike | Sequence[tuple[str, str, str]],
    source_hierarchy: PathLike | Sequence[Mapping[str, str]],
    target_hierarchy: PathLike | Sequence[Mapping[str, str]],
) -> dict[str, float]: ...
```

Each function should:

- return a metric dictionary without printing or writing files;
- accept either paths or already parsed values;
- avoid Java and subprocess startup for the structural proxy;
- raise `NotImplementedError` for a deliberately unavailable capability;
- expose the package version through `oaei_bioml_eval.__version__`.

## Current adapter mapping

The pinned revision already supplies `score_global_files`, `local_ranking_metrics`, and
`typed.report.score_files`. Exact calls those behind `BioMLApi`. Exact's local TSV stores scored
`(target, score)` values in one cell, so it is parsed in Exact and passed to the pure
`local_ranking_metrics` function; passing it to `score_local_files` would treat tuple literals as
identifiers. The official coherence path starts ROBOT/Java and is intentionally not called.
`score_structural_proxy_files` is capability-probed and currently reported as skipped because its
rule set remains a stub.

## Integration deviation

This repository commits the proposal but does not create an upstream issue automatically. Filing
an issue or pull request changes an external project and therefore requires explicit maintainer
authorization.
