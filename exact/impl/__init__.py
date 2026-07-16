"""Concrete Exact components with an explicit, lazy bootstrap.

Keeping this package initializer light is important: importing one optional
integration (for example an evaluator) must not import transformer models and
dataset backends as a side effect.
"""

from __future__ import annotations

import importlib
from typing import Any

_BOOTSTRAPPED = False
_SUBMODULES = {"datasets", "evaluators", "metrics", "models", "seed", "trainer"}


def bootstrap_components() -> None:
    """Import and register all in-tree pipeline components once."""

    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for name in ("datasets", "metrics", "models", "seed", "trainer", "evaluators"):
        importlib.import_module(f"{__name__}.{name}")

    from exact.core.entities.registry import ComponentRegistry, ComponentType

    dependencies = {
        "SemanticScorer": ("ContextDataset", "SemanticAlignmentRunner"),
        "PairAdaptiveSemanticScorer": (
            "PairAdaptiveContextDataset",
            "SemanticAlignmentRunner",
        ),
        "CandidateSetSelector": (
            "PairAdaptiveContextDataset",
            "SemanticAlignmentRunner",
        ),
        "SecondPassReranker": (
            "PairAdaptiveContextDataset",
            "SemanticAlignmentRunner",
        ),
    }
    for model_name, (dataset_name, trainer_name) in dependencies.items():
        ComponentRegistry.register_dependency(
            model_name,
            {
                ComponentType.DATASET: dataset_name,
                ComponentType.TRAINER: trainer_name,
            },
        )
    _BOOTSTRAPPED = True


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)


__all__ = ["bootstrap_components"]
