"""Exact-OM public package surface.

Delivery classes are loaded lazily so ontology and utility imports stay light and
Java-free.
"""

import warnings
from pathlib import Path
from typing import Any

EXACT_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = EXACT_DIR / "default_config.yaml"

__all__ = ["AlignmentRunner", "EvaluationRunner"]


def init_jvm(*_: Any, **__: Any) -> None:
    """Deprecated compatibility stub retained until 2.1."""

    raise RuntimeError("Exact-OM no longer needs Java; remove init_jvm calls")


def __getattr__(name: str) -> Any:
    if name in {"AlignmentRunner", "EvaluationRunner", "EvalutionRunner"}:
        from .delivery.api import AlignmentRunner, EvaluationRunner

        if name == "AlignmentRunner":
            return AlignmentRunner
        if name == "EvaluationRunner":
            return EvaluationRunner
        warnings.warn(
            "EvalutionRunner is deprecated; use EvaluationRunner instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return EvaluationRunner
    raise AttributeError(name)
