import warnings

from .align import AlignmentRunner
from .eval import EvaluationRunner

__all__ = ["AlignmentRunner", "EvaluationRunner"]


def __getattr__(name: str):
    if name == "EvalutionRunner":
        warnings.warn(
            "EvalutionRunner is deprecated; use EvaluationRunner instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return EvaluationRunner
    raise AttributeError(name)
