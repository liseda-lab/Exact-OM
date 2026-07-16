import warnings
from pathlib import Path

EXACT_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = EXACT_DIR / "default_config.yaml"

# Load default configuration file

from .utils.data import read_yaml

config = read_yaml(DEFAULT_CONFIG_PATH)

# Get Jpype init

from mowl import init_jvm

from .delivery.api import AlignmentRunner, EvaluationRunner

# Get AlignmentRunner


__all__ = ["AlignmentRunner", "EvaluationRunner", "init_jvm"]


def __getattr__(name: str):
    if name == "EvalutionRunner":
        warnings.warn(
            "EvalutionRunner is deprecated; use EvaluationRunner instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return EvaluationRunner
    raise AttributeError(name)
