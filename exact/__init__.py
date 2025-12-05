from pathlib import Path

EXACT_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = EXACT_DIR / "default_config.yaml"

## Load default configuration file

from .utils.data import read_yaml

config = read_yaml(DEFAULT_CONFIG_PATH)

# Get Jpype init

from mowl import init_jvm

# Get AlignmentRunner

from .delivery.api import AlignmentRunner, EvalutionRunner
