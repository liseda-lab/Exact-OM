import os
import tarfile
import urllib.request
from pathlib import Path

MATCHA_DL_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = MATCHA_DL_DIR / "default_config.yaml"

# If matchaJar and dependencies don't exist, download them.

if not (MATCHA_DL_DIR / "impl/matcha/matcha/").exists():

    print("Matcha-DL jar and dependencies not found. Downloading...")

    from .core.values import MATCHA_URL
    from .utils.data import DataDownloader

    # Create an instance of DataDownloader
    downloader = DataDownloader(str(MATCHA_DL_DIR / "impl/matcha/"))
    downloader.download_matcha(MATCHA_URL)

## Load default configuration file

from .utils.data import read_yaml

config = read_yaml(DEFAULT_CONFIG_PATH)

# Get Jpype init

from mowl import init_jvm

# Get AlignmentRunner

from .delivery.api import AlignmentRunner, EvalutionRunner
