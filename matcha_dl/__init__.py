import os
import tarfile
import urllib.request
from pathlib import Path

import yaml

MATCHA_DL_DIR = Path(__file__).parent

# If matchaJar and dependencies don't exist, download them.

if not (MATCHA_DL_DIR / "impl/matcha/matcha/").exists():

    print("Matcha-DL jar and dependencies not found. Downloading...")

    from .core.values import MATCHA_URL
    from .utils.data import DataDownloader

    # Create an instance of DataDownloader
    downloader = DataDownloader(str(MATCHA_DL_DIR / "impl/matcha/"))
    downloader.download_matcha(MATCHA_URL)

## Load default configuration file


def read_yaml(file_path: Path):
    with open(str(file_path), "r") as file:
        return yaml.safe_load(file)


def get_config_path():
    current_file = Path(__file__)
    parent_directory = current_file.parent
    config_path = parent_directory / "default_config.yaml"
    return config_path


## get current directory

config = read_yaml(get_config_path())

# Get Jpype init

from mowl import init_jvm

# Get AlignmentRunner

from .delivery.api import AlignmentRunner
