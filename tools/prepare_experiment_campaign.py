#!/usr/bin/env python3
"""Prepare strict v2 campaign declarations without running experiments."""
import argparse
from pathlib import Path

from exact.core.entities.configs.yaml_io import load_yaml_mapping
from exact.experiments.preparation import prepare_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint", type=Path, default=Path("specs/experiments/campaign-v2.yaml")
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        prepare_campaign(
            args.blueprint, args.base_config, dict(load_yaml_mapping(args.bindings)), args.output
        )
    )


if __name__ == "__main__":
    main()
