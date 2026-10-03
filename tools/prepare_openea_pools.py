#!/usr/bin/env python3
"""Generate E23's independent pools with the existing retrieval recipe; no scoring or fits."""

import argparse
from pathlib import Path

from exact.experiments.openea_pools import prepare_pools
from exact.core.entities.configs.yaml_io import load_yaml_mapping
from exact.experiments.harness import deep_merge
from exact.experiments.preparation import prepare_campaign
from exact.utils.fitted_artifacts import freeze_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--campaign-bindings", type=Path)
    parser.add_argument("--campaign-output", type=Path)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "specs/experiments/campaign-v2.yaml",
    )
    args = parser.parse_args()
    if bool(args.campaign_bindings) != bool(args.campaign_output):
        parser.error("--campaign-bindings and --campaign-output must be supplied together")
    fragment = prepare_pools(args.config, args.prepared, args.output, device=args.device)
    print(args.output / "bindings-fragment.json")
    if args.campaign_bindings:
        bindings = deep_merge(dict(load_yaml_mapping(args.campaign_bindings)), fragment)
        freeze_json(args.output / "campaign-bindings.json", bindings)
        print(prepare_campaign(args.blueprint, args.config, bindings, args.campaign_output))


if __name__ == "__main__":
    main()
