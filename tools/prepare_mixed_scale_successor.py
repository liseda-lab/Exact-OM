#!/usr/bin/env python3
"""Prepare an offline corrected successor bundle; never edit the live queue."""
from __future__ import annotations

import argparse
import json

from exact.experiments.mixed_scale import binding, prepare_successor_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--public-inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-freeze")
    parser.add_argument("--deployments", help="JSON map from logical cell IDs to manifest bindings")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    deployments = None
    if args.deployments:
        with open(args.deployments) as stream:
            deployments = json.load(stream)
    result = prepare_successor_bundle(
        args.output, registry=args.registry, public_inputs=args.public_inputs,
        selection_freeze=binding(args.selection_freeze) if args.selection_freeze else None,
        deployments=deployments, source_revision=args.source_revision,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
