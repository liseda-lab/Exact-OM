#!/usr/bin/env python3
"""Prepare the approved E23 OpenEA case without model execution or held-out label access."""

import argparse
from pathlib import Path

from exact.experiments.openea import download_archive, prepare_openea


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/exact-openea"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_openea(args.archive or download_archive(args.cache), args.output)
    print(
        f"Prepared {manifest['recipe']['task']}: {manifest['train_links']} train, {manifest['valid_links']} validation links; {args.output / 'bindings-fragment.json'}"
    )


if __name__ == "__main__":
    main()
