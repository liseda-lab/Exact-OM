#!/usr/bin/env python3
"""Prepare full native global populations or original-query local inference shards."""
import argparse
from pathlib import Path

from exact.experiments.public_inference import prepare_public_inference
from exact.experiments.submission import TRACKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output", "source", "target"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--public-candidates", type=Path)
    args = parser.parse_args()
    print(
        prepare_public_inference(
            args.config,
            args.output,
            source=args.source,
            target=args.target,
            track=args.track,
            public_candidates=args.public_candidates,
        )
    )


if __name__ == "__main__":
    main()
