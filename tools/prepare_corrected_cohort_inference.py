#!/usr/bin/env python3
"""Prepare immutable reference-free configs for an already bound diagnostic cohort."""
import argparse
from pathlib import Path

from exact.experiments.corrected_worker import prepare_cohort_inference
from exact.experiments.mixed_scale import binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output", "cohort", "selection-freeze", "source", "target"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("global_alignment", "local_ranking"))
    parser.add_argument("--fitted-artifacts", type=Path)
    args = parser.parse_args()
    print(prepare_cohort_inference(args.config, args.output, cohort=binding(args.cohort),
        selection_freeze=binding(args.selection_freeze), source=args.source, target=args.target,
        mode=args.mode, fitted_artifacts=args.fitted_artifacts))


if __name__ == "__main__":
    main()
