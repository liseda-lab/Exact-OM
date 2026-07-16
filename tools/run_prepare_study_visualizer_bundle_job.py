#!/usr/bin/env python3
"""Deprecated YAML-job wrapper for ``exact-inspect bundle --job-config``."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated Exact Inspect YAML job wrapper.")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sbatch-script", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings.warn(
        "tools/run_prepare_study_visualizer_bundle_job.py moved to "
        "`exact-inspect bundle --job-config`.",
        DeprecationWarning,
        stacklevel=2,
    )
    command = ["bundle", "--job-config", str(args.run_config)]
    if args.dry_run:
        command.append("--dry-run")
    if args.sbatch_script:
        command.extend(("--sbatch-script", str(args.sbatch_script)))
    from exact_inspect.cli import main as inspect_main

    return inspect_main(command)


if __name__ == "__main__":
    sys.exit(main())
