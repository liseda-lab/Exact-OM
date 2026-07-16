#!/usr/bin/env python3
"""Deprecated wrapper for ``exact-inspect bundle``."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated Exact Inspect bundle wrapper.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--bundle-name")
    parser.add_argument("--source-ontology", type=Path)
    parser.add_argument("--target-ontology", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--logging-level", default="INFO")
    parser.add_argument("--jvm-heap-size", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warnings.warn(
        "tools/prepare_study_visualizer_bundle.py moved to `exact-inspect bundle`.",
        DeprecationWarning,
        stacklevel=2,
    )
    if args.jvm_heap_size:
        warnings.warn(
            "--jvm-heap-size is ignored because Exact-OM no longer needs Java.",
            DeprecationWarning,
            stacklevel=2,
        )
    command = [
        "bundle",
        str(args.run_dir),
        str(args.bundle_dir),
        "--log-level",
        args.logging_level,
    ]
    for value, flag in (
        (args.analysis_dir, "--analysis-dir"),
        (args.config_path, "--config-path"),
        (args.bundle_name, "--bundle-name"),
        (args.source_ontology, "--source-ontology"),
        (args.target_ontology, "--target-ontology"),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    if args.overwrite:
        command.append("--overwrite")
    from exact_inspect.cli import main as inspect_main

    return inspect_main(command)


if __name__ == "__main__":
    sys.exit(main())
