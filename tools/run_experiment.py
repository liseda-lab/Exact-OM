#!/usr/bin/env python3
"""Run the frozen Exact-OM screen/confirm experiment workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from exact.experiments.harness import load_suite_or_experiment, run_stage  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or inspect a staged Exact-OM paper experiment matrix."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=Path,
        help="One exp/experiments/EXX/exp.yaml experiment declaration.",
    )
    source.add_argument(
        "--suite",
        type=Path,
        help="A dependency-aware experiment suite manifest.",
    )
    parser.add_argument(
        "--stage",
        choices=("screen", "confirm"),
        required=True,
        help="Screen development arms or confirm a frozen selection.",
    )
    parser.add_argument(
        "--selection-record",
        type=Path,
        help="Frozen screen/selection.json (required for confirm; invalid for screen).",
    )
    parser.add_argument(
        "--confirmed-components-record",
        type=Path,
        help=(
            "Immutable confirmed component record required when E17 is runnable; "
            "accepted for both public stages."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPOSITORY_ROOT / "exp" / "results",
        help="Artifact root; defaults to exp/results.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum concurrent cells; GPU/LLM resource keys remain serialized.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only completed or partial cells with an identical provenance fingerprint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the full matrix without creating artifacts or running Exact.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        suite = load_suite_or_experiment(
            suite_path=args.suite,
            experiment_path=args.config,
        )
        selection_path = run_stage(
            suite,
            stage=args.stage,
            output_root=args.output_root,
            jobs=args.jobs,
            resume=args.resume,
            workdir=_REPOSITORY_ROOT,
            selection_record_path=args.selection_record,
            confirmed_components_record_path=args.confirmed_components_record,
            dry_run=args.dry_run,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"run_experiment: {exc}", file=sys.stderr)
        return 2

    if selection_path is not None:
        print(f"Selection record: {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
