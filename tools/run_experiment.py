#!/usr/bin/env python3
"""Run the frozen Exact-OM screen/confirm experiment workflow."""

from __future__ import annotations

import argparse
import json
import os
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
    source.add_argument("--campaign", type=Path, help="A strict v2 campaign lock.")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Read an OpenRouter key into the worker environment; never write it to artifacts.",
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Write strict run declarations without executing them.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Import verified stage artifacts from an earlier campaign directory.",
    )
    parser.add_argument(
        "--repair-record", type=Path, help="Dependency-scoped implementation repair record."
    )
    parser.add_argument(
        "--reuse-plan-only",
        action="store_true",
        help="Write the repair/reuse plan without executing work.",
    )
    parser.add_argument(
        "--stop-after-checkpoint", action="store_true", help="Stop at the next durable checkpoint."
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
        if args.api_key_file is not None:
            key = args.api_key_file.expanduser().read_text(encoding="utf-8").strip()
            if not key:
                raise ValueError("API-key file is empty")
            os.environ["OPENROUTER_API_KEY"] = key
        if args.campaign:
            from exact.experiments.campaign import (
                campaign_plan,
                execute_campaign,
                materialize_campaign,
            )

            if args.dry_run:
                plan = campaign_plan(args.campaign, stage=args.stage)
                print(json.dumps(plan, indent=2, sort_keys=True))
                return (
                    2 if plan["budget_errors"] or any(row["issues"] for row in plan["rows"]) else 0
                )
            if args.materialize_only:
                suite = materialize_campaign(
                    args.campaign, args.output_root / "declarations" / args.stage, stage=args.stage
                )
                print(
                    f"Materialized {len(suite.sources)} strict declarations in {args.output_root / 'declarations' / args.stage}"
                )
                return 0
            return execute_campaign(
                args.campaign,
                stage=args.stage,
                output_root=args.output_root,
                workdir=_REPOSITORY_ROOT,
                jobs=args.jobs,
                resume=args.resume,
                resume_from=args.resume_from,
                repair_record=args.repair_record,
                reuse_plan_only=args.reuse_plan_only,
                stop_after_checkpoint=args.stop_after_checkpoint,
            )
        if any(
            (
                args.materialize_only,
                args.resume_from,
                args.repair_record,
                args.reuse_plan_only,
                args.stop_after_checkpoint,
            )
        ):
            raise ValueError("v2 preparation/recovery flags require --campaign")
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
