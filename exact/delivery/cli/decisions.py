"""Prepare, run and apply source-gated local Elephant-1 decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from exact.decisions import apply, prepare, run
from exact.decisions.pipeline import POLICIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exact decisions", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "prepare", help="Freeze the label-free source gate and pair requests"
    )
    plan.add_argument("--candidates", type=Path, required=True, help="JSONL, contiguous by Src")
    plan.add_argument(
        "--sources", type=Path, required=True, help="Full eligible source IRIs, one per line"
    )
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--policy", choices=POLICIES, default="global")
    plan.add_argument(
        "--anchors", type=Path, help="Original anchor JSON sidecar; not corrected global"
    )
    inference = commands.add_parser("run", help="Resume local decisions or verify cached responses")
    inference.add_argument("--plan", type=Path, required=True)
    inference.add_argument("--cache", type=Path, required=True)
    inference.add_argument("--model-dir", type=Path)
    inference.add_argument("--device", default="cuda:0")
    inference.add_argument("--cache-only", action="store_true")
    application = commands.add_parser("apply", help="Apply clip(beta*U) mixing to admitted pairs")
    application.add_argument("--candidates", type=Path, required=True)
    application.add_argument("--plan", type=Path, required=True)
    application.add_argument("--cache", type=Path, required=True)
    application.add_argument("--output", type=Path, required=True)
    application.add_argument("--beta", type=float, default=0.8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = {
            "plan": str(
                prepare(
                    args.candidates,
                    args.sources,
                    args.output,
                    policy=args.policy,
                    anchors_path=args.anchors,
                )
            )
        }
    elif args.command == "run":
        result = run(
            args.plan,
            args.cache,
            model_dir=args.model_dir,
            device=args.device,
            cache_only=args.cache_only,
        )
    else:
        result = apply(args.candidates, args.plan, args.cache, args.output, beta=args.beta)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
