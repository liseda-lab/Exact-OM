"""Run/resume explicitly scheduled frozen-case repair comparisons."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.study import load_schedule, run_study


def main(argv: list[str] | None = None) -> int:
    """Execute only the supplied v2 schedule; never fetch data or run a matcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", help="exact-repair/study-schedule/v2 JSON")
    parser.add_argument("--output", required=True, help="resumable result directory")
    parser.add_argument(
        "--max-attempts", type=int, default=2, help="cap interrupted attempts per scheduled arm"
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="declare matched model/profile controls, preserving unavailable arms",
    )
    parser.add_argument("--campaign-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    started = time.monotonic()
    cases, arms, seed = load_schedule(args.schedule)
    if args.matrix:
        from exact.repair.comparisons import freeze_controls, research_arms

        cases = tuple(freeze_controls(case) for case in cases)
        arms = research_arms()
    write_artifact(
        Path(args.output) / "loading.json",
        {
            "schema": "exact-repair/study-loading/v2",
            "schedule_hash": canonical_hash(Path(args.schedule).read_bytes().hex()),
            "loading_seconds": time.monotonic() - started,
            "scheduled_cases": len(cases),
        },
    )
    result = run_study(
        cases,
        args.output,
        arms=arms,
        seed=seed,
        max_attempts=args.max_attempts,
        campaign_seconds=args.campaign_seconds,
    )
    print(
        json.dumps(
            {
                "scheduled": result["scheduled"],
                "recorded": result["recorded"],
                "plan_hash": result["plan_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
