#!/usr/bin/env python3
"""Compare behavior-sensitive Exact-OM run outputs after a refactor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

BYTE_EXACT_ARTIFACTS = (
    "src2tgt.maps_global.tsv",
    "src2tgt.maps_local.tsv",
    "full_explanations.json",
)
RUN_STATS_ARTIFACT = "run_stats.json"
TIMING_KEYS = frozenset({"timing", "timings"})


def _single_artifact(root: Path, name: str) -> Path | None:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if not matches:
        return None
    if len(matches) > 1:
        rendered = ", ".join(str(path.relative_to(root)) for path in matches)
        raise ValueError(f"Multiple {name} artifacts below {root}: {rendered}")
    return matches[0]


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if str(key).lower() not in TIMING_KEYS
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def compare_run_outputs(baseline: Path, candidate: Path) -> list[str]:
    """Return human-readable differences; an empty list means parity."""
    baseline = baseline.resolve()
    candidate = candidate.resolve()
    differences: list[str] = []

    for name in BYTE_EXACT_ARTIFACTS:
        left = _single_artifact(baseline, name)
        right = _single_artifact(candidate, name)
        if left is None or right is None:
            missing = []
            if left is None:
                missing.append("baseline")
            if right is None:
                missing.append("candidate")
            differences.append(f"{name}: missing from {' and '.join(missing)}")
            continue
        if left.read_bytes() != right.read_bytes():
            differences.append(f"{name}: byte content differs")

    left_stats = _single_artifact(baseline, RUN_STATS_ARTIFACT)
    right_stats = _single_artifact(candidate, RUN_STATS_ARTIFACT)
    if left_stats is None or right_stats is None:
        missing = []
        if left_stats is None:
            missing.append("baseline")
        if right_stats is None:
            missing.append("candidate")
        differences.append(f"{RUN_STATS_ARTIFACT}: missing from {' and '.join(missing)}")
    else:
        left_payload = _without_timing(json.loads(left_stats.read_text(encoding="utf-8")))
        right_payload = _without_timing(json.loads(right_stats.read_text(encoding="utf-8")))
        if left_payload != right_payload:
            differences.append(f"{RUN_STATS_ARTIFACT}: non-timing content differs")

    return differences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare alignment TSVs and explanations byte-for-byte, and run_stats.json "
            "after removing timing blocks."
        )
    )
    parser.add_argument("baseline", type=Path, help="Baseline run directory")
    parser.add_argument("candidate", type=Path, help="Refactored run directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    differences = compare_run_outputs(args.baseline, args.candidate)
    if differences:
        for difference in differences:
            print(difference)
        return 1
    print("Run outputs match (timing blocks excluded from run_stats.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
