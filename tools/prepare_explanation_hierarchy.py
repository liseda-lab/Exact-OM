#!/usr/bin/env python3
"""Prepare indexed roots and optional native inference in a new context package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact_inspect.context import OntologyContext  # noqa: E402
from exact_inspect.contracts import VisibilityPolicy  # noqa: E402
from exact_inspect.navigation_prepare import prepare_navigation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--infer", action="store_true", help="Classify the admitted scope with native HermiT"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="VisibilityPolicy JSON; defaults to the physically filtered context policy",
    )
    parser.add_argument(
        "--source-ontology",
        type=Path,
        help="Optional admitted source for logical axioms whose original bytes were redacted",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--memory-mib", type=int, default=2048)
    args = parser.parse_args()
    policy = VisibilityPolicy.model_validate_json(args.policy.read_bytes()) if args.policy else None
    snapshot = None
    if args.source_ontology:
        import pyowl_core as core

        snapshot = core.load_snapshot(
            args.source_ontology,
            options=core.LoadOptions(
                backend=core.BackendPreference.NATIVE,
                imports=core.ImportPolicy.RESOLVE_STRICT,
                offline=True,
                limits=core.ParseLimits(
                    deadline_seconds=args.timeout_seconds,
                    max_memory_bytes=args.memory_mib * 1024**2,
                ),
            ),
        )
    prepared = prepare_navigation(
        OntologyContext(args.context),
        args.destination,
        policy=policy,
        infer=args.infer,
        source_snapshot=snapshot,
        timeout_seconds=args.timeout_seconds,
        max_memory_bytes=args.memory_mib * 1024**2,
    )
    print(json.dumps(prepared.navigation_metadata(policy), sort_keys=True, indent=2))
    return (
        1
        if args.infer and prepared.navigation_metadata(policy)["inference"]["status"] != "available"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
