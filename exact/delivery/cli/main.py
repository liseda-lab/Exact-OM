"""Public Exact dispatcher; imports heavy backends only when needed."""
from __future__ import annotations

import argparse
import importlib
import sys
from importlib.metadata import version
from typing import Optional, Sequence

COMMANDS = {
    "align": "Match ontologies with the configurable native pipeline",
    "score": "Generate and checkpoint native pair evidence",
    "decisions": "Prepare, cache and apply gated Elephant decisions",
    "acceptance": "Fit and apply calibrated global acceptance",
    "rank": "Fit and apply original-query local ranking",
    "eval": "Evaluate a submission",
    "config": "Print or migrate configuration",
    "data": "Discover, retrieve and verify datasets",
    "run": "Inspect, export or clean run artifacts",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch a public command, preserving original flat alignment flags."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="exact", description="Exact-OM: explainable ontology matching")
    parser.add_argument("--version", action="version", version=version("Exact-OM"))
    parser.add_argument("command", nargs="?", choices=COMMANDS,
                        help="; ".join(f"{k}: {v}" for k, v in COMMANDS.items()))
    if not arguments or arguments[0] in {"-h", "--help", "--version"}:
        parser.parse_args(arguments)
        if not arguments:
            parser.print_help()
        return 0
    command = arguments.pop(0) if arguments[0] in COMMANDS else "align"
    if command == "align" and arguments and not arguments[0].startswith("-"):
        parser.error(f"unknown command: {arguments[0]}")
    module = importlib.import_module(f"exact.delivery.cli.{command}")
    try:
        return module.main(arguments)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"exact: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
