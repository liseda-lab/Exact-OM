"""Configuration schema utilities exposed by ``exact config``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from exact.core.entities.configs.yaml_io import (
    dump_yaml_document,
    load_round_trip_mapping,
    migrate_round_trip_document,
    render_default_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exact config", description="Inspect and migrate configs")
    commands = parser.add_subparsers(dest="command", required=True)

    default_parser = commands.add_parser("default", help="render the complete default v2 config")
    default_parser.add_argument(
        "--format", choices=("yaml",), default="yaml", help="output serialization format"
    )
    default_parser.add_argument("-o", "--output", type=Path, help="write to a file instead of stdout")

    migrate_parser = commands.add_parser("migrate", help="migrate an unversioned v1 YAML config")
    migrate_parser.add_argument("input", type=Path, help="v1 or already-v2 YAML configuration")
    migrate_parser.add_argument("-o", "--output", type=Path, help="write v2 YAML to this file")
    return parser


def _write(text: str, output: Optional[Path], stream: TextIO) -> None:
    if output is None:
        stream.write(text)
        if text and not text.endswith("\n"):
            stream.write("\n")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "default":
        _write(render_default_yaml(), args.output, sys.stdout)
        return 0

    document = load_round_trip_mapping(args.input)
    migrated, report = migrate_round_trip_document(document)
    _write(dump_yaml_document(migrated), args.output, sys.stdout)
    report_stream = sys.stdout if args.output is not None else sys.stderr
    report_stream.write(report.render() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
