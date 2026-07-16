"""Inspect, export, and safely clean Exact run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from exact.runs import ExplanationStore, RunLayout, RunReader, clean_run
from exact.runs.gc import format_bytes


def _write_records(
    records: Iterable[dict[str, Any]],
    output: Path,
    format: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        with output.open("w", encoding="utf-8") as stream:
            json.dump(
                list(records),
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        return
    if format == "jsonl":
        with output.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                )
                stream.write("\n")
        return
    materialized = list(records)
    fieldnames = sorted({key for record in materialized for key in record})
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in materialized:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in record.items()
                }
            )


def _info(args: argparse.Namespace) -> int:
    reader = RunReader.open(Path(args.run_dir))
    manifest = reader.manifest()
    artifacts = list(manifest.get("artifacts") or [])
    total_bytes = sum(int(artifact.get("bytes", 0) or 0) for artifact in artifacts)
    kinds = Counter(str(artifact.get("kind", "unknown")) for artifact in artifacts)
    sessions = [str(item) for item in manifest.get("sessions") or []]

    print(f"Run: {reader.layout.root}")
    print(
        f"Layout: v{manifest.get('layout_version', reader.layout.version)} "
        f"(manifest schema v{manifest.get('schema_version', 'unknown')})"
    )
    print(f"Exact: {manifest.get('exact_version', 'unknown')}")
    print(f"Sessions: {len(sessions)}" + (f" ({', '.join(sessions)})" if sessions else ""))
    print(f"Artifacts: {len(artifacts)} ({format_bytes(total_bytes)})")
    for kind, count in sorted(kinds.items()):
        kind_bytes = sum(
            int(artifact.get("bytes", 0) or 0)
            for artifact in artifacts
            if artifact.get("kind") == kind
        )
        print(f"  {kind}: {count} ({format_bytes(kind_bytes)})")
    if manifest.get("synthesized"):
        print("Manifest: synthesized from a layout-v1 run")
    return 0


def _clean(args: argparse.Namespace) -> int:
    result = clean_run(
        Path(args.run_dir),
        keep_resume=bool(args.keep_resume),
        include_dataset_cache=bool(args.all),
        dry_run=bool(args.dry_run),
    )
    verb = "Would remove" if result.dry_run else "Removed"
    print(f"{verb} {result.count} file(s), freeing {format_bytes(result.bytes)}.")
    for path in result.paths:
        print(f"  {path}")
    return 0


def _export(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    suffix = "jsonl" if args.format == "jsonl" else args.format
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / f"explanations_export.{suffix}"
    )
    layout = RunLayout.open(run_dir)
    if layout.explanation_index_path.is_file():
        store = ExplanationStore(layout.explanations_dir)
        store.export(output, format=args.format, src_iri=args.src)
    else:
        reader = RunReader.open(run_dir)
        records: Iterable[dict[str, Any]] = (
            reader.explanations_for(args.src) if args.src else reader.iter_explanations()
        )
        _write_records(records, output, args.format)
    print(f"Exported explanations to {output}")
    return 0


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``info``, ``clean``, and ``export`` commands to ``parser``."""

    commands = parser.add_subparsers(dest="run_command", required=True)

    info = commands.add_parser("info", help="Show a run manifest and artifact sizes")
    info.add_argument("run_dir", help="Run directory")
    info.set_defaults(handler=_info)

    clean = commands.add_parser("clean", help="Remove known resume artifacts safely")
    clean.add_argument("run_dir", help="Run directory")
    clean.add_argument(
        "--keep-resume",
        action="store_true",
        help="Keep checkpoint and resume artifacts",
    )
    clean.add_argument("--dry-run", action="store_true", help="Print without deleting")
    clean.add_argument(
        "--all",
        action="store_true",
        help="Also remove dataset-cache files listed in the manifest",
    )
    clean.set_defaults(handler=_clean)

    export = commands.add_parser("export", help="Generate a derived run-artifact view")
    export.add_argument("run_dir", help="Run directory")
    export.add_argument(
        "--what",
        choices=["explanations"],
        default="explanations",
        help="Artifact view to export",
    )
    export.add_argument("--src", help="Export one source IRI")
    export.add_argument(
        "--format",
        choices=["json", "jsonl", "csv"],
        default="json",
    )
    export.add_argument("--output", help="Output file (defaults inside the run directory)")
    export.set_defaults(handler=_export)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exact run", description=__doc__)
    return configure_parser(parser)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "configure_parser", "main"]
