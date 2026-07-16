"""Dataset-track retrieval commands for the unified ``exact`` CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from exact.tracks import get_track, list_tracks, provider_from_descriptor
from exact.tracks.lockfile import read_lock
from exact.tracks.provider import TaskLayout, TrackProvider, VerificationReport


def build_parser() -> argparse.ArgumentParser:
    """Build the ``exact data`` subcommand parser."""

    parser = argparse.ArgumentParser(
        prog="exact data", description="Materialize and verify Exact dataset tracks"
    )
    commands = parser.add_subparsers(dest="data_command", required=True)

    list_parser = commands.add_parser("list", help="List built-in and plugin tracks")
    list_parser.add_argument("--descriptor", type=Path, help="Also list a custom YAML descriptor")

    pull = commands.add_parser("pull", help="Materialize a track or one track task")
    pull.add_argument("target", help="TRACK or TRACK/TASK")
    pull.add_argument("--root", type=Path, default=Path("data"), help="Dataset root")
    pull.add_argument("--revision", help="Override the descriptor revision for this pin")
    pull.add_argument(
        "--update",
        action="store_true",
        help="Explicitly repin an already materialized task to the current upstream revision",
    )
    pull.add_argument("--descriptor", type=Path, help="Custom YAML descriptor")

    for name, description in (
        ("verify", "Verify hashes and mutable upstream revisions"),
        ("status", "Report the four-state status of materialized tasks"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("target", nargs="?", help="Optional TRACK or TRACK/TASK")
        command.add_argument("--root", type=Path, default=Path("data"), help="Dataset root")
        command.add_argument("--descriptor", type=Path, help="Custom YAML descriptor")
    return parser


def _provider_and_task(
    target: str, descriptor: Optional[Path]
) -> tuple[TrackProvider, Optional[str]]:
    if descriptor is not None:
        provider = provider_from_descriptor(descriptor.expanduser().resolve())
        if "/" not in target:
            if target == provider.name:
                return provider, None
            return provider, target
        track, task = target.split("/", 1)
        if track != provider.name:
            raise ValueError(
                f"Target track {track!r} does not match descriptor name {provider.name!r}"
            )
        return provider, task
    track, separator, task = target.partition("/")
    provider = get_track(track)
    return provider, task if separator else None


def _selected_tasks(provider: TrackProvider, task: Optional[str]) -> list[str]:
    if task is not None:
        if task not in provider.tasks():
            available = ", ".join(provider.tasks()) or "none"
            raise KeyError(
                f"Unknown task {task!r} for track {provider.name!r}; available: {available}"
            )
        return [task]
    return provider.tasks()


def _layout_payload(provider: TrackProvider, task: str, layout: TaskLayout) -> dict:
    return {
        "track": provider.name,
        "task": task,
        "source": str(layout.source),
        "target": str(layout.target),
        "refs": {name: str(path) for name, path in layout.refs.items()},
        "candidates": str(layout.candidates) if layout.candidates else None,
        "provenance": dict(layout.provenance),
    }


def _print_report(report: VerificationReport) -> None:
    print(f"{report.provider}/{report.task}\t{report.status}")
    for issue in report.issues:
        print(f"  error: {issue}")
    for warning in report.warnings:
        print(f"  warning: {warning}")


def _locked_targets(root: Path) -> list[str]:
    lock = read_lock(root)
    return sorted(str(key) for key in lock["tasks"])


def run(args: argparse.Namespace) -> int:
    """Execute a parsed dataset-track command."""

    if args.data_command == "list":
        names = list_tracks()
        providers = [get_track(name) for name in names]
        if args.descriptor:
            providers.append(provider_from_descriptor(args.descriptor.expanduser().resolve()))
        for provider in providers:
            tasks = ", ".join(provider.tasks()) or "(no published tasks)"
            print(f"{provider.name}\t{tasks}")
        return 0

    root = args.root.expanduser().resolve()
    if args.data_command == "pull":
        provider, selected = _provider_and_task(args.target, args.descriptor)
        for task in _selected_tasks(provider, selected):
            layout = provider.materialize(
                task,
                root,
                revision=args.revision,
                update=args.update,
            )
            print(json.dumps(_layout_payload(provider, task, layout), sort_keys=True, default=str))
        return 0

    targets = [args.target] if args.target else _locked_targets(root)
    if args.descriptor is not None and args.target is None:
        descriptor_provider = provider_from_descriptor(args.descriptor.expanduser().resolve())
        prefix = f"{descriptor_provider.name}/"
        targets = [target for target in targets if target.startswith(prefix)]
    if not targets:
        print("No materialized dataset tasks found.")
        return 0
    reports: list[VerificationReport] = []
    for target in targets:
        provider, selected = _provider_and_task(target, args.descriptor)
        for task in _selected_tasks(provider, selected):
            report = provider.verify(task, root)
            reports.append(report)
            _print_report(report)
    if args.data_command == "verify" and any(not report.ok for report in reports):
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ``exact data`` and return a process exit status."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
