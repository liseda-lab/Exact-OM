#!/usr/bin/env python3
"""Build and validate a multi-pair BioKG alignment submission.

The competition kit remains the final authority. This tool mirrors its stable
file-level checks (header, relation vocabulary, finite scores, duplicate pairs,
and optional candidate membership) before concatenating per-pair outputs.
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from exact.io.writers.base import WriterOptionsError
from exact.io.writers.typed_tsv import TYPED_COLUMNS, validate_typed_frame


def _assignment(value: str) -> tuple[str, Path]:
    task, separator, raw_path = value.partition("=")
    if not separator or not task.strip() or not raw_path.strip():
        raise ValueError(f"Expected TASK=PATH, got {value!r}")
    return task.strip(), Path(raw_path).expanduser()


def _read_typed(path: Path, *, task: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Typed alignment for {task!r} does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    if tuple(frame.columns) != TYPED_COLUMNS:
        raise WriterOptionsError(
            f"{path} must have exactly this header: {' | '.join(TYPED_COLUMNS)}"
        )
    return validate_typed_frame(frame, sort=True, reject_duplicates=True)


def _candidate_pairs(path: Path) -> set[tuple[str, str]]:
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    if {"SrcEntity", "TgtEntity"}.issubset(frame.columns) and "TgtCandidates" not in frame:
        return {(str(row.SrcEntity), str(row.TgtEntity)) for row in frame.itertuples(index=False)}
    required = {"SrcEntity", "TgtCandidates"}
    if not required.issubset(frame.columns):
        raise WriterOptionsError(
            f"Candidate file {path} needs SrcEntity plus TgtEntity or TgtCandidates"
        )
    pairs: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        try:
            candidates = literal_eval(str(row.TgtCandidates))
        except (SyntaxError, ValueError) as exc:
            raise WriterOptionsError(
                f"Could not parse TgtCandidates for {row.SrcEntity!r} in {path}"
            ) from exc
        if not isinstance(candidates, (list, tuple)):
            raise WriterOptionsError(f"TgtCandidates in {path} must be a list or tuple")
        for candidate in candidates:
            target = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
            pairs.add((str(row.SrcEntity), str(target)))
    return pairs


def build_submission(
    inputs: Mapping[str, Path],
    output_path: Path,
    *,
    candidate_files: Mapping[str, Path] | None = None,
) -> Path:
    """Validate and concatenate per-task typed TSV alignments."""

    if not inputs:
        raise WriterOptionsError("At least one task alignment is required")
    candidates = dict(candidate_files or {})
    unknown_candidates = sorted(set(candidates) - set(inputs))
    if unknown_candidates:
        raise WriterOptionsError(
            "Candidate files were supplied for unknown task(s): " + ", ".join(unknown_candidates)
        )
    frames: list[pd.DataFrame] = []
    for task, path in sorted(inputs.items()):
        frame = _read_typed(Path(path), task=task)
        candidate_path = candidates.get(task)
        if candidate_path is not None:
            allowed = _candidate_pairs(Path(candidate_path))
            emitted = set(zip(frame["SrcEntity"].map(str), frame["TgtEntity"].map(str)))
            outside = sorted(emitted - allowed)
            if outside:
                source, target = outside[0]
                raise WriterOptionsError(
                    f"Task {task!r} emits a pair outside its candidate pool: "
                    f"{source} -> {target}"
                )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = validate_typed_frame(combined, sort=True, reject_duplicates=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, sep="\t", index=False)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="TASK=PATH",
        help="Per-task typed TSV; repeat for each task pair.",
    )
    parser.add_argument(
        "--candidates",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Optional candidate table used for membership checks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _assignments(values: Sequence[str], *, parser: argparse.ArgumentParser) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        try:
            task, path = _assignment(value)
        except ValueError as exc:
            parser.error(str(exc))
        if task in result:
            parser.error(f"Task {task!r} was supplied more than once")
        result[task] = path
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the submission builder CLI."""

    parser = _parser()
    args = parser.parse_args(argv)
    inputs = _assignments(args.input, parser=parser)
    candidates = _assignments(args.candidates, parser=parser)
    try:
        destination = build_submission(
            inputs,
            args.output,
            candidate_files=candidates,
        )
    except (FileNotFoundError, WriterOptionsError, ValueError) as exc:
        parser.error(str(exc))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
