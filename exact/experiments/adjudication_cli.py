"""Offline CLI for blinded disagreement adjudication artifacts.

This module performs no matching, inference, hosted calls, or annotation.  It
only exports a deterministic sample, validates completed annotation TSVs, and
analyzes a verified merged artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adjudication import (
    adjudication_sample_size,
    build_blinded_sample,
    import_annotations,
    inverse_probability_weighted_precision,
    merge_adjudications,
    verify_sample_manifest,
    write_blinded_sample,
)


def _json_payload(path: Path) -> Any:
    resolved = Path(path).expanduser().resolve()
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required JSON artifact does not exist: {resolved}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact at {resolved}: {exc}") from exc


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = _json_payload(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return {str(key): value for key, value in payload.items()}


def _disagreements(path: Path) -> list[Mapping[str, Any]]:
    payload = _json_payload(path)
    if isinstance(payload, Mapping):
        payload = payload.get("disagreements")
    if not isinstance(payload, list) or not payload:
        raise ValueError("disagreement input must be a non-empty JSON list")
    if not all(isinstance(item, Mapping) for item in payload):
        raise ValueError("every disagreement must be a JSON object")
    return list(payload)


def _require_new(*paths: Path) -> None:
    existing = [str(Path(path).expanduser().resolve()) for path in paths if Path(path).exists()]
    if existing:
        raise ValueError(f"refusing to overwrite existing adjudication artifacts: {existing}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    _require_new(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            stream.write("\n")
        os.replace(temporary, output)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def _export_command(args: argparse.Namespace) -> dict[str, Any]:
    disagreements = _disagreements(args.disagreements)
    design = adjudication_sample_size(
        len(disagreements),
        target_ci_width=args.target_ci_width,
        confidence=args.confidence,
        anticipated_precision=args.anticipated_precision,
        minimum_when_available=args.minimum_when_available,
    )
    sample = build_blinded_sample(
        disagreements,
        sample_size=int(design["sample_size"]),
        seed=args.seed,
        sample_design=design,
    )
    output = Path(args.output).expanduser().resolve()
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    _require_new(output, manifest)
    table, manifest = write_blinded_sample(sample, output)
    return {
        "command": "export",
        "status": "awaiting_annotation",
        "annotations": str(table),
        "sample_manifest": str(manifest),
        "sample_hash": sample["sample_hash"],
        "population_hash": sample["population_hash"],
        "sample_design": design,
    }


def _import_command(args: argparse.Namespace) -> dict[str, Any]:
    sample = _json_object(args.sample_manifest, label="sample manifest")
    sample_hash = verify_sample_manifest(sample)
    annotations = import_annotations(args.annotations, sample)
    merged = merge_adjudications(
        sample,
        annotations,
        minimum_annotators=args.minimum_annotators,
    )
    output = _write_json(args.output, merged)
    return {
        "command": "import",
        "status": merged["status"],
        "output": str(output),
        "sample_hash": sample_hash,
        "case_count": len(merged["cases"]),
        "inter_annotator_agreement": merged["inter_annotator_agreement"],
    }


def _analyze_command(args: argparse.Namespace) -> dict[str, Any]:
    merged = _json_object(args.adjudication, label="merged adjudication")
    result = inverse_probability_weighted_precision(
        merged,
        resamples=args.resamples,
        seed=args.seed,
    )
    output = _write_json(args.output, result)
    return {
        "command": "analyze",
        "status": result["status"],
        "output": str(output),
        "sample_hash": result["sample_hash"],
        "precision": result["precision"],
        "raw_precision": result["raw_precision"],
        "confidence_interval": result["confidence_interval"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m exact.experiments.adjudication_cli",
        description="Export, validate, and analyze blinded adjudication fixtures/artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="export a deterministic blinded sample")
    export.add_argument("--disagreements", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True, help="new annotation TSV path")
    export.add_argument("--target-ci-width", type=float, required=True)
    export.add_argument("--confidence", type=float, default=0.95)
    export.add_argument("--anticipated-precision", type=float, default=0.5)
    export.add_argument("--minimum-when-available", type=int, default=50)
    export.add_argument("--seed", type=int, default=17)
    export.set_defaults(handler=_export_command)

    import_parser = commands.add_parser(
        "import", help="validate annotations and merge primary/adjudicator decisions"
    )
    import_parser.add_argument("--sample-manifest", type=Path, required=True)
    import_parser.add_argument("--annotations", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True, help="new merged JSON path")
    import_parser.add_argument("--minimum-annotators", type=int, default=2)
    import_parser.set_defaults(handler=_import_command)

    analyze = commands.add_parser("analyze", help="compute verified IPW precision and bootstrap CI")
    analyze.add_argument("--adjudication", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True, help="new analysis JSON path")
    analyze.add_argument("--resamples", type=int, default=10_000)
    analyze.add_argument("--seed", type=int, default=17)
    analyze.set_defaults(handler=_analyze_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() fixtures
    raise SystemExit(main())
