#!/usr/bin/env python3

"""Batch evaluation of filtered global alignments."""

from __future__ import annotations

import argparse
import csv
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


def _as_list(values: Optional[Iterable[Any]], default: Optional[Sequence[Any]]) -> List[Any]:
    if values is None:
        return list(default) if default is not None else []
    return list(values)


def _parse_thresholds(values: Optional[Sequence[str]]) -> List[Optional[float]]:
    parsed = []
    for value in _as_list(values, default=[None]):
        if value is None:
            parsed.append(None)
            continue
        value_str = str(value).strip().lower()
        if value_str in {"none", "all", "null"}:
            parsed.append(None)
        else:
            parsed.append(float(value_str))
    return parsed


def _parse_cardinalities(values: Optional[Sequence[str]]) -> List[Optional[int]]:
    parsed = []
    for value in _as_list(values, default=[None]):
        if value is None:
            parsed.append(None)
            continue
        value_str = str(value).strip().lower()
        if value_str in {"none", "all", "null"}:
            parsed.append(None)
        else:
            parsed.append(int(value_str))
    return parsed


def _parse_k(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(v) for v in values]


def _resolve_optional_path(path_value: Optional[Union[str, Path]]) -> Optional[Path]:
    if path_value is None:
        return None
    if isinstance(path_value, str):
        normalized = path_value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def _format_label(value: Optional[Union[float, int]]) -> str:
    if value is None:
        return "all"
    if isinstance(value, float):
        label = f"{value:.6f}".rstrip("0").rstrip(".")
        return label or "0"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply cardinality/threshold sweeps to a global alignment and evaluate every configuration."
    )
    parser.add_argument(
        "--alignment-file",
        required=True,
        help="Path to src2tgt.maps_global.tsv (or any mappings file with SrcEntity/TgtEntity/Score columns).",
    )
    parser.add_argument(
        "--output-base-dir",
        required=True,
        help="Directory where per-configuration evaluation outputs will be written.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        help="List of score thresholds (e.g. --thresholds 0.0 0.2 none). 'none' keeps all scores.",
    )
    parser.add_argument(
        "--cardinalities",
        nargs="*",
        help="List of source->target cardinalities (e.g. --cardinalities 1 5 none). 'none' keeps all matches.",
    )
    parser.add_argument(
        "--source-ontology-path",
        help="Optional source ontology file (required for global evaluation if target is provided).",
    )
    parser.add_argument(
        "--target-ontology-path",
        help="Optional target ontology file (required for global evaluation if source is provided).",
    )
    parser.add_argument("--train-reference-file", help="Training reference mappings (optional).")
    parser.add_argument(
        "--full-reference-file", help="Gold/test reference mappings for global evaluation."
    )
    parser.add_argument(
        "--reference-candidates-file", help="Reference candidates file for local evaluation."
    )
    parser.add_argument(
        "--k-values",
        nargs="*",
        help="K cut-offs for local evaluation metrics (e.g. --k-values 1 5 10).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level passed to EvaluationAction (default: INFO).",
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Store evaluation.log inside each combination output directory.",
    )
    parser.add_argument(
        "--error-on-fail",
        action="store_true",
        help="Raise an exception if evaluation fails for any configuration.",
    )
    parser.add_argument(
        "--jvm-heap-size",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()

    alignment_path = Path(args.alignment_file).expanduser().resolve()
    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    output_dir = Path(args.output_base_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_thresholds(args.thresholds)
    cardinalities = _parse_cardinalities(args.cardinalities)
    k_values = _parse_k(args.k_values)

    source_path = _resolve_optional_path(args.source_ontology_path)
    target_path = _resolve_optional_path(args.target_ontology_path)
    train_reference_path = _resolve_optional_path(args.train_reference_file)
    full_reference_path = _resolve_optional_path(args.full_reference_file)
    reference_candidates_path = _resolve_optional_path(args.reference_candidates_file)

    if full_reference_path is None and reference_candidates_path is None:
        raise ValueError(
            "Provide either --full-reference-file for global evaluation or --reference-candidates-file for local evaluation."
        )

    if args.jvm_heap_size is not None:
        warnings.warn(
            "--jvm-heap-size is deprecated and ignored; Exact-OM no longer needs Java.",
            DeprecationWarning,
            stacklevel=2,
        )

    from exact.core.actions.evaluation import EvaluationAction
    from exact.core.entities.mappings import EntityMapping

    summary_rows: List[Dict[str, Any]] = []

    for threshold_value, cardinality_value in product(thresholds, cardinalities):
        label = f"thr_{_format_label(threshold_value)}__card_{_format_label(cardinality_value)}"
        combo_dir = output_dir / label
        combo_dir.mkdir(parents=True, exist_ok=True)

        filtered_alignment = EntityMapping.read_table_mappings(
            alignment_path, threshold=threshold_value, cardinality=cardinality_value
        )

        print(
            f"[run_cardinality_threshold_tests] threshold={threshold_value} cardinality={cardinality_value} -> "
            f"{len(filtered_alignment)} mappings"
        )

        log_file_path = combo_dir / "evaluation.log" if args.save_logs else None

        results = EvaluationAction.run(
            alignment=filtered_alignment,
            output_dir_path=combo_dir,
            error_on_fail=args.error_on_fail,
            K=k_values,
            source_file_path=source_path,
            target_file_path=target_path,
            train_reference_file_path=train_reference_path,
            full_reference_file_path=full_reference_path,
            reference_candidates=reference_candidates_path,
            log_file_path=log_file_path,
            log_level=args.log_level,
        )

        if results:
            for metric_name, metric_value in results.items():
                summary_rows.append(
                    {
                        "threshold": threshold_value,
                        "cardinality": cardinality_value,
                        "metric": metric_name,
                        "value": metric_value,
                        "status": "ok",
                        "output_dir": combo_dir,
                    }
                )
        else:
            summary_rows.append(
                {
                    "threshold": threshold_value,
                    "cardinality": cardinality_value,
                    "metric": "",
                    "value": "",
                    "status": "failed",
                    "output_dir": combo_dir,
                }
            )

    summary_file = output_dir / "cardinality_threshold_summary.csv"
    with summary_file.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["threshold", "cardinality", "metric", "value", "status", "output_dir"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[run_cardinality_threshold_tests] Summary saved to {summary_file}")


if __name__ == "__main__":
    run()
