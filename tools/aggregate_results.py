#!/usr/bin/env python3
"""Aggregate Exact tuning run metrics into a single CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from exact.runs import RunReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Exact hparam tuning results."
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        type=Path,
        help="Path to the experiment directory that contains the runs subdirectory.",
    )
    parser.add_argument(
        "--runs-subdir",
        default="runs",
        type=str,
        help="Name of the subdirectory under the experiment directory that stores runs.",
    )
    parser.add_argument(
        "--results-filename",
        default="evaluation_results.csv",
        type=str,
        help="Filename of the per-run results table to collect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the aggregated CSV. Defaults to <exp-dir>/aggregated_results.csv",
    )
    return parser.parse_args()


def normalize_value(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def find_results_file(run_dir: Path, filename: str) -> Path | None:
    reader = RunReader.open(run_dir)
    if filename in {"evaluation_results.csv", "evaluation_results.json"}:
        suffix = "json" if filename.endswith(".json") else "csv"
        path = reader.layout.evaluation_path(suffix)
        return path if path.is_file() else None
    for artifact in reader.manifest().get("artifacts", []):
        relative = artifact.get("path")
        if not isinstance(relative, str) or Path(relative).name != filename:
            continue
        path = reader.layout.resolve_relative(relative)
        if path.is_file():
            return path
    return None


def load_metrics(path: Path) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers:
            return metrics
        first_header = headers[0].strip().lower()
        if "metric" in first_header and len(headers) >= 2:
            value_header = headers[1]
            for row in reader:
                key = row.get(headers[0])
                value = row.get(value_header)
                if key is None or value is None:
                    continue
                metrics[key] = value
        else:
            # Interpret other layouts as a single summary row.
            rows = list(reader)
            if rows:
                metrics = {
                    key: value for key, value in rows[0].items() if key is not None
                }
    return metrics


def load_params(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    params = meta.get("params", {})
    return {f"param.{key}": normalize_value(value) for key, value in params.items()}


@dataclass
class RunRecord:
    name: str
    path: Path
    params: Dict[str, str]
    metrics: Dict[str, str]


def collect_runs(runs_dir: Path, results_filename: str) -> List[RunRecord]:
    records: List[RunRecord] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        trial_path = entry / "trial.json"
        results_path = find_results_file(entry, results_filename)
        if not trial_path.exists():
            print(f"[WARN] Skipping {entry} (missing trial.json)")
            continue
        if not results_path:
            print(f"[WARN] Skipping {entry} (missing {results_filename})")
            continue
        params = load_params(trial_path)
        metrics = {
            f"metric.{key}": value for key, value in load_metrics(results_path).items()
        }
        records.append(
            RunRecord(name=entry.name, path=entry, params=params, metrics=metrics)
        )
    return records


def write_aggregate(output_path: Path, records: Sequence[RunRecord]) -> None:
    if not records:
        raise SystemExit("No runs with evaluation results were found.")
    param_columns: List[str] = []
    metric_columns: List[str] = []
    for record in records:
        for key in record.params:
            if key not in param_columns:
                param_columns.append(key)
        for key in record.metrics:
            if key not in metric_columns:
                metric_columns.append(key)
    fieldnames = ["run_name", "run_path"] + param_columns + metric_columns
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {"run_name": record.name, "run_path": str(record.path)}
            row.update(record.params)
            row.update(record.metrics)
            writer.writerow(row)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    runs_dir = exp_dir / args.runs_subdir
    if not runs_dir.is_dir():
        raise SystemExit(f"Runs directory '{runs_dir}' does not exist.")
    output_path = args.output or (exp_dir / "aggregated_results.csv")
    records = collect_runs(runs_dir, args.results_filename)
    write_aggregate(output_path, records)
    print(f"Wrote aggregated results for {len(records)} runs to {output_path}")


if __name__ == "__main__":
    main()
