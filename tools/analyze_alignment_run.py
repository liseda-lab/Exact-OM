#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from exact.analysis.alignment_diagnostics import analyze_alignment_run, format_diagnostics_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report candidate-oracle and miss-bucket diagnostics for an Exact run.")
    parser.add_argument("run_dir", type=Path, help="Run output directory.")
    parser.add_argument("--reference", type=Path, default=None, help="Full reference TSV/CSV path.")
    parser.add_argument("--train-reference", type=Path, default=None, help="Training/null reference TSV/CSV path.")
    parser.add_argument("--summary", type=Path, default=None, help="summary_metrics.csv path.")
    parser.add_argument("--alignment", type=Path, default=None, help="Saved alignment TSV/CSV path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path for machine-readable diagnostics.")
    args = parser.parse_args()

    reference = args.reference or _infer_dataset_path(args.run_dir, "full_reference")
    if reference is None:
        raise SystemExit("Could not infer reference path; pass --reference.")
    train_reference = args.train_reference or _infer_dataset_path(args.run_dir, "train_reference")

    diagnostics = analyze_alignment_run(
        run_dir=args.run_dir,
        reference_path=reference,
        train_reference_path=train_reference,
        summary_path=args.summary,
        alignment_path=args.alignment,
    )
    print(format_diagnostics_report(diagnostics))
    if args.json_output:
        args.json_output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")


def _infer_dataset_path(run_dir: Path, key: str) -> Optional[Path]:
    try:
        import yaml
    except ImportError:
        return None

    for config_path in sorted(Path(run_dir).glob("*.yaml")):
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        dataset = payload.get("dataset") if isinstance(payload, dict) else None
        if not isinstance(dataset, dict):
            continue
        data_dir = dataset.get("data_dir")
        value = dataset.get(key)
        if data_dir and value:
            candidate = Path(data_dir) / str(value)
            return candidate
    return None


if __name__ == "__main__":
    main()
