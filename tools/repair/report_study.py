"""Report grouped paired effects and every scheduled status from a frozen repair study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.evaluation import holm_adjust, paired_group_effects


def main(argv: list[str] | None = None) -> int:
    """Compute evaluator-only summaries without changing frozen runs or fitting models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--contrast", nargs=2, action="append", metavar=("LEFT", "RIGHT"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--confidence-level", type=float)
    parser.add_argument("--sign-flip-replicates", type=int, default=10000)
    parser.add_argument("--metric", default="exact_external_regret")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contrasts = list(args.contrast or ())
    if bool(args.left) != bool(args.right):
        parser.error("--left and --right must be supplied together")
    if args.left:
        contrasts.append((args.left, args.right))
    if not contrasts:
        parser.error("provide --left/--right or one or more --contrast pairs")
    analysis = {}
    if args.protocol:
        from tools.repair.prepare import load_protocol

        analysis = load_protocol(args.protocol)["analysis"]
        if analysis.get("multiplicity", "holm") != "holm":
            parser.error("only the declared Holm multiplicity procedure is supported")
    payload = json.loads(args.results.read_text())
    if payload.get("schema") != "exact-repair/study/v2":
        raise ValueError("unsupported repair result schema")
    rows = [
        dict(row, **row.get("metrics", {}), **row.get("resources", {})) for row in payload["rows"]
    ]
    reports = {}
    for left, right in contrasts:
        key = f"{left}__vs__{right}"
        if key in reports:
            raise ValueError("duplicate paired comparison")
        reports[key] = paired_group_effects(
            rows,
            left,
            right,
            args.metric,
            bootstrap_replicates=(
                args.replicates
                if args.replicates is not None
                else analysis.get("bootstrap_replicates", 1000)
            ),
            seed=args.seed if args.seed is not None else analysis.get("bootstrap_seed", 13),
            confidence_level=(
                args.confidence_level
                if args.confidence_level is not None
                else analysis.get("interval", 0.95)
            ),
            sign_flip_replicates=args.sign_flip_replicates,
        )
    adjusted = holm_adjust({key: value["paired_pvalue"] for key, value in reports.items()})
    for key, value in reports.items():
        value["holm_adjusted_pvalue"] = adjusted[key]
    summary = dict(next(iter(reports.values()))) if len(reports) == 1 else {}
    summary["contrasts"] = reports
    summary["multiplicity"] = {
        "method": "holm",
        "alpha": analysis.get("alpha", 0.05),
        "family": list(reports),
    }
    summary.update(
        schema="exact-repair/paired-report/v2",
        plan_hash=payload["plan_hash"],
        recorded_jobs=payload["recorded"],
        scheduled_jobs=payload["scheduled"],
    )
    write_artifact(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
