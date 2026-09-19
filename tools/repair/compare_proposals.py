"""Compare explicitly selected proposal arms for one local frozen case/checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run a single bounded comparison; no model fitting, downloads, or campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "problem", type=Path, help="v2 repair input record or input/objective bundle"
    )
    parser.add_argument("checkpoint", type=Path, help="weights-only repair checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--constructors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument(
        "--teacher-cache", type=Path, help="evaluation-only bounded teacher cache JSON"
    )
    parser.add_argument(
        "--coverage-labels",
        type=Path,
        help="evaluation-only useful candidate IDs and symbol encodings",
    )
    args = parser.parse_args(argv)
    import pyowl_core as owl
    import torch

    from exact.repair.api import write_artifact
    from exact.repair.model import RepairModel
    from exact.repair.proposals import PROPOSAL_ARMS, compare_proposals
    from exact.repair.records import ObjectiveV2, RepairInputV2, read_record

    payload = json.loads(args.problem.read_text())
    problem = read_record(payload.get("input", payload))
    if not isinstance(problem, RepairInputV2):
        parser.error("problem must contain a repair input")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = RepairModel(checkpoint["metadata"], **checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    labels = json.loads(args.coverage_labels.read_text()) if args.coverage_labels else {}
    teacher = None
    if args.teacher_cache:
        from tools.repair.prepare import cache_from_dict

        teacher = cache_from_dict(json.loads(args.teacher_cache.read_text()))
    profile: tuple[tuple[str, float], ...] = ()
    if "objective" in payload:
        objective = read_record(payload["objective"])
        if not isinstance(objective, ObjectiveV2):
            parser.error("objective must contain a frozen objective record")
        profile = objective.profile
    result = compare_proposals(
        problem,
        model,
        arms=args.arms or PROPOSAL_ARMS,
        seconds_per_arm=args.seconds,
        useful_candidate_ids=labels.get("useful_candidate_ids", {}),
        required_symbols={
            key: tuple(owl.decode_canonical(bytes.fromhex(value)) for value in values)
            for key, values in labels.get("required_symbol_hex", {}).items()
        },
        solve=args.solve,
        teacher_cache=teacher,
        options={
            "draws_per_object": args.draws,
            "candidate_cap": args.candidate_cap,
            "max_depth": args.depth,
            "max_constructors": args.constructors,
            "seed": args.seed,
            "profile": profile,
        },
    )
    write_artifact(args.output, result)
    print(
        json.dumps(
            {
                "scheduled": result["scheduled"],
                "recorded": result["recorded"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
