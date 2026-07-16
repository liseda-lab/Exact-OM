import argparse
import logging
import warnings
from pathlib import Path

from exact.analysis.user_study import run_user_study_analysis


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable user-study and failure-analysis artifacts from an existing Exact run."
    )
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Path to the existing run directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Optional output directory for the analysis artifacts. Defaults to <run-dir>/analysis/user_study.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked target candidates per source to analyze.",
    )
    parser.add_argument(
        "--per-rank",
        type=int,
        default=4,
        help="Final number of selected sources per gold-rank bucket.",
    )
    parser.add_argument(
        "--shortlist-per-rank",
        type=int,
        default=8,
        help="Automatic shortlist size per gold-rank bucket before manual review.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Deterministic seed placeholder for stable selection."
    )
    parser.add_argument(
        "--skip-explanation-backfill",
        dest="backfill_explanations",
        action="store_false",
        help="Skip targeted backfill of missing provenance-backed explanation fields.",
    )
    parser.set_defaults(backfill_explanations=True)
    parser.add_argument(
        "--generate-rationales",
        dest="generate_rationales",
        action="store_true",
        help="Backfill only the missing rationales for the final selected records.",
    )
    parser.add_argument(
        "--skip-rationales",
        dest="generate_rationales",
        action="store_false",
        help="Skip rationale backfill and reuse the selected records as-is.",
    )
    parser.set_defaults(generate_rationales=True)
    parser.add_argument(
        "--config-file",
        type=str,
        required=False,
        help="Optional config file override. Defaults to <run-dir>/config.yaml.",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        help="Optional GPU device ID for local rationale fallback. Leave empty for CPU.",
    )
    parser.add_argument(
        "--jvm-heap-size",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logger level: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.jvm_heap_size is not None:
        warnings.warn(
            "--jvm-heap-size is deprecated and ignored; Exact-OM no longer needs Java.",
            DeprecationWarning,
            stacklevel=2,
        )
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "analysis" / "user_study")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(args.logging_level).upper())
    logger = logging.getLogger("exact.user_study")
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "exact_user_study.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
    logger.handlers = [stream_handler, file_handler]
    logger.info("Logging to file %s", output_dir / "exact_user_study.log")

    outputs = run_user_study_analysis(
        run_dir=run_dir,
        output_dir=output_dir,
        top_k=args.top_k,
        per_rank=args.per_rank,
        shortlist_per_rank=args.shortlist_per_rank,
        seed=args.seed,
        backfill_explanations=args.backfill_explanations,
        generate_rationales=args.generate_rationales,
        config_path=Path(args.config_file).resolve() if args.config_file else None,
        device=args.device,
        logger=logger,
    )

    print("Wrote user-study analysis artifacts:", flush=True)
    for name, path in outputs.items():
        print(f"- {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
