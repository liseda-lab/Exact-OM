"""Command-line entry point for Exact evaluation backends."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from exact.delivery.common import (
    execute_evaluation,
    prepare_evaluation_namespace,
    warn_ignored_jvm,
)


def run_evaluation(args: argparse.Namespace):
    invocation = getattr(args, "_exact_invocation", None) or prepare_evaluation_namespace(args)
    return execute_evaluation(invocation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an Exact alignment")
    parser.add_argument(
        "--alignment_file",
        "--alignment-file",
        "-a",
        required=True,
        help="Path to the alignment file",
    )
    parser.add_argument(
        "--output_dir", "--output-dir", "-o", required=True, help="Evaluation output directory"
    )
    parser.add_argument(
        "--error_on_fail",
        "--error-on-fail",
        "-e",
        action="store_true",
        help="Raise if a backend fails",
    )
    parser.add_argument(
        "--K", "-k", nargs="+", type=int, default=None, help="Ranking cut-offs (default: 1 5 10)"
    )
    parser.add_argument("--source_ontology_file", "--source-ontology-file", "-s")
    parser.add_argument("--target_ontology_file", "--target-ontology-file", "-t")
    parser.add_argument("--train_reference_file", "--train-reference-file", "-r")
    parser.add_argument("--full_reference_file", "--full-reference-file", "-f")
    parser.add_argument("--reference_candidates", "--reference-candidates", "-c")
    parser.add_argument(
        "--eval-backends",
        "--eval_backends",
        nargs="+",
        default=["builtin"],
        metavar="BACKEND",
        help="Ordered evaluator backends (default: builtin)",
    )
    parser.add_argument("--bioml-typed-submission")
    parser.add_argument("--bioml-typed-answers")
    parser.add_argument("--bioml-preferred-pairs")
    parser.add_argument("--bioml-graded-relevance")
    parser.add_argument("--bioml-hierarchy")
    parser.add_argument("--bioml-candidate-count", type=int)
    parser.add_argument(
        "--bioml-coherence-reasoner",
        choices=("hermit", "elk"),
        default="hermit",
        help="Official Bio-ML coherence reasoner (default: hermit)",
    )
    parser.add_argument(
        "--bioml-coherence-timeout",
        type=float,
        default=7200.0,
        metavar="SECONDS",
        help="HermiT timeout before the labelled ELK fallback (default: 7200)",
    )
    parser.add_argument(
        "--bioml-skip-invalid-iris",
        action="store_true",
        help="Drop malformed alignment IRIs during official coherence scoring",
    )
    parser.add_argument("--save_logs", "--save-logs", "-l", action="store_true")
    parser.add_argument("--log_level", "--log-level", "-v", default="INFO")
    parser.add_argument(
        "--jvm_heap_size",
        "--jvm-heap-size",
        "-m",
        help=argparse.SUPPRESS,
    )
    return parser


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_arguments(argv)
    warn_ignored_jvm(args.jvm_heap_size or None, "--jvm-heap-size")
    args._exact_invocation = prepare_evaluation_namespace(args)
    return run_evaluation(args)


if __name__ == "__main__":
    main()
