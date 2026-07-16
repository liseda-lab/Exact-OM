"""Command-line entry point for Exact evaluation backends."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Sequence


def _optional_path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser().resolve() if value else None


def run_evaluation(args: argparse.Namespace):
    from exact.core.actions.evaluation import EvaluationAction

    bioml_options = {
        key: value
        for key, value in {
            "typed_submission_path": _optional_path(args.bioml_typed_submission),
            "typed_answers_path": _optional_path(args.bioml_typed_answers),
            "preferred_pairs_path": _optional_path(args.bioml_preferred_pairs),
            "graded_relevance_path": _optional_path(args.bioml_graded_relevance),
            "hierarchy_path": _optional_path(args.bioml_hierarchy),
            "candidate_count": args.bioml_candidate_count,
        }.items()
        if value is not None
    }
    return EvaluationAction.run(
        alignment=Path(args.alignment_file).resolve(),
        output_dir_path=Path(args.output_dir).resolve(),
        error_on_fail=args.error_on_fail,
        K=args.K,
        source_file_path=_optional_path(args.source_ontology_file),
        target_file_path=_optional_path(args.target_ontology_file),
        train_reference_file_path=_optional_path(args.train_reference_file),
        full_reference_file_path=_optional_path(args.full_reference_file),
        reference_candidates=_optional_path(args.reference_candidates),
        log_file_path=(
            Path(args.output_dir).resolve() / "OAEI_bio_ml_eval.log" if args.save_logs else None
        ),
        log_level=args.log_level,
        backends=args.eval_backends,
        backend_options={"bioml": bioml_options},
    )


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


def _validate_path(value: Optional[str], label: str) -> None:
    if value and not Path(value).exists():
        raise FileNotFoundError(f"{label} {value} does not exist")


def main(argv: Optional[Sequence[str]] = None):
    args = parse_arguments(argv)
    _validate_path(args.alignment_file, "Alignment file")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _validate_path(args.source_ontology_file, "Source ontology file")
    _validate_path(args.target_ontology_file, "Target ontology file")
    _validate_path(args.train_reference_file, "Training reference file")
    _validate_path(args.full_reference_file, "Full reference file")
    _validate_path(args.reference_candidates, "Reference candidates file")
    _validate_path(args.bioml_typed_submission, "BioML typed submission")
    _validate_path(args.bioml_typed_answers, "BioML typed answers")
    if args.jvm_heap_size:
        warnings.warn(
            "--jvm-heap-size is deprecated and ignored; Exact no longer starts a JVM.",
            DeprecationWarning,
            stacklevel=2,
        )
    return run_evaluation(args)


if __name__ == "__main__":
    main()
