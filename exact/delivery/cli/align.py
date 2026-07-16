import argparse
from typing import Optional, Sequence

from exact.delivery.common import (
    execute_alignment,
    prepare_alignment_namespace,
    warn_ignored_jvm,
)


def run_alignment(args) -> None:
    invocation = getattr(args, "_exact_invocation", None) or prepare_alignment_namespace(args)
    execute_alignment(invocation)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the alignment between two ontologies")
    parser.add_argument(
        "--source_ontology_file",
        "-s",
        type=str,
        required=False,
        help="Source ontology path; optional when supplied by the config data block",
    )
    parser.add_argument(
        "--target_ontology_file",
        "-t",
        type=str,
        required=False,
        help="Target ontology path; optional when supplied by the config data block",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Please provide the path to the output directory",
    )
    parser.add_argument(
        "--training_reference_file",
        "-r",
        type=str,
        required=False,
        help="Please provide the path to the training reference file",
    )
    parser.add_argument(
        "--full_reference_file",
        "-f",
        type=str,
        required=False,
        help="Please provide the path to the full reference file",
    )
    parser.add_argument(
        "--candidates_file",
        "-c",
        type=str,
        required=False,
        help="Please provide the path to the candidates file",
    )
    parser.add_argument(
        "--config_file",
        "-y",
        type=str,
        required=False,
        help="Please provide the path to the yaml configuration file",
    )
    parser.add_argument(
        "--save_logs",
        "-l",
        action="store_true",
        help="Whether to save logs",
    )
    parser.add_argument(
        "--run_eval",
        "-e",
        action="store_true",
        help="Whether to run evaluation",
    )
    parser.add_argument(
        "--jvm_heap_size",
        "-m",
        type=str,
        required=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        required=False,
        help="GPU device ID to use (leave empty for CPU)",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "owl", "rdf", "csv-kg"],
        help="Input adapter for both source and target (default: infer from each path)",
    )
    parser.add_argument(
        "--source-options",
        nargs="+",
        metavar="KEY=VALUE|YAML",
        help="Source adapter key=value options, or one YAML mapping file",
    )
    parser.add_argument(
        "--target-options",
        nargs="+",
        metavar="KEY=VALUE|YAML",
        help="Target adapter key=value options, or one YAML mapping file",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        metavar="FORMAT",
        help="Alignment writers, such as tsv-global, oaei-rdf, typed-tsv, or json",
    )
    parser.add_argument(
        "--relation-prediction",
        choices=["none", "hierarchy_heuristic"],
        help="Post-scoring relation typing mode",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_arguments(argv)
    warn_ignored_jvm(args.jvm_heap_size, "--jvm_heap_size/-m")
    args._exact_invocation = prepare_alignment_namespace(args)
    return run_alignment(args)


if __name__ == "__main__":
    main()
