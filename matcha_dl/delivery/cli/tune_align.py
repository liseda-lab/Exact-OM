import argparse
from pathlib import Path

from matcha_dl import init_jvm


def run_tuning(args):
    from matcha_dl.core.actions.alignment import TuningAlignmentAction

    TuningAlignmentAction.run(
        source_file_path=Path(args.source_ontology_file).resolve(),
        target_file_path=Path(args.target_ontology_file).resolve(),
        output_dir_path=Path(args.output_dir).resolve(),
        configs_file_path=Path(args.config_file).resolve(),
        reference_file_path=Path(args.reference_file).resolve(),
        full_reference_file_path=Path(args.full_reference_file).resolve(),
        candidates_file_path=Path(args.candidates_file).resolve(),
        save_logs=args.save_logs,
        devices=args.devices,
        max_workers=args.max_workers,
        max_combinations=args.max_combinations
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute the alignment between two ontologies")
    parser.add_argument(
        "--source_ontology_file",
        "-s",
        type=str,
        required=True,
        help="Please provide the path to the source ontology file",
    )
    parser.add_argument(
        "--target_ontology_file",
        "-t",
        type=str,
        required=True,
        help="Please provide the path to the target ontology file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Please provide the path to the output directory",
    )
    parser.add_argument(
        "--reference_file",
        "-r",
        type=str,
        required=True,
        help="Please provide the path to the training reference file",
    )
    parser.add_argument(
        "--full_reference_file",
        "-f",
        type=str,
        required=True,
        help="Please provide the path to the full reference file",
    )
    parser.add_argument(
        "--candidates_file",
        "-c",
        type=str,
        required=True,
        help="Please provide the path to the candidates file",
    )
    parser.add_argument(
        "--config_file",
        "-y",
        type=str,
        required=True,
        help="Please provide the path to the configuration file",
    )
    parser.add_argument(
        "--save_logs",
        "-l",
        action="store_true",
        help="Save logs to a file",
    
    )
    parser.add_argument(
        "--devices",
        "-d",
        nargs="+",
        type=list,
        required=False,
        help="List of GPU device IDs to use (leave empty for CPU)",
    )
    parser.add_argument(
        "--max_workers",
        "-w",
        type=int,
        required=False,
        help="Number of workers to use for parallel processing",
    )
    parser.add_argument(
        "--max_combinations",
        "-x",
        type=int,
        required=False,
        help="Maximum number of combinations to evaluate. If None, all combinations are used.",
    )
    parser.add_argument(
        "--jvm_heap_size",
        "-m",
        type=str,
        required=False,
        help="JVM heap size",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not Path(args.source_ontology_file).exists():
        raise FileNotFoundError(f"Source ontology file not found: {args.source_ontology_file}")
    if not Path(args.target_ontology_file).exists():
        raise FileNotFoundError(f"Target ontology file not found: {args.target_ontology_file}")
    if not Path(args.reference_file).exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference_file}")
    if not Path(args.full_reference_file).exists():
        raise FileNotFoundError(f"Full reference file not found: {args.full_reference_file}")
    if not Path(args.candidates_file).exists():
        raise FileNotFoundError(f"Candidates file not found: {args.candidates_file}")
    if not Path(args.config_file).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config_file}")
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    if args.jvm_heap_size:
        if args.jvm_heap_size.isdigit():
            args.jvm_heap_size += 'G'
        elif not (args.jvm_heap_size[:-1].isdigit() and args.jvm_heap_size[-1].lower() == 'g'):
            raise Exception(f"JVM heap size {args.jvm_heap_size} is not valid, please provide a valid format")
    else:
        args.jvm_heap_size = '32G'

    init_jvm(args.jvm_heap_size)

    run_tuning(args)

if __name__ == "__main__":
    main()