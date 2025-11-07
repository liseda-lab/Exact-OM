import argparse
from pathlib import Path

from exact import init_jvm

def run_evaluation(args):
    from exact.core.actions.evaluation import EvaluationAction

    EvaluationAction.run(
        alignment=Path(args.alignment_file).resolve(),
        output_dir_path=Path(args.output_dir).resolve(),
        error_on_fail=args.error_on_fail,
        K=args.K,
        source_file_path=Path(args.source_ontology_file).resolve() if args.source_ontology_file else None,
        target_file_path=Path(args.target_ontology_file).resolve() if args.target_ontology_file else None,
        train_reference_file_path=Path(args.train_reference_file).resolve() if args.train_reference_file else None,
        full_reference_file_path=Path(args.full_reference_file).resolve() if args.full_reference_file else None,
        reference_candidates=Path(args.reference_candidates).resolve() if args.reference_candidates else None,
        log_file_path=Path(args.output_dir).resolve() / "OAEI_bio_ml_eval.log" if args.save_logs else None,
        log_level=args.log_level,
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the alignment between two ontologies")
    parser.add_argument(
        "--alignment_file",
        "-a",
        type=str,
        required=True,
        help="Please provide the path to the alignment file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Please provide the path to the output directory",
    )
    parser.add_argument(
        "--error_on_fail",
        "-e",
        action="store_true",
        help="Raise an error if evaluation fails",
    )
    parser.add_argument(
        "--K",
        "-k",
        nargs="+",
        type=list,
        required=False,
        help="The number of top-K elements to consider in the evaluation",
    )
    parser.add_argument(
        "--source_ontology_file",
        "-s",
        type=str,
        required=False,
        help="Please provide the path to the source ontology file",
    )
    parser.add_argument(
        "--target_ontology_file",
        "-t",
        type=str,
        required=False,
        help="Please provide the path to the target ontology file",
    )
    parser.add_argument(
        "--train_reference_file",
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
        "--reference_candidates",
        "-c",
        type=str,
        required=False,
        help="Please provide the path to the reference candidates file",
    )
    parser.add_argument(
        "--save_logs",
        "-l",
        action="store_true",
        help="Save logs to a file",
    )
    parser.add_argument(
        "--log_level",
        "-v",
        type=str,
        default="INFO",
        help="Set the log level",
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

    if not Path(args.alignment_file).exists():
        raise Exception(f"Alignment file {args.alignment_file} does not exist")
    if not Path(args.output_dir).exists():
        raise Exception(f"Output directory {args.output_dir} does not exist")
    if args.source_ontology_file and not Path(args.source_ontology_file).exists():
        raise Exception(f"Source ontology file {args.source_ontology_file} does not exist")
    if args.target_ontology_file and not Path(args.target_ontology_file).exists():
        raise Exception(f"Target ontology file {args.target_ontology_file} does not exist")
    if args.train_reference_file and not Path(args.train_reference_file).exists():
        raise Exception(f"Training reference file {args.train_reference_file} does not exist")
    if args.full_reference_file and not Path(args.full_reference_file).exists():
        raise Exception(f"Full reference file {args.full_reference_file} does not exist")
    if args.reference_candidates and not Path(args.reference_candidates).exists():
        raise Exception(f"Reference candidates file {args.reference_candidates} does not exist")
    
    if args.jvm_heap_size:
        if args.jvm_heap_size.isdigit():
            args.jvm_heap_size += 'G'
        elif not (args.jvm_heap_size[:-1].isdigit() and args.jvm_heap_size[-1].lower() == 'g'):
            raise Exception(f"JVM heap size {args.jvm_heap_size} is not valid, please provide a valid format")
    else:
        args.jvm_heap_size = '32G'

    init_jvm(args.jvm_heap_size)

    run_evaluation(args)

if __name__ == "__main__":
    main()