
import argparse
from pathlib import Path

from matcha_dl import init_jvm


def run_alignment(args):
    from matcha_dl.core.actions.alignment import DirectoryAlignmentAction

    DirectoryAlignmentAction.run(
        data_dir=Path(args.data_dir).resolve(),
        output_dir_path=Path(args.output_dir).resolve(),
        configs_file_path=Path(args.config_file).resolve() if args.config_file else None,
        run_eval=args.run_eval,
        save_logs=args.save_logs,
        devices=args.devices
    )

def parse_arguments():

    parser = argparse.ArgumentParser(description="Compute the alignment between two ontologies")
    parser.add_argument(
        "--data_dir",
        "-p",
        type=str,
        required=True,
        help="Please provide the path to the directory containing OAEI data",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Please provide the path to the output directory",
    )
    parser.add_argument(
        "--config_file",
        "-y",
        type=str,
        required=False,
        help="Please provide the path to the configuration file",
    )
    parser.add_argument(
        "--run_eval",
        "-e",
        action='store_true',
        help="Run evaluation after alignment"
    )
    parser.add_argument(
        "--save_logs",
        "-l",
        action='store_true',
        help="Save logs to a file"
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
        "--jvm_heap_size",
        "-m",
        type=str,
        required=False,
        help="JVM heap size",
    )

    return parser


def main():
    args = parse_arguments()

    if not Path(args.data_dir).is_dir():
        raise ValueError("The provided data directory does not exist or is not a directory.")
    if not Path(args.output_dir).is_dir():
        Path(args.output_dir).mkdir(parents=True)

    if args.config_file:
        config_file = Path(args.config_file)
        if not config_file.exists():
            raise Exception(f"Configuration file {args.config_file} does not exist")
        
    if args.jvm_heap_size:
        if args.jvm_heap_size.isdigit():
            args.jvm_heap_size += 'G'
        elif not (args.jvm_heap_size[:-1].isdigit() and args.jvm_heap_size[-1].lower() == 'g'):
            raise Exception(f"JVM heap size {args.jvm_heap_size} is not valid, please provide a valid format")
    else:
        args.jvm_heap_size = '32G'

    init_jvm(args.jvm_heap_size)

    run_alignment(args)

if __name__ == "__main__":
    main()

    


