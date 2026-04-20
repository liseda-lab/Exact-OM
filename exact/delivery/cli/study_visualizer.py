import argparse
import logging
import os
from pathlib import Path

import uvicorn
from jpype import JVMNotFoundException
from mowl import init_jvm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the Exact study visualizer as a FastAPI app backed by one fixed study run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=os.getenv("EXACT_STUDY_RUN_DIR"),
        help="Path to the fixed Exact study run directory. Defaults to EXACT_STUDY_RUN_DIR.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=os.getenv("EXACT_STUDY_ANALYSIS_DIR"),
        help="Optional analysis directory. Defaults to <run-dir>/analysis/user_study.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("EXACT_STUDY_HOST", "0.0.0.0"),
        help="Host interface for the FastAPI server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("EXACT_STUDY_PORT", "8000")),
        help="Port for the FastAPI server.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn reload for local development.",
    )
    parser.add_argument(
        "--jvm-heap-size",
        type=str,
        default=os.getenv("EXACT_STUDY_JVM_HEAP_SIZE", "32G"),
        help="JVM heap size used for ontology-backed study expansion.",
    )
    parser.add_argument(
        "--disable-ontology-info",
        dest="enable_ontology_info",
        action="store_false",
        help="Disable ontology-backed node info and one-hop expansion.",
    )
    parser.set_defaults(enable_ontology_info=os.getenv("EXACT_STUDY_ENABLE_ONTOLOGY_INFO", "true").lower() in {"1", "true", "yes", "y", "on"})
    parser.add_argument(
        "--logging-level",
        type=str,
        default=os.getenv("EXACT_STUDY_LOG_LEVEL", "INFO"),
        help="Logger level: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.run_dir:
        raise RuntimeError("Study visualizer requires --run-dir or EXACT_STUDY_RUN_DIR.")

    if args.jvm_heap_size:
        if args.jvm_heap_size.isdigit():
            args.jvm_heap_size += "G"
        elif not (args.jvm_heap_size[:-1].isdigit() and args.jvm_heap_size[-1].lower() == "g"):
            raise Exception(
                f"JVM heap size {args.jvm_heap_size} is not valid, please provide a valid format"
            )
    else:
        args.jvm_heap_size = "32G"

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else None

    level = getattr(logging, str(args.logging_level).upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("exact.study_visualizer")
    logger.setLevel(level)
    logger.info("Starting study visualizer for run_dir=%s analysis_dir=%s", run_dir, analysis_dir)

    if args.enable_ontology_info:
        logger.info("Initializing JVM for study visualizer with heap=%s", args.jvm_heap_size)
        try:
            init_jvm(args.jvm_heap_size)
        except JVMNotFoundException as exc:
            raise RuntimeError(
                "Study visualizer requires a working Java runtime when ontology info is enabled. "
                "Install a JDK/JRE and set JAVA_HOME so libjvm.so is available, or rerun with --disable-ontology-info."
            ) from exc

    from exact.analysis.study_visualizer import create_study_visualizer_app

    app = create_study_visualizer_app(
        run_dir=run_dir,
        analysis_dir=analysis_dir,
        enable_ontology_info=args.enable_ontology_info,
        logger=logger,
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level=str(args.logging_level).lower())


if __name__ == "__main__":
    main()
