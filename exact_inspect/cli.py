"""Prepare, serve and export local Exact Explain bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(prog="exact-inspect", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the local explorer and bundle library")
    serve.add_argument("--package", type=Path, help="Prepared package.json to open initially")
    serve.add_argument("--library-dir", type=Path)
    serve.add_argument("--frontend-dir", type=Path)
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug"))
    prepare = commands.add_parser(
        "prepare", help="Prepare ontologies and an optional saved alignment run"
    )
    prepare.add_argument("--source-ontology", type=Path)
    prepare.add_argument("--target-ontology", type=Path)
    prepare.add_argument("--run-dir", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--template",
        type=Path,
        help="Advanced JSON input template, including explicit imports or generation",
    )
    export = commands.add_parser("export", help="Export a prepared bundle as a portable ZIP")
    export.add_argument("--package", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def prepare_bundle(args):
    from exact.runs import RunLayout

    from .artifacts import atomic_json
    from .preparation import Preparation, bind_execution_lock

    if args.template:
        if args.source_ontology or args.target_ontology or args.run_dir:
            raise ValueError("Use --template or explicit ontology/run arguments, not both")
        template = json.loads(args.template.read_bytes())
        input_root = args.template.resolve().parent
    else:
        if not args.source_ontology or not args.target_ontology:
            raise ValueError("Pass --source-ontology and --target-ontology, or --template")
        input_root = Path.cwd()
        template = {
            "design_revision": "exact-explain-local/1",
            "ontologies": [
                {"name": "Source ontology", "root": {"path": str(args.source_ontology.resolve())}},
                {"name": "Target ontology", "root": {"path": str(args.target_ontology.resolve())}},
            ],
        }
        if args.run_dir:
            layout = RunLayout.open(args.run_dir)
            if not layout.manifest_path.is_file():
                raise ValueError(
                    "Expected an `exact align` run with run_manifest.json. Staged score, "
                    "decisions, acceptance and rank directories are not inspection bundles."
                )
            template["run"] = {
                "binding": {"path": str(layout.manifest_path)},
                "source_ontology": "Source ontology",
                "target_ontology": "Target ontology",
            }
    lock = bind_execution_lock(template, input_root=input_root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "execution.json", lock)
    report = Preparation(lock, output, input_root=input_root).run()
    manifest = output / report["outputs"]["portable-export"] / "package.json"
    print(json.dumps({"package": str(manifest)}, indent=2))
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Exact Explain requires Python 3.11 or 3.12; the matcher also supports 3.10")
    try:
        if args.command == "prepare":
            return prepare_bundle(args)
        if args.command == "export":
            from .artifacts import export_archive

            export_archive(args.package, args.output)
            return 0
        import uvicorn

        from .app import create_app
        from .settings import InspectSettings

        settings = InspectSettings(
            **{
                key: value
                for key, value in vars(args).items()
                if key != "command" and value is not None
            }
        )
        app = create_app(settings)
        print(f"Exact Explain: http://{settings.host}:{settings.port}/", flush=True)
        uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
        return 0
    except ImportError as exc:
        parser.exit(
            2, f"Missing optional dependency: {exc}. Install Exact-OM with the [viz] extra.\n"
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
