"""Command-line entry point for local and deployed alignment inspection."""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Sequence

import yaml

from .settings import InspectSettings


def _add_server_arguments(parser: argparse.ArgumentParser, *, include_run_dir: bool) -> None:
    if include_run_dir:
        parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--frontend-dir", type=Path)
    parser.add_argument("--source-ontology", dest="source_ontology_path", type=Path)
    parser.add_argument("--target-ontology", dest="target_ontology_path", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--reload", action="store_true")
    ontology = parser.add_mutually_exclusive_group()
    ontology.add_argument(
        "--enable-ontology-info", dest="enable_ontology_info", action="store_true"
    )
    ontology.add_argument(
        "--disable-ontology-info", dest="enable_ontology_info", action="store_false"
    )
    parser.set_defaults(enable_ontology_info=None)
    parser.add_argument("--log-level", "--logging-level", dest="log_level")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exact-inspect",
        description="Inspect Exact alignment runs locally or serve a fixed bundle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Serve the configured run or deployment bundle.")
    _add_server_arguments(serve, include_run_dir=True)

    open_parser = commands.add_parser("open", help="Open an alignment run in the local viewer.")
    open_parser.add_argument("run_dir", type=Path)
    _add_server_arguments(open_parser, include_run_dir=False)
    open_parser.add_argument("--no-browser", action="store_true")

    bundle = commands.add_parser("bundle", help="Export a portable inspection bundle.")
    bundle.add_argument("run_dir", type=Path, nargs="?")
    bundle.add_argument("bundle_dir", type=Path, nargs="?")
    bundle.add_argument("--analysis-dir", type=Path)
    bundle.add_argument("--config-path", type=Path)
    bundle.add_argument("--bundle-name")
    bundle.add_argument("--source-ontology", dest="source_ontology_path", type=Path)
    bundle.add_argument("--target-ontology", dest="target_ontology_path", type=Path)
    bundle.add_argument("--overwrite", action="store_true")
    bundle.add_argument("--log-level", "--logging-level", dest="log_level", default="INFO")
    bundle.add_argument(
        "--job-config",
        "--run-config",
        dest="job_config",
        type=Path,
        help="Read bundle and optional Slurm parameters from the historical YAML job format.",
    )
    bundle.add_argument("--dry-run", action="store_true")
    bundle.add_argument("--sbatch-script", type=Path)
    return parser


def _settings_from_args(args: argparse.Namespace) -> InspectSettings:
    values = {
        key: getattr(args, key, None)
        for key in (
            "run_dir",
            "analysis_dir",
            "frontend_dir",
            "source_ontology_path",
            "target_ontology_path",
            "enable_ontology_info",
            "host",
            "port",
            "log_level",
        )
        if getattr(args, key, None) is not None
    }
    return InspectSettings(**values)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _uvicorn_run(settings: InspectSettings, *, reload: bool) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - isolated package test
        raise RuntimeError("exact-inspect requires Uvicorn. Install `exact-om[viz]`.") from exc
    from .app import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


def _load_job(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.expanduser().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read bundle job config {path}: {exc}") from exc
    if not isinstance(payload.get("bundle"), dict):
        raise ValueError(f"Bundle job config must define a `bundle` mapping: {path}")
    return payload


def _bundle_values(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    values = {
        "run_dir": args.run_dir,
        "bundle_dir": args.bundle_dir,
        "analysis_dir": args.analysis_dir,
        "config_path": args.config_path,
        "bundle_name": args.bundle_name,
        "source_ontology_path": args.source_ontology_path,
        "target_ontology_path": args.target_ontology_path,
        "overwrite": args.overwrite,
        "log_level": args.log_level,
    }
    job: dict[str, Any] = {}
    if args.job_config:
        payload = _load_job(args.job_config)
        job = dict(payload.get("job") or {})
        configured = dict(payload["bundle"])
        for key in (
            "run_dir",
            "bundle_dir",
            "analysis_dir",
            "config_path",
            "bundle_name",
            "source_ontology_path",
            "target_ontology_path",
        ):
            if values[key] is None and configured.get(key) is not None:
                values[key] = configured[key]
        if not values["overwrite"]:
            values["overwrite"] = bool(configured.get("overwrite", False))
        if values["log_level"] == "INFO":
            values["log_level"] = job.get("logging_level", configured.get("logging_level", "INFO"))
    if values["run_dir"] is None or values["bundle_dir"] is None:
        raise ValueError(
            "`exact-inspect bundle` requires RUN_DIR and BUNDLE_DIR, either as positional "
            "arguments or in --job-config."
        )
    return values, job


def _bundle_command(values: dict[str, Any]) -> list[str]:
    command = [
        "exact-inspect",
        "bundle",
        str(Path(values["run_dir"]).expanduser().resolve()),
        str(Path(values["bundle_dir"]).expanduser().resolve()),
        "--log-level",
        str(values["log_level"]),
    ]
    for key, flag in (
        ("analysis_dir", "--analysis-dir"),
        ("config_path", "--config-path"),
        ("bundle_name", "--bundle-name"),
        ("source_ontology_path", "--source-ontology"),
        ("target_ontology_path", "--target-ontology"),
    ):
        if values.get(key) is not None:
            command.extend((flag, str(values[key])))
    if values.get("overwrite"):
        command.append("--overwrite")
    return command


def _submit_bundle_job(script: Path, values: dict[str, Any], job: dict[str, Any]) -> None:
    slurm = dict(job.get("slurm") or {})
    export_values = {
        "JOB_NAME": str(job.get("name") or f"bundle_{Path(values['run_dir']).name}"),
        "RUN_DIR": str(Path(values["run_dir"]).expanduser().resolve()),
        "BUNDLE_DIR": str(Path(values["bundle_dir"]).expanduser().resolve()),
        "ANALYSIS_DIR": str(values.get("analysis_dir") or ""),
        "CONFIG_PATH": str(values.get("config_path") or ""),
        "BUNDLE_NAME": str(values.get("bundle_name") or ""),
        "LOGGING_LEVEL": str(values["log_level"]),
        "OVERWRITE": "1" if values.get("overwrite") else "0",
    }
    command = ["sbatch"]
    sbatch_args = slurm.get("sbatch_args") or []
    if isinstance(sbatch_args, str):
        sbatch_args = [sbatch_args]
    command.extend(str(value) for value in sbatch_args)
    exported = ",".join(f"{key}={value}" for key, value in export_values.items())
    command.extend((f"--export=ALL,{exported}", str(script.expanduser().resolve())))
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"serve", "open"}:
        settings = _settings_from_args(args)
        _configure_logging(settings.log_level)
        if args.command == "open" and not args.no_browser:
            webbrowser.open(f"http://{settings.host}:{settings.port}/")
        _uvicorn_run(settings, reload=args.reload)
        return 0

    values, job = _bundle_values(args)
    command = _bundle_command(values)
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0
    if args.sbatch_script:
        _submit_bundle_job(args.sbatch_script, values, job)
        return 0

    from .bundles import export_bundle

    _configure_logging(str(values["log_level"]))
    export_bundle(
        Path(values["run_dir"]),
        Path(values["bundle_dir"]),
        analysis_dir=Path(values["analysis_dir"]) if values.get("analysis_dir") else None,
        config_path=Path(values["config_path"]) if values.get("config_path") else None,
        bundle_name=values.get("bundle_name"),
        source_ontology_path=(
            Path(values["source_ontology_path"]) if values.get("source_ontology_path") else None
        ),
        target_ontology_path=(
            Path(values["target_ontology_path"]) if values.get("target_ontology_path") else None
        ),
        overwrite=bool(values["overwrite"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main"]
