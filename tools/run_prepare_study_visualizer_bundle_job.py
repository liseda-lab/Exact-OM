#!/usr/bin/env python3
"""
Helper to launch a study-visualizer bundle export from a YAML description.
"""
import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a study visualizer bundle from a YAML config."
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        required=True,
        help="YAML file describing the existing run directory and bundle-export parameters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    parser.add_argument(
        "--sbatch-script",
        type=Path,
        help="Optional sbatch script to submit instead of running locally.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _normalize_sbatch_args(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    raise TypeError(f"Unsupported sbatch_args value: {type(value)!r}")


def build_bundle_command(cfg: Dict) -> List[str]:
    bundle_cfg = cfg["bundle"]
    job_cfg = cfg.get("job", {})

    cmd = [
        "poetry",
        "run",
        "python",
        "tools/prepare_study_visualizer_bundle.py",
        "--run-dir",
        str(Path(bundle_cfg["run_dir"]).resolve()),
        "--bundle-dir",
        str(Path(bundle_cfg["bundle_dir"]).resolve()),
        "--jvm-heap-size",
        str(bundle_cfg.get("jvm_heap_size", "8G")),
        "--logging-level",
        str(job_cfg.get("logging_level", bundle_cfg.get("logging_level", "INFO"))),
    ]

    analysis_dir = bundle_cfg.get("analysis_dir")
    if analysis_dir:
        cmd.extend(["--analysis-dir", str(Path(analysis_dir).resolve())])

    config_path = bundle_cfg.get("config_path")
    if config_path:
        cmd.extend(["--config-path", str(Path(config_path).resolve())])

    bundle_name = bundle_cfg.get("bundle_name")
    if bundle_name:
        cmd.extend(["--bundle-name", str(bundle_name)])

    if bundle_cfg.get("overwrite", False):
        cmd.append("--overwrite")

    return [str(part) for part in cmd if part]


def submit_via_sbatch(
    script: Path,
    env: Dict[str, str],
    job_name: str | None = None,
    sbatch_args: List[str] | None = None,
    stdout: Path | None = None,
    stderr: Path | None = None,
) -> None:
    if stdout:
        stdout.parent.mkdir(parents=True, exist_ok=True)
    if stderr:
        stderr.parent.mkdir(parents=True, exist_ok=True)
    export = ",".join(f"{k}={v}" for k, v in env.items())
    cmd = ["sbatch"]
    if job_name:
        cmd.extend(["--job-name", job_name])
    if stdout:
        cmd.extend(["--output", str(stdout)])
    if stderr:
        cmd.extend(["--error", str(stderr)])
    if sbatch_args:
        cmd.extend(sbatch_args)
    cmd.extend([f"--export=ALL,{export}", str(script)])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.run_config)
    cmd = build_bundle_command(cfg)

    print("[run_prepare_study_visualizer_bundle_job] Command:")
    print(" ", " ".join(shlex.quote(part) for part in cmd))

    if args.dry_run:
        return

    if args.sbatch_script:
        bundle_cfg = cfg["bundle"]
        job_cfg = cfg.get("job", {})
        slurm_cfg = job_cfg.get("slurm", {})
        run_dir = Path(bundle_cfg["run_dir"]).resolve()
        bundle_dir = Path(bundle_cfg["bundle_dir"]).resolve()
        bundle_dir.parent.mkdir(parents=True, exist_ok=True)

        env = {
            "JOB_NAME": _stringify(job_cfg.get("name", f"bundle_{run_dir.name}")),
            "RUN_DIR": str(run_dir),
            "BUNDLE_DIR": str(bundle_dir),
            "ANALYSIS_DIR": _stringify(bundle_cfg.get("analysis_dir")),
            "CONFIG_PATH": _stringify(bundle_cfg.get("config_path")),
            "BUNDLE_NAME": _stringify(bundle_cfg.get("bundle_name")),
            "JVM_HEAP_SIZE": _stringify(bundle_cfg.get("jvm_heap_size", "8G")),
            "LOGGING_LEVEL": _stringify(
                job_cfg.get("logging_level", bundle_cfg.get("logging_level", "INFO"))
            ),
            "OVERWRITE": "1" if bundle_cfg.get("overwrite", False) else "0",
        }

        stdout = run_dir / "slurm_%j.out"
        stderr = run_dir / "slurm_%j.err"
        job_name = _stringify(job_cfg.get("name", f"bundle_{run_dir.name}"))
        sbatch_args = _normalize_sbatch_args(slurm_cfg.get("sbatch_args"))
        submit_via_sbatch(
            args.sbatch_script,
            env,
            job_name=job_name,
            sbatch_args=sbatch_args,
            stdout=stdout,
            stderr=stderr,
        )
    else:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
