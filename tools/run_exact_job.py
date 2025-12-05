#!/usr/bin/env python3
"""
Helper to launch a single Exact run from a YAML description.
"""
import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exact with a YAML config.")
    parser.add_argument(
        "--run-config",
        type=Path,
        required=True,
        help="YAML file describing dataset + runtime parameters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing.",
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


def resolve(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def build_exact_command(cfg: Dict) -> List[str]:
    dataset_cfg = cfg["dataset"]
    job_cfg = cfg["job"]

    data_dir = Path(dataset_cfg["data_dir"]).resolve()

    def dataset_path(key: str) -> str | None:
        value = dataset_cfg.get(key)
        if not value:
            return None
        p = Path(value)
        if not p.is_absolute():
            p = data_dir / p
        return str(p.resolve())

    cmd = [
        "exact",
        "-s",
        dataset_path("source"),
        "-t",
        dataset_path("target"),
        "-o",
        str(Path(job_cfg["output_dir"]).resolve()),
        "-y",
        str(Path(job_cfg["config_file"]).resolve()),
        "-m",
        str(job_cfg.get("memory", "60G")),
    ]

    if dataset_cfg.get("train_reference"):
        cmd.extend(["-r", dataset_path("train_reference")])
    if dataset_cfg.get("full_reference"):
        cmd.extend(["-f", dataset_path("full_reference")])
    if dataset_cfg.get("candidates"):
        cmd.extend(["-c", dataset_path("candidates")])
    if job_cfg.get("save_logs", False):
        cmd.append("-l")
    if job_cfg.get("run_eval", False):
        cmd.append("-e")
    if "device" in job_cfg and job_cfg["device"] is not None:
        cmd.extend(["-d", str(job_cfg["device"])])

    return [str(part) for part in cmd if part]


def submit_via_sbatch(
    script: Path, env: Dict[str, str], stdout: Path | None = None, stderr: Path | None = None
) -> None:
    if stdout:
        stdout.parent.mkdir(parents=True, exist_ok=True)
    if stderr:
        stderr.parent.mkdir(parents=True, exist_ok=True)
    export = ",".join(f"{k}={v}" for k, v in env.items())
    cmd = ["sbatch"]
    if stdout:
        cmd.extend(["--output", str(stdout)])
    if stderr:
        cmd.extend(["--error", str(stderr)])
    cmd.extend([f"--export=ALL,{export}", str(script)])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.run_config)
    cmd = build_exact_command(cfg)

    print("[run_exact_job] Command:")
    print(" ", " ".join(shlex.quote(part) for part in cmd))

    if args.dry_run:
        return

    if args.sbatch_script:
        dataset_cfg = cfg["dataset"]
        job_cfg = cfg["job"]
        exp_dir = Path(job_cfg["output_dir"]).resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)

        def env_value(value):
            if value is None:
                return ""
            if isinstance(value, Path):
                return str(value)
            return str(value)

        env = {
            "JOB_NAME": job_cfg.get("name", Path(job_cfg["output_dir"]).name),
            "EXP_DIR": str(exp_dir),
            "CONFIG_FILE": str(Path(job_cfg["config_file"]).resolve()),
            "DATA_DIR": str(Path(dataset_cfg["data_dir"]).resolve()),
            "SOURCE": dataset_cfg["source"],
            "TARGET": dataset_cfg["target"],
            "TRAIN_REFERENCE": env_value(dataset_cfg.get("train_reference")),
            "FULL_REFERENCE": env_value(dataset_cfg.get("full_reference")),
            "CANDIDATES": env_value(dataset_cfg.get("candidates")),
            "MEMORY": str(job_cfg.get("memory", "60G")),
            "RUN_EVAL": "1" if job_cfg.get("run_eval") else "0",
            "SAVE_LOGS": "1" if job_cfg.get("save_logs") else "0",
            "DEVICE": env_value(job_cfg.get("device")),
        }
        stdout = exp_dir / "slurm_%j.out"
        stderr = exp_dir / "slurm_%j.err"
        submit_via_sbatch(args.sbatch_script, env, stdout=stdout, stderr=stderr)
    else:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
