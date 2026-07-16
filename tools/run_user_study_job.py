#!/usr/bin/env python3
"""
Helper to launch a user-study analysis job from a YAML description.
"""

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exact user-study analysis from a YAML config."
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        required=True,
        help="YAML file describing the existing run directory and analysis parameters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved exact-user-study command without executing it.",
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


def build_user_study_command(cfg: Dict) -> List[str]:
    analysis_cfg = cfg["analysis"]
    job_cfg = cfg.get("job", {})

    run_dir = Path(analysis_cfg["run_dir"]).resolve()
    cmd = [
        "exact-user-study",
        "--run-dir",
        str(run_dir),
        "--top-k",
        str(analysis_cfg.get("top_k", 5)),
        "--per-rank",
        str(analysis_cfg.get("per_rank", 4)),
        "--shortlist-per-rank",
        str(analysis_cfg.get("shortlist_per_rank", 8)),
        "--seed",
        str(analysis_cfg.get("seed", 0)),
        "--logging-level",
        str(job_cfg.get("logging_level", analysis_cfg.get("logging_level", "INFO"))),
    ]

    output_dir = analysis_cfg.get("output_dir")
    if output_dir:
        cmd.extend(["--output-dir", str(Path(output_dir).resolve())])

    config_file = analysis_cfg.get("config_file")
    if config_file:
        cmd.extend(["--config-file", str(Path(config_file).resolve())])

    if analysis_cfg.get("device") is not None:
        cmd.extend(["--device", str(analysis_cfg["device"])])

    if analysis_cfg.get("generate_rationales", True):
        cmd.append("--generate-rationales")
    else:
        cmd.append("--skip-rationales")

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
    cmd = build_user_study_command(cfg)

    print("[run_user_study_job] Command:")
    print(" ", " ".join(shlex.quote(part) for part in cmd))

    if args.dry_run:
        return

    if args.sbatch_script:
        analysis_cfg = cfg["analysis"]
        job_cfg = cfg.get("job", {})
        slurm_cfg = job_cfg.get("slurm", {})
        run_dir = Path(analysis_cfg["run_dir"]).resolve()
        output_dir = analysis_cfg.get("output_dir")
        if output_dir:
            analysis_output_dir = Path(output_dir).resolve()
        else:
            analysis_output_dir = run_dir / "analysis" / "user_study"
        analysis_output_dir.mkdir(parents=True, exist_ok=True)

        env = {
            "JOB_NAME": _stringify(job_cfg.get("name", analysis_output_dir.name)),
            "RUN_DIR": str(run_dir),
            "OUTPUT_DIR": _stringify(analysis_cfg.get("output_dir")),
            "TOP_K": _stringify(analysis_cfg.get("top_k", 5)),
            "PER_RANK": _stringify(analysis_cfg.get("per_rank", 4)),
            "SHORTLIST_PER_RANK": _stringify(analysis_cfg.get("shortlist_per_rank", 8)),
            "SEED": _stringify(analysis_cfg.get("seed", 0)),
            "GENERATE_RATIONALES": "1" if analysis_cfg.get("generate_rationales", True) else "0",
            "CONFIG_FILE": _stringify(analysis_cfg.get("config_file")),
            "DEVICE": _stringify(analysis_cfg.get("device")),
            "LOGGING_LEVEL": _stringify(
                job_cfg.get("logging_level", analysis_cfg.get("logging_level", "INFO"))
            ),
        }

        log_dir = run_dir
        stdout = log_dir / "slurm_%j.out"
        stderr = log_dir / "slurm_%j.err"
        job_name = _stringify(job_cfg.get("name", analysis_output_dir.name))
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
