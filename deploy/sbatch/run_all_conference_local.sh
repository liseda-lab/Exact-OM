#!/bin/sh
# Submit all conference local experiments using deploy/sbatch/exact_tune_run.sh.
# Usage: ./deploy/sbatch/run_all_conference_local.sh [root_dir]
#   root_dir: defaults to exp/local_conference
#
# Optional environment variables:
#   SBATCH_SCRIPT : sbatch wrapper to use (default: deploy/sbatch/exact_tune_run.sh)
#   DRY_RUN       : set to 1 to only print commands without submitting
#   VENV          : path to virtualenv to activate before submission

ROOT_DIR="${1:-exp/local_conference}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-deploy/sbatch/exact_tune_run.sh}"
DRY_RUN="${DRY_RUN:-0}"
VENV="${VENV:-$HOME/Exact-OM/.venv}"

if [ -d "$VENV" ]; then
  # Activate venv so sbatch job finds the right python/exact binary.
  # shellcheck source=/dev/null
  . "$VENV/bin/activate"
fi

if [ ! -d "$ROOT_DIR" ]; then
  echo "Root directory '$ROOT_DIR' not found." >&2
  exit 1
fi

find "$ROOT_DIR" -maxdepth 3 -type f -name "*.yaml" ! -name "config.yaml" | sort | while read -r JOB_FILE; do
  LOCAL_DIR="$(dirname "$JOB_FILE")"
  CONFIG_FILE="$LOCAL_DIR/config.yaml"

  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Skipping $JOB_FILE (missing $CONFIG_FILE)" >&2
    continue
  fi

  python - "$JOB_FILE" "$CONFIG_FILE" "$SBATCH_SCRIPT" "$DRY_RUN" <<'PY'
import sys
import shlex
import subprocess
from pathlib import Path

import yaml

job_file = Path(sys.argv[1])
config_file = Path(sys.argv[2])
sbatch_script = sys.argv[3]
dry_run = sys.argv[4] == "1"

job_cfg = yaml.safe_load(job_file.read_text())
dataset = job_cfg["dataset"]
job = job_cfg["job"]
exp_dir = Path(job.get("output_dir", str(job_file.parent)))
exp_dir.mkdir(parents=True, exist_ok=True)
slurm_out = exp_dir / "slurm-%j.out"
slurm_err = exp_dir / "slurm-%j.err"

cmd = [
    "sbatch",
    "--output",
    str(slurm_out),
    "--error",
    str(slurm_err),
    sbatch_script,
    "--job-name",
    job.get("name", job_file.stem),
    "--exp-dir",
    str(exp_dir),
    "--config-file",
    str(config_file),
    "--data-dir",
    dataset["data_dir"],
    "--source",
    dataset["source"],
    "--target",
    dataset["target"],
]

if dataset.get("train_reference"):
    cmd += ["--train-reference", dataset["train_reference"]]
if dataset.get("full_reference"):
    cmd += ["--full-reference", dataset["full_reference"]]
if dataset.get("candidates"):
    cmd += ["--candidates", dataset["candidates"]]

if job.get("memory"):
    cmd += ["--memory", str(job["memory"])]
if job.get("device") is not None:
    cmd += ["--device", str(job["device"])]
if job.get("run_eval", True):
    cmd.append("--run-eval")
if job.get("save_logs", True):
    cmd.append("--save-logs")

print(" ".join(shlex.quote(c) for c in cmd))
if not dry_run:
    subprocess.run(cmd, check=True)
PY
done
