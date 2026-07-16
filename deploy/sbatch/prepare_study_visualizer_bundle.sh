#!/bin/bash
#SBATCH --job-name=exact
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch deploy/sbatch/prepare_study_visualizer_bundle.sh [options]

Options (all may also be provided via env vars):
  --job-name NAME
  --run-dir PATH
  --bundle-dir PATH
  --analysis-dir PATH
  --config-path PATH
  --bundle-name NAME
  --logging-level LEVEL
  --overwrite | --no-overwrite
  --help

Examples:
  sbatch deploy/sbatch/prepare_study_visualizer_bundle.sh \
    --run-dir /home/pgcotovio/Exact-OM/exp/test/Full_local_bioml_with_exp/omim-ordo \
    --bundle-dir /home/pgcotovio/Exact-OM/deploy/render/study_bundles/omim-ordo \
    --overwrite

  srun --cpus-per-task=8 --mem=32G --time=06:00:00 \
    bash deploy/sbatch/prepare_study_visualizer_bundle.sh \
      --run-dir /home/pgcotovio/Exact-OM/exp/test/Full_local_bioml_with_exp/omim-ordo \
      --bundle-dir /home/pgcotovio/Exact-OM/deploy/render/study_bundles/omim-ordo \
      --overwrite
EOF
}

JOB_NAME=${JOB_NAME:-prepare_study_bundle}
RUN_DIR=${RUN_DIR:-}
BUNDLE_DIR=${BUNDLE_DIR:-}
ANALYSIS_DIR=${ANALYSIS_DIR:-}
CONFIG_PATH=${CONFIG_PATH:-}
BUNDLE_NAME=${BUNDLE_NAME:-}
LOGGING_LEVEL=${LOGGING_LEVEL:-INFO}
OVERWRITE=${OVERWRITE:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
    --analysis-dir) ANALYSIS_DIR="$2"; shift 2 ;;
    --config-path) CONFIG_PATH="$2"; shift 2 ;;
    --bundle-name) BUNDLE_NAME="$2"; shift 2 ;;
    --logging-level) LOGGING_LEVEL="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --no-overwrite) OVERWRITE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "Missing --run-dir" >&2
  exit 1
fi

if [[ -z "$BUNDLE_DIR" ]]; then
  echo "Missing --bundle-dir" >&2
  exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" && -n "$JOB_NAME" ]]; then
  scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME" >/dev/null 2>&1 || true
fi

cmd=(poetry run python tools/prepare_study_visualizer_bundle.py --run-dir "$RUN_DIR" --bundle-dir "$BUNDLE_DIR" --logging-level "$LOGGING_LEVEL")

if [[ -n "$ANALYSIS_DIR" ]]; then
  cmd+=(--analysis-dir "$ANALYSIS_DIR")
fi
if [[ -n "$CONFIG_PATH" ]]; then
  cmd+=(--config-path "$CONFIG_PATH")
fi
if [[ -n "$BUNDLE_NAME" ]]; then
  cmd+=(--bundle-name "$BUNDLE_NAME")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "[prepare_study_visualizer_bundle] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
