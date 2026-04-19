#!/bin/bash
#SBATCH --job-name=exact_user_study
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch deploy/sbatch/exact_user_study_run.sh [options]

Options (all may also be provided via env vars):
  --job-name NAME
  --run-dir PATH
  --output-dir PATH
  --top-k INT
  --per-rank INT
  --shortlist-per-rank INT
  --seed INT
  --generate-rationales | --skip-rationales
  --config-file PATH
  --device ID
  --jvm-heap-size SIZE
  --logging-level LEVEL
  --help
EOF
}

JOB_NAME=${JOB_NAME:-exact_user_study}
RUN_DIR=${RUN_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
TOP_K=${TOP_K:-5}
PER_RANK=${PER_RANK:-4}
SHORTLIST_PER_RANK=${SHORTLIST_PER_RANK:-8}
SEED=${SEED:-0}
GENERATE_RATIONALES=${GENERATE_RATIONALES:-1}
CONFIG_FILE=${CONFIG_FILE:-}
DEVICE=${DEVICE:-}
JVM_HEAP_SIZE=${JVM_HEAP_SIZE:-32G}
LOGGING_LEVEL=${LOGGING_LEVEL:-INFO}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --per-rank) PER_RANK="$2"; shift 2 ;;
    --shortlist-per-rank) SHORTLIST_PER_RANK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --generate-rationales) GENERATE_RATIONALES=1; shift ;;
    --skip-rationales) GENERATE_RATIONALES=0; shift ;;
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --jvm-heap-size) JVM_HEAP_SIZE="$2"; shift 2 ;;
    --logging-level) LOGGING_LEVEL="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "Missing --run-dir" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

cmd=(exact-user-study --run-dir "$RUN_DIR" --top-k "$TOP_K" --per-rank "$PER_RANK" --shortlist-per-rank "$SHORTLIST_PER_RANK" --seed "$SEED" --jvm-heap-size "$JVM_HEAP_SIZE" --logging-level "$LOGGING_LEVEL")

if [[ -n "$OUTPUT_DIR" ]]; then
  cmd+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$CONFIG_FILE" ]]; then
  cmd+=(--config-file "$CONFIG_FILE")
fi
if [[ -n "$DEVICE" ]]; then
  cmd+=(--device "$DEVICE")
fi
if [[ "$GENERATE_RATIONALES" == "1" ]]; then
  cmd+=(--generate-rationales)
else
  cmd+=(--skip-rationales)
fi

echo "[exact_user_study_run] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
