#!/bin/bash
#SBATCH --job-name=exact_tune
#SBATCH --partition=tier3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:1

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch deploy/sbatch/exact_tune_run.sh [options]

Options (all may also be provided via env vars):
  --job-name NAME
  --exp-dir PATH
  --config-file PATH
  --data-dir PATH
  --source FILE
  --target FILE
  --train-reference FILE
  --full-reference FILE
  --candidates FILE
  --memory SIZE
  --device ID
  --run-eval | --no-run-eval
  --save-logs | --no-save-logs
  --help
EOF
}

JOB_NAME=${JOB_NAME:-exact_tune}
EXP_DIR=${EXP_DIR:-}
CONFIG_FILE=${CONFIG_FILE:-}
DATA_DIR=${DATA_DIR:-data}
SOURCE=${SOURCE:-}
TARGET=${TARGET:-}
TRAIN_REFERENCE=${TRAIN_REFERENCE:-}
FULL_REFERENCE=${FULL_REFERENCE:-}
CANDIDATES=${CANDIDATES:-}
MEMORY=${MEMORY:-60G}
DEVICE=${DEVICE:-}
RUN_EVAL=${RUN_EVAL:-0}
SAVE_LOGS=${SAVE_LOGS:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --train-reference) TRAIN_REFERENCE="$2"; shift 2 ;;
    --full-reference) FULL_REFERENCE="$2"; shift 2 ;;
    --candidates) CANDIDATES="$2"; shift 2 ;;
    --memory) MEMORY="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --run-eval) RUN_EVAL=1; shift ;;
    --no-run-eval) RUN_EVAL=0; shift ;;
    --save-logs) SAVE_LOGS=1; shift ;;
    --no-save-logs) SAVE_LOGS=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$EXP_DIR" ]]; then
  EXP_DIR="exp/tuning/$JOB_NAME"
fi
mkdir -p "$EXP_DIR"

if [[ -z "$CONFIG_FILE" ]]; then
  echo "Missing --config-file" >&2
  exit 1
fi

resolve_path() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo ""
  elif [[ "$value" == /* ]]; then
    echo "$value"
  else
    echo "$DATA_DIR/$value"
  fi
}

SOURCE_PATH=$(resolve_path "$SOURCE")
TARGET_PATH=$(resolve_path "$TARGET")
TRAIN_REFERENCE_PATH=$(resolve_path "$TRAIN_REFERENCE")
FULL_REFERENCE_PATH=$(resolve_path "$FULL_REFERENCE")
CANDIDATES_PATH=$(resolve_path "$CANDIDATES")

cmd=(exact -s "$SOURCE_PATH" -t "$TARGET_PATH" -o "$EXP_DIR")

if [[ -n "$TRAIN_REFERENCE_PATH" ]]; then
  cmd+=(-r "$TRAIN_REFERENCE_PATH")
fi
if [[ -n "$FULL_REFERENCE_PATH" ]]; then
  cmd+=(-f "$FULL_REFERENCE_PATH")
fi
if [[ -n "$CANDIDATES_PATH" ]]; then
  cmd+=(-c "$CANDIDATES_PATH")
fi
cmd+=(-y "$CONFIG_FILE")
if [[ "$SAVE_LOGS" == "1" ]]; then
  cmd+=(-l)
fi
if [[ "$RUN_EVAL" == "1" ]]; then
  cmd+=(-e)
fi
cmd+=(-m "$MEMORY")
if [[ -n "$DEVICE" ]]; then
  cmd+=(-d "$DEVICE")
fi

echo "[exact_tune_run] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
