#!/bin/bash
#SBATCH --job-name=exact_candidate_recall
#SBATCH --partition=gpu_hi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:1

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch deploy/sbatch/exact_candidate_recall_experiment.sh [options]

Options:
  --run-config PATH
  --output-dir PATH
  --top-k "20 40 60" or --top-k 20 40 60
  --jvm-heap-size SIZE
  --device ID
  --python-bin PATH
  --help
EOF
}

RUN_CONFIG=${RUN_CONFIG:-exp/test/Full_global/ncit_doid_global.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-exp/test/candidate_recall}
TOP_K_VALUES=()
JVM_HEAP_SIZE=${JVM_HEAP_SIZE:-32G}
DEVICE=${DEVICE:-}
PYTHON_BIN=${PYTHON_BIN:-./.venv/bin/python}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-config) RUN_CONFIG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --top-k)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TOP_K_VALUES+=("$1")
        shift
      done
      ;;
    --jvm-heap-size) JVM_HEAP_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#TOP_K_VALUES[@]} -eq 0 ]]; then
  TOP_K_VALUES=(20 40 60)
fi

mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi

cmd=("$PYTHON_BIN" tools/run_candidate_recall_experiment.py
  --run-config "$RUN_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --top-k "${TOP_K_VALUES[@]}"
  --jvm-heap-size "$JVM_HEAP_SIZE")

if [[ -n "$DEVICE" ]]; then
  cmd+=(--device "$DEVICE")
fi

echo "[exact_candidate_recall_experiment] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
