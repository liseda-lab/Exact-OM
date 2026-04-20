#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${EXACT_STUDY_RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "[study-visualizer] EXACT_STUDY_RUN_DIR is required" >&2
  exit 1
fi

ANALYSIS_DIR="${EXACT_STUDY_ANALYSIS_DIR:-${RUN_DIR}/analysis/user_study}"
HOST="${EXACT_STUDY_HOST:-0.0.0.0}"
PORT_VALUE="${PORT:-${EXACT_STUDY_PORT:-10000}}"
LOG_LEVEL="${EXACT_STUDY_LOG_LEVEL:-INFO}"

ARGS=(
  --run-dir "${RUN_DIR}"
  --analysis-dir "${ANALYSIS_DIR}"
  --host "${HOST}"
  --port "${PORT_VALUE}"
  --logging-level "${LOG_LEVEL}"
)

ENABLE_ONTOLOGY="${EXACT_STUDY_ENABLE_ONTOLOGY_INFO:-true}"
case "${ENABLE_ONTOLOGY,,}" in
  1|true|yes|y|on) ;;
  *) ARGS+=(--disable-ontology-info) ;;
esac

echo "[study-visualizer] starting study_visualizer_runtime" >&2
echo "[study-visualizer] run_dir=${RUN_DIR}" >&2
echo "[study-visualizer] analysis_dir=${ANALYSIS_DIR}" >&2
echo "[study-visualizer] ontology_info=${ENABLE_ONTOLOGY}" >&2
echo "[study-visualizer] port=${PORT_VALUE}" >&2

exec python -m study_visualizer_runtime.cli "${ARGS[@]}"
