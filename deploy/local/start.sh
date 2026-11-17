#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo"
if [ ! -f explanations_visualizer/out/index.html ]; then
    echo 'Build the frontend first: cd explanations_visualizer && npm ci && npm run build' >&2
    exit 1
fi
library=${EXACT_UI_LIBRARY:-${XDG_DATA_HOME:-$HOME/.local/share}/exact-explain/library}
mkdir -p "$library"
exec "${EXACT_UI_PYTHON:-python3}" -m exact_inspect.cli serve \
    --profile local_app --library-dir "$library" \
    --frontend-dir "$repo/explanations_visualizer/out" --host 127.0.0.1 --port 8000 "$@"
