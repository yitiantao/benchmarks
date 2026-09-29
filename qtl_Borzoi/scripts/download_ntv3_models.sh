#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_ENV=${NTV3_CONDA_ENV:-ntv3}

command -v conda >/dev/null 2>&1 || { echo "找不到 conda。" >&2; exit 1; }
cd "$ROOT_DIR"
exec conda run --no-capture-output -n "$CONDA_ENV" \
  python "$ROOT_DIR/scripts/download_ntv3_models.py" "$@"
