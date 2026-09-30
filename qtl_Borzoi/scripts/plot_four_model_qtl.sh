#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Backward-compatible alias. The comparison now contains six result sets.
exec "$ROOT_DIR/scripts/plot_six_result_qtl.sh" "$@"
