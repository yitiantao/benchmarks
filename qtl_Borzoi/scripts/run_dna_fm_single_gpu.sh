#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GPU_ID=${DNA_FM_GPU_ID:-0}
exec "$ROOT_DIR/scripts/run_dna_fm_multi_gpu.sh" --gpus "$GPU_ID" "$@"
