#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GPU_ID=${BORZOI_GPU_ID:-0}

# One-click single-GPU Borzoi launcher. All benchmark arguments are forwarded.
exec "$ROOT_DIR/scripts/run_borzoi_multi_gpu.sh" --gpus "$GPU_ID" "$@"
