#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=ntv3
TASK=all
VERSION=2026-02
GPU=0
LIMIT=0
OUTPUT_DIR="$ROOT/runs"
RESTART=()
while (($#)); do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --force) RESTART=(--force); shift ;;
    --resume) RESTART=(--resume); shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$MODEL" in
  ntv3) PY=/root/anaconda3/envs/ntv3/bin/python ;;
  borzoi) PY=/root/anaconda3/envs/borzoi_py310/bin/python ;;
  alphagenome) PY=/root/anaconda3/envs/alphagenome/bin/python ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m clinvar_benchmark.cli score \
  --model "$MODEL" --task "$TASK" --version "$VERSION" --limit "$LIMIT" \
  --device cuda --output-dir "$OUTPUT_DIR" "${RESTART[@]}"
"$PY" -m clinvar_benchmark.cli evaluate \
  --model "$MODEL" --task "$TASK" --version "$VERSION" --limit "$LIMIT" \
  --output-dir "$OUTPUT_DIR"
