#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MAX_VARIANTS=2000
GPU_IDS=${DNA_FM_GPU_IDS:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=outputs/dna_fm_step_00150000_fp32_all_qtl
MODEL_CONFIG=configs/dna_fm_100m_post_step_00150000_fp32.json

echo "[run] model=DNA-FM-step150000-FP32 tasks=eqtl,sqtl,paqtl,ipaqtl"
echo "[run] gpus=$GPU_IDS max_variants=$MAX_VARIANTS batch_size=1"
echo "[run] model_config=$MODEL_CONFIG"
echo "[run] output=$OUTPUT_DIR"

./scripts/run_dna_fm_multi_gpu.sh \
  --gpus "$GPU_IDS" \
  --tasks all \
  --max-variants "$MAX_VARIANTS" \
  --batch-size 1 \
  --model-config "$MODEL_CONFIG" \
  --data-dir data \
  --output-dir "$OUTPUT_DIR"
