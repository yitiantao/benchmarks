#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Keep this aligned with scripts/all_alphagenome_qtl.sh so the two models use
# the same four tasks, input data, per-tissue eQTL limit, and evaluation code.
MAX_VARIANTS=2000
GPU_IDS=0,1,2,3,4,5,6,7
OUTPUT_DIR=outputs/dna_fm_step_0034000_all_qtl

echo "[run] model=DNA-FM tasks=eqtl,sqtl,paqtl,ipaqtl"
echo "[run] gpus=$GPU_IDS max_variants=$MAX_VARIANTS batch_size=1"
echo "[run] model_config=configs/dna_fm_100m_post_step_0034000.json"
echo "[run] output=$OUTPUT_DIR"

for task in eqtl sqtl paqtl ipaqtl; do
    ./scripts/run_dna_fm_multi_gpu.sh \
      --gpus "$GPU_IDS" \
      --tasks "$task" \
      --max-variants "$MAX_VARIANTS" \
      --batch-size 1 \
      --model-config configs/dna_fm_100m_post_step_0034000.json \
      --data-dir data \
      --output-dir "$OUTPUT_DIR"
done
