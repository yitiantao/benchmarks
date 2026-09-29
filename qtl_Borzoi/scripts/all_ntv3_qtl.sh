#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Formal NTv3 100M post-training run. Keep the benchmark selection aligned
# with AlphaGenome and DNA-FM, and use DNA-FM's 131,072 bp context.
MAX_VARIANTS=2000
GPU_IDS=0,1,2,3,4,5,6,7
SEQUENCE_LENGTH=131072
OUTPUT_DIR=outputs/ntv3_100m_post_all_qtl

echo "[run] model=ntv3-100m-post tasks=eqtl,sqtl,paqtl,ipaqtl"
echo "[run] gpus=$GPU_IDS max_variants=$MAX_VARIANTS batch_size=1"
echo "[run] sequence_length=$SEQUENCE_LENGTH"
echo "[run] model_config=configs/ntv3_100m_post.json"
echo "[run] output=$OUTPUT_DIR"

for task in eqtl sqtl paqtl ipaqtl; do
    ./scripts/run_ntv3_multi_gpu.sh \
      --gpus "$GPU_IDS" \
      --tasks "$task" \
      --max-variants "$MAX_VARIANTS" \
      --batch-size 1 \
      --sequence-length "$SEQUENCE_LENGTH" \
      --model-config configs/ntv3_100m_post.json \
      --data-dir data \
      --output-dir "$OUTPUT_DIR"
done
