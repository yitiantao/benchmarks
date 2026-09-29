#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MAX_VARIANTS=${ALPHAGENOME_ALL_QTL_MAX_VARIANTS:-2000}

# AlphaGenome's eQTL loader applies this limit per tissue and split. The
# formal Borzoi launcher uses the same per-tissue scope.
echo "[run] eqtl_limit_scope=per-tissue max_variants=$MAX_VARIANTS"

for task in eqtl sqtl paqtl ipaqtl; do
    ./scripts/run_alphagenome_multi_gpu.sh \
      --gpus 0,1,2,3,4,5,6,7 \
      --tasks "$task" \
      --max-variants "$MAX_VARIANTS" \
      --variant-batch-size 2 \
      --model-config configs/alphagenome_local.json \
      --checkpoint-path /data/yitian_workspace/DNA/alphagenome_models/all_folds \
      --model-name alphagenome-all-folds \
      --data-dir data \
      --output-dir outputs/alphagenome_all_qtl
done
