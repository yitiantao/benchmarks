#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MAX_VARIANTS=${BORZOI_ALL_QTL_MAX_VARIANTS:-2000}
MODEL_FILE=${BORZOI_ALL_QTL_MODEL:-models/replicate_0/model0_best.h5}
MODEL_NAME=${BORZOI_ALL_QTL_MODEL_NAME:-borzoi-replicate-0}
OUTPUT_DIR=${BORZOI_ALL_QTL_OUTPUT_DIR:-outputs/borzoi_replicate_0_all_qtl}

echo "[run] borzoi_model=$MODEL_FILE model_name=$MODEL_NAME output=$OUTPUT_DIR"

./scripts/run_borzoi_multi_gpu.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --tasks all \
  --max-variants "$MAX_VARIANTS" \
  --eqtl-limit-scope per-tissue \
  --variant-batch-size 2 \
  --model "$MODEL_FILE" \
  --model-name "$MODEL_NAME" \
  --params configs/params_pred.json \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir "$OUTPUT_DIR"

# Produce the matched eQTL comparison automatically when the corresponding
# AlphaGenome cache is already available. Native outputs remain untouched.
if [[ ${BORZOI_COMPARE_EQTL:-1} == 1 ]]; then
  ./scripts/compare_alphagenome_borzoi_eqtl.sh \
    --max-variants "$MAX_VARIANTS" \
    --limit-scope per-tissue \
    --borzoi-model-name "$MODEL_NAME" \
    --borzoi-output "$OUTPUT_DIR"
fi
