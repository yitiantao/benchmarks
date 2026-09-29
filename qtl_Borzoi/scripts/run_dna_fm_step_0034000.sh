#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Small, checkpoint-specific smoke test. Arguments supplied by the caller are
# appended last, so --tasks/--max-variants/--output-dir can override defaults.
exec "$ROOT_DIR/scripts/run_dna_fm_multi_gpu.sh" \
  --tasks all \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-variants 2000 \
  --batch-size 1 \
  --model-config "$ROOT_DIR/configs/dna_fm_100m_post_step_0034000.json" \
  --data-dir "$ROOT_DIR/data" \
  --output-dir "$ROOT_DIR/outputs/dna_fm_step_0034000_fp32_smoke" \
  "$@"
