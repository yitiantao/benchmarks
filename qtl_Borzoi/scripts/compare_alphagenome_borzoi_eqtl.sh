#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MAX_VARIANTS=2000
LIMIT_SCOPE=global
MIN_VARIANTS=32
CONDA_ENV=${BORZOI_CONDA_ENV:-borzoi_py310}
ALPHAGENOME_OUTPUT="$ROOT_DIR/outputs/alphagenome_all_qtl"
BORZOI_OUTPUT="$ROOT_DIR/outputs/borzoi_all_qtl"
OUTPUT_DIR="$ROOT_DIR/outputs/alphagenome_borzoi_eqtl_comparison"
BORZOI_MODEL_NAME=borzoi-replicate-0

usage() {
  sed -n '2,30p' "$0" | sed -n 's/^# //p'
}

# Evaluate AlphaGenome and Borzoi on the exact same Borzoi eQTL subset.
#
# The comparison keeps each model's native result and adds an AlphaGenome
# view grouped with Borzoi's broad tissue keywords.
#
# Options:
#   --max-variants N       Borzoi subset tag to compare (default: 2000).
#   --limit-scope S        global or per-tissue (default: global).
#   --min-variants N       Minimum causal rows per tissue (default: 32).
#   --alphagenome-output D AlphaGenome output root.
#   --borzoi-output D      Borzoi output root.
#   --borzoi-model-name N  Unified Borzoi model name (default: borzoi-replicate-0).
#   --output-dir D         Comparison output root.
#   --conda-env NAME       Environment used for evaluation.
#   -h, --help             Show this help.

while (($#)); do
  case "$1" in
    --max-variants) MAX_VARIANTS=$2; shift 2 ;;
    --limit-scope) LIMIT_SCOPE=$2; shift 2 ;;
    --min-variants) MIN_VARIANTS=$2; shift 2 ;;
    --alphagenome-output) ALPHAGENOME_OUTPUT=$2; shift 2 ;;
    --borzoi-output) BORZOI_OUTPUT=$2; shift 2 ;;
    --borzoi-model-name) BORZOI_MODEL_NAME=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --conda-env) CONDA_ENV=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$MAX_VARIANTS" =~ ^[0-9]+$ ]] || {
  echo "--max-variants 必须是大于或等于 0 的整数。" >&2
  exit 2
}
[[ "$MIN_VARIANTS" =~ ^[0-9]+$ ]] || {
  echo "--min-variants 必须是大于或等于 0 的整数。" >&2
  exit 2
}
case "$LIMIT_SCOPE" in
  global|per-tissue) ;;
  *) echo "--limit-scope 必须是 global 或 per-tissue。" >&2; exit 2 ;;
esac

if ((MAX_VARIANTS)); then BASE_TAG="max_$MAX_VARIANTS"; else BASE_TAG=all; fi
if ((MAX_VARIANTS)) && [[ "$LIMIT_SCOPE" == per-tissue ]]; then
  SHARED_TAG="per_tissue_max_$MAX_VARIANTS"
else
  SHARED_TAG=$BASE_TAG
fi
DATA_DIR="$BORZOI_OUTPUT/eqtl/inputs/$SHARED_TAG/eqtl_pip90"
ALPHA_STORE="$ALPHAGENOME_OUTPUT/$BASE_TAG/alphagenome-all-folds/gtex-eqtl-$BASE_TAG/predictions.sqlite"
BORZOI_STORE="$BORZOI_OUTPUT/unified/$SHARED_TAG/$BORZOI_MODEL_NAME/gtex-eqtl-$SHARED_TAG/predictions.sqlite"
RESULT_DIR="$OUTPUT_DIR/$SHARED_TAG/$BORZOI_MODEL_NAME"

[[ -d "$DATA_DIR" ]] || { echo "缺少共享 eQTL 输入: $DATA_DIR" >&2; exit 1; }
[[ -f "$ALPHA_STORE" ]] || { echo "缺少 AlphaGenome 缓存: $ALPHA_STORE" >&2; exit 1; }
[[ -f "$BORZOI_STORE" ]] || { echo "缺少 Borzoi 缓存: $BORZOI_STORE" >&2; exit 1; }

export PYTHONPATH="$ROOT_DIR/benchmark:$ROOT_DIR/src:${PYTHONPATH:-}"
conda run --no-capture-output -n "$CONDA_ENV" python -m \
  qtl_benchmark.compare_eqtl_models \
  --data-dir "$DATA_DIR" \
  --alphagenome-store "$ALPHA_STORE" \
  --borzoi-store "$BORZOI_STORE" \
  --output-dir "$RESULT_DIR" \
  --min-variants "$MIN_VARIANTS" \
  --limit-scope "$LIMIT_SCOPE" \
  --max-variants "$MAX_VARIANTS"
