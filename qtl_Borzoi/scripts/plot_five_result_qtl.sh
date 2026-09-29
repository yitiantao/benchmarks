#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

OUTPUT_DIR="$ROOT_DIR/outputs/five_result_qtl_plots"
MPLCONFIGDIR="$OUTPUT_DIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

echo "[plot] AlphaGenome, Borzoi, NTv3 100M post, DNA-FM step150000/step340000"
echo "[plot] eQTL + sQTL + paQTL + iPaQTL"
echo "[plot] output=$OUTPUT_DIR"

PYTHONPATH="$ROOT_DIR/benchmark:$ROOT_DIR/src" \
  conda run --no-capture-output -n borzoi_py310 \
  python -m qtl_benchmark.visualize_four_models \
  --root "$ROOT_DIR" \
  --output-dir "$OUTPUT_DIR"
