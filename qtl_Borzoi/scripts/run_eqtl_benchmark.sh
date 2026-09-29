#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR="$ROOT_DIR/data/eqtl"
OUTPUT_DIR="$ROOT_DIR/outputs/borzoi_eqtl"
CONDA_ENV="${BORZOI_CONDA_ENV:-borzoi_py310}"
PARAMS_FILE="$ROOT_DIR/configs/params_pred.json"
TARGETS_GTEX="$ROOT_DIR/configs/targets_gtex.txt"
TARGETS_HUMAN="$ROOT_DIR/configs/targets_human.txt"
MODES="sed"
MAX_VARIANTS=0
LIMIT_SCOPE=global
VARIANT_BATCH_SIZE=${BORZOI_VARIANT_BATCH_SIZE:-1}
ACTION="run"
MODEL_DIR="$ROOT_DIR/models"
declare -a MODEL_FILES=()

usage() {
  sed -n '2,59p' "$0" | sed -n 's/^# //p'
}

# 独立 Borzoi eQTL benchmark 入口（本地顺序调度，不要求 Slurm）。
#
# 常用命令：
#   ./scripts/run_eqtl_benchmark.sh
#   ./scripts/run_eqtl_benchmark.sh --modes all
#   ./scripts/run_eqtl_benchmark.sh --max-variants 100
#   ./scripts/run_eqtl_benchmark.sh --check
#
# 参数：
#   --modes LIST          sed、sad、sed,sad 或 all；默认 sed。
#   --max-variants N      每个 scope 的变异上限；0 为全量，默认 0。
#   --limit-scope SCOPE   global 或 per-tissue；正式跨模型比较使用 per-tissue。
#   --variant-batch-size N  每次模型调用的变异数；实际输入 2*N 条 REF/ALT 序列，默认 1。
#   --model FILE          Borzoi .h5 模型；可重复指定以组成 ensemble。
#   --model-dir DIR       搜索 model0_best.h5/model_best.h5；默认 models/。
#   --params FILE         模型 JSON；默认 configs/params_pred.json。
#   --targets-gtex FILE   SED target 表；默认 configs/targets_gtex.txt。
#   --targets-human FILE  SAD target 表；默认 configs/targets_human.txt。
#   --conda-env NAME      Conda 环境；默认 borzoi_py310。
#   --data-dir DIR        eQTL 数据目录；默认 data/eqtl。
#   --output-dir DIR      输出目录；默认 outputs/borzoi_eqtl。
#   --prepare-only        只解压/筛选 VCF，不运行模型。
#   --check               检查文件、环境和 TensorFlow GPU。
#   -h, --help            显示帮助。

while (($#)); do
  case "$1" in
    --modes) MODES=$2; shift 2 ;;
    --max-variants) MAX_VARIANTS=$2; shift 2 ;;
    --limit-scope) LIMIT_SCOPE=$2; shift 2 ;;
    --variant-batch-size) VARIANT_BATCH_SIZE=$2; shift 2 ;;
    --model) MODEL_FILES+=("$2"); shift 2 ;;
    --model-dir) MODEL_DIR=$2; shift 2 ;;
    --params) PARAMS_FILE=$2; shift 2 ;;
    --targets-gtex) TARGETS_GTEX=$2; shift 2 ;;
    --targets-human) TARGETS_HUMAN=$2; shift 2 ;;
    --conda-env) CONDA_ENV=$2; shift 2 ;;
    --data-dir) DATA_DIR=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --prepare-only) ACTION="prepare"; shift ;;
    --check) ACTION="check"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$MAX_VARIANTS" =~ ^[0-9]+$ ]] || {
  echo "--max-variants 必须是大于或等于 0 的整数。" >&2
  exit 2
}
[[ "$VARIANT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "--variant-batch-size 必须是大于 0 的整数。" >&2
  exit 2
}
case "$LIMIT_SCOPE" in
  global|per-tissue) ;;
  *) echo "--limit-scope 必须是 global 或 per-tissue。" >&2; exit 2 ;;
esac

abspath() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

DATA_DIR=$(abspath "$DATA_DIR")
OUTPUT_DIR=$(abspath "$OUTPUT_DIR")
MODEL_DIR=$(abspath "$MODEL_DIR")
PARAMS_FILE=$(abspath "$PARAMS_FILE")
TARGETS_GTEX=$(abspath "$TARGETS_GTEX")
TARGETS_HUMAN=$(abspath "$TARGETS_HUMAN")

if [[ "$MODES" == "all" ]]; then
  MODES="sed,sad"
fi
IFS=',' read -r -a MODE_LIST <<< "$MODES"
for eqtl_mode in "${MODE_LIST[@]}"; do
  case "$eqtl_mode" in
    sed|sad) ;;
    *) echo "未知 eQTL mode: $eqtl_mode" >&2; exit 2 ;;
  esac
done

discover_models() {
  local found
  if ((${#MODEL_FILES[@]} == 0)); then
    while IFS= read -r found; do MODEL_FILES+=("$found"); done < <(
      find "$MODEL_DIR" -type f \( -name 'model0_best.h5' -o -name 'model_best.h5' \) | sort
    )
  fi
}

require_input_files() {
  local required
  for required in \
    "$DATA_DIR/pos_merge.vcf.gz" \
    "$DATA_DIR/neg_merge.vcf.gz"; do
    [[ -s "$required" ]] || { echo "缺少文件: $required" >&2; exit 1; }
  done
  [[ -d "$DATA_DIR/tables" ]] || {
    echo "缺少 SuSiE 表目录: $DATA_DIR/tables" >&2
    exit 1
  }
}

require_runtime_files() {
  local required
  for required in \
    "$PARAMS_FILE" \
    "$TARGETS_GTEX" \
    "$TARGETS_HUMAN" \
    "$ROOT_DIR/reference/hg38/assembly/ucsc/hg38.fa" \
    "$ROOT_DIR/reference/hg38/assembly/ucsc/hg38.fa.fai" \
    "$ROOT_DIR/reference/hg38/genes/gencode41/gencode41_basic_nort.gtf"; do
    [[ -s "$required" ]] || { echo "缺少文件: $required" >&2; exit 1; }
  done
  discover_models
  ((${#MODEL_FILES[@]})) || { echo "没有找到模型；请使用 --model 或 --model-dir。" >&2; exit 1; }
  for required in "${MODEL_FILES[@]}"; do
    [[ -s "$required" ]] || { echo "模型不存在或为空: $required" >&2; exit 1; }
  done
}

check_environment() {
  command -v conda >/dev/null 2>&1 || { echo "找不到 conda。" >&2; exit 1; }
  PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python -c \
    'import h5py, intervaltree, matplotlib, natsort, numpy, pandas, pybedtools, pysam, scipy, seaborn, sklearn, tensorflow; from baskerville import seqnn' \
    >/dev/null
  conda run -n "$CONDA_ENV" bedtools --version >/dev/null
  PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" conda run -n "$CONDA_ENV" python -c \
    'import tensorflow as tf; g=tf.config.list_physical_devices("GPU"); print("TensorFlow GPUs:", g); raise SystemExit(0 if g else 1)'
}

require_input_files
if [[ "$ACTION" == "check" ]]; then
  require_runtime_files
  "$ROOT_DIR/scripts/verify_package.py"
  check_environment
  echo "检查通过，可以运行 benchmark。"
  exit 0
fi

if ((MAX_VARIANTS)); then
  if [[ "$LIMIT_SCOPE" == per-tissue ]]; then
    LIMIT_TAG="per_tissue_max_$MAX_VARIANTS"
  else
    LIMIT_TAG="max_$MAX_VARIANTS"
  fi
else
  LIMIT_TAG="all"
fi
INPUT_ROOT="$OUTPUT_DIR/inputs/$LIMIT_TAG"
# Upstream coefficient scripts parse the PIP threshold from this directory
# name, so it must end in _pip90 just like borzoi-paper.
INPUT_DIR="$INPUT_ROOT/eqtl_pip90"
EXPERIMENT_DIR="$OUTPUT_DIR/experiment/$LIMIT_TAG"
RUNTIME_BIN="$OUTPUT_DIR/runtime_bin"
export MPLCONFIGDIR="$OUTPUT_DIR/matplotlib"
export TF_CPP_MIN_LOG_LEVEL="${BORZOI_TF_CPP_MIN_LOG_LEVEL:-${TF_CPP_MIN_LOG_LEVEL:-3}}"

mkdir -p "$INPUT_DIR" "$MPLCONFIGDIR"
echo "[input] limit_scope=$LIMIT_SCOPE max_variants=$MAX_VARIANTS tag=$LIMIT_TAG"
python3 "$ROOT_DIR/benchmark/qtl_benchmark/subset_vcfs.py" \
  --max-variants "$MAX_VARIANTS" --limit-scope "$LIMIT_SCOPE" \
  "$DATA_DIR" "$INPUT_DIR"

if [[ "$ACTION" == "prepare" ]]; then
  echo "VCF 已准备到: $INPUT_DIR"
  echo "子集清单: $INPUT_DIR/subset_manifest.json"
  exit 0
fi

require_runtime_files
check_environment
PYTHONPATH="$ROOT_DIR/benchmark:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "$CONDA_ENV" python \
  "$ROOT_DIR/scripts/validate_borzoi_batch.py" \
  "$PARAMS_FILE" "$VARIANT_BATCH_SIZE"
mkdir -p "$EXPERIMENT_DIR" "$RUNTIME_BIN"
for script_name in \
  borzoi_sad.py borzoi_sed.py borzoi_bench_classify.py \
  borzoi_gtex_coef_sad.py borzoi_gtex_coef_sed.py \
  borzoi_bench_gtex_folds_sad.py borzoi_bench_gtex_folds_sed.py; do
  ln -sfn "$ROOT_DIR/benchmark/qtl_benchmark/run_borzoi_script.py" "$RUNTIME_BIN/$script_name"
done

declare -a ABS_MODELS=()
for model in "${MODEL_FILES[@]}"; do
  ABS_MODELS+=("$(abspath "$model")")
done
for index in "${!ABS_MODELS[@]}"; do
  stage_dir="$EXPERIMENT_DIR/f0c$index/train"
  stage_link="$stage_dir/model0_best.h5"
  mkdir -p "$stage_dir"
  if [[ -e "$stage_link" || -L "$stage_link" ]]; then
    existing=$(readlink -f "$stage_link")
    if [[ "$existing" != "${ABS_MODELS[$index]}" ]]; then
      echo "输出目录已包含另一个模型: $stage_link" >&2
      echo "请更换 --output-dir，避免混合实验结果。" >&2
      exit 1
    fi
  else
    ln -s "${ABS_MODELS[$index]}" "$stage_link"
  fi
done

CONDA_BASE=$(conda info --base)
export BORZOI_SCRIPTS_DIR="$ROOT_DIR/src/borzoi_scripts"
export BORZOI_HG38="$ROOT_DIR/reference/hg38"
export BORZOI_CONDA="$CONDA_BASE/etc/profile.d/conda.sh"
export PATH="$RUNTIME_BIN:$PATH"
export PYTHONPATH="$ROOT_DIR/benchmark:$ROOT_DIR/src:${PYTHONPATH:-}"

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

MODEL_COUNT=${#ABS_MODELS[@]}
if ((MAX_VARIANTS)); then
  SUSIE_DIR="$INPUT_DIR/tables"
else
  SUSIE_DIR="$DATA_DIR/tables"
fi
for eqtl_mode in "${MODE_LIST[@]}"; do
  echo "[run] variant_batch_size=$VARIANT_BATCH_SIZE sequence_batch_size=$((2 * VARIANT_BATCH_SIZE))"
  echo "===== 运行 eQTL $eqtl_mode，模型数: $MODEL_COUNT ====="
  case "$eqtl_mode" in
    sed)
      run_python "$RUNTIME_BIN/borzoi_bench_gtex_folds_sed.py" \
        -d 0 -e "$CONDA_ENV" --gtex "$INPUT_DIR" --susie "$SUSIE_DIR" \
        -p "${QTL_MAX_PROCS:-1}" --variant-batch-size "$VARIANT_BATCH_SIZE" -o eqtl_sed --max_proc "${QTL_MAX_PROCS:-1}" --f_list 0 -c "$MODEL_COUNT" \
        --rc -u --stats SED,logSED -t "$TARGETS_GTEX" \
        "$PARAMS_FILE" "$EXPERIMENT_DIR"
      ;;
    sad)
      run_python "$RUNTIME_BIN/borzoi_bench_gtex_folds_sad.py" \
        -d 0 -e "$CONDA_ENV" -g "$INPUT_DIR" --susie "$SUSIE_DIR" \
        -p "${QTL_MAX_PROCS:-1}" --variant-batch-size "$VARIANT_BATCH_SIZE" -o eqtl_sad --max_proc "${QTL_MAX_PROCS:-1}" --f_list 0 -c "$MODEL_COUNT" \
        --rc -u --msl 12 --stats SAD,logSAD,D2,logD2 -t "$TARGETS_HUMAN" \
        "$PARAMS_FILE" "$EXPERIMENT_DIR"
      ;;
  esac
done

echo "===== 完成 ====="
echo "结果目录: $OUTPUT_DIR"
find "$OUTPUT_DIR" -type f \( -name metrics.tsv -o -name stats.txt \) -print | sort
