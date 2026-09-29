#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR="$ROOT_DIR/data"
OUTPUT_DIR="$ROOT_DIR/outputs/borzoi_qtl"
CONDA_ENV=${BORZOI_CONDA_ENV:-borzoi_py310}
PARAMS_FILE="$ROOT_DIR/configs/params_pred.json"
MODEL_DIR="$ROOT_DIR/models"
MODEL_NAME=borzoi
TASKS=eqtl
MAX_VARIANTS=0
EQTL_LIMIT_SCOPE=global
VARIANT_BATCH_SIZE=${BORZOI_VARIANT_BATCH_SIZE:-1}
PREPARE_ONLY=0
FORCE_UNIFIED=0
GPU_IDS=${BORZOI_GPU_IDS:-${QTL_GPU_IDS:-0,1}}
declare -a MODEL_FILES=()

usage() {
  sed -n '2,45p' "$0" | sed -n 's/^# //p'
}

# Run selected Borzoi QTL tasks and unified evaluation in one command.
#
# Examples:
#   ./scripts/run_borzoi_multi_gpu.sh --prepare-only --max-variants 10
#   ./scripts/run_borzoi_multi_gpu.sh --tasks sqtl,paqtl,ipaqtl
#   ./scripts/run_borzoi_multi_gpu.sh --tasks all --max-variants 0
#
# Options:
#   --tasks LIST       eqtl,sqtl,paqtl,ipaqtl or all (default: eqtl).
#   --max-variants N   eQTL scope limit or matched pairs per other task; 0 means all.
#   --eqtl-limit-scope S  global or per-tissue; formal comparisons use per-tissue.
#   --variant-batch-size N  Variants per model call; uses 2*N REF/ALT sequences (default: 1).
#   --model FILE       Borzoi .h5 model; repeat for an ensemble.
#   --model-dir DIR    Search model0_best.h5/model_best.h5 below DIR.
#   --model-name NAME  Result/log label for this model or ensemble.
#   --params FILE      Borzoi parameter JSON.
#   --conda-env NAME   Runtime environment (default: borzoi_py310).
#   --data-dir DIR     Directory containing eqtl/sqtl/paqtl/ipaqtl.
#   --output-dir DIR   Artifact root (default: outputs/borzoi_qtl).
#   --gpus LIST        Comma-separated physical GPU IDs (default: 0,1).
#   --prepare-only     Only create uncompressed, consistently paired VCFs.
#   --force-unified    Discard Borzoi's unified prediction cache before evaluation.
#   -h, --help         Show this help.

while (($#)); do
  case "$1" in
    --tasks) TASKS=$2; shift 2 ;;
    --max-variants) MAX_VARIANTS=$2; shift 2 ;;
    --eqtl-limit-scope) EQTL_LIMIT_SCOPE=$2; shift 2 ;;
    --variant-batch-size) VARIANT_BATCH_SIZE=$2; shift 2 ;;
    --model) MODEL_FILES+=("$2"); shift 2 ;;
    --model-dir) MODEL_DIR=$2; shift 2 ;;
    --model-name) MODEL_NAME=$2; shift 2 ;;
    --params) PARAMS_FILE=$2; shift 2 ;;
    --conda-env) CONDA_ENV=$2; shift 2 ;;
    --data-dir) DATA_DIR=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --gpus) GPU_IDS=$2; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --force-unified) FORCE_UNIFIED=1; shift ;;
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
case "$EQTL_LIMIT_SCOPE" in
  global|per-tissue) ;;
  *) echo "--eqtl-limit-scope 必须是 global 或 per-tissue。" >&2; exit 2 ;;
esac
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
((${#GPU_LIST[@]})) || { echo "--gpus 不能为空。" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_LIST[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || {
    echo "--gpus 必须是逗号分隔的非负整数，例如 0 或 0,1,2,3。" >&2
    exit 2
  }
  [[ -z "${SEEN_GPUS[$gpu_id]:-}" ]] || {
    echo "--gpus 包含重复 GPU: $gpu_id" >&2
    exit 2
  }
  SEEN_GPUS[$gpu_id]=1
done
export QTL_GPU_IDS=$(IFS=,; echo "${GPU_LIST[*]}")
export QTL_MAX_PROCS=${#GPU_LIST[@]}
export BORZOI_MODEL_NAME="$MODEL_NAME"
# TensorFlow 2.15 may emit duplicate cuDNN/cuFFT/cuBLAS registration messages
# for every worker even when GPU initialization succeeds. Keep Python
# exceptions visible while silencing those C++ startup diagnostics. Set
# BORZOI_TF_CPP_MIN_LOG_LEVEL=0 when full TensorFlow diagnostics are needed.
export TF_CPP_MIN_LOG_LEVEL=${BORZOI_TF_CPP_MIN_LOG_LEVEL:-3}

abspath() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

DATA_DIR=$(abspath "$DATA_DIR")
OUTPUT_DIR=$(abspath "$OUTPUT_DIR")
MODEL_DIR=$(abspath "$MODEL_DIR")
PARAMS_FILE=$(abspath "$PARAMS_FILE")
if [[ "$TASKS" == all ]]; then TASKS=eqtl,sqtl,paqtl,ipaqtl; fi
[[ -n "$TASKS" ]] || {
  echo "--tasks 不能为空；请使用 eqtl、sqtl、paqtl、ipaqtl 或 all。" >&2
  exit 2
}
IFS=',' read -r -a TASK_LIST <<< "$TASKS"
for task in "${TASK_LIST[@]}"; do
  case "$task" in
    eqtl|sqtl|paqtl|ipaqtl) ;;
    *) echo "未知 QTL task: $task" >&2; exit 2 ;;
  esac
done

if ((MAX_VARIANTS)); then LIMIT_TAG="max_$MAX_VARIANTS"; else LIMIT_TAG=all; fi
if ((MAX_VARIANTS)) && [[ "$EQTL_LIMIT_SCOPE" == per-tissue ]]; then
  EQTL_LIMIT_TAG="per_tissue_max_$MAX_VARIANTS"
else
  EQTL_LIMIT_TAG=$LIMIT_TAG
fi
declare -a UNIFIED_FORCE_ARGS=()
if ((FORCE_UNIFIED)); then UNIFIED_FORCE_ARGS+=(--force); fi
INPUT_ROOT="$OUTPUT_DIR/inputs/$LIMIT_TAG"
EXPERIMENT_DIR="$OUTPUT_DIR/experiment/$LIMIT_TAG"
RUNTIME_BIN="$OUTPUT_DIR/runtime_bin"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$INPUT_ROOT" "$LOG_DIR"
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="$LOG_DIR/${RUN_STAMP}_${TASKS//,/_}_${LIMIT_TAG}_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[run] tasks=$TASKS max_variants=$MAX_VARIANTS model_name=$MODEL_NAME output=$OUTPUT_DIR"
echo "[run] eqtl_limit_scope=$EQTL_LIMIT_SCOPE eqtl_limit_tag=$EQTL_LIMIT_TAG"
echo "[run] gpus=$QTL_GPU_IDS workers=$QTL_MAX_PROCS"
echo "[run] variant_batch_size=$VARIANT_BATCH_SIZE sequence_batch_size=$((2 * VARIANT_BATCH_SIZE))"
echo "[run] log=$LOG_FILE"

has_task() {
  local wanted=$1 item
  for item in "${TASK_LIST[@]}"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}

for task in sqtl paqtl ipaqtl; do
  has_task "$task" || continue
  case "$task" in
    sqtl) suffix=sqtl_pip90ea ;;
    paqtl) suffix=paqtl_pip90ea ;;
    ipaqtl) suffix=ipaqtl_pip90ea ;;
  esac
  python3 "$ROOT_DIR/benchmark/qtl_benchmark/subset_vcfs.py" \
    --matched-pairs --max-variants "$MAX_VARIANTS" \
    "$DATA_DIR/$task" "$INPUT_ROOT/$suffix"
done

if ((PREPARE_ONLY)); then
  if has_task eqtl; then
    "$ROOT_DIR/scripts/run_eqtl_benchmark.sh" \
      --max-variants "$MAX_VARIANTS" --data-dir "$DATA_DIR/eqtl" \
      --limit-scope "$EQTL_LIMIT_SCOPE" \
      --output-dir "$OUTPUT_DIR/eqtl" --prepare-only
  fi
  echo "非 eQTL 输入: $INPUT_ROOT"
  if has_task eqtl; then
    echo "eQTL 输入: $OUTPUT_DIR/eqtl/inputs/$EQTL_LIMIT_TAG/eqtl_pip90"
  fi
  exit 0
fi

command -v conda >/dev/null 2>&1 || { echo "找不到 conda。" >&2; exit 1; }
if ! conda run -n "$CONDA_ENV" python -c \
  'import h5py, numpy, pandas, pybedtools, pyranges, scipy, sklearn, tensorflow; from baskerville import seqnn' \
  >/dev/null; then
  echo "Conda 环境 $CONDA_ENV 无法导入 Borzoi 运行依赖。" >&2
  echo "若看到 NumPy/Pandas ABI 错误，请用 setup_qtl_benchmark.sh 创建新的干净环境。" >&2
  exit 1
fi
[[ -s "$PARAMS_FILE" ]] || { echo "缺少参数文件: $PARAMS_FILE" >&2; exit 1; }
PYTHONPATH="$ROOT_DIR/benchmark:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "$CONDA_ENV" python \
  "$ROOT_DIR/scripts/validate_borzoi_batch.py" \
  "$PARAMS_FILE" "$VARIANT_BATCH_SIZE"
if ((${#MODEL_FILES[@]} == 0)); then
  while IFS= read -r model; do MODEL_FILES+=("$model"); done < <(
    find "$MODEL_DIR" -type f \( -name model0_best.h5 -o -name model_best.h5 \) | sort
  )
fi
((${#MODEL_FILES[@]})) || { echo "未找到 Borzoi 模型。" >&2; exit 1; }
for index in "${!MODEL_FILES[@]}"; do
  model=$(abspath "${MODEL_FILES[$index]}")
  [[ -s "$model" ]] || { echo "模型不存在: $model" >&2; exit 1; }
  MODEL_FILES[$index]=$model
  echo "[model/$index] $model"
done

if has_task eqtl; then
  declare -a EQTL_ARGS=(
    --max-variants "$MAX_VARIANTS" --data-dir "$DATA_DIR/eqtl"
    --limit-scope "$EQTL_LIMIT_SCOPE"
    --output-dir "$OUTPUT_DIR/eqtl" --params "$PARAMS_FILE"
    --conda-env "$CONDA_ENV" --variant-batch-size "$VARIANT_BATCH_SIZE"
  )
  for model in "${MODEL_FILES[@]}"; do EQTL_ARGS+=(--model "$model"); done
  "$ROOT_DIR/scripts/run_eqtl_benchmark.sh" "${EQTL_ARGS[@]}"
fi

NON_EQTL=0
for task in "${TASK_LIST[@]}"; do
  [[ "$task" == eqtl ]] || NON_EQTL=1
done
if has_task eqtl && ((!NON_EQTL)); then
  export PYTHONPATH="$ROOT_DIR/benchmark:$ROOT_DIR/src:${PYTHONPATH:-}"
  conda run --no-capture-output -n "$CONDA_ENV" python -m \
    qtl_benchmark.run_borzoi_unified eqtl \
    --data-dir "$OUTPUT_DIR/eqtl/inputs/$EQTL_LIMIT_TAG/eqtl_pip90" \
    --artifact-dir "$OUTPUT_DIR/eqtl/experiment/$EQTL_LIMIT_TAG/ensemble/eqtl_sed" \
    --max-variants "$MAX_VARIANTS" \
    --limit-tag "$EQTL_LIMIT_TAG" \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR/unified/$EQTL_LIMIT_TAG" \
    "${UNIFIED_FORCE_ARGS[@]}"
  exit 0
fi

mkdir -p "$EXPERIMENT_DIR" "$RUNTIME_BIN"
for script_name in \
  borzoi_sed.py borzoi_sed_paqtl_cov.py borzoi_sed_ipaqtl_cov.py \
  borzoi_bench_classify.py borzoi_bench_sqtl_folds.py \
  borzoi_bench_paqtl_folds.py borzoi_bench_ipaqtl_folds.py; do
  ln -sfn "$ROOT_DIR/benchmark/qtl_benchmark/run_borzoi_script.py" \
    "$RUNTIME_BIN/$script_name"
done

for index in "${!MODEL_FILES[@]}"; do
  model=${MODEL_FILES[$index]}
  stage_dir="$EXPERIMENT_DIR/f0c$index/train"
  stage_link="$stage_dir/model0_best.h5"
  mkdir -p "$stage_dir"
  if [[ -e "$stage_link" || -L "$stage_link" ]]; then
    existing=$(readlink -f "$stage_link")
    if [[ "$existing" != "$model" ]]; then
      echo "输出目录已包含另一个模型: $stage_link" >&2
      echo "请更换 --output-dir，避免混合实验结果。" >&2
      exit 1
    fi
  else
    ln -s "$model" "$stage_link"
  fi
done

CONDA_BASE=$(conda info --base)
export BORZOI_SCRIPTS_DIR="$ROOT_DIR/src/borzoi_scripts"
export BORZOI_HG38="$ROOT_DIR/reference/hg38"
export BORZOI_CONDA="$CONDA_BASE/etc/profile.d/conda.sh"
export PATH="$RUNTIME_BIN:$PATH"
export PYTHONPATH="$ROOT_DIR/benchmark:$ROOT_DIR/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$OUTPUT_DIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

run_python() {
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

MODEL_COUNT=${#MODEL_FILES[@]}
for task in "${TASK_LIST[@]}"; do
  echo "===== 运行 $task，模型数: $MODEL_COUNT ====="
  case "$task" in
    eqtl) ;;
    sqtl)
      run_python "$RUNTIME_BIN/borzoi_bench_sqtl_folds.py" \
        -r -p "$QTL_MAX_PROCS" --variant-batch-size "$VARIANT_BATCH_SIZE" --span --no_untransform --vcf "$INPUT_ROOT/sqtl_pip90ea" \
        -o sqtl_span -d 0 -e "$CONDA_ENV" --rc -u --msl 4 \
        --max_proc "$QTL_MAX_PROCS" --f_list 0 -c "$MODEL_COUNT" --stats nDi \
        -t "$ROOT_DIR/configs/targets_rna.txt" "$PARAMS_FILE" "$EXPERIMENT_DIR"
      ;;
    paqtl)
      run_python "$RUNTIME_BIN/borzoi_bench_paqtl_folds.py" \
        -r -p "$QTL_MAX_PROCS" --variant-batch-size "$VARIANT_BATCH_SIZE" --vcf "$INPUT_ROOT/paqtl_pip90ea" -o paqtl -d 0 \
        -e "$CONDA_ENV" --rc -u --msl 12 --max_proc "$QTL_MAX_PROCS" --f_list 0 \
        -c "$MODEL_COUNT" --stats COVR --utr3 \
        -t "$ROOT_DIR/configs/targets_gtex.txt" "$PARAMS_FILE" "$EXPERIMENT_DIR"
      ;;
    ipaqtl)
      run_python "$RUNTIME_BIN/borzoi_bench_ipaqtl_folds.py" \
        -r -p "$QTL_MAX_PROCS" --variant-batch-size "$VARIANT_BATCH_SIZE" --vcf "$INPUT_ROOT/ipaqtl_pip90ea" -o ipaqtl -d 0 \
        -e "$CONDA_ENV" --rc -u --msl 12 --max_proc "$QTL_MAX_PROCS" --f_list 0 \
        -c "$MODEL_COUNT" --stats COVR \
        -t "$ROOT_DIR/configs/targets_gtex.txt" "$PARAMS_FILE" "$EXPERIMENT_DIR"
      ;;
  esac
done

echo "===== 进入统一评估层 ====="
for task in "${TASK_LIST[@]}"; do
  case "$task" in
    eqtl)
      run_python -m qtl_benchmark.run_borzoi_unified eqtl \
        --data-dir "$OUTPUT_DIR/eqtl/inputs/$EQTL_LIMIT_TAG/eqtl_pip90" \
        --artifact-dir "$OUTPUT_DIR/eqtl/experiment/$EQTL_LIMIT_TAG/ensemble/eqtl_sed" \
        --max-variants "$MAX_VARIANTS" \
        --limit-tag "$EQTL_LIMIT_TAG" \
        --model-name "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR/unified/$EQTL_LIMIT_TAG" \
        "${UNIFIED_FORCE_ARGS[@]}"
      ;;
    sqtl|paqtl|ipaqtl)
      run_python -m qtl_benchmark.run_borzoi_unified "$task" \
        --data-dir "$DATA_DIR" --experiment-dir "$EXPERIMENT_DIR" \
        --max-variants "$MAX_VARIANTS" \
        --model-name "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR/unified/$LIMIT_TAG" \
        "${UNIFIED_FORCE_ARGS[@]}"
      ;;
  esac
done

echo "Borzoi artifacts: $EXPERIMENT_DIR"
echo "统一结果: $OUTPUT_DIR/unified/$LIMIT_TAG"
find "$OUTPUT_DIR/unified/$LIMIT_TAG" -name metrics.tsv -print | sort
if [[ "$EQTL_LIMIT_TAG" != "$LIMIT_TAG" ]]; then
  echo "eQTL 统一结果: $OUTPUT_DIR/unified/$EQTL_LIMIT_TAG"
  find "$OUTPUT_DIR/unified/$EQTL_LIMIT_TAG" -name metrics.tsv -print | sort
fi
