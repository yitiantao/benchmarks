#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT_DIR/scripts/run_ntv3_multi_gpu.sh"

usage() {
  cat <<EOF
Usage: $0 [--gpus 0,1,...] [NTv3 benchmark options]

Run one deterministic prediction shard per GPU, then evaluate once.
GPU list default: ${NTV3_GPU_IDS:-0,1}

Example:
  $0 --gpus 0,1 --tasks all --max-variants 1 --batch-size 1
EOF
}

worker_main() {
  local conda_env=${NTV3_CONDA_ENV:-ntv3}
  local output_dir="$ROOT_DIR/outputs/ntv3_qtl"
  local has_config=0
  local -a forward=()

  while (($#)); do
    case "$1" in
      --conda-env)
        [[ $# -ge 2 ]] || { echo "--conda-env requires a value" >&2; exit 2; }
        conda_env=$2
        shift 2
        ;;
      --model-config)
        [[ $# -ge 2 && -n "$2" ]] || { echo "--model-config requires a value" >&2; exit 2; }
        has_config=1
        forward+=("$1" "$2")
        shift 2
        ;;
      --output-dir)
        [[ $# -ge 2 && -n "$2" ]] || { echo "--output-dir requires a value" >&2; exit 2; }
        output_dir=$2
        forward+=("$1" "$2")
        shift 2
        ;;
      *) forward+=("$1"); shift ;;
    esac
  done

  if ((!has_config)); then
    forward+=(--model-config "$ROOT_DIR/configs/ntv3_100m_post.json")
  fi
  command -v conda >/dev/null 2>&1 || { echo "找不到 conda。" >&2; exit 1; }
  if ! conda run -n "$conda_env" python -c \
    'import numpy, pandas, pyfaidx, torch, transformers' >/dev/null; then
    echo "Conda 环境 $conda_env 无法导入 NTv3 运行依赖。" >&2
    echo "请运行: ./scripts/setup_ntv3_benchmark.sh $conda_env" >&2
    exit 1
  fi

  output_dir=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$output_dir")
  local log_dir="$output_dir/logs"
  mkdir -p "$log_dir"
  local run_stamp log_file
  run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
  log_file="$log_dir/${run_stamp}_ntv3_$$.log"
  exec > >(tee -a "$log_file") 2>&1
  echo "[run] conda_env=$conda_env output=$output_dir"
  echo "[run] log=$log_file"

  export PYTHONPATH="$ROOT_DIR/benchmark:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=false
  export HF_MODULES_CACHE=${HF_MODULES_CACHE:-$ROOT_DIR/.benchmark_deps/ntv3_hf/modules}
  mkdir -p "$HF_MODULES_CACHE"
  cd "$ROOT_DIR"
  exec conda run --no-capture-output -n "$conda_env" \
    python -m qtl_benchmark.run_ntv3_unified "${forward[@]}"
}

if [[ "${1:-}" == --internal-worker ]]; then
  shift
  worker_main "$@"
fi

GPU_IDS=${NTV3_GPU_IDS:-0,1}
declare -a FORWARD=()

while (($#)); do
  case "$1" in
    --gpus)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--gpus requires a value" >&2; exit 2; }
      GPU_IDS=$2
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    --predict-only|--evaluate-only|--num-shards|--shard-index|--force)
      echo "$1 is managed by the multi-GPU launcher and cannot be supplied." >&2
      exit 2
      ;;
    *) FORWARD+=("$1"); shift ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
((${#GPUS[@]})) || { echo "GPU list cannot be empty." >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || {
    echo "GPU IDs must be comma-separated non-negative integers." >&2
    exit 2
  }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || {
    echo "Duplicate GPU ID: $gpu" >&2
    exit 2
  }
  SEEN_GPUS[$gpu]=1
done

worker_count=${#GPUS[@]}
declare -a PIDS=()
echo "[ntv3/multi] gpus=$GPU_IDS shards=$worker_count"
for shard_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_index]}
  echo "[ntv3/multi] launch shard=$((shard_index + 1))/$worker_count gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$SELF" --internal-worker \
    "${FORWARD[@]}" --predict-only \
    --num-shards "$worker_count" --shard-index "$shard_index" &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if ((failed)); then
  echo "At least one NTv3 prediction shard failed; evaluation was not run." >&2
  exit 1
fi

echo "[ntv3/multi] all prediction shards complete; evaluating once"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" exec \
  "$SELF" --internal-worker \
  "${FORWARD[@]}" --evaluate-only
