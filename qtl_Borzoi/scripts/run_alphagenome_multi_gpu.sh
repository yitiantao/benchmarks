#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT_DIR/scripts/run_alphagenome_multi_gpu.sh"

worker_main() {
  local conda_env=${ALPHAGENOME_CONDA_ENV:-alphagenome}
  command -v conda >/dev/null 2>&1 || { echo "找不到 conda。" >&2; exit 1; }
  unset LD_LIBRARY_PATH
  export PYTHONPATH="$ROOT_DIR/benchmark:${PYTHONPATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export JAX_PLATFORMS=cuda

  local has_model_config=0
  local output_dir="$ROOT_DIR/outputs/alphagenome_qtl"
  local -a original_args=("$@")
  local index argument
  for ((index=0; index<${#original_args[@]}; index++)); do
    argument=${original_args[$index]}
    [[ "$argument" == --model-config ]] && has_model_config=1
    if [[ "$argument" == --output-dir ]]; then
      output_dir=${original_args[$((index + 1))]}
    fi
  done
  local -a forward=("$@")
  if ((!has_model_config)); then
    forward+=(--model-config "$ROOT_DIR/configs/alphagenome_local.json")
  fi

  output_dir=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$output_dir")
  local log_dir="$output_dir/logs"
  mkdir -p "$log_dir"
  local run_stamp log_file
  run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
  log_file="$log_dir/${run_stamp}_alphagenome_$$.log"
  exec > >(tee -a "$log_file") 2>&1
  echo "[run] output=$output_dir"
  echo "[run] log=$log_file"

  cd "$ROOT_DIR"
  exec conda run --no-capture-output -n "$conda_env" \
    python -m qtl_benchmark.run_alphagenome_unified "${forward[@]}"
}

if [[ "${1:-}" == --internal-worker ]]; then
  shift
  worker_main "$@"
fi

GPU_IDS=${ALPHAGENOME_GPU_IDS:-0,1}
declare -a FORWARD=()

usage() {
  cat <<EOF
Usage: $0 [--gpus 0,1,...] [AlphaGenome benchmark options]

Run one deterministic prediction shard per GPU, then evaluate once.
GPU list default: ${ALPHAGENOME_GPU_IDS:-0,1}

Example:
  $0 --gpus 0,1 --tasks sqtl --max-variants 10 \\
    --output-dir outputs/my_alphagenome_sqtl
EOF
}

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
echo "[alphagenome/multi] gpus=$GPU_IDS shards=$worker_count"
for shard_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_index]}
  echo "[alphagenome/multi] launch shard=$((shard_index + 1))/$worker_count gpu=$gpu"
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
  echo "At least one AlphaGenome prediction shard failed; evaluation was not run." >&2
  exit 1
fi

echo "[alphagenome/multi] all prediction shards complete; evaluating once"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" exec \
  "$SELF" --internal-worker \
  "${FORWARD[@]}" --evaluate-only
