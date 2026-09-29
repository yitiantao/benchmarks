#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT_DIR/scripts/run_dna_fm_multi_gpu.sh"

usage() {
  cat <<EOF
Usage: $0 [--gpus 0,1,...] [DNA-FM benchmark options]

Run one deterministic prediction shard per GPU, then evaluate once.
GPU list default: ${DNA_FM_GPU_IDS:-0,1}

Example:
  $0 --gpus 0,1 --tasks all --max-variants 1 --batch-size 1
EOF
}

worker_main() {
  local python_bin output_dir has_config=0
  if [[ -n "${DNA_FM_PYTHON:-}" ]]; then
    python_bin=$DNA_FM_PYTHON
  elif command -v conda >/dev/null 2>&1 && \
       [[ -x "$(conda info --base)/envs/ntv3/bin/python" ]]; then
    python_bin="$(conda info --base)/envs/ntv3/bin/python"
  else
    python_bin=/data/pengcheng_workspace/miniconda3/envs/dna_t5/bin/python
  fi
  output_dir="$ROOT_DIR/outputs/dna_fm_qtl"
  local -a forward=()

  while (($#)); do
    case "$1" in
      --python)
        [[ $# -ge 2 && -n "$2" ]] || { echo "--python requires a value" >&2; exit 2; }
        python_bin=$2
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
    forward+=(--model-config "$ROOT_DIR/configs/dna_fm_100m_post_step_0034000.json")
  fi
  if [[ ! -x "$python_bin" ]]; then
    echo "DNA-FM Python is not executable: $python_bin" >&2
    echo "Set DNA_FM_PYTHON or pass --python /path/to/python." >&2
    exit 1
  fi
  if ! "$python_bin" -c \
    'import numpy, pandas, pyfaidx, safetensors, torch, yaml' >/dev/null; then
    echo "DNA-FM Python cannot import the inference dependencies." >&2
    exit 1
  fi

  output_dir=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$output_dir")
  local log_dir="$output_dir/logs"
  mkdir -p "$log_dir"
  local run_stamp log_file
  run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
  log_file="$log_dir/${run_stamp}_dna_fm_$$.log"
  exec > >(tee -a "$log_file") 2>&1
  echo "[run] python=$python_bin output=$output_dir"
  echo "[run] log=$log_file"

  export PYTHONPATH="$ROOT_DIR/benchmark:${PYTHONPATH:-}"
  cd "$ROOT_DIR"
  exec "$python_bin" -m qtl_benchmark.run_dna_fm_unified "${forward[@]}"
}

if [[ "${1:-}" == --internal-worker ]]; then
  shift
  worker_main "$@"
fi

GPU_IDS=${DNA_FM_GPU_IDS:-0,1}
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
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo "Duplicate GPU ID: $gpu" >&2; exit 2; }
  SEEN_GPUS[$gpu]=1
done

worker_count=${#GPUS[@]}
declare -a PIDS=()
echo "[dna-fm/multi] gpus=$GPU_IDS shards=$worker_count"
for shard_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_index]}
  echo "[dna-fm/multi] launch shard=$((shard_index + 1))/$worker_count gpu=$gpu"
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
  echo "At least one DNA-FM prediction shard failed; evaluation was not run." >&2
  exit 1
fi

echo "[dna-fm/multi] all prediction shards complete; evaluating once"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" exec \
  "$SELF" --internal-worker \
  "${FORWARD[@]}" --evaluate-only
