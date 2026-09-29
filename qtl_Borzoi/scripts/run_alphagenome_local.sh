#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_ENV=${ALPHAGENOME_CONDA_ENV:-alphagenome}
command -v conda >/dev/null 2>&1 || {
  echo "找不到 conda。" >&2
  exit 1
}

# JAX's pip-provided CUDA libraries can be shadowed by an unrelated CUDA in
# LD_LIBRARY_PATH.  Keep inference isolated and avoid reserving all GPU memory.
unset LD_LIBRARY_PATH
export PYTHONPATH="$ROOT_DIR/benchmark:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

check_host_memory() {
  local memory_current_file=/sys/fs/cgroup/memory.current
  local memory_max_file=/sys/fs/cgroup/memory.max
  local minimum_gib=${ALPHAGENOME_MIN_FREE_GIB:-10}
  [[ "$minimum_gib" =~ ^[0-9]+$ ]] || {
    echo "ALPHAGENOME_MIN_FREE_GIB 必须是非负整数。" >&2
    exit 2
  }
  [[ -r "$memory_current_file" && -r "$memory_max_file" ]] || return 0
  local memory_current memory_max minimum_bytes available_bytes
  memory_current=$(<"$memory_current_file")
  memory_max=$(<"$memory_max_file")
  [[ "$memory_max" == "max" ]] && return 0
  minimum_bytes=$((minimum_gib * 1024 * 1024 * 1024))
  available_bytes=$((memory_max - memory_current))
  if ((available_bytes < minimum_bytes)); then
    echo "AlphaGenome 启动前的容器可用主存不足。" >&2
    echo "当前可用约 $((available_bytes / 1024 / 1024 / 1024)) GiB，建议至少 ${minimum_gib} GiB。" >&2
    echo "请关闭占用主存的进程（常见为 VS Code/Pylance）后重试。" >&2
    echo "确认资源足够时可设置 ALPHAGENOME_MIN_FREE_GIB=0 跳过此检查。" >&2
    exit 1
  fi
}

check_host_memory
cd "$ROOT_DIR"
exec conda run --no-capture-output -n "$CONDA_ENV" \
  python -m qtl_benchmark.alphagenome_eqtl \
  --model-class qtl_benchmark.models.alphagenome:AlphaGenomeModel \
  "$@"
