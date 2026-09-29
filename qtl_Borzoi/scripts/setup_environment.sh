#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${1:-borzoi_py310}

if command -v mamba >/dev/null 2>&1; then
  CONDA_COMMAND=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_COMMAND=conda
else
  echo "需要先安装 Conda 或 Mamba。" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "更新已有环境: $ENV_NAME"
  "$CONDA_COMMAND" env update --name "$ENV_NAME" --file "$ROOT_DIR/environment.yml" --prune
else
  echo "创建环境: $ENV_NAME"
  "$CONDA_COMMAND" env create --name "$ENV_NAME" --file "$ROOT_DIR/environment.yml"
fi

echo "环境已就绪。请在 GPU 节点运行："
echo "  $ROOT_DIR/scripts/run_eqtl_benchmark.sh --conda-env $ENV_NAME --check"
