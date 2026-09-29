#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${1:-ntv3}

if command -v mamba >/dev/null 2>&1; then
  CONDA_COMMAND=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_COMMAND=conda
else
  echo "需要先安装 Conda 或 Mamba。" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "更新已有 NTv3 环境: $ENV_NAME"
  "$CONDA_COMMAND" env update --name "$ENV_NAME" \
    --file "$ROOT_DIR/environment_ntv3.yml" --prune
else
  echo "创建独立 NTv3 环境: $ENV_NAME"
  "$CONDA_COMMAND" env create --name "$ENV_NAME" \
    --file "$ROOT_DIR/environment_ntv3.yml"
fi

conda run -n "$ENV_NAME" python -c \
  'import h5py, numpy, pandas, pyfaidx, torch, transformers; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__, "h5py", h5py.__version__)'

echo "NTv3 环境已就绪: $ENV_NAME"
echo "模型受 Hugging Face gated access 保护。接受条款后执行："
echo "  conda run -n $ENV_NAME hf auth login"
echo "  NTV3_CONDA_ENV=$ENV_NAME $ROOT_DIR/scripts/download_ntv3_models.sh --suite qtl"
