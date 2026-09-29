#!/usr/bin/env bash
set -euo pipefail

# This file lives in <benchmark-root>/scripts. Keep benchmark-local paths
# relative to ROOT_DIR; scripts/ itself is not the repository root.
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DNA_DIR=$(cd -- "$ROOT_DIR/../.." && pwd)

BORZOI_DIR=${BORZOI_DIR:-"$DNA_DIR/borzoi"}
BORZOI_PAPER_DIR=${BORZOI_PAPER_DIR:-"$DNA_DIR/borzoi-paper"}
CONDA_ENV=${BORZOI_CONDA_ENV:-borzoi_py310}
BASKERVILLE_DIR=${BASKERVILLE_DIR:-"$ROOT_DIR/.benchmark_deps/baskerville"}
DOWNLOAD_ASSETS=1
CHECK_ONLY=0

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Prepare the Python environment and upstream resources used by this benchmark.

Options:
  --conda-env NAME         Conda environment (default: $CONDA_ENV)
  --borzoi-dir DIR         Borzoi source checkout (default: $BORZOI_DIR)
  --borzoi-paper-dir DIR   Borzoi paper checkout (default: $BORZOI_PAPER_DIR)
  --baskerville-dir DIR    Baskerville checkout (default: $BASKERVILLE_DIR)
  --no-download-assets     Do not run Borzoi's download_models.sh
  --check                  Only validate paths and the existing environment
  -h, --help               Show this help

The paper checkout is used as a fallback source for an existing Baskerville
checkout. BORZOI_DIR, BORZOI_PAPER_DIR, BASKERVILLE_DIR and BORZOI_CONDA_ENV
may also be set as environment variables.
EOF
}

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || {
    echo "Option $1 requires a value." >&2
    exit 2
  }
}

while (($#)); do
  case "$1" in
    --conda-env) require_value "$@"; CONDA_ENV=$2; shift 2 ;;
    --borzoi-dir) require_value "$@"; BORZOI_DIR=$2; shift 2 ;;
    --borzoi-paper-dir) require_value "$@"; BORZOI_PAPER_DIR=$2; shift 2 ;;
    --baskerville-dir) require_value "$@"; BASKERVILLE_DIR=$2; shift 2 ;;
    --no-download-assets) DOWNLOAD_ASSETS=0; shift ;;
    --check) CHECK_ONLY=1; DOWNLOAD_ASSETS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

abspath() {
  python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"
}

BORZOI_DIR=$(abspath "$BORZOI_DIR")
BORZOI_PAPER_DIR=$(abspath "$BORZOI_PAPER_DIR")
BASKERVILLE_DIR=$(abspath "$BASKERVILLE_DIR")

[[ -f "$ROOT_DIR/environment.yml" ]] || {
  echo "Missing benchmark environment file: $ROOT_DIR/environment.yml" >&2
  exit 1
}
[[ -f "$BORZOI_DIR/pyproject.toml" && -x "$BORZOI_DIR/download_models.sh" ]] || {
  echo "Borzoi checkout not found or incomplete: $BORZOI_DIR" >&2
  exit 1
}
[[ -d "$BORZOI_PAPER_DIR/data/qtl" ]] || {
  echo "Borzoi paper checkout not found or incomplete: $BORZOI_PAPER_DIR" >&2
  exit 1
}

# Reuse the paper repository's already cloned Baskerville when the benchmark-
# local checkout is absent. An explicit --baskerville-dir always wins.
if [[ ! -f "$BASKERVILLE_DIR/pyproject.toml" && \
      -f "$BORZOI_PAPER_DIR/.benchmark_deps/baskerville/pyproject.toml" ]]; then
  BASKERVILLE_DIR="$BORZOI_PAPER_DIR/.benchmark_deps/baskerville"
fi

if command -v mamba >/dev/null 2>&1; then
  CONDA_COMMAND=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_COMMAND=conda
else
  echo "Conda or Mamba is required." >&2
  exit 1
fi

echo "Benchmark root : $ROOT_DIR"
echo "Borzoi source  : $BORZOI_DIR"
echo "Borzoi paper   : $BORZOI_PAPER_DIR"
echo "Baskerville    : $BASKERVILLE_DIR"
echo "Conda env      : $CONDA_ENV"

if ((CHECK_ONLY)); then
  [[ -f "$BASKERVILLE_DIR/pyproject.toml" ]] || {
    echo "Baskerville checkout not found: $BASKERVILLE_DIR" >&2
    exit 1
  }
  conda run -n "$CONDA_ENV" python -c \
    'import h5py, numpy, pandas, pybedtools, pyranges, pysam, sklearn, tensorflow; from baskerville import seqnn; import borzoi'
  conda run -n "$CONDA_ENV" bedtools --version
  "$ROOT_DIR/scripts/verify_package.py"
  echo "Setup check passed. For the full GPU check, run:"
  echo "  $ROOT_DIR/scripts/run_eqtl_benchmark.sh --conda-env $CONDA_ENV --check"
  exit 0
fi

if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
  "$CONDA_COMMAND" env update --name "$CONDA_ENV" \
    --file "$ROOT_DIR/environment.yml" --prune
else
  "$CONDA_COMMAND" env create -y --name "$CONDA_ENV" \
    --file "$ROOT_DIR/environment.yml"
fi

if [[ ! -f "$BASKERVILLE_DIR/pyproject.toml" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo "git is required to clone Baskerville." >&2
    exit 1
  }
  mkdir -p "$(dirname -- "$BASKERVILLE_DIR")"
  git clone https://github.com/calico/baskerville.git "$BASKERVILLE_DIR"
fi

# The upstream projects infer their versions with setuptools_scm. Some compute
# nodes do not expose a git executable, and an isolated pip build may also pull
# incompatible newest setuptools_scm/vcs-versioning releases from a mirror.
# Use the 8.x backend installed by environment.yml and provide deterministic
# package-specific versions, as supported by setuptools_scm.
conda run --no-capture-output -n "$CONDA_ENV" \
  python -c 'import setuptools, setuptools_scm'
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BASKERVILLE=1.0.0 \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BORZOI=1.0.0 \
  conda run --no-capture-output -n "$CONDA_ENV" \
  pip install --no-build-isolation --no-deps \
  -e "$BASKERVILLE_DIR" -e "$BORZOI_DIR"

# Borzoi pins TensorFlow 2.15 but does not install the optional CUDA user-space
# libraries. Match those extras to the installed TensorFlow wheel exactly.
TF_VERSION=$(conda run --no-capture-output -n "$CONDA_ENV" \
  python -c 'from importlib.metadata import version; print(version("tensorflow"))')
if [[ ! "$TF_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.+-][0-9A-Za-z.-]+)?$ ]]; then
  echo "Could not determine TensorFlow version in environment: $CONDA_ENV" >&2
  exit 1
fi
conda run --no-capture-output -n "$CONDA_ENV" \
  pip install "tensorflow[and-cuda]==$TF_VERSION"

if ((DOWNLOAD_ASSETS)); then
  echo "Downloading published models and hg38 annotations into $BORZOI_DIR ..."
  (cd "$BORZOI_DIR" && ./download_models.sh)
fi

# Expose upstream downloads at paths consumed by run_*_benchmark.sh. Never
# replace a file or directory already present in this benchmark checkout.
mkdir -p "$ROOT_DIR/models" "$ROOT_DIR/reference"
link_asset() {
  local source_path=$1 target_path=$2
  if [[ -s "$source_path" && ! -e "$target_path" ]]; then
    mkdir -p "$(dirname -- "$target_path")"
    ln -s "$source_path" "$target_path"
  fi
}

for index in 0 1 2 3; do
  source_model="$BORZOI_DIR/examples/saved_models/f3c$index/train/model0_best.h5"
  link_asset "$source_model" \
    "$ROOT_DIR/models/replicate_$index/model0_best.h5"
done
for relative_path in \
  assembly/ucsc/hg38.fa \
  assembly/ucsc/hg38.fa.fai \
  genes/gencode41/gencode41_basic_nort.gtf \
  genes/gencode41/gencode41_basic_protein_splice.csv.gz \
  genes/gencode41/gencode41_basic_protein_splice.gff \
  genes/polyadb/polyadb_human_v3.csv.gz; do
  link_asset "$BORZOI_DIR/examples/hg38/$relative_path" \
    "$ROOT_DIR/reference/hg38/$relative_path"
done

echo "Setup complete. Validate it on a GPU node with:"
echo "  $ROOT_DIR/scripts/setup_qtl_benchmark.sh --check --conda-env $CONDA_ENV \\"
echo "    --borzoi-dir $BORZOI_DIR --borzoi-paper-dir $BORZOI_PAPER_DIR"
