#!/usr/bin/env python3
"""Fast structural checks, with optional SHA256 verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def require(path: str, minimum_size: int = 1) -> None:
    target = ROOT / path
    if not target.is_file():
        raise RuntimeError(f"缺少文件: {target}")
    size = target.stat().st_size
    if size < minimum_size:
        raise RuntimeError(f"文件尺寸异常: {target} ({size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="运行 sha256sum -c")
    args = parser.parse_args()

    require("configs/params_pred.json")
    require("configs/targets_gtex.txt")
    require("configs/targets_human.txt")
    require("configs/targets_rna.txt")
    require("data/eqtl/pos_merge.vcf.gz")
    require("data/eqtl/neg_merge.vcf.gz")
    require("reference/hg38/assembly/ucsc/hg38.fa", 3_000_000_000)
    require("reference/hg38/assembly/ucsc/hg38.fa.fai", 10_000)
    require(
        "reference/hg38/genes/gencode41/gencode41_basic_nort.gtf",
        800_000_000,
    )
    require("reference/hg38/genes/polyadb/polyadb_human_v3.csv.gz", 5_000_000)
    require("reference/alphagenome/polyadb_human_v3.feather", 10_000_000)
    require("src/borzoi_scripts/borzoi_sed.py")
    require("src/borzoi_scripts/borzoi_sad.py")
    require("src/baskerville/seqnn.py")

    models = sorted((ROOT / "models").glob("replicate_*/model0_best.h5"))
    if len(models) != 4:
        raise RuntimeError(f"模型数量应为 4，实际为 {len(models)}")
    for model in models:
        if model.stat().st_size < 700_000_000:
            raise RuntimeError(f"模型尺寸异常: {model}")

    positive = sorted((ROOT / "data/eqtl").glob("*_pos.vcf.gz"))
    negative = sorted((ROOT / "data/eqtl").glob("*_neg.vcf.gz"))
    tissue_tables = sorted((ROOT / "data/eqtl/tables").glob("*.tsv.gz"))
    if len(positive) != 49 or len(negative) != 49 or len(tissue_tables) != 49:
        raise RuntimeError(
            "eQTL tissue 文件数量异常: "
            f"pos={len(positive)}, neg={len(negative)}, tables={len(tissue_tables)}"
        )

    for task in ("sqtl", "paqtl", "ipaqtl"):
        require(f"data/{task}/pos_merge.vcf.gz")
        require(f"data/{task}/neg_merge.vcf.gz")
        task_positive = sorted((ROOT / "data" / task).glob("*_pos.vcf.gz"))
        task_negative = sorted((ROOT / "data" / task).glob("*_neg.vcf.gz"))
        if len(task_positive) != 49 or len(task_negative) != 49:
            raise RuntimeError(
                f"{task} tissue 文件数量异常: "
                f"pos={len(task_positive)}, neg={len(task_negative)}"
            )

    print("结构检查通过：4 个模型、4 类 QTL、49 个 tissue 和参考文件齐全。")
    if args.deep:
        checksum_file = ROOT / "SHA256SUMS"
        require("SHA256SUMS")
        subprocess.run(
            ["sha256sum", "--check", str(checksum_file)], cwd=ROOT, check=True
        )
        print("SHA256 校验通过。")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"检查失败: {error}", file=sys.stderr)
        raise SystemExit(1)
