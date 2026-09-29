#!/usr/bin/env python3
"""用最少依赖快速查看 ClinVar variant_summary.txt.gz。"""

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FILE = SCRIPT_DIR / "data/raw/variant_summary_latest_2026-09-14.txt.gz"
SUMMARY_FIELDS = [
    "Assembly",
    "Type",
    "ClinicalSignificance",
    "ReviewStatus",
    "OriginSimple",
    "Chromosome",
]
EXAMPLE_FIELDS = [
    "#AlleleID",
    "VariationID",
    "Type",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "ReviewStatus",
    "Assembly",
    "Chromosome",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
]


def print_counter(title, counter, total, limit=12):
    print(f"\n{title}")
    for value, count in counter.most_common(limit):
        print(f"  {value:<55} {count:>10,}  ({count / total:>7.2%})")
    if len(counter) > limit:
        print(f"  ... 另外还有 {len(counter) - limit} 种取值")


def main():
    parser = argparse.ArgumentParser(
        description="直观查看 ClinVar variant_summary.txt.gz 的结构与常见取值"
    )
    parser.add_argument("file", nargs="?", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--rows", type=int, default=2, help="展示多少条示例记录，默认 2")
    parser.add_argument(
        "--scan",
        type=int,
        default=100_000,
        help="统计前多少行；0 表示扫描整个文件，默认 100000",
    )
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"找不到文件：{args.file}")
    if args.rows < 0 or args.scan < 0:
        parser.error("--rows 和 --scan 不能是负数")

    counters = {field: Counter() for field in SUMMARY_FIELDS}
    missing = Counter()
    examples = []
    scanned = 0

    with gzip.open(args.file, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []

        print("ClinVar variant_summary 快速探查")
        print(f"文件：{args.file}")
        print(f"压缩后大小：{args.file.stat().st_size / 1024**2:,.1f} MiB")
        print("格式：gzip + UTF-8 + TSV（Tab 分隔，首行为字段名）")
        print(f"字段数：{len(fields)}")

        print("\n字段列表")
        for number, field in enumerate(fields, start=1):
            print(f"  {number:>2}. {field}")

        for row in reader:
            if len(examples) < args.rows:
                examples.append(row)
            scanned += 1
            for field in fields:
                if row[field] in {"", "-", "na"}:
                    missing[field] += 1
            for field in SUMMARY_FIELDS:
                counters[field][row[field] or "<空字符串>"] += 1
            if args.scan and scanned >= args.scan:
                break

    print(f"\n本次统计行数：{scanned:,}" + ("（全文件）" if args.scan == 0 else "（文件开头样本）"))
    if scanned == 0:
        return

    for field in SUMMARY_FIELDS:
        print_counter(f"{field} 常见取值", counters[field], scanned)

    print("\n缺失占比较高的字段（把空字符串、'-'、'na' 当作占位缺失）")
    for field, count in missing.most_common():
        if count / scanned >= 0.5:
            print(f"  {field:<45} {count:>10,}  ({count / scanned:>7.2%})")

    print("\n示例记录（只展示最常用字段）")
    for index, row in enumerate(examples, start=1):
        print(f"\n  ---- 记录 {index} ----")
        for field in EXAMPLE_FIELDS:
            print(f"  {field:<24} {row.get(field, '')}")

    print("\n提示：默认只统计前 100,000 行；需要完整统计时使用 --scan 0。")
    print("字段逐列解释见 README.md。")


if __name__ == "__main__":
    main()
