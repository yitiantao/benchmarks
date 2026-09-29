#!/usr/bin/env python3
"""Decompress QTL VCFs and keep a consistent variant subset."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def read_lines(path: Path) -> tuple[list[str], list[str]]:
    with gzip.open(path, "rt") as handle:
        lines = list(handle)
    return [line for line in lines if line.startswith("#")], [
        line for line in lines if not line.startswith("#")
    ]


def variant_id(line: str) -> str:
    return line.split("\t", 3)[2]


def matched_positive_id(line: str) -> str | None:
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 8:
        return None
    for item in fields[7].split(";"):
        if item.startswith("PI="):
            return item[3:]
    return None


def select_matched_ids(source: Path, limit: int) -> dict[str, set[str] | None]:
    if not limit:
        return {"pos": None, "neg": None}
    negative = read_lines(source / "neg_merge.vcf.gz")[1]
    selected_negative: list[str] = []
    selected_positive: set[str] = set()
    seen: set[str] = set()
    for line in negative:
        identifier = variant_id(line)
        if identifier in seen:
            continue
        partner = matched_positive_id(line)
        if partner is None:
            continue
        seen.add(identifier)
        selected_negative.append(identifier)
        selected_positive.add(partner)
        if len(selected_negative) >= limit:
            break
    return {"pos": selected_positive, "neg": set(selected_negative)}


def select_variant_ids(source: Path, split: str, limit: int) -> set[str] | None:
    if not limit:
        return None

    groups = [read_lines(path)[1] for path in sorted(source.glob(f"*_{split}.vcf.gz"))]
    indexes = [0] * len(groups)
    selected: set[str] = set()
    while len(selected) < limit:
        added = False
        for group_index, records in enumerate(groups):
            while indexes[group_index] < len(records):
                record = records[indexes[group_index]]
                indexes[group_index] += 1
                identifier = variant_id(record)
                if identifier not in selected:
                    selected.add(identifier)
                    added = True
                    break
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def select_variant_ids_per_tissue(
    source: Path, split: str, limit: int
) -> dict[str, set[str]]:
    """Select the same first-N tissue records consumed by AlphaGenome."""
    selected: dict[str, set[str]] = {}
    for path in sorted(source.glob(f"*_{split}.vcf.gz")):
        records = read_lines(path)[1]
        if limit:
            records = records[:limit]
        selected[path.name] = {variant_id(record) for record in records}
    return selected


def write_vcf(source: Path, destination: Path, selected: set[str] | None) -> int:
    headers, records = read_lines(source)
    if selected is not None:
        records = [line for line in records if variant_id(line) in selected]
    destination.write_text("".join(headers + records))
    return len(records)


def subset_eqtl_tables(
    source: Path,
    destination: Path,
    selected: set[str] | dict[str, set[str]],
) -> dict[str, int]:
    """Filter SuSiE tables to the positive variants that were actually scored."""
    source_dir = source / "tables"
    if not source_dir.is_dir():
        return {}
    destination_dir = destination / "tables"
    destination_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for table_path in sorted(source_dir.glob("*.tsv.gz")):
        output_path = destination_dir / table_path.name
        tissue = table_path.name.removesuffix(".tsv.gz")
        selected_for_table = (
            selected.get(f"{tissue}_pos.vcf.gz", set())
            if isinstance(selected, dict)
            else selected
        )
        count = 0
        with gzip.open(table_path, "rt") as input_handle:
            header = input_handle.readline()
            columns = header.rstrip("\n").split("\t")
            try:
                variant_index = columns.index("variant")
            except ValueError as error:
                raise ValueError(f"Missing 'variant' column in {table_path}") from error
            with gzip.open(output_path, "wt") as output_handle:
                output_handle.write(header)
                for line in input_handle:
                    fields = line.rstrip("\n").split("\t")
                    if (
                        len(fields) > variant_index
                        and fields[variant_index] in selected_for_table
                    ):
                        output_handle.write(line)
                        count += 1
        counts[tissue] = count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-variants", type=int, default=1000)
    parser.add_argument(
        "--limit-scope",
        choices=("global", "per-tissue"),
        default="global",
        help=(
            "eQTL limit scope: global round-robin or the first N records "
            "per tissue and split (default: global)"
        ),
    )
    parser.add_argument(
        "--matched-pairs",
        action="store_true",
        help="select negatives and their PI-linked positives (sQTL/paQTL/iPaQTL)",
    )
    args = parser.parse_args()

    if args.max_variants < 0:
        parser.error("--max-variants must be 0 or greater")
    if args.matched_pairs and args.limit_scope != "global":
        parser.error("--limit-scope applies only to eQTL, not matched pairs")

    args.destination.mkdir(parents=True, exist_ok=True)
    selected_by_split: dict[str, set[str] | None]
    if args.matched_pairs:
        selected_by_split = select_matched_ids(args.source, args.max_variants)
    else:
        selected_by_split = {}
    selected_by_tissue: dict[str, dict[str, set[str]]] = {}
    merged_counts = {}
    for split in ("pos", "neg"):
        source = args.source / f"{split}_merge.vcf.gz"
        headers, records = read_lines(source)
        if (
            not args.matched_pairs
            and args.limit_scope == "per-tissue"
            and args.max_variants
        ):
            selected_by_tissue[split] = select_variant_ids_per_tissue(
                args.source, split, args.max_variants
            )
            selected_by_split[split] = set().union(
                *selected_by_tissue[split].values()
            )
        elif not args.matched_pairs:
            selected_by_split[split] = select_variant_ids(
                args.source, split, args.max_variants
            )
        if selected_by_split[split] is not None:
            records = [
                line
                for line in records
                if variant_id(line) in selected_by_split[split]
            ]
        (args.destination / f"{split}_merge.vcf").write_text(
            "".join(headers + records)
        )
        merged_counts[split] = len(records)

    tissue_counts: dict[str, int] = {}
    for source in sorted(args.source.glob("*.vcf.gz")):
        if source.name in {"pos_merge.vcf.gz", "neg_merge.vcf.gz"}:
            continue
        if source.name.endswith("_pos.vcf.gz"):
            split = "pos"
        elif source.name.endswith("_neg.vcf.gz"):
            split = "neg"
        else:
            continue
        destination = args.destination / source.name.removesuffix(".gz")
        selected_for_tissue = (
            selected_by_tissue[split][source.name]
            if (
                args.limit_scope == "per-tissue"
                and not args.matched_pairs
                and args.max_variants
            )
            else selected_by_split[split]
        )
        tissue_counts[destination.name] = write_vcf(
            source, destination, selected_for_tissue
        )

    table_counts = {}
    if selected_by_split["pos"] is not None:
        table_counts = subset_eqtl_tables(
            args.source,
            args.destination,
            (
                selected_by_tissue["pos"]
                if (
                    args.limit_scope == "per-tissue"
                    and not args.matched_pairs
                    and args.max_variants
                )
                else selected_by_split["pos"]
            ),
        )

    manifest = {
        "source": str(args.source.resolve()),
        "destination": str(args.destination.resolve()),
        "limit_scope": (
            "matched-pairs" if args.matched_pairs else args.limit_scope
        ),
        "max_variants_per_split": (
            args.max_variants if args.limit_scope == "global" else None
        ),
        "max_variants_per_tissue_per_split": (
            args.max_variants
            if args.limit_scope == "per-tissue" and not args.matched_pairs
            else None
        ),
        "selection": (
            "matched_negative_positive_pairs"
            if args.matched_pairs
            else (
                "first_n_per_tissue_per_split"
                if args.limit_scope == "per-tissue"
                else "round_robin_across_tissues"
            )
        ),
        "merged_counts": merged_counts,
        "selected_unique_ids": {
            split: None if selected is None else len(selected)
            for split, selected in selected_by_split.items()
        },
        "tissue_vcf_counts": tissue_counts,
        "selected_unique_ids_by_tissue": {
            split: {
                name.removesuffix(".vcf.gz"): len(selected)
                for name, selected in sorted(by_tissue.items())
            }
            for split, by_tissue in selected_by_tissue.items()
        },
        "causal_table_row_counts": table_counts,
    }
    (args.destination / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        "Prepared variants: "
        f"positive={merged_counts['pos']} negative={merged_counts['neg']}"
    )


if __name__ == "__main__":
    main()
