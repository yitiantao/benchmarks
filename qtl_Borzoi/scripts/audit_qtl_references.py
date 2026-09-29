#!/usr/bin/env python3
"""Audit benchmark VCF alleles against hg38 and record exact source positions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
from typing import Iterator

from pyfaidx import Fasta


EXCEPTION_COLUMNS = (
    "task",
    "split",
    "tissue",
    "status",
    "file_path",
    "file_path_absolute",
    "record_index_0based",
    "record_number_1based",
    "file_line_number_1based",
    "chrom",
    "pos_1based",
    "variant_id",
    "variant_id_ref",
    "variant_id_alt",
    "vcf_vs_variant_id",
    "vcf_ref",
    "vcf_alt",
    "fasta_ref",
)


def _variant_id_alleles(variant_id: str) -> tuple[str, str]:
    """Return alleles encoded by GTEx-style chr_pos_ref_alt_build IDs."""
    fields = variant_id.rsplit("_", 3)
    if len(fields) == 4 and fields[-1].startswith("b"):
        return fields[-3].upper(), fields[-2].upper()
    return "", ""


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _source_files(data_dir: Path) -> Iterator[tuple[str, str, str, Path]]:
    eqtl_dir = data_dir / "eqtl"
    for split in ("pos", "neg"):
        suffixes = (f"_{split}.vcf.gz", f"_{split}.vcf")
        selected: dict[str, Path] = {}
        for suffix in suffixes:
            for path in sorted(eqtl_dir.glob(f"*{suffix}")):
                tissue = path.name[: -len(suffix)]
                if tissue in {"pos_merge", "neg_merge"}:
                    continue
                # Prefer compressed source when both forms exist.
                selected.setdefault(tissue, path)
        for tissue, path in sorted(selected.items()):
            yield "eqtl", split, tissue, path

    for task in ("sqtl", "paqtl", "ipaqtl"):
        for split in ("pos", "neg"):
            plain = data_dir / task / f"{split}_merge.vcf"
            compressed = Path(str(plain) + ".gz")
            path = compressed if compressed.is_file() else plain
            if not path.is_file():
                raise FileNotFoundError(f"Missing benchmark VCF: {path}")
            yield task, split, "", path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("reference/hg38/assembly/ucsc/hg38.fa"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/qtl_reference_audit")
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    fasta_path = args.fasta.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    genome = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    status_counts: Counter[str] = Counter()
    task_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_split_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    file_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scanned_files: Counter[str] = Counter()
    exceptions: list[dict[str, object]] = []
    unique_by_status: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    task_unique_by_status: dict[str, dict[str, set[tuple[object, ...]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    base_cache: dict[tuple[str, int], str] = {}
    variant_id_relationships: Counter[str] = Counter()

    for task, split, tissue, path in _source_files(data_dir):
        record_index = 0
        with _open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                chrom, position, variant_id, ref, alt = fields[:5]
                position = int(position)
                ref, alt = ref.upper(), alt.upper()
                variant_id_ref, variant_id_alt = _variant_id_alleles(variant_id)
                if (ref, alt) == (variant_id_ref, variant_id_alt):
                    vcf_vs_variant_id = "same_as_variant_id"
                elif (ref, alt) == (variant_id_alt, variant_id_ref):
                    vcf_vs_variant_id = "exact_ref_alt_swap"
                elif variant_id_ref and variant_id_alt:
                    vcf_vs_variant_id = "different_from_variant_id"
                else:
                    vcf_vs_variant_id = "variant_id_unparsed"
                variant_id_relationships[vcf_vs_variant_id] += 1
                if len(ref) != 1 or len(alt) != 1 or "," in alt:
                    status = "unsupported_non_biallelic_snv"
                    fasta_ref = ""
                else:
                    cache_key = (chrom, position)
                    try:
                        fasta_ref = base_cache.get(cache_key)
                        if fasta_ref is None:
                            fasta_ref = str(genome[chrom][position - 1 : position]).upper()
                            base_cache[cache_key] = fasta_ref
                        if fasta_ref == ref:
                            status = "ref_matches_fasta"
                        elif fasta_ref == alt:
                            status = "alt_matches_fasta"
                        else:
                            status = "neither_allele_matches_fasta"
                    except (KeyError, IndexError):
                        fasta_ref = ""
                        status = "reference_unavailable"

                status_counts[status] += 1
                task_status_counts[task][status] += 1
                task_split_status_counts[f"{task}/{split}"][status] += 1
                file_status_counts[str(path)][status] += 1
                scanned_files[str(path)] += 1
                variant_key = (task, chrom, position, variant_id, ref, alt)
                unique_by_status[status].add(variant_key)
                task_unique_by_status[task][status].add(variant_key)
                if status != "ref_matches_fasta":
                    try:
                        relative = str(path.resolve().relative_to(root))
                    except ValueError:
                        relative = str(path)
                    exceptions.append(
                        {
                            "task": task,
                            "split": split,
                            "tissue": tissue,
                            "status": status,
                            "file_path": relative,
                            "file_path_absolute": str(path.resolve()),
                            "record_index_0based": record_index,
                            "record_number_1based": record_index + 1,
                            "file_line_number_1based": line_number,
                            "chrom": chrom,
                            "pos_1based": position,
                            "variant_id": variant_id,
                            "variant_id_ref": variant_id_ref,
                            "variant_id_alt": variant_id_alt,
                            "vcf_vs_variant_id": vcf_vs_variant_id,
                            "vcf_ref": ref,
                            "vcf_alt": alt,
                            "fasta_ref": fasta_ref,
                        }
                    )
                record_index += 1

    exceptions_path = output_dir / "exceptions.tsv"
    with exceptions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXCEPTION_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(exceptions)

    summary = {
        "data_dir": str(data_dir),
        "fasta": str(fasta_path),
        "source_scope": (
            "eQTL tissue *_pos/_neg VCFs and sQTL/paQTL/iPaQTL "
            "pos_merge/neg_merge VCFs used by the unified benchmark"
        ),
        "files": len(scanned_files),
        "record_occurrences": sum(status_counts.values()),
        "status_occurrences": dict(sorted(status_counts.items())),
        "status_occurrence_fraction": {
            status: count / sum(status_counts.values())
            for status, count in sorted(status_counts.items())
        },
        "status_unique_variants": {
            status: len(keys) for status, keys in sorted(unique_by_status.items())
        },
        "vcf_vs_variant_id_occurrences": dict(
            sorted(variant_id_relationships.items())
        ),
        "by_task_occurrences": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(task_status_counts.items())
        },
        "by_task_unique_variants": {
            task: {
                status: len(keys) for status, keys in sorted(status_sets.items())
            }
            for task, status_sets in sorted(task_unique_by_status.items())
        },
        "by_task_split_occurrences": {
            task_split: dict(sorted(counts.items()))
            for task_split, counts in sorted(task_split_status_counts.items())
        },
        "by_file_occurrences": {
            str(Path(path).resolve()): dict(sorted(counts.items()))
            for path, counts in sorted(file_status_counts.items())
        },
        "exceptions_tsv": str(exceptions_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
