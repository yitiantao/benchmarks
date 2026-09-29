#!/usr/bin/env python3
"""Select NTv3 BigWigs by species/assay and extract a genomic signal matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import pyBigWig
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install dependencies with `python -m pip install numpy pyBigWig`.") from exc


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalog", type=Path, help="Catalog directory made by build_coverage_catalog.py")
    p.add_argument("--dataset", type=Path, required=True, help="NTv3_benchmark_dataset directory")
    p.add_argument("--species", required=True, help="e.g. human")
    p.add_argument("--assay", required=True, help="Exact assay, e.g. ATAC-seq")
    p.add_argument("--region", required=True, help="0-based half-open region, e.g. chr19:6700000-6831072")
    p.add_argument("--output", type=Path, required=True, help="Output .npz")
    p.add_argument("--fill", default="nan", help="Missing-value fill: nan or a number (default: nan)")
    p.add_argument("--bin-size", type=int, default=1, help="Mean-pool into this many bp (default: 1)")
    return p.parse_args()


def parse_region(text):
    chrom, coords = text.rsplit(":", 1)
    start, end = map(lambda x: int(x.replace(",", "")), coords.split("-", 1))
    if start < 0 or end <= start:
        raise ValueError("region must satisfy 0 <= start < end")
    return chrom, start, end


def main():
    args = arguments()
    chrom, start, end = parse_region(args.region)
    fill = np.nan if args.fill.lower() == "nan" else float(args.fill)
    with (args.catalog / "tracks.tsv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t")
                if r["species_common_name"] == args.species and r["assay"] == args.assay]
    if not rows:
        raise SystemExit(f"No tracks for species={args.species!r}, assay={args.assay!r}")

    length = end - start
    matrix = np.full((len(rows), length), fill, dtype=np.float32)
    present = np.zeros((len(rows), length), dtype=bool)
    messages = []
    for i, row in enumerate(rows):
        path = args.dataset / row["relative_path"]
        with pyBigWig.open(str(path)) as bw:
            chroms = bw.chroms()
            if chrom not in chroms:
                messages.append(f"{row['file_id']}: chromosome {chrom} absent; filled with {args.fill}")
                continue
            query_end = min(end, chroms[chrom])
            if start >= query_end:
                messages.append(f"{row['file_id']}: region outside chromosome; filled with {args.fill}")
                continue
            values = np.asarray(bw.values(chrom, start, query_end, numpy=True), dtype=np.float32)
            ok = ~np.isnan(values)
            matrix[i, :len(values)][ok] = values[ok]
            present[i, :len(values)] = ok
            missing = length - int(ok.sum())
            if missing:
                messages.append(f"{row['file_id']}: {missing}/{length} bp not covered; filled with {args.fill}")

    if args.bin_size > 1:
        bins = math.ceil(length / args.bin_size)
        padded = bins * args.bin_size
        x = np.full((len(rows), padded), fill, dtype=np.float32)
        mask = np.zeros((len(rows), padded), dtype=bool)
        x[:, :length], mask[:, :length] = matrix, present
        sums = np.where(mask, x, 0).reshape(len(rows), bins, args.bin_size).sum(axis=2)
        counts = mask.reshape(len(rows), bins, args.bin_size).sum(axis=2)
        matrix = np.full_like(sums, fill, dtype=np.float32)
        np.divide(sums, counts, out=matrix, where=counts > 0)
        present = counts > 0

    metadata = {
        "species": args.species, "assay": args.assay, "chrom": chrom,
        "start": start, "end": end, "bin_size": args.bin_size,
        "coordinate_system": "0-based half-open", "missing_fill": args.fill,
    }
    np.savez_compressed(args.output, signal=matrix, covered=present,
                        file_ids=np.asarray([r["file_id"] for r in rows]),
                        metadata=json.dumps(metadata))
    if messages:
        print("WARNING: requested region has missing BigWig coverage:", file=sys.stderr)
        print("\n".join(f"  - {m}" for m in messages), file=sys.stderr)
    else:
        print("All selected BigWigs cover the complete requested region.", file=sys.stderr)
    print(f"Wrote {matrix.shape} signal matrix to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
