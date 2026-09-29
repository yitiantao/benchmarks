#!/usr/bin/env python3
"""Create reusable official-region and deterministic NTv3 window manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("species", pa.string()), ("assembly", pa.string()),
    ("split", pa.string()), ("chrom", pa.string()), ("input_start", pa.int64()),
    ("input_end", pa.int64()), ("target_start", pa.int64()),
    ("target_end", pa.int64()), ("region_id", pa.string()),
    ("sampling_policy", pa.string()), ("seed", pa.int64()),
])


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--window-size", type=int, default=32768)
    p.add_argument("--target-fraction", type=float, default=0.375)
    p.add_argument("--val-samples", type=int, default=1000)
    p.add_argument("--test-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=["official_region_uniform", "coordinate_uniform"],
                   default="official_region_uniform")
    p.add_argument("--paper-eval", action="store_true",
                   help="Tile val/test with stride=window size, as described in the paper")
    p.add_argument("--species", action="append")
    return p.parse_args()


def metadata(dataset):
    rows = {}
    with (dataset / "benchmark_metadata.tsv").open() as f:
        header = f.readline().rstrip().split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip().split("\t")))
            rows.setdefault(row["species_common_name"], row["file_assembly"])
    return rows


def regions(path, species, window):
    result = []
    with path.open() as f:
        for i, line in enumerate(f):
            chrom, start, end, split = line.rstrip().split("\t")
            start, end = int(float(start)), int(float(end))
            if end - start >= window:
                result.append({"species": species, "split": split, "chrom": chrom,
                               "start": start, "end": end,
                               "region_id": f"{species}:region:{i}"})
    return result


def choose(rng, candidates, policy, window):
    if policy == "official_region_uniform":
        region = rng.choice(candidates)
    else:
        total = sum(r["end"] - r["start"] - window + 1 for r in candidates)
        pick = rng.randrange(total)
        for region in candidates:
            count = region["end"] - region["start"] - window + 1
            if pick < count:
                break
            pick -= count
    start = rng.randint(region["start"], region["end"] - window)
    return region, start


def create_rows(species, assembly, split, candidates, n, policy, seed, window, target_fraction):
    rng = random.Random(f"{seed}:{species}:{split}:{policy}:{window}")
    crop = round(window * (1 - target_fraction) / 2)
    target_size = window - 2 * crop
    rows = []
    for i in range(n):
        region, start = choose(rng, candidates, policy, window)
        sid = hashlib.sha1(f"{species}:{split}:{start}:{region['chrom']}:{window}".encode()).hexdigest()[:20]
        rows.append({"sample_id": sid, "species": species, "assembly": assembly,
                     "split": split, "chrom": region["chrom"], "input_start": start,
                     "input_end": start + window, "target_start": start + crop,
                     "target_end": start + crop + target_size,
                     "region_id": region["region_id"], "sampling_policy": policy, "seed": seed})
    return rows


def tile_rows(species, assembly, split, candidates, window, target_fraction, seed):
    crop = round(window * (1 - target_fraction) / 2)
    rows = []
    for region in candidates:
        for start in range(region["start"], region["end"] - window + 1, window):
            sid = hashlib.sha1(f"{species}:{split}:{start}:{region['chrom']}:{window}".encode()).hexdigest()[:20]
            rows.append({"sample_id": sid, "species": species, "assembly": assembly,
                         "split": split, "chrom": region["chrom"], "input_start": start,
                         "input_end": start + window, "target_start": start + crop,
                         "target_end": start + window - crop, "region_id": region["region_id"],
                         "sampling_policy": "paper_stride_1.0", "seed": seed})
    return rows


def main():
    a = args(); a.output.mkdir(parents=True, exist_ok=True)
    assemblies = metadata(a.dataset)
    species_list = a.species or sorted(p.name for p in a.dataset.iterdir() if (p / "splits.bed").exists())
    all_regions = []
    for species in species_list:
        rs = regions(a.dataset / species / "splits.bed", species, a.window_size)
        all_regions.extend(rs)
        for split, n in (("val", a.val_samples), ("test", a.test_samples)):
            candidates = [r for r in rs if r["split"] == split]
            if a.paper_eval:
                rows = tile_rows(species, assemblies.get(species, "unknown"), split, candidates,
                                 a.window_size, a.target_fraction, a.seed)
            else:
                rows = create_rows(species, assemblies.get(species, "unknown"), split, candidates,
                                   n, a.policy, a.seed, a.window_size, a.target_fraction)
            out = a.output / f"{species}.{split}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out, compression="zstd")
    pq.write_table(pa.Table.from_pylist(all_regions), a.output / "official_regions.parquet", compression="zstd")
    config = vars(a).copy()
    config["dataset"], config["output"] = str(a.dataset.resolve()), str(a.output.resolve())
    (a.output / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
