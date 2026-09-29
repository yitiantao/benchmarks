"""Utilities for collecting data-parallel Borzoi SED worker outputs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


SNP_KEYS = frozenset({"snp", "chr", "pos", "ref_allele", "alt_allele"})
COMMON_KEYS = frozenset({"target_ids", "target_labels"})


def collect_sed_shards(output_dir: str | Path, num_shards: int) -> Path:
    """Merge ``jobN/sed.h5`` files and re-index gene rows to merged SNPs."""
    output_dir = Path(output_dir)
    shard_paths = [output_dir / f"job{index}" / "sed.h5" for index in range(num_shards)]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Borzoi shard outputs: {missing}")

    snp_counts: list[int] = []
    row_counts: list[int] = []
    for path in shard_paths:
        with h5py.File(path, "r") as handle:
            snp_counts.append(len(handle["snp"]))
            row_counts.append(len(handle["si"]))

    destination = output_dir / "sed.h5"
    temporary = output_dir / "sed.h5.tmp"
    with h5py.File(shard_paths[0], "r") as first, h5py.File(temporary, "w") as merged:
        for key in COMMON_KEYS:
            merged.create_dataset(key, data=first[key])

        for key in first.keys():
            if key in COMMON_KEYS or key in SNP_KEYS:
                continue
            shape = list(first[key].shape)
            shape[0] = sum(row_counts)
            merged.create_dataset(key, shape=tuple(shape), dtype=first[key].dtype)

        snp_values: dict[str, list[np.ndarray]] = {key: [] for key in SNP_KEYS}
        row_offset = 0
        snp_offset = 0
        for path, snp_count, row_count in zip(shard_paths, snp_counts, row_counts):
            with h5py.File(path, "r") as shard:
                for key in COMMON_KEYS:
                    if not np.array_equal(first[key][:], shard[key][:]):
                        raise ValueError(f"Inconsistent {key!r} in Borzoi shard {path}")
                for key in SNP_KEYS:
                    snp_values[key].append(shard[key][:])
                for key in shard.keys():
                    if key in COMMON_KEYS or key in SNP_KEYS:
                        continue
                    values = shard[key][:]
                    if key == "si":
                        values = values + snp_offset
                    merged[key][row_offset : row_offset + row_count] = values
            row_offset += row_count
            snp_offset += snp_count

        for key, arrays in snp_values.items():
            merged.create_dataset(key, data=np.concatenate(arrays, axis=0))

    temporary.replace(destination)
    return destination
