#!/usr/bin/env python3
"""TensorStore-backed NTv3 random-access source and Grain loader helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import grain
import numpy as np
import pyarrow.parquet as pq
import tensorstore as ts
from pyfaidx import Fasta


def _open_zarr(path: Path):
    return ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}}).result()


class NTv3WindowSource:
    """Pickle-safe Grain random-access source for fixed or dynamic windows."""

    def __init__(self, tensorstore_root, species, *, manifest=None, regions=None,
                 split="train", num_samples=100_000, window_size=32768,
                 target_fraction=0.375, seed=0, sampling_policy="official_region_uniform",
                 assay=None, normalize=True, use_mask=True,
                 load_functional=True, load_annotations=False):
        self.root = str(Path(tensorstore_root).resolve())
        self.species = species
        self.manifest_path = str(Path(manifest).resolve()) if manifest else None
        self.regions_path = str(Path(regions).resolve()) if regions else None
        self.split, self.num_samples = split, num_samples
        self.window_size, self.target_fraction, self.seed = window_size, target_fraction, seed
        self.sampling_policy, self.assay = sampling_policy, assay
        self.normalize, self.use_mask = normalize, use_mask
        self.load_functional, self.load_annotations = load_functional, load_annotations
        if not load_functional and not load_annotations:
            raise ValueError("At least one of load_functional/load_annotations must be true")
        self._reset_handles()
        self._load_static_metadata()

    def _reset_handles(self):
        self._signal = self._coverage = self._annotations = self._fasta = None

    def _load_static_metadata(self):
        species_dir = Path(self.root) / self.species
        functional_meta = species_dir / "metadata.json"
        annotation_meta = species_dir / "annotation_metadata.json"
        if self.load_functional and not functional_meta.exists():
            raise FileNotFoundError(f"Functional metadata not found: {functional_meta}")
        if self.load_annotations and not annotation_meta.exists():
            raise FileNotFoundError(f"Annotation metadata not found: {annotation_meta}")
        self.meta = json.loads((functional_meta if self.load_functional else annotation_meta).read_text())
        if not self.meta.get("chromosomes") or self.meta.get("shape", [0])[0] <= 0:
            raise ValueError(
                f"Invalid empty prepared store for species={self.species!r}; rebuild with --overwrite"
            )
        self.chroms = {c["name"]: c for c in self.meta["chromosomes"]}
        if self.load_functional:
            selected = [t for t in self.meta["tracks"] if self.assay is None or t["assay"] == self.assay]
            if not selected:
                raise ValueError(f"No tracks for species={self.species!r}, assay={self.assay!r}")
            self.track_indices = np.asarray([t["index"] for t in selected], dtype=np.int64)
            self.track_ids = [t["file_id"] for t in selected]
            self.track_assays = [t["assay"] for t in selected]
            self.track_means = np.asarray([t["mean"] for t in selected], dtype=np.float32)
            self.is_rna_seq = np.asarray(["rna-seq" in t["assay"].lower() for t in selected], dtype=bool)
        if self.load_annotations:
            self.annotation_meta = json.loads(annotation_meta.read_text())
            self.annotation_chroms = {
                c["name"]: c for c in self.annotation_meta["chromosomes"]
            }
            # The physical concatenation order and offsets may differ. Joint
            # loading requires matching public contigs/sizes, then slices each
            # store with its own offset in __getitem__.
            sizes = lambda cs: {name: int(c["size"]) for name, c in cs.items()}
            if self.load_functional and sizes(self.annotation_chroms) != sizes(self.chroms):
                raise ValueError("Functional and annotation chromosome names/sizes do not match")
            self.annotation_names = [e["name"] for e in self.annotation_meta["elements"]]
        if self.manifest_path:
            table = pq.read_table(self.manifest_path, filters=[("species", "=", self.species)])
            self.windows = table.to_pylist()
            manifest_chroms = {row["chrom"] for row in self.windows}
            missing = sorted(manifest_chroms - set(self.chroms))
            if missing:
                raise ValueError(
                    f"Prepared store for {self.species!r} lacks {len(missing)} manifest contig(s), "
                    f"including {missing[:5]}; rebuild legacy stores with --overwrite"
                )
        else:
            table = pq.read_table(self.regions_path,
                                  filters=[("species", "=", self.species), ("split", "=", self.split)])
            self.regions = table.to_pylist()
            region_chroms = {row["chrom"] for row in self.regions}
            missing = sorted(region_chroms - set(self.chroms))
            if missing:
                raise ValueError(
                    f"Prepared store for {self.species!r} lacks {len(missing)} region contig(s), "
                    f"including {missing[:5]}; rebuild legacy stores with --overwrite"
                )
            if self.sampling_policy == "paper_stride":
                self.paper_stride = max(1, round(self.window_size * 0.001)) if self.split == "train" else self.window_size
                self.region_window_counts = [   # 统计所有可能的区间总共可以画出多少数据区段
                    1 + (r["end"] - r["start"] - self.window_size) // self.paper_stride
                    for r in self.regions
                ]
                self.region_window_prefix = np.cumsum(self.region_window_counts)

    def _ensure_open(self):
        species_dir = Path(self.root) / self.species
        if self._signal is None:
            if self.load_functional:
                self._signal = _open_zarr(species_dir / "signal")
                if self.use_mask and self.meta["has_coverage_mask"]:
                    self._coverage = _open_zarr(species_dir / "coverage")
            else:
                self._signal = False
            if self.load_annotations:
                self._annotations = _open_zarr(species_dir / "annotations")
            self._fasta = Fasta(self.meta["fasta_path"], as_raw=True, sequence_always_upper=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_signal"] = state["_coverage"] = state["_annotations"] = state["_fasta"] = None
        return state

    def __len__(self):
        if self.manifest_path:
            return len(self.windows)
        if self.sampling_policy == "paper_stride":
            return int(self.region_window_prefix[-1])
        return self.num_samples

    def _dynamic_window(self, index):
        if self.sampling_policy == "paper_stride":
            region_index = int(np.searchsorted(self.region_window_prefix, index, side="right"))
            previous = 0 if region_index == 0 else int(self.region_window_prefix[region_index - 1])
            region = self.regions[region_index]
            start = region["start"] + (index - previous) * self.paper_stride
            crop = round(self.window_size * (1 - self.target_fraction) / 2)
            return {"chrom": region["chrom"], "input_start": start,
                    "input_end": start + self.window_size, "target_start": start + crop,
                    "target_end": start + self.window_size - crop, "split": self.split,
                    "sample_id": f"paper:{self.species}:{self.split}:{index}"}
        digest = hashlib.blake2b(f"{self.seed}:{self.species}:{self.split}:{index}".encode(),
                                 digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, "little"))
        if self.sampling_policy == "official_region_uniform":
            region = rng.choice(self.regions)
        else:
            weights = [r["end"] - r["start"] - self.window_size + 1 for r in self.regions]
            region = rng.choices(self.regions, weights=weights, k=1)[0]
        start = rng.randint(region["start"], region["end"] - self.window_size)
        crop = round(self.window_size * (1 - self.target_fraction) / 2)
        return {"chrom": region["chrom"], "input_start": start,
                "input_end": start + self.window_size, "target_start": start + crop,
                "target_end": start + self.window_size - crop, "split": self.split,
                "sample_id": f"dynamic:{index}"}

    def __getitem__(self, index):
        self._ensure_open()
        row = self.windows[index] if self.manifest_path else self._dynamic_window(index)
        chrom = row["chrom"]
        c = self.chroms[chrom]
        start, end = int(row["input_start"]), int(row["input_end"])
        global_start, global_end = c["offset"] + start, c["offset"] + end
        target_left = int(row["target_start"]) - start
        target_right = int(row["target_end"]) - start
        sequence = str(self._fasta[c.get("fasta_name", chrom)][start:end])
        result = {"sample_id": row["sample_id"], "species": self.species, "chrom": chrom,
                  "input_start": start, "input_end": end, "sequence": sequence}
        if self.load_functional:
            signal = np.asarray(self._signal[global_start:global_end, self.track_indices].read().result(),
                                dtype=np.float32)
            if self._coverage is None:
                coverage = np.ones(signal.shape, dtype=bool)
            else:
                coverage = np.asarray(
                    self._coverage[global_start:global_end, self.track_indices].read().result(), dtype=bool)
            signal, coverage = signal[target_left:target_right], coverage[target_left:target_right]
            if self.normalize:
                scaled = signal / self.track_means[None, :]
                scaled[:, self.is_rna_seq] = np.power(np.maximum(scaled[:, self.is_rna_seq], 0), 0.75)
                signal = np.where(scaled > 10, 2 * np.sqrt(scaled * 10) - 10, scaled).astype(np.float32)
            result.update({"targets": signal, "target_mask": coverage,
                           "track_ids": np.asarray(self.track_ids),
                           "track_assays": np.asarray(self.track_assays),
                           "track_indices": self.track_indices.copy()})
        if self.load_annotations:
            annotation_c = self.annotation_chroms[chrom]
            annotation_start = int(annotation_c["offset"]) + start
            annotation_end = int(annotation_c["offset"]) + end
            annotations = np.asarray(
                self._annotations[annotation_start:annotation_end, :].read().result(),
                dtype=np.uint8)
            result.update({"annotation_targets": annotations[target_left:target_right],
                           "annotation_names": np.asarray(self.annotation_names)})
        return result

    def __repr__(self):
        return (f"NTv3WindowSource(species={self.species!r}, split={self.split!r}, "
                f"samples={len(self)}, assay={self.assay!r})")


def make_grain_loader(source, batch_size, *, shuffle, seed=0, workers=4,
                      prefetch_buffer=16, drop_remainder=True):
    """Build a multiprocess Grain pipeline from an NTv3WindowSource."""
    dataset = grain.MapDataset.source(source)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    iterator = dataset.to_iter_dataset(
        read_options=grain.ReadOptions(num_threads=4, prefetch_buffer_size=prefetch_buffer))
    if workers:
        iterator = iterator.mp_prefetch(
            grain.MultiprocessingOptions(num_workers=workers,
                                         per_worker_buffer_size=max(1, prefetch_buffer // workers)))
    return iterator
