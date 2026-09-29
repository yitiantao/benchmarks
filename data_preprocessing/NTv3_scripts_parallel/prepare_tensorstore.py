#!/usr/bin/env python3
"""Convert NTv3 benchmark BigWigs to one chunked TensorStore per species.

Layout follows the NTv3 paper: float16, zstd level 3, and chunks spanning
8192 genomic bases by all tracks for a species. Chromosomes are concatenated
with chunk-aligned offsets and described in metadata.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import resource
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyBigWig
import tensorstore as ts
from pyfaidx import Fasta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--species", action="append", help="Repeatable; default: all with BigWigs")
    p.add_argument("--chunk-bp", type=int, default=8192)
    p.add_argument("--block-chunks", type=int, default=16,
                   help="Write this many chunks per transaction (default: 16)")
    p.add_argument("--workers", type=int, default=1,
                   help="BigWig reader processes; 1 keeps sequential reading")
    p.add_argument("--prefetch", type=int, default=None,
                   help="Maximum in-flight/computed blocks (default: 2 * workers)")
    p.add_argument("--no-mask", action="store_true",
                   help="Do not store the explicit BigWig coverage mask")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-bp", type=int, default=None,
                   help="Testing only: stop each chromosome after N bp")
    p.add_argument("--all-contigs", action="store_true",
                   help="Include FASTA contigs absent from splits.bed")
    return p.parse_args()


def read_tracks(dataset: Path) -> dict[str, list[dict[str, str]]]:
    with (dataset / "benchmark_metadata.tsv").open(newline="") as f:
        metadata = {r["file_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    result: dict[str, list[dict[str, str]]] = {}
    for path in sorted(dataset.glob("*/functional_tracks/*.bigwig")):
        row = metadata.get(path.stem)
        if row is None:
            print(f"WARNING: missing metadata; skipping {path}", file=sys.stderr)
            continue
        species = row["species_common_name"]
        result.setdefault(species, []).append({**row, "path": str(path.resolve())})
    return result


def zarr_spec(path: Path, shape: tuple[int, int], chunks: tuple[int, int], dtype: str):
    fill = 0 if dtype == "|u1" else 0.0
    return {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path.resolve())},
        "metadata": {
            "dtype": dtype,
            "shape": list(shape),
            "chunks": list(chunks),
            "compressor": {"id": "zstd", "level": 3},
            "fill_value": fill,
            "order": "C",
        },
        "create": True,
        "delete_existing": True,
    }


def open_existing_zarr(path: Path):
    return ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(path.resolve())}}).result()


def align_up(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def canonical_contig(name: str) -> str:
    value = str(name).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    value = value.casefold()
    return "mt" if value in {"m", "mt"} else value


def alias_index(names):
    result = defaultdict(list)
    for name in names:
        result[canonical_contig(name)].append(str(name))
    return result


def resolve_required(public_names, source_names, species, source_label):
    source_names = [str(x) for x in source_names]
    exact, aliases = set(source_names), alias_index(source_names)
    resolved, missing = [], []
    for public_name in public_names:
        if public_name in exact:
            source_name = public_name
        else:
            candidates = aliases[canonical_contig(public_name)]
            if len(candidates) != 1:
                missing.append(public_name)
                continue
            source_name = candidates[0]
        resolved.append((public_name, source_name))
    if missing:
        raise ValueError(
            f"{species}: {len(missing)} contig(s) cannot be mapped to {source_label}; "
            f"examples: {', '.join(missing[:8])}"
        )
    source_order = {name: i for i, name in enumerate(source_names)}
    return sorted(resolved, key=lambda pair: source_order[pair[1]])


def resolve_optional(public_name, source_names, aliases):
    if public_name in source_names:
        return public_name
    candidates = aliases[canonical_contig(public_name)]
    return candidates[0] if len(candidates) == 1 else None


_BW_HANDLES = None
_BW_CHROM_MAPS = None


def init_bigwig_worker(track_paths, track_chrom_maps):
    """Open process-local handles; pyBigWig handles must never cross processes."""
    global _BW_HANDLES, _BW_CHROM_MAPS
    _BW_HANDLES = [pyBigWig.open(path) for path in track_paths]
    _BW_CHROM_MAPS = track_chrom_maps


def close_bigwig_worker():
    global _BW_HANDLES
    if _BW_HANDLES:
        for handle in _BW_HANDLES:
            handle.close()
    _BW_HANDLES = None


def read_functional_block(task):
    """Read one block in a worker; only the parent writes the TensorStore."""
    chrom_index, chrom, start, end = task
    values = np.zeros((end - start, len(_BW_HANDLES)), dtype=np.float16)
    coverage = np.zeros((end - start, len(_BW_HANDLES)), dtype=np.uint8)
    clipped = 0
    covered = np.zeros(len(_BW_HANDLES), dtype=np.int64)
    for track_index, bw in enumerate(_BW_HANDLES):
        bw_chrom = _BW_CHROM_MAPS[track_index].get(chrom)
        if bw_chrom is None:
            continue
        bw_size = bw.chroms(bw_chrom)
        if start >= bw_size:
            continue
        query_end = min(end, bw_size)
        x = np.asarray(bw.values(bw_chrom, start, query_end, numpy=True), dtype=np.float32)
        present = np.isfinite(x)
        covered[track_index] = int(present.sum())
        clean = np.where(present, x, 0)
        too_large = np.abs(clean) > np.finfo(np.float16).max
        clipped += int(too_large.sum())
        clean = np.clip(clean, -np.finfo(np.float16).max, np.finfo(np.float16).max)
        values[:len(x), track_index] = clean.astype(np.float16)
        coverage[:len(x), track_index] = present
    return chrom_index, chrom, start, end, values, coverage, clipped, covered


def bounded_ordered_map(executor, function, iterable, max_pending):
    """Like executor.map, but bounds large array results and preserves task order."""
    iterator, pending = iter(iterable), deque()
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def prepare_species(dataset: Path, output: Path, species: str, tracks, args) -> None:
    species_dir = output / species
    meta_path = species_dir / "metadata.json"
    progress_path = species_dir / "progress.json"
    if meta_path.exists() and not args.overwrite:
        existing = json.loads(meta_path.read_text())
        if not existing.get("chromosomes") or existing.get("shape", [0])[0] == 0:
            raise RuntimeError(f"{species}: existing functional store is empty/invalid; use --overwrite")
        split_names = {line.split("\t", 1)[0] for line in
                       (dataset / species / "splits.bed").read_text().splitlines() if line}
        stored_names = {c["name"] for c in existing["chromosomes"]}
        if not split_names.issubset(stored_names):
            raise RuntimeError(
                f"{species}: existing store uses contig names incompatible with splits/manifest "
                f"(likely a legacy chr-prefix bug); rebuild with --overwrite"
            )
        print(f"{species}: already complete; use --overwrite to rebuild", file=sys.stderr)
        return
    species_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = dataset / species / "genome.fasta"
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    split_contigs = list(dict.fromkeys(
        line.split("\t", 1)[0] for line in
        (dataset / species / "splits.bed").read_text().splitlines() if line
    ))
    fasta_contigs = [str(chrom) for chrom in fasta.keys()]
    contig_pairs = ([(name, name) for name in fasta_contigs] if args.all_contigs else
                    resolve_required(split_contigs, fasta_contigs, species, "FASTA"))
    chroms = []
    offset = 0
    for public_name, fasta_name in contig_pairs:
        actual_size = len(fasta[fasta_name])
        size = min(actual_size, args.limit_bp) if args.limit_bp else actual_size
        offset = align_up(offset, args.chunk_bp)
        chroms.append({"name": public_name, "fasta_name": fasta_name, "size": size,
                       "reference_size": actual_size, "offset": offset})
        offset += align_up(size, args.chunk_bp)
    if not chroms or offset == 0:
        fasta.close()
        raise RuntimeError(f"{species}: zero chromosomes selected; refusing to create an empty store")

    # Preflight every BigWig before creating/deleting a TensorStore.
    handle_chroms = []
    track_chrom_maps = []
    for track in tracks:
        with pyBigWig.open(track["path"]) as bw:
            bw_chroms = bw.chroms()
        aliases = alias_index(bw_chroms)
        mapping = {c["name"]: resolved for c in chroms
                   if (resolved := resolve_optional(c["name"], bw_chroms, aliases)) is not None}
        if not mapping:
            fasta.close()
            raise RuntimeError(
                f"{species}/{track['file_id']}: no selected contig maps to BigWig; refusing empty track"
            )
        handle_chroms.append(bw_chroms)
        track_chrom_maps.append(mapping)
    shape = (offset, len(tracks))
    chunks = (args.chunk_bp, len(tracks))
    print(f"{species}: shape={shape}, chunks={chunks}, tracks={len(tracks)}", file=sys.stderr)

    soft_fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    fd_worker_limit = (args.workers if soft_fd_limit == resource.RLIM_INFINITY else
                       max(1, (soft_fd_limit - 64) // max(1, len(tracks))))
    worker_count = min(args.workers, fd_worker_limit)
    if worker_count < args.workers:
        print(f"WARNING: reducing workers {args.workers} -> {worker_count}; each worker "
              f"opens {len(tracks)} BigWigs and RLIMIT_NOFILE={soft_fd_limit}", file=sys.stderr)

    resume = progress_path.exists() and (species_dir / "signal" / ".zarray").exists() and not args.overwrite
    if resume:
        progress = json.loads(progress_path.read_text())
        signal = open_existing_zarr(species_dir / "signal")
        print(f"{species}: resuming after {progress}", file=sys.stderr)
    else:
        progress = {"chrom_index": 0, "position": 0}
        signal = ts.open(zarr_spec(species_dir / "signal", shape, chunks, "<f2")).result()
    mask = None
    if not args.no_mask:
        if resume:
            mask = open_existing_zarr(species_dir / "coverage")
        else:
            mask = ts.open(zarr_spec(species_dir / "coverage", shape, chunks, "|u1")).result()

    if not resume:
        atomic_json(progress_path, progress)
    clipped_values = int(progress.get("clipped_values", 0))
    covered_values = np.asarray(progress.get("covered_values", [0] * len(tracks)), dtype=np.int64)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    prefetch = args.prefetch if args.prefetch is not None else max(1, 2 * worker_count)
    if prefetch < 1:
        raise ValueError("--prefetch must be >= 1")

    resume_chrom_index = int(progress["chrom_index"])
    resume_position = int(progress["position"])
    block_bp = args.chunk_bp * args.block_chunks

    def block_tasks():
        for chrom_index, chrom_info in enumerate(chroms):
            if resume and chrom_index < resume_chrom_index:
                continue
            first_start = (resume_position
                           if resume and chrom_index == resume_chrom_index else 0)
            for start in range(first_start, chrom_info["size"], block_bp):
                yield (chrom_index, chrom_info["name"], start,
                       min(start + block_bp, chrom_info["size"]))

    track_paths = [t["path"] for t in tracks]
    executor = None
    try:
        if worker_count == 1:
            init_bigwig_worker(track_paths, track_chrom_maps)
            results = map(read_functional_block, block_tasks())
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=init_bigwig_worker,
                initargs=(track_paths, track_chrom_maps),
            )
            results = bounded_ordered_map(
                executor, read_functional_block, block_tasks(), prefetch
            )
        last_chrom_index = None
        for chrom_index, chrom, start, end, values, coverage, clipped, covered in results:
            chrom_info = chroms[chrom_index]
            dest_start = chrom_info["offset"] + start
            signal[dest_start:dest_start + len(values), :].write(values).result()
            if mask is not None:
                mask[dest_start:dest_start + len(values), :].write(coverage).result()
            clipped_values += clipped
            covered_values += covered
            progress = {"chrom_index": chrom_index, "chrom": chrom, "position": end,
                        "clipped_values": clipped_values,
                        "covered_values": covered_values.tolist()}
            atomic_json(progress_path, progress)
            if last_chrom_index is not None and last_chrom_index != chrom_index:
                print(file=sys.stderr)
            last_chrom_index = chrom_index
            print(f"\r{species} {chrom}: {end:,}/{chrom_info['size']:,}",
                  end="", file=sys.stderr)
        if last_chrom_index is not None:
            print(file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        else:
            close_bigwig_worker()
        fasta.close()

    if int(covered_values.sum()) == 0:
        raise RuntimeError(
            f"{species}: conversion produced zero covered BigWig values; metadata not written"
        )

    metadata = {
        "format_version": 1,
        "species": species,
        "assembly": tracks[0]["file_assembly"],
        "coordinate_system": "0-based half-open",
        "dtype": "float16",
        "compressor": {"id": "zstd", "level": 3},
        "chunk_shape": list(chunks),
        "shape": list(shape),
        "missing_fill": 0.0,
        "float16_overflow_values_clipped": clipped_values,
        "has_coverage_mask": mask is not None,
        "fasta_path": str(fasta_path.resolve()),
        "chromosomes": chroms,
        "tracks": [
            {"index": i, "file_id": t["file_id"], "assay": t["assay"],
             "mean": float(t["mean"]), "std": float(t["std"]),
             "bigwig_path": t["path"],
             "mapped_contig_count": len(track_chrom_maps[i]),
             "covered_values": int(covered_values[i])}
            for i, t in enumerate(tracks)
        ],
    }
    atomic_json(meta_path, metadata)
    progress_path.unlink(missing_ok=True)
    print(f"{species}: complete -> {species_dir}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or (args.prefetch is not None and args.prefetch < 1):
        raise SystemExit("--workers and --prefetch must be >= 1")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    all_tracks = read_tracks(dataset)
    requested = args.species or sorted(all_tracks)
    unknown = sorted(set(requested) - set(all_tracks))
    if unknown:
        raise SystemExit(f"No functional tracks for species: {', '.join(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    for species in requested:
        prepare_species(dataset, output, species, all_tracks[species], args)


if __name__ == "__main__":
    main()
