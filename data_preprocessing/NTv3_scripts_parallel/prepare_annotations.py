#!/usr/bin/env python3
"""Convert NTv3 BED annotations to per-species uint8 TensorStores."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tensorstore as ts
from pyfaidx import Fasta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--species", action="append", help="Repeatable; default: all with annotation BEDs")
    p.add_argument("--chunk-bp", type=int, default=8192)
    p.add_argument("--block-chunks", type=int, default=64)
    p.add_argument("--workers", type=int, default=1,
                   help="Chromosome rasterizer/writer processes; 1 is sequential")
    p.add_argument("--prefetch", type=int, default=None,
                   help="Maximum in-flight chromosomes (default: 2 * workers)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit-bp", type=int, default=None, help="Testing only")
    p.add_argument("--all-contigs", action="store_true",
                   help="Include FASTA contigs absent from splits.bed")
    return p.parse_args()


def align_up(value, alignment):
    return math.ceil(value / alignment) * alignment


def canonical_contig(name):
    """Conservative alias key for common FASTA/BED `chr` prefix differences."""
    value = str(name).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    value = value.casefold()
    return "mt" if value in {"m", "mt"} else value


def resolve_fasta_contigs(public_contigs, fasta_contigs, species):
    """Map split/BED-facing contig names to the actual FASTA record names."""
    fasta_contigs = [str(x) for x in fasta_contigs]
    exact = set(fasta_contigs)
    aliases = defaultdict(list)
    for fasta_name in fasta_contigs:
        aliases[canonical_contig(fasta_name)].append(fasta_name)
    resolved, missing = [], []
    for public_name in public_contigs:
        if public_name in exact:
            fasta_name = public_name
        else:
            candidates = aliases[canonical_contig(public_name)]
            if len(candidates) != 1:
                missing.append(public_name)
                continue
            fasta_name = candidates[0]
        resolved.append((public_name, fasta_name))
    if missing:
        examples = ", ".join(missing[:8])
        raise ValueError(
            f"{species}: {len(missing)} split contig(s) cannot be mapped to FASTA; "
            f"examples: {examples}"
        )
    fasta_order = {name: i for i, name in enumerate(fasta_contigs)}
    return sorted(resolved, key=lambda pair: fasta_order[pair[1]])


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def create_store(path, shape, chunks):
    return ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path.resolve())},
        "metadata": {
            "dtype": "|u1", "shape": list(shape), "chunks": list(chunks),
            "compressor": {"id": "zstd", "level": 3},
            "fill_value": 0, "order": "C",
        },
        "create": True, "delete_existing": True,
    }).result()


def render_annotation_block(task):
    """Rasterize one block in a worker; only the parent writes TensorStore."""
    chrom_index, chrom, start, end, overlapping, element_count = task
    labels = np.zeros((end - start, element_count), dtype=np.uint8)
    for element_index, ivs in enumerate(overlapping):
        for iv_start, iv_end in ivs:
            left, right = max(start, iv_start), min(end, iv_end)
            if right > left:
                labels[left - start:right - start, element_index] = 1
    return chrom_index, chrom, start, end, labels


def bounded_ordered_map(executor, function, iterable, max_pending):
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


def load_beds(annotation_dir, elements):
    intervals = {element: defaultdict(list) for element in elements}
    counts = {}
    for element in elements:
        print(f"Loading {element}.bed ...", file=sys.stderr, flush=True)
        count = 0
        with (annotation_dir / f"{element}.bed").open() as handle:
            for line_no, line in enumerate(handle, 1):
                fields = line.rstrip().split("\t")
                if len(fields) < 3:
                    raise ValueError(f"Malformed BED: {element}.bed:{line_no}")
                chrom, start, end = fields[0], int(fields[1]), int(fields[2])
                if start < 0 or end <= start:
                    raise ValueError(f"Invalid BED interval: {element}.bed:{line_no}")
                intervals[element][chrom].append((start, end))
                count += 1
        for chrom in intervals[element]:
            intervals[element][chrom].sort()
        counts[element] = count
        print(f"Loaded {element}: {count:,} intervals", file=sys.stderr, flush=True)
    return intervals, counts


def prepare_species(dataset, output, species, args):
    annotation_dir = dataset / species / "genome_annotation"
    elements = sorted(p.stem for p in annotation_dir.glob("*.bed"))
    species_dir = output / species
    metadata_path = species_dir / "annotation_metadata.json"
    if metadata_path.exists() and not args.overwrite:
        existing = json.loads(metadata_path.read_text())
        if not existing.get("chromosomes") or existing.get("shape", [0])[0] == 0:
            raise RuntimeError(
                f"{species}: existing annotation store is empty/invalid; rerun with --overwrite"
            )
        print(f"{species}: annotations already complete; use --overwrite", file=sys.stderr)
        return
    species_dir.mkdir(parents=True, exist_ok=True)
    print(f"{species}: loading BED annotations", file=sys.stderr, flush=True)
    intervals, interval_counts = load_beds(annotation_dir, elements)

    fasta_path = dataset / species / "genome.fasta"
    fasta = Fasta(str(fasta_path), as_raw=True)
    split_contigs = list(dict.fromkeys(
        line.split("\t", 1)[0] for line in
        (dataset / species / "splits.bed").read_text().splitlines() if line
    ))
    fasta_contigs = [str(chrom) for chrom in fasta.keys()]
    if args.all_contigs:
        contig_pairs = [(name, name) for name in fasta_contigs]
    else:
        contig_pairs = resolve_fasta_contigs(split_contigs, fasta_contigs, species)
    chromosomes, offset = [], 0
    for public_name, fasta_name in contig_pairs:
        reference_size = len(fasta[fasta_name])
        size = min(reference_size, args.limit_bp) if args.limit_bp else reference_size
        offset = align_up(offset, args.chunk_bp)
        chromosomes.append({"name": public_name, "fasta_name": fasta_name, "size": size,
                            "reference_size": reference_size, "offset": offset})
        offset += align_up(size, args.chunk_bp)
    fasta.close()

    if not chromosomes or offset == 0:
        raise RuntimeError(f"{species}: zero chromosomes selected; refusing to create an empty store")
    selected_names = {c["name"] for c in chromosomes}
    matched_interval_counts = {
        element: sum(len(ivs) for chrom, ivs in intervals[element].items()
                     if chrom in selected_names)
        for element in elements
    }
    if sum(matched_interval_counts.values()) == 0:
        raise RuntimeError(
            f"{species}: no BED contigs match selected split/FASTA contigs; refusing empty labels"
        )

    shape, chunks = (offset, len(elements)), (args.chunk_bp, len(elements))
    print(f"{species}: creating TensorStore shape={shape}, chunks={chunks}",
          file=sys.stderr, flush=True)
    store_path = species_dir / "annotations"
    store = create_store(store_path, shape, chunks)
    block_bp = args.chunk_bp * args.block_chunks
    positive_bases = np.zeros(len(elements), dtype=np.int64)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    prefetch = args.prefetch if args.prefetch is not None else max(1, 2 * args.workers)
    if prefetch < 1:
        raise ValueError("--prefetch must be >= 1")

    def block_tasks():
        for chrom_index, chrom_info in enumerate(chromosomes):
            chrom = chrom_info["name"]
            chrom_intervals = [intervals[e].get(chrom, []) for e in elements]
            cursors = [0] * len(elements)
            active = [[] for _ in elements]
            for start in range(0, chrom_info["size"], block_bp):
                end = min(start + block_bp, chrom_info["size"])
                overlapping = []
                for element_index, ivs in enumerate(chrom_intervals):
                    current = [iv for iv in active[element_index] if iv[1] > start]
                    cursor = cursors[element_index]
                    while cursor < len(ivs) and ivs[cursor][0] < end:
                        if ivs[cursor][1] > start:
                            current.append(ivs[cursor])
                        cursor += 1
                    cursors[element_index] = cursor
                    active[element_index] = current
                    overlapping.append(current)
                yield (chrom_index, chrom, start, end, overlapping, len(elements))

    executor = None
    try:
        print(f"{species}: starting block pipeline with {args.workers} worker(s), "
              f"prefetch={prefetch}", file=sys.stderr, flush=True)
        if args.workers == 1:
            results = map(render_annotation_block, block_tasks())
        else:
            executor = ProcessPoolExecutor(
                max_workers=args.workers, mp_context=mp.get_context("spawn")
            )
            results = bounded_ordered_map(
                executor, render_annotation_block, block_tasks(), prefetch
            )
        last_chrom_index = None
        for chrom_index, chrom, start, end, labels in results:
            chrom_info = chromosomes[chrom_index]
            destination = chrom_info["offset"] + start
            store[destination:destination + len(labels), :].write(labels).result()
            positive_bases += labels.sum(axis=0, dtype=np.int64)
            if last_chrom_index is not None and last_chrom_index != chrom_index:
                print(file=sys.stderr)
            last_chrom_index = chrom_index
            print(f"\r{species} annotations {chrom}: {end:,}/{chrom_info['size']:,}",
                  end="", file=sys.stderr, flush=True)
        if last_chrom_index is not None:
            print(file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    metadata = {
        "format_version": 1, "species": species,
        "coordinate_system": "0-based half-open",
        "dtype": "uint8", "compressor": {"id": "zstd", "level": 3},
        "chunk_shape": list(chunks), "shape": list(shape),
        "fasta_path": str(fasta_path.resolve()), "chromosomes": chromosomes,
        "elements": [{"index": i, "name": e, "interval_count": interval_counts[e],
                      "matched_interval_count": matched_interval_counts[e],
                      "positive_bases": int(positive_bases[i])}
                     for i, e in enumerate(elements)],
        "strand_encoding": "collapsed; a base is positive if either strand is annotated",
        "label_semantics": "independent binary labels; zero means annotated negative/background",
    }
    atomic_json(metadata_path, metadata)
    print(f"{species}: annotation TensorStore complete -> {species_dir}", file=sys.stderr)


def main():
    args = parse_args()
    if args.workers < 1 or (args.prefetch is not None and args.prefetch < 1):
        raise SystemExit("--workers and --prefetch must be >= 1")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    available = sorted(p.parent.parent.name for p in dataset.glob("*/genome_annotation/*.bed"))
    available = sorted(set(available))
    requested = args.species or available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise SystemExit(f"No genome annotations for: {', '.join(unknown)}")
    for species in requested:
        prepare_species(dataset, output, species, args)


if __name__ == "__main__":
    main()
