#!/usr/bin/env python3
"""Build a species/assay coverage catalog for the NTv3 BigWig tracks."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

try:
    import pyBigWig
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: install it with `python -m pip install pyBigWig`.") from exc


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="NTv3_benchmark_dataset directory")
    parser.add_argument("--output", type=Path, default=Path("bigwig_coverage_catalog"))
    parser.add_argument("--chunk-bp", type=int, default=10_000_000,
                        help="BigWig query chunk size (default: 10,000,000)")
    return parser.parse_args()


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["file_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def merged_covered_intervals(bw, chrom: str, chrom_size: int, chunk_bp: int):
    """Yield the union of explicitly stored BigWig intervals, in coordinate order."""
    pending_start = pending_end = None
    for chunk_start in range(0, chrom_size, chunk_bp):
        chunk_end = min(chunk_start + chunk_bp, chrom_size)
        intervals = bw.intervals(chrom, chunk_start, chunk_end) or ()
        for start, end, _value in intervals:
            start, end = max(start, chunk_start), min(end, chunk_end)
            if end <= start:
                continue
            if pending_start is None:
                pending_start, pending_end = start, end
            elif start <= pending_end:
                pending_end = max(pending_end, end)
            else:
                yield pending_start, pending_end
                pending_start, pending_end = start, end
    if pending_start is not None:
        yield pending_start, pending_end


def write_track_coverage(track, dataset: Path, tmp: Path, chunk_bp: int):
    path = dataset / track["relative_path"]
    tmp_path = tmp / f'{track["file_id"]}.bed'
    covered = 0
    first = {}
    last = {}
    with pyBigWig.open(str(path)) as bw, tmp_path.open("w") as out:
        chroms = bw.chroms()
        for chrom, size in chroms.items():
            for start, end in merged_covered_intervals(bw, chrom, size, chunk_bp):
                out.write(f"{chrom}\t{start}\t{end}\n")
                covered += end - start
                first.setdefault(chrom, start)
                last[chrom] = end
    return tmp_path, covered, first, last, chroms


def iter_bed(path: Path):
    with path.open() as handle:
        for line in handle:
            chrom, start, end = line.rstrip().split("\t")
            yield chrom, int(start), int(end)


def grouped_coverage(paths: list[Path]):
    """Yield chrom,start,end,number-of-tracks-covered using an external merge."""
    events = []
    for track_no, path in enumerate(paths):
        for chrom, start, end in iter_bed(path):
            events.append((chrom, start, 1, track_no))
            events.append((chrom, end, -1, track_no))
    events.sort(key=lambda x: (x[0], x[1]))
    active = set()
    previous_chrom = None
    previous_pos = None
    i = 0
    while i < len(events):
        chrom, pos = events[i][0], events[i][1]
        if chrom == previous_chrom and previous_pos is not None and pos > previous_pos and active:
            yield chrom, previous_pos, pos, len(active)
        if chrom != previous_chrom:
            active.clear()
        while i < len(events) and events[i][0] == chrom and events[i][1] == pos:
            _, _, delta, track_no = events[i]
            (active.add if delta > 0 else active.discard)(track_no)
            i += 1
        previous_chrom, previous_pos = chrom, pos


def main() -> None:
    args = arguments()
    dataset = args.dataset.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(dataset / "benchmark_metadata.tsv")
    tracks = []
    for path in sorted(dataset.glob("*/functional_tracks/*.bigwig")):
        file_id = path.stem
        if file_id not in metadata:
            print(f"WARNING: no metadata for {path}; skipped", file=sys.stderr)
            continue
        row = metadata[file_id]
        tracks.append({**row, "relative_path": str(path.relative_to(dataset))})

    groups = defaultdict(list)
    detail_rows = []
    chrom_rows = {}
    with tempfile.TemporaryDirectory(prefix="ntv3_bw_coverage_") as tmp_name:
        tmp = Path(tmp_name)
        for index, track in enumerate(tracks, 1):
            print(f"[{index}/{len(tracks)}] {track['relative_path']}", file=sys.stderr)
            bed, covered, first, last, chroms = write_track_coverage(track, dataset, tmp, args.chunk_bp)
            track["covered_bases"] = str(covered)
            groups[(track["species_common_name"], track["assay"])].append((track, bed))
            for chrom, size in chroms.items():
                chrom_rows[(track["species_common_name"], track["assay"], track["file_id"], chrom)] = (
                    size, first.get(chrom, ""), last.get(chrom, "")
                )

        with gzip.open(args.output / "coverage_segments.tsv.gz", "wt", newline="") as all_out, \
             gzip.open(args.output / "all_tracks_covered.bed.gz", "wt", newline="") as common_out:
            all_out.write("species\tassay\tchrom\tstart\tend\tcovered_track_count\ttrack_count\n")
            common_out.write("#species\tassay\tchrom\tstart\tend\n")
            for (species, assay), members in sorted(groups.items()):
                n_tracks = len(members)
                pending = None
                for chrom, start, end, count in grouped_coverage([bed for _, bed in members]):
                    current = (chrom, start, end, count)
                    if pending and pending[0] == chrom and pending[2] == start and pending[3] == count:
                        pending = (chrom, pending[1], end, count)
                        continue
                    if pending:
                        c, s, e, k = pending
                        all_out.write(f"{species}\t{assay}\t{c}\t{s}\t{e}\t{k}\t{n_tracks}\n")
                        if k == n_tracks:
                            common_out.write(f"{species}\t{assay}\t{c}\t{s}\t{e}\n")
                    pending = current
                if pending:
                    c, s, e, k = pending
                    all_out.write(f"{species}\t{assay}\t{c}\t{s}\t{e}\t{k}\t{n_tracks}\n")
                    if k == n_tracks:
                        common_out.write(f"{species}\t{assay}\t{c}\t{s}\t{e}\n")

    fields = list(tracks[0]) if tracks else []
    with (args.output / "tracks.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(tracks)
    with gzip.open(args.output / "track_chromosomes.tsv.gz", "wt", newline="") as handle:
        handle.write("species\tassay\tfile_id\tchrom\tchrom_size\tfirst_covered\tlast_covered\n")
        for key, values in sorted(chrom_rows.items()):
            handle.write("\t".join(map(str, (*key, *values))) + "\n")
    print(f"Catalog written to {args.output.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
