#!/usr/bin/env python3
"""Convert the bundled GENCODE GTF to AlphaGenome's Feather representation."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GTF = (
    ROOT
    / "reference/hg38/genes/gencode41/gencode41_basic_nort.gtf"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reference/alphagenome/gencode41_basic_nort.gtf.feather"
)
DEFAULT_SPLICE_STARTS_OUTPUT = (
    ROOT
    / "reference/alphagenome/gencode41_basic_nort.splice_sites_starts.feather"
)
DEFAULT_SPLICE_ENDS_OUTPUT = (
    ROOT
    / "reference/alphagenome/gencode41_basic_nort.splice_sites_ends.feather"
)
DEFAULT_PAS_SOURCE = (
    ROOT / "reference/hg38/genes/polyadb/polyadb_human_v3.csv.gz"
)
DEFAULT_PAS_OUTPUT = ROOT / "reference/alphagenome/polyadb_human_v3.feather"
REQUIRED_COLUMNS = {
    "Chromosome",
    "Start",
    "End",
    "Strand",
    "Feature",
    "gene_id",
    "gene_name",
    "gene_type",
    "transcript_id",
}


def repair_missing_gene_rows(frame):
    """Add gene metadata rows required by AlphaGenome for orphan transcripts.

    Some filtered GENCODE GTFs retain ``basic`` transcript annotations while
    dropping the parent ``gene`` row (whose row does not itself carry the
    ``basic`` tag).  AlphaGenome discovers genes from transcript TSS records,
    then looks up their metadata in the gene rows, so such input otherwise
    fails at scoring time with ``KeyError: ... not in index``.
    """
    gene_rows = frame.loc[
        frame["Feature"].eq("gene") & frame["gene_id"].notna()
    ]
    known_gene_ids = set(gene_rows["gene_id"])
    orphan_mask = frame["gene_id"].notna() & ~frame["gene_id"].isin(
        known_gene_ids
    )
    orphan_gene_ids = tuple(
        frame.loc[orphan_mask, "gene_id"].drop_duplicates().astype(str)
    )
    if not orphan_gene_ids:
        return frame, orphan_gene_ids

    gene_level_columns = {
        "Chromosome",
        "Source",
        "Feature",
        "Start",
        "End",
        "Score",
        "Strand",
        "Frame",
        "gene_id",
        "gene_type",
        "gene_name",
        "level",
        "hgnc_id",
        "havana_gene",
        "artif_dupl",
    }
    synthetic_rows = []
    for gene_id in orphan_gene_ids:
        annotations = frame.loc[frame["gene_id"].eq(gene_id)]
        for column in ("Chromosome", "Strand", "gene_name", "gene_type"):
            values = annotations[column].dropna().unique()
            if len(values) != 1:
                raise ValueError(
                    f"Cannot reconstruct gene row for {gene_id}: "
                    f"{column} has values {values.tolist()}"
                )

        gene_row = annotations.iloc[[0]].copy()
        gene_row.loc[:, "Feature"] = "gene"
        gene_row.loc[:, "Start"] = int(annotations["Start"].min())
        gene_row.loc[:, "End"] = int(annotations["End"].max())
        for column in frame.columns:
            if column not in gene_level_columns:
                gene_row.loc[:, column] = None
        synthetic_rows.append(gene_row)

    # Keep the original column dtypes wherever possible by deriving every
    # synthetic row from an existing annotation row.
    import pandas as pd

    repaired = pd.concat([frame, *synthetic_rows], ignore_index=True)
    repaired = repaired.sort_values(
        ["Chromosome", "Start", "End", "Feature"], kind="stable"
    ).reset_index(drop=True)
    return repaired, orphan_gene_ids


def extract_splice_sites(frame):
    """Build AlphaGenome's intron-boundary tables from transcript exons."""
    exons = frame.loc[
        frame["Feature"].eq("exon") & frame["transcript_id"].notna(),
        ["Chromosome", "Start", "End", "Strand", "transcript_id"],
    ].drop_duplicates()
    exons = exons.sort_values(
        ["Chromosome", "transcript_id", "Start", "End"], kind="stable"
    )
    previous_end = exons.groupby(
        ["Chromosome", "transcript_id"], sort=False
    )["End"].shift()
    introns = exons.assign(intron_start=previous_end)
    introns = introns.loc[
        introns["intron_start"].notna()
        & introns["Start"].gt(introns["intron_start"])
    ].copy()
    introns["intron_start"] = introns["intron_start"].astype("int64")

    # AlphaGenome expects the first intronic base in Start and the first base
    # of the following exon in End, both in 0-based coordinates.
    starts = (
        introns.loc[:, ["Chromosome", "intron_start", "Strand"]]
        .rename(columns={"intron_start": "Start"})
        .drop_duplicates()
        .sort_values(["Chromosome", "Start", "Strand"], kind="stable")
        .reset_index(drop=True)
    )
    ends = (
        introns.loc[:, ["Chromosome", "Start", "Strand"]]
        .rename(columns={"Start": "End"})
        .drop_duplicates()
        .sort_values(["Chromosome", "End", "Strand"], kind="stable")
        .reset_index(drop=True)
    )
    return starts, ends


def convert_pas(source: Path, output: Path) -> None:
    """Convert Borzoi's PolyADB table to AlphaGenome's PAS annotation."""
    import pandas as pd

    if not source.is_file():
        raise SystemExit(f"PolyADB 不存在: {source}")
    frame = pd.read_csv(source, sep="\t")
    required = {"chrom", "position_hg38", "strand", "ensemble_id"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"PolyADB 缺少 AlphaGenome 所需列: {sorted(missing)}")
    frame = frame.rename(
        columns={
            "chrom": "Chromosome",
            "position_hg38": "Start",
            "strand": "pas_strand",
            "ensemble_id": "gene_id",
        }
    )
    frame["Start"] = pd.to_numeric(frame.Start, errors="raise").astype("int64")
    frame["End"] = frame.Start + 1
    frame["gene_id"] = frame.gene_id.astype(str)
    frame["gene_id_nopatch"] = frame.gene_id.str.split(".").str[0]
    frame = frame.sort_values(
        ["Chromosome", "Start", "gene_id"], kind="stable"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(output)
    print(
        f"已写入 {output}（{len(frame):,} PAS, "
        f"{output.stat().st_size / 2**20:.1f} MiB）"
    )


def convert_gtf(
    source: Path,
    output: Path,
    splice_starts_output: Path,
    splice_ends_output: Path,
) -> None:
    try:
        import pyranges as pr
    except ImportError as error:
        raise SystemExit(
            "缺少 pyranges；请使用 AlphaGenome 环境运行本脚本。"
        ) from error

    if not source.is_file():
        raise SystemExit(f"GTF 不存在: {source}")

    print(f"读取 GTF: {source}", flush=True)
    frame = pr.read_gtf(str(source), full=True, as_df=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise SystemExit(f"GTF 缺少 AlphaGenome 所需列: {sorted(missing)}")

    frame, repaired_gene_ids = repair_missing_gene_rows(frame)
    if repaired_gene_ids:
        print(
            "已补全缺失的 gene 元数据行："
            f"{len(repaired_gene_ids)} genes（例如 {repaired_gene_ids[0]}）",
            flush=True,
        )

    # PyRanges converts 1-based inclusive GTF coordinates to the 0-based,
    # half-open convention expected by AlphaGenome's genome.Interval.
    frame = frame.sort_values(
        ["Chromosome", "Start", "End", "Feature"], kind="stable"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(output)
    splice_starts, splice_ends = extract_splice_sites(frame)
    splice_starts_output.parent.mkdir(parents=True, exist_ok=True)
    splice_ends_output.parent.mkdir(parents=True, exist_ok=True)
    splice_starts.to_feather(splice_starts_output)
    splice_ends.to_feather(splice_ends_output)
    print(
        f"已写入 {output}（{len(frame):,} rows, "
        f"{output.stat().st_size / 2**20:.1f} MiB）"
    )
    print(
        "已写入 splice-site 注释："
        f"starts={splice_starts_output} ({len(splice_starts):,} rows), "
        f"ends={splice_ends_output} ({len(splice_ends):,} rows)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, default=DEFAULT_GTF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--splice-starts-output",
        type=Path,
        default=DEFAULT_SPLICE_STARTS_OUTPUT,
    )
    parser.add_argument(
        "--splice-ends-output",
        type=Path,
        default=DEFAULT_SPLICE_ENDS_OUTPUT,
    )
    parser.add_argument("--pas-source", type=Path, default=DEFAULT_PAS_SOURCE)
    parser.add_argument("--pas-output", type=Path, default=DEFAULT_PAS_OUTPUT)
    args = parser.parse_args()
    convert_gtf(
        args.gtf.resolve(),
        args.output.resolve(),
        args.splice_starts_output.resolve(),
        args.splice_ends_output.resolve(),
    )
    convert_pas(args.pas_source.resolve(), args.pas_output.resolve())


if __name__ == "__main__":
    main()
