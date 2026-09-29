#!/usr/bin/env python3
"""Reproduce VEP-eval ClinVar functional groups without modifying VEP-eval.

The classification rules are transcribed from
VEP_ClinVar_Benchmarking_RefSeq.ipynb.  The interval implementation is indexed
instead of scanning the full MANE DataFrame once per variant, but preserves the
notebook's feature and group semantics (including its promoter non-match).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


BASIC_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y"}
BASES = {"A", "C", "G", "T"}
AMINO_ACIDS = set("GQNIVREACWDFKPMLYTHS")
AA3_TO_AA1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*",
}
CLINICAL_SIGNIFICANCE_TO_LABEL = {
    "Benign": 0,
    "Likely benign": 0.1,
    "Benign/Likely benign": 0.1,
    "Pathogenic": 1,
    "Likely pathogenic": 1.1,
    "Pathogenic/Likely pathogenic": 1.1,
    "Uncertain significance": 2,
}
REVIEW_STATUS_TO_GOLD_STARS = {
    "criteria provided, single submitter": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": np.nan,
    "reviewed by expert panel": 3,
    "no assertion provided": np.nan,
    "no interpretation for the single variant": np.nan,
    "practice guideline": 4,
}
INPUT_COLUMNS = [
    "Name", "Assembly", "Chromosome", "VariationID", "PositionVCF",
    "ReferenceAlleleVCF", "AlternateAlleleVCF", "ClinicalSignificance",
    "ReviewStatus", "OriginSimple", "Type",
]

FINAL_SUBGROUPS = [
    "group: coding", "group: noncoding",
    "group: start loss", "group: start loss + 3'UTR", "group: start loss + 5'UTR",
    "group: start loss + intron (non-splice)",
    "group: missense", "group: missense + 3'UTR", "group: missense + 5'UTR",
    "group: missense + intron (non-splice)",
    "group: synonymous", "group: synonymous + 3'UTR", "group: synonymous + 5'UTR",
    "group: synonymous + intron (non-splice)",
    "group: stop gain", "group: stop gain + 3'UTR", "group: stop gain + 5'UTR",
    "group: stop gain + intron (non-splice)",
    "group: stop loss", "group: stop loss + 3'UTR", "group: stop loss + 5'UTR",
    "group: stop synonymous", "group: stop synonymous + 3'UTR",
    "group: 5'UTR", "group: 5'UTR + 3'UTR", "group: 5'UTR + intron (non-splice)",
    "group: 5'UTR + splice", "group: 3'UTR", "group: 3'UTR + intron (non-splice)",
    "group: 3'UTR + splice", "group: 3'UTR + RNA gene", "group: splice",
    "group: intron (non-splice)", "group: intron (non-splice) + RNA gene",
    "group: RNA gene",
]
RULE_BASED_GROUPS = [
    "group: start loss", "group: missense", "group: missense + 3'UTR",
    "group: missense + intron (non-splice)", "group: synonymous",
    "group: stop gain", "group: stop loss", "group: 5'UTR", "group: 3'UTR",
    "group: 3'UTR + RNA gene", "group: splice", "group: intron (non-splice)",
    "group: RNA gene",
]
PRIMARY_TASKS = {
    "coding", "noncoding", "intron (non-splice)", "splice", "5'UTR", "3'UTR",
    "RNA gene", "synonymous", "start loss", "stop loss", "stop gain", "missense",
    "missense + intron (non-splice)", "missense + 3'UTR",
}


@dataclass
class Transcript:
    chrom: str
    start: int
    end: int
    strand: str
    transcript_id: str
    record_id: str
    exons: list[tuple[int, int]] = field(default_factory=list)
    cds: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class RNARecord:
    chrom: str
    start: int
    end: int
    feature: str
    transcript_id: str
    record_id: str
    exons: list[tuple[int, int]] = field(default_factory=list)


def parse_attributes(raw: str) -> dict[str, str]:
    result = {}
    for item in raw.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def load_mane(path: Path):
    transcripts: dict[str, Transcript] = {}
    rna_records: dict[str, RNARecord] = {}
    child_exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    child_cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    gene_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    all_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    feature_counts = Counter()

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chrom, _source, feature_name, start, end, _score, strand, _frame, raw_attrs = line.rstrip("\n").split("\t")
            start_i, end_i = int(start), int(end)
            attrs = parse_attributes(raw_attrs)
            feature_counts[feature_name] += 1
            all_intervals[chrom].append((start_i, end_i))
            if feature_name == "gene":
                gene_intervals[chrom].append((start_i, end_i))
            elif feature_name == "mRNA":
                record_id = attrs.get("ID", "")
                transcript_id = attrs.get("transcript_id", "")
                transcripts[record_id] = Transcript(
                    chrom, start_i, end_i, strand, transcript_id, record_id
                )
            elif feature_name in {"snRNA", "snoRNA"}:
                record_id = attrs.get("ID", "")
                transcript_id = attrs.get("transcript_id", "")
                rna_records[record_id] = RNARecord(
                    chrom, start_i, end_i, feature_name, transcript_id, record_id
                )
            elif feature_name == "exon":
                child_exons[attrs.get("Parent", "")].append((start_i, end_i))
            elif feature_name == "CDS":
                child_cds[attrs.get("Parent", "")].append((start_i, end_i))

    for record_id, transcript in transcripts.items():
        transcript.exons = sorted(child_exons.get(record_id, []))
        transcript.cds = sorted(child_cds.get(record_id, []))
    for record_id, record in rna_records.items():
        # This matches the notebook's Parent == f"rna-{transcript_id}" test.
        expected_parent = f"rna-{record.transcript_id}"
        record.exons = sorted(child_exons.get(expected_parent, []))

    return {
        "transcripts": list(transcripts.values()),
        "rna_records": list(rna_records.values()),
        "gene_intervals": {chrom: merge_intervals(v) for chrom, v in gene_intervals.items()},
        "all_intervals": {chrom: merge_intervals(v) for chrom, v in all_intervals.items()},
        "feature_counts": feature_counts,
    }


def load_fai(path: Path) -> dict[str, tuple[int, int, int, int]]:
    result = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            name, length, offset, line_bases, line_width, *_ = line.rstrip("\n").split("\t")
            result[name] = tuple(map(int, (length, offset, line_bases, line_width)))
    return result


def build_fai(fasta: Path, fai_path: Path):
    """Create the five-column FASTA index needed for random base lookup."""
    records = []
    with fasta.open("rb") as handle:
        name = None
        length = offset = line_bases = line_width = 0
        while True:
            line = handle.readline()
            if not line:
                break
            if line.startswith(b">"):
                if name is not None:
                    records.append((name, length, offset, line_bases, line_width))
                name = line[1:].split()[0].decode("ascii")
                length = 0
                offset = handle.tell()
                line_bases = line_width = 0
            else:
                sequence = line.rstrip(b"\r\n")
                if line_bases == 0:
                    line_bases = len(sequence)
                    line_width = len(line)
                length += len(sequence)
        if name is not None:
            records.append((name, length, offset, line_bases, line_width))
    fai_path.write_text(
        "".join("\t".join(map(str, record)) + "\n" for record in records),
        encoding="utf-8",
    )


def validate_reference(df: pd.DataFrame, fasta: Path, fai_path: Path):
    fai = load_fai(fai_path)
    mapped = np.memmap(fasta, mode="r", dtype=np.uint8)
    status = np.full(len(df), "LENGTH_ERROR", dtype=object)
    flank = 65536 // 2
    for chrom, indices in df.groupby("#CHROM", sort=False).indices.items():
        fasta_chrom = str(chrom)
        if not fasta_chrom.startswith("chr"):
            fasta_chrom = "chr" + fasta_chrom
        if fasta_chrom not in fai:
            continue
        length, offset, line_bases, line_width = fai[fasta_chrom]
        idx = np.asarray(indices)
        pos = df.iloc[idx]["POS"].to_numpy(dtype=np.int64)
        boundary_ok = (pos - flank - 1 >= 0) & (pos + flank <= length)
        valid_idx = idx[boundary_ok]
        valid_pos = pos[boundary_ok]
        zero_based = valid_pos - 1
        byte_offsets = offset + (zero_based // line_bases) * line_width + (zero_based % line_bases)
        reference_bases = np.asarray(mapped[byte_offsets], dtype=np.uint8)
        # The UCSC FASTA is soft-masked; the notebook calls .seq.upper().
        is_lower = (reference_bases >= ord("a")) & (reference_bases <= ord("z"))
        reference_bases = reference_bases.copy()
        reference_bases[is_lower] -= ord("a") - ord("A")
        expected_bases = np.fromiter(
            (ord(x) for x in df.iloc[valid_idx]["REF"].astype(str)),
            dtype=np.uint8,
            count=len(valid_idx),
        )
        status[valid_idx] = np.where(reference_bases == expected_bases, "VALID", "MISMATCH_ERROR")
    del mapped
    return status


def load_clinvar_candidates(path: Path, chunksize: int):
    funnel = Counter()
    pieces = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=INPUT_COLUMNS,
        dtype={"Chromosome": "string"},
        compression="gzip",
        chunksize=chunksize,
        low_memory=False,
    ):
        funnel["raw"] += len(chunk)
        frame = chunk.loc[chunk["Assembly"].eq("GRCh38")].copy()
        funnel["GRCh38"] += len(frame)
        frame = frame.loc[frame["OriginSimple"].ne("somatic")]
        funnel["exclude exact somatic"] += len(frame)
        frame = frame.loc[frame["Chromosome"].isin(BASIC_CHROMOSOMES)]
        funnel["basic chromosomes"] += len(frame)
        frame = frame.loc[frame["Type"].eq("single nucleotide variant")].copy()
        funnel["SNV"] += len(frame)
        frame["ClinVar_label"] = frame["ClinicalSignificance"].map(
            CLINICAL_SIGNIFICANCE_TO_LABEL
        )
        frame = frame.loc[frame["ClinVar_label"].isin({0, 0.1, 1, 1.1})].copy()
        funnel["selected clinical significance"] += len(frame)
        frame["ClinVar_gold_stars"] = frame["ReviewStatus"].map(
            REVIEW_STATUS_TO_GOLD_STARS
        )
        frame = frame.loc[frame["ClinVar_gold_stars"].isin({1, 2, 3, 4})].copy()
        funnel["gold stars >= 1"] += len(frame)
        frame = frame.loc[
            frame["ReferenceAlleleVCF"].isin(BASES)
            & frame["AlternateAlleleVCF"].isin(BASES)
        ].copy()
        funnel["canonical A/C/G/T alleles"] += len(frame)
        pieces.append(frame)

    selected = pd.concat(pieces, ignore_index=True)
    selected = selected.rename(
        columns={
            "Chromosome": "#CHROM",
            "PositionVCF": "POS",
            "VariationID": "ID",
            "ReferenceAlleleVCF": "REF",
            "AlternateAlleleVCF": "ALT",
        }
    )
    selected["POS"] = pd.to_numeric(selected["POS"], errors="raise").astype(np.int64)
    return selected, funnel


def parse_protein_change(raw):
    if pd.isna(raw):
        return False, pd.NA, pd.NA, None, None, None
    try:
        last_digit = list(re.finditer(r"\d", raw))[-1].start()
        raw_ref = raw[: last_digit + 1]
        raw_alt = raw[last_digit + 1 :]
        parts = raw_ref.split("_")
        assert 1 <= len(parts) <= 2
        parsed = [(AA3_TO_AA1[p[:3]], int(p[3:])) for p in parts]
        start_aa, start_pos = parsed[0]
        end_aa, end_pos = parsed[-1]
        if raw_alt.endswith("fs"):
            alt = "fs"
        elif raw_alt == "=":
            alt = "="
        elif raw_alt == "dup":
            alt = "dup"
        else:
            is_insert = "ins" in raw_alt and "del" not in raw_alt
            amino_text = raw_alt.replace("del", "").replace("ins", "")
            assert len(amino_text) % 3 == 0
            alt = "".join(AA3_TO_AA1[amino_text[i : i + 3]] for i in range(0, len(amino_text), 3))
            if is_insert:
                assert end_pos == start_pos + 1
                alt = start_aa + alt + end_aa
        return False, start_pos, end_pos, start_aa, end_aa, alt
    except Exception:
        return True, pd.NA, pd.NA, None, None, None


def add_hgvs_annotations(df: pd.DataFrame):
    names = df["Name"].fillna("").astype(str)
    raw_protein = names.str.extract(r"\(p\.(.+)\)", expand=False)
    parsed = [parse_protein_change(value) for value in raw_protein]
    df["ClinVarName_failed_parsing_protein_change"] = np.fromiter(
        (x[0] for x in parsed), dtype=bool, count=len(parsed)
    )
    df["ClinVarName_AAPOS"] = pd.array([x[1] for x in parsed], dtype="Int64")
    df["ClinVarName_AAREF"] = [x[3] for x in parsed]
    df["ClinVarName_AAALT"] = [x[5] for x in parsed]
    df["ClinVarName_coding_sequence"] = (
        df[["ClinVarName_AAPOS", "ClinVarName_AAREF", "ClinVarName_AAALT"]]
        .notna()
        .all(axis=1)
        .astype(np.uint8)
    )
    df["ClinVarName_splice"] = names.str.contains(
        r"\d+(?:\+1|\+2|\-1|\-2)[ATCG]", regex=True, na=False
    ).astype(np.uint8)
    df["ClinVarName_RNA_gene"] = names.str.contains("NR_", regex=False, na=False).astype(np.uint8)
    df["ClinVarName_refseq_ids"] = names.str.extract(r"(NM_\d+\.?\d*)", expand=False)


def mark_merged_intervals(target, global_indices, positions, intervals):
    for start, end in intervals:
        left = np.searchsorted(positions, start, side="left")
        right = np.searchsorted(positions, end, side="right")
        if left < right:
            target[global_indices[left:right]] = 1


def annotate_mane(df: pd.DataFrame, mane):
    columns = [
        "gene", "mRNA", "mRNA_exon", "coding_sequence", "start_codon", "stop_codon",
        "five_prime_UTR", "three_prime_UTR", "mRNA_intron", "mRNA_splice",
        "snRNA", "snRNA_exon", "snoRNA", "snoRNA_exon", "other",
    ]
    arrays = {name: np.zeros(len(df), dtype=np.uint8) for name in columns}
    annotated = np.zeros(len(df), dtype=np.uint8)
    transcript_count = np.zeros(len(df), dtype=np.uint16)
    groups = {}
    for chrom, indices in df.groupby("#CHROM", sort=False).indices.items():
        idx = np.asarray(indices)
        groups[f"chr{chrom}"] = (idx, df.iloc[idx]["POS"].to_numpy(dtype=np.int64))

    for chrom, intervals in mane["all_intervals"].items():
        if chrom in groups:
            idx, positions = groups[chrom]
            mark_merged_intervals(annotated, idx, positions, intervals)
    for chrom, intervals in mane["gene_intervals"].items():
        if chrom in groups:
            idx, positions = groups[chrom]
            mark_merged_intervals(arrays["gene"], idx, positions, intervals)

    for transcript in mane["transcripts"]:
        if transcript.chrom not in groups:
            continue
        idx, positions = groups[transcript.chrom]
        left = np.searchsorted(positions, transcript.start, side="left")
        right = np.searchsorted(positions, transcript.end, side="right")
        if left >= right:
            continue
        hit_idx = idx[left:right]
        hit_pos = positions[left:right]
        arrays["mRNA"][hit_idx] = 1
        transcript_count[hit_idx] += 1
        in_exon = np.zeros(len(hit_idx), dtype=bool)
        in_cds = np.zeros(len(hit_idx), dtype=bool)
        for start, end in transcript.exons:
            a = np.searchsorted(hit_pos, start, side="left")
            b = np.searchsorted(hit_pos, end, side="right")
            in_exon[a:b] = True
        for start, end in transcript.cds:
            a = np.searchsorted(hit_pos, start, side="left")
            b = np.searchsorted(hit_pos, end, side="right")
            in_cds[a:b] = True

        coding = in_exon & in_cds
        utr = in_exon & ~in_cds
        intron = ~in_exon & ~in_cds
        arrays["mRNA_exon"][hit_idx[in_exon]] = 1
        arrays["coding_sequence"][hit_idx[coding]] = 1
        arrays["mRNA_intron"][hit_idx[intron]] = 1

        if transcript.cds and transcript.exons:
            min_cds_start = min(x[0] for x in transcript.cds)
            max_cds_end = max(x[1] for x in transcript.cds)
            min_exon_start = min(x[0] for x in transcript.exons)
            max_exon_end = max(x[1] for x in transcript.exons)
            if transcript.strand == "+":
                start_mask = coding & (hit_pos >= min_cds_start) & (hit_pos <= min_cds_start + 2)
                stop_mask = coding & (hit_pos >= max_cds_end - 2) & (hit_pos <= max_cds_end)
                five_mask = utr & (hit_pos >= min_exon_start) & (hit_pos <= min_cds_start - 1)
                three_mask = utr & (hit_pos >= max_cds_end + 1) & (hit_pos <= max_exon_end)
            else:
                start_mask = coding & (hit_pos >= max_cds_end - 2) & (hit_pos <= max_cds_end)
                stop_mask = coding & (hit_pos >= min_cds_start) & (hit_pos <= min_cds_start + 2)
                five_mask = utr & (hit_pos >= max_cds_end + 1) & (hit_pos <= max_exon_end)
                three_mask = utr & (hit_pos >= min_exon_start) & (hit_pos <= min_cds_start - 1)
            arrays["start_codon"][hit_idx[start_mask]] = 1
            arrays["stop_codon"][hit_idx[stop_mask]] = 1
            arrays["five_prime_UTR"][hit_idx[five_mask]] = 1
            arrays["three_prime_UTR"][hit_idx[three_mask]] = 1

        splice_positions = set()
        for start, end in transcript.exons:
            splice_positions.update((start - 1, start - 2, end + 1, end + 2))
        splice_mask = intron & np.isin(hit_pos, list(splice_positions))
        arrays["mRNA_splice"][hit_idx[splice_mask]] = 1

    for record in mane["rna_records"]:
        if record.chrom not in groups:
            continue
        idx, positions = groups[record.chrom]
        left = np.searchsorted(positions, record.start, side="left")
        right = np.searchsorted(positions, record.end, side="right")
        if left >= right:
            continue
        hit_idx = idx[left:right]
        hit_pos = positions[left:right]
        arrays[record.feature][hit_idx] = 1
        exon_hit = np.zeros(len(hit_idx), dtype=bool)
        for start, end in record.exons:
            a = np.searchsorted(hit_pos, start, side="left")
            b = np.searchsorted(hit_pos, end, side="right")
            exon_hit[a:b] = True
        arrays[f"{record.feature}_exon"][hit_idx[exon_hit]] = 1

    arrays["other"] = (annotated == 0).astype(np.uint8)
    rename = {
        "mRNA": "MANE_mRNA", "mRNA_exon": "MANE_mRNA_exon",
        "coding_sequence": "MANE_coding_sequence", "start_codon": "MANE_start_codon",
        "stop_codon": "MANE_stop_codon", "five_prime_UTR": "MANE_five_prime_UTR",
        "three_prime_UTR": "MANE_three_prime_UTR", "mRNA_intron": "MANE_mRNA_intron",
        "mRNA_splice": "MANE_mRNA_splice", "snRNA": "MANE_snRNA",
        "snRNA_exon": "MANE_snRNA_exon", "snoRNA": "MANE_snoRNA",
        "snoRNA_exon": "MANE_snoRNA_exon", "other": "MANE_other",
    }
    for name, values in arrays.items():
        df[rename.get(name, name)] = values
    df["MANE_transcript_count"] = transcript_count


def add_final_groups(df: pd.DataFrame):
    for col in FINAL_SUBGROUPS:
        df[col] = np.uint8(0)

    df["ClinVarName_missense"] = (
        df["ClinVarName_AAREF"].isin(AMINO_ACIDS)
        & df["ClinVarName_AAALT"].isin(AMINO_ACIDS)
    )
    df["ClinVarName_synonymous"] = df["ClinVarName_AAALT"].eq("=")
    df["ClinVarName_stop_gain"] = df["ClinVarName_AAALT"].eq("*")
    df["Union_stop_loss"] = df["ClinVarName_AAREF"].eq("*") | df["MANE_stop_codon"].eq(1)
    df["Union_splice"] = df["MANE_mRNA_splice"].eq(1) | df["ClinVarName_splice"].eq(1)
    df["MANE_intron"] = df["MANE_mRNA_intron"].eq(1)
    df["Union_RNA_gene"] = (
        df["ClinVarName_RNA_gene"].eq(1)
        | df["MANE_snRNA"].eq(1)
        | df["MANE_snoRNA"].eq(1)
    )

    start = df["MANE_start_codon"].eq(1)
    missense = df["ClinVarName_missense"]
    synonymous = df["ClinVarName_synonymous"]
    stop_gain = df["ClinVarName_stop_gain"]
    stop_loss = df["Union_stop_loss"]
    five = df["MANE_five_prime_UTR"].eq(1)
    three = df["MANE_three_prime_UTR"].eq(1)
    splice = df["Union_splice"]
    intron = df["MANE_intron"]
    rna = df["Union_RNA_gene"]

    df["group: coding"] = (
        df["ClinVarName_coding_sequence"].eq(1) | df["MANE_coding_sequence"].eq(1)
    ).astype(np.uint8)
    df["group: noncoding"] = (~df["group: coding"].astype(bool)).astype(np.uint8)

    def assign(name, condition):
        df[name] = condition.astype(np.uint8)

    assign("group: start loss", start & ~five & ~three & ~intron)
    assign("group: start loss + 3'UTR", start & ~five & three & ~intron)
    assign("group: start loss + 5'UTR", start & five & ~three & ~intron)
    assign("group: start loss + intron (non-splice)", start & ~five & ~three & intron)
    assign("group: missense", ~start & missense & ~five & ~three & ~intron)
    assign("group: missense + 3'UTR", ~start & missense & ~five & three & ~intron)
    assign("group: missense + 5'UTR", ~start & missense & five & ~three & ~intron)
    assign("group: missense + intron (non-splice)", ~start & missense & ~five & ~three & intron)
    assign("group: synonymous", synonymous & ~five & ~three & ~intron & ~stop_loss)
    assign("group: synonymous + 3'UTR", synonymous & ~five & three & ~intron & ~stop_loss)
    assign("group: synonymous + 5'UTR", synonymous & five & ~three & ~intron & ~stop_loss)
    assign("group: synonymous + intron (non-splice)", synonymous & ~five & ~three & intron & ~stop_loss)
    assign("group: stop synonymous", synonymous & ~five & ~three & ~intron & stop_loss)
    assign("group: stop synonymous + 3'UTR", synonymous & ~five & three & ~intron & stop_loss)
    assign("group: stop gain", stop_gain & ~five & ~three & ~intron)
    assign("group: stop gain + 3'UTR", stop_gain & ~five & three & ~intron)
    assign("group: stop gain + 5'UTR", stop_gain & five & ~three & ~intron)
    assign("group: stop gain + intron (non-splice)", stop_gain & ~five & ~three & intron)
    assign("group: stop loss", ~synonymous & stop_loss & ~five & ~three & ~intron)
    assign("group: stop loss + 3'UTR", ~synonymous & stop_loss & ~five & three & ~intron)
    assign("group: stop loss + 5'UTR", ~synonymous & stop_loss & five & ~three & ~intron)

    base = ~start & ~missense & ~synonymous & ~stop_gain & ~stop_loss
    assign("group: 5'UTR", base & five & ~three & ~splice & ~intron)
    assign("group: 5'UTR + 3'UTR", base & five & three & ~splice & ~intron)
    assign("group: 5'UTR + intron (non-splice)", base & five & ~three & ~splice & intron)
    assign("group: 5'UTR + splice", base & five & ~three & splice & intron)
    assign("group: 3'UTR", base & ~five & three & ~splice & ~intron & ~rna)
    assign("group: 3'UTR + intron (non-splice)", base & ~five & three & ~splice & intron & ~rna)
    assign("group: 3'UTR + splice", base & ~five & three & splice & intron & ~rna)
    assign("group: 3'UTR + RNA gene", base & ~five & three & ~splice & ~intron & rna)
    assign("group: splice", base & ~five & ~three & splice & intron & ~rna)
    assign("group: intron (non-splice)", base & ~five & ~three & ~splice & intron & ~rna)
    assign("group: intron (non-splice) + RNA gene", base & ~five & ~three & ~splice & intron & rna)
    assign("group: RNA gene", base & ~five & ~three & ~splice & ~intron & rna)


def load_strand_maps(path: Path):
    exact, base = {}, {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            refseq_id, strand = row[1], row[3]
            exact.setdefault(refseq_id, strand)
            base.setdefault(refseq_id.split(".")[0], strand)
    return exact, base


def add_strand_groups(df: pd.DataFrame, strand_maps):
    exact, base = strand_maps

    def lookup(value):
        if pd.isna(value):
            return np.nan
        value = str(value).strip()
        return exact.get(value, base.get(value.split(".")[0], np.nan))

    df["Strand"] = df["ClinVarName_refseq_ids"].map(lookup)
    df["group: +"] = df["Strand"].eq("+").astype(np.uint8)
    df["group: -"] = df["Strand"].eq("-").astype(np.uint8)


def add_rule_based_baseline(frame: pd.DataFrame):
    category = pd.Series(pd.NA, index=frame.index, dtype="string")
    for group in RULE_BASED_GROUPS:
        mask = frame[group].eq(1)
        if (category.notna() & mask).any():
            raise ValueError("variant-type groups are not mutually exclusive")
        category.loc[mask] = group.removeprefix("group: ")
    category.loc[category.isna() & frame["group: coding"].eq(1)] = "other coding"
    category.loc[category.isna() & frame["group: noncoding"].eq(1)] = "other noncoding"
    if category.isna().any():
        raise ValueError("some variants do not have a rule-based category")
    prevalence = frame["ClinVar_label"].groupby(category).mean()
    frame["Rule_based_category"] = category
    frame["Rule_based"] = category.map(prevalence).astype(float)


def group_count_table(df: pd.DataFrame, release: str):
    rows = []
    for column in FINAL_SUBGROUPS + ["group: +", "group: -"]:
        subset = df.loc[df[column].eq(1)]
        counts = subset["ClinVar_label"].value_counts()
        benign = int(counts.get(0, 0))
        pathogenic = int(counts.get(1, 0))
        rows.append(
            {
                "release": release,
                "group": column.removeprefix("group: "),
                "is_primary_task": column.removeprefix("group: ") in PRIMARY_TASKS,
                "Benign": benign,
                "Likely_Benign": int(counts.get(0.1, 0)),
                "Pathogenic": pathogenic,
                "Likely_Pathogenic": int(counts.get(1.1, 0)),
                "exact_total": benign + pathogenic,
                "all_label_total": len(subset),
                "pathogenic_fraction_exact": pathogenic / (benign + pathogenic)
                if benign + pathogenic
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def feature_count_table(df: pd.DataFrame, release: str):
    fields = [
        "gene", "MANE_mRNA", "MANE_mRNA_exon", "MANE_coding_sequence",
        "MANE_start_codon", "MANE_stop_codon", "MANE_five_prime_UTR",
        "MANE_three_prime_UTR", "MANE_mRNA_intron", "MANE_mRNA_splice",
        "MANE_snRNA", "MANE_snRNA_exon", "MANE_snoRNA", "MANE_snoRNA_exon",
        "MANE_other", "ClinVarName_coding_sequence", "ClinVarName_splice",
        "ClinVarName_RNA_gene", "ClinVarName_missense", "ClinVarName_synonymous",
        "ClinVarName_stop_gain", "Union_stop_loss", "Union_splice", "Union_RNA_gene",
    ]
    return pd.DataFrame(
        {
            "release": release,
            "feature": fields,
            "count": [int(df[field].sum()) for field in fields],
        }
    )


def process_release(
    release, clinvar_path, mane, fasta, fai, strand_maps, output_root, chunksize
):
    print(f"\n[{release}] loading and filtering {clinvar_path}", flush=True)
    df, funnel = load_clinvar_candidates(clinvar_path, chunksize)
    print(f"[{release}] canonical candidates: {len(df):,}", flush=True)

    status = validate_reference(df, fasta, fai)
    status_counts = Counter(status)
    for key, value in status_counts.items():
        funnel[f"reference validation: {key}"] = value
    df = df.loc[status == "VALID"].copy()
    df = df.sort_values(["#CHROM", "POS", "REF", "ALT"]).reset_index(drop=True)
    duplicate_rows = int(df.duplicated(["#CHROM", "POS", "REF", "ALT"], keep=False).sum())
    funnel["duplicate variant-key rows"] = duplicate_rows
    print(
        f"[{release}] reference valid: {len(df):,}; mismatch: {status_counts['MISMATCH_ERROR']:,}; "
        f"boundary: {status_counts['LENGTH_ERROR']:,}",
        flush=True,
    )

    print(f"[{release}] parsing HGVS and annotating MANE intervals", flush=True)
    add_hgvs_annotations(df)
    annotate_mane(df, mane)
    add_final_groups(df)
    add_strand_groups(df, strand_maps)

    release_dir = output_root / release
    release_dir.mkdir(parents=True, exist_ok=True)
    group_counts = group_count_table(df, release)
    feature_counts = feature_count_table(df, release)
    pd.DataFrame(
        [{"release": release, "step": step, "count": int(count)} for step, count in funnel.items()]
    ).to_csv(release_dir / "filter_funnel.csv", index=False)
    group_counts.to_csv(release_dir / "group_counts.csv", index=False)
    feature_counts.to_csv(release_dir / "feature_counts.csv", index=False)

    exact_pb = df.loc[df["ClinVar_label"].isin([0, 1])].copy()
    add_rule_based_baseline(exact_pb)
    output_columns = [
        "#CHROM", "POS", "ID", "REF", "ALT", "ClinVar_label", "ClinVar_gold_stars",
        "ClinVarName_refseq_ids", "ClinVarName_AAPOS", "ClinVarName_AAREF",
        "ClinVarName_AAALT", "ClinVarName_coding_sequence", "ClinVarName_splice",
        "ClinVarName_RNA_gene", "ClinVarName_missense", "ClinVarName_synonymous",
        "ClinVarName_stop_gain", "Union_stop_loss", "Union_splice", "Union_RNA_gene",
        "gene", "MANE_mRNA", "MANE_mRNA_exon", "MANE_coding_sequence",
        "MANE_start_codon", "MANE_stop_codon", "MANE_five_prime_UTR",
        "MANE_three_prime_UTR", "MANE_mRNA_intron", "MANE_mRNA_splice", "MANE_snRNA",
        "MANE_snRNA_exon", "MANE_snoRNA", "MANE_snoRNA_exon", "MANE_other",
        "MANE_transcript_count", "Strand", "group: +", "group: -",
    ] + FINAL_SUBGROUPS
    df[output_columns].to_csv(
        release_dir / "clinvar_benchmark_all_labels.csv.gz", index=False, compression="gzip"
    )
    exact_pb[output_columns + ["Rule_based_category", "Rule_based"]].to_csv(
        release_dir / "clinvar_benchmark.csv.gz", index=False, compression="gzip"
    )
    exact_pb[["#CHROM", "POS", "REF", "ALT", "ClinVar_label"]].to_csv(
        release_dir / "clinvar_variants.csv.gz", index=False, compression="gzip"
    )
    print(f"[{release}] outputs written to {release_dir}", flush=True)
    return group_counts, feature_counts, funnel, len(df), len(exact_pb)


def parse_release_spec(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = spec.split("=", 1)
    return name, Path(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_item(path: Path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="append", type=parse_release_spec, required=True)
    parser.add_argument("--mane", type=Path, required=True)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--fai", type=Path)
    parser.add_argument("--refseq", type=Path, required=True)
    parser.add_argument(
        "--source-notebook",
        type=Path,
        default=Path("VEP_ClinVar_Benchmarking_RefSeq.ipynb"),
    )
    parser.add_argument("--output", type=Path, default=Path("processed/vep_eval"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    fai = args.fai or Path(str(args.genome) + ".fai")
    for path in [args.mane, args.genome, args.refseq, *(p for _, p in args.release)]:
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if not fai.is_file():
        print(f"Building FASTA index {fai}", flush=True)
        build_fai(args.genome, fai)

    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading MANE and RefSeq references", flush=True)
    mane = load_mane(args.mane)
    strand_maps = load_strand_maps(args.refseq)
    all_groups, all_features, run_summary = [], [], []
    for release, path in args.release:
        groups, features, funnel, valid_count, exact_count = process_release(
            release, path, mane, args.genome, fai, strand_maps, args.output, args.chunksize
        )
        all_groups.append(groups)
        all_features.append(features)
        run_summary.append(
            {
                "release": release,
                "input": str(path),
                "reference_valid_all_labels": valid_count,
                "reference_valid_exact_benign_pathogenic": exact_count,
                "duplicate_variant_key_rows": funnel["duplicate variant-key rows"],
            }
        )
    pd.concat(all_groups, ignore_index=True).to_csv(args.output / "all_group_counts.csv", index=False)
    pd.concat(all_features, ignore_index=True).to_csv(args.output / "all_feature_counts.csv", index=False)
    pd.DataFrame(run_summary).to_csv(args.output / "run_summary.csv", index=False)
    manifest = {
        "MANE": manifest_item(args.mane),
        "genome": manifest_item(args.genome),
        "fai": manifest_item(fai),
        "ncbiRefSeq": manifest_item(args.refseq),
        "note": "Group rules transcribed from VEP_ClinVar_Benchmarking_RefSeq.ipynb",
    }
    if args.source_notebook.is_file():
        manifest["source_notebook"] = manifest_item(args.source_notebook)
    (args.output / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nCombined statistics written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
