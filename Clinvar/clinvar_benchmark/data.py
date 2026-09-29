"""Dataset selection and GRCh38 reference-sequence preparation."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FASTA = ROOT / "data/raw/hg38.fa"
VERSIONS = ("2025-03", "2026-02", "2026-09-14")
def list_tasks(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        names = next(csv.reader(handle))
    return ["all"] + [name.removeprefix("group: ") for name in names if name.startswith("group: ")]


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int
    ref: str
    alt: str
    label: int
    group: str
    stars: int

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"

    @property
    def chr_name(self) -> str:
        return self.chrom if self.chrom.startswith("chr") else "chr" + self.chrom


def dataset_path(version: str) -> Path:
    if version not in VERSIONS:
        raise ValueError(f"Unknown version {version!r}; choose from {VERSIONS}")
    return ROOT / "processed/vep_eval" / version / "clinvar_benchmark.csv.gz"


def iter_variants(path: Path, group: str = "all", limit: int = 0):
    if group not in list_tasks(path):
        raise ValueError(f"Unknown group {group!r}; choose from {list_tasks(path)}")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    opener = gzip.open if str(path).endswith(".gz") else open
    seen = set()
    yielded = 0
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"#CHROM", "POS", "REF", "ALT", "ClinVar_label", "ClinVar_gold_stars"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")
        group_col = f"group: {group}"
        if group != "all" and group_col not in reader.fieldnames:
            raise ValueError(f"Missing group column {group_col!r}")
        for row in reader:
            if group != "all" and row[group_col] not in ("1", "1.0", "True"):
                continue
            label = float(row["ClinVar_label"])
            if label not in (0.0, 1.0):
                continue
            chrom = row["#CHROM"].removeprefix("chr")
            ref, alt = row["REF"].upper(), row["ALT"].upper()
            if chrom not in {str(i) for i in range(1, 23)} | {"X", "Y"}:
                continue
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                continue
            variant = Variant(chrom, int(row["POS"]), ref, alt, int(label), group,
                              int(float(row["ClinVar_gold_stars"])))
            if variant.key in seen:
                continue
            seen.add(variant.key)
            yield variant
            yielded += 1
            if limit and yielded >= limit:
                return


class Reference:
    def __init__(self, fasta: Path = FASTA):
        from pyfaidx import Fasta
        if not fasta.is_file() or not Path(str(fasta) + ".fai").is_file():
            raise FileNotFoundError(f"FASTA and .fai required: {fasta}")
        self.path = fasta
        self.genome = Fasta(str(fasta), as_raw=True, sequence_always_upper=True)

    def window(self, variant: Variant, length: int) -> tuple[str, str, int]:
        if length < 2 or length % 2:
            raise ValueError("window length must be even and >= 2")
        # Zero-based position of VCF POS within a centered, fixed-length window.
        offset = length // 2 - 1
        start = variant.pos - 1 - offset
        end = start + length
        chromosome = self.genome[variant.chr_name]
        size = len(chromosome)
        sequence = ("N" * max(0, -start) +
                    chromosome[max(0, start):min(size, end)] +
                    "N" * max(0, end - size)).upper()
        if len(sequence) != length:
            raise ValueError(f"Incorrect window length for {variant.key}")
        if sequence[offset] != variant.ref:
            raise ValueError(f"Reference mismatch at {variant.key}: FASTA={sequence[offset]}")
        alternate = sequence[:offset] + variant.alt + sequence[offset + 1:]
        return sequence, alternate, offset
