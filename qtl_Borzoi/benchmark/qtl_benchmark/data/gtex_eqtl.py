"""GTEx/SuSiE eQTL data module used by every model adapter."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..core import BenchmarkDataset, variant_key
from ..model_base import VariantRecord
from .base import DataModule


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _patchless(value: object) -> str:
    return str(value).split(".", 1)[0]


def discover_tissues(data_dir: Path) -> list[str]:
    tissues: set[str] = set()
    for suffix in ("_pos.vcf.gz", "_pos.vcf"):
        tissues.update(path.name[: -len(suffix)] for path in data_dir.glob(f"*{suffix}"))
    return sorted(tissues)


def tissue_vcf_path(data_dir: Path, tissue: str, split: str) -> Path:
    for suffix in (".vcf.gz", ".vcf"):
        candidate = data_dir / f"{tissue}_{split}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {tissue} {split} VCF below {data_dir}")


def read_tissue_vcf(
    path: Path, split: str, tissue: str, max_variants: int = 0
) -> tuple[VariantRecord, ...]:
    records: list[VariantRecord] = []
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            records.append(
                VariantRecord(
                    task="eqtl",
                    split=split,
                    chrom=fields[0],
                    pos=int(fields[1]),
                    variant_id=fields[2],
                    ref=fields[3],
                    alt=fields[4],
                    tissue=tissue,
                )
            )
            if max_variants and len(records) >= max_variants:
                break
    return tuple(records)


def _causal_targets(
    data_dir: Path,
    tissue: str,
    positive: Sequence[VariantRecord],
    pip_threshold: float,
) -> pd.DataFrame:
    candidates = (
        data_dir / "tables" / f"{tissue}.tsv.gz",
        data_dir / "tables" / f"{tissue}.tsv",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing causal table for {tissue}: {candidates}")
    table = pd.read_csv(path, sep="\t", index_col=0)
    required = {"variant", "gene", "pip", "beta_posterior", "allele1"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    by_id = {record.variant_id: record for record in positive}
    table = table[(table.pip > pip_threshold) & table.variant.isin(by_id)].copy()
    return pd.DataFrame(
        {
            "variant_key": table.variant.map(lambda item: variant_key(by_id[item])),
            "variant_id": table.variant.astype(str),
            "gene_id": table.gene.map(_patchless),
            "tissue": tissue,
            "effect_size": pd.to_numeric(table.beta_posterior),
            "effect_allele": table.allele1.astype(str),
            "ref": table.variant.map(lambda item: by_id[item].ref),
        }
    )


class GTExEQTLDataModule(DataModule):
    """Load tissue VCF memberships and causal effect labels once.

    ``max_variants`` is applied per tissue and split, matching the historical
    AlphaGenome command. A value of zero keeps all rows.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/eqtl",
        tissues: Sequence[str] | None = None,
        all_tissues: bool = False,
        max_variants: int = 0,
        pip_threshold: float = 0.9,
        name: str = "gtex-eqtl",
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.requested_tissues = tuple(tissues or ())
        self.all_tissues = bool(all_tissues)
        self.max_variants = int(max_variants)
        self.pip_threshold = float(pip_threshold)
        self.name = name
        if self.max_variants < 0:
            raise ValueError("max_variants must be 0 or greater")
        if not 0 < self.pip_threshold <= 1:
            raise ValueError("pip_threshold must be in (0, 1]")
        if self.all_tissues and self.requested_tissues:
            raise ValueError("Choose tissues or all_tissues, not both")

    def load(self) -> BenchmarkDataset:
        available = discover_tissues(self.data_dir)
        if not available:
            raise FileNotFoundError(
                "No GTEx tissue VCFs found below "
                f"{self.data_dir}; expected files named <tissue>_pos.vcf(.gz) "
                "and <tissue>_neg.vcf(.gz)"
            )
        tissues = available if self.all_tissues else list(self.requested_tissues or ("Liver",))
        tissues = list(dict.fromkeys(tissues))
        unknown = sorted(set(tissues) - set(available))
        if unknown:
            raise ValueError(f"Unknown tissues: {unknown}; available: {available}")

        unique: dict[str, VariantRecord] = {}
        membership_rows: list[dict[str, object]] = []
        target_tables: list[pd.DataFrame] = []
        for tissue in tissues:
            by_split: dict[str, tuple[VariantRecord, ...]] = {}
            for split in ("pos", "neg"):
                records = read_tissue_vcf(
                    tissue_vcf_path(self.data_dir, tissue, split),
                    split,
                    tissue,
                    self.max_variants,
                )
                by_split[split] = records
                for record in records:
                    key = variant_key(record)
                    unique.setdefault(key, record)
                    membership_rows.append(
                        {
                            "variant_key": key,
                            "variant_id": record.variant_id,
                            "split": split,
                            "tissue": tissue,
                        }
                    )
            target_tables.append(
                _causal_targets(
                    self.data_dir, tissue, by_split["pos"], self.pip_threshold
                )
            )

        memberships = pd.DataFrame(
            membership_rows,
            columns=["variant_key", "variant_id", "split", "tissue"],
        ).drop_duplicates()
        targets = pd.concat(target_tables, ignore_index=True)
        return BenchmarkDataset(
            name=self.name,
            task="eqtl",
            data_dir=self.data_dir,
            variants=tuple(unique.values()),
            memberships=memberships,
            targets=targets,
            metadata={
                "max_variants_per_split_per_tissue": self.max_variants,
                "pip_threshold": self.pip_threshold,
            },
        )
