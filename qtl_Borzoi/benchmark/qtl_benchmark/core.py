"""Core contracts shared by datasets, models, evaluators, and the runner.

The benchmark deliberately exchanges small Python objects and tidy tables.  A
model never needs to know where labels came from, while an evaluator never
needs to import a model framework such as TensorFlow or JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .model_base import VariantRecord


MEMBERSHIP_COLUMNS = ("variant_key", "variant_id", "split", "tissue")
TARGET_COLUMNS = (
    "variant_key",
    "variant_id",
    "gene_id",
    "tissue",
    "effect_size",
    "effect_allele",
    "ref",
)
PREDICTION_COLUMNS = ("variant_key", "gene_id", "tissue", "score", "n_tracks")


def variant_key(record: VariantRecord) -> str:
    """Return the stable identity used to deduplicate model inference."""
    return f"{record.chrom}:{record.pos}:{record.ref}:{record.alt}"


@dataclass(frozen=True)
class BenchmarkDataset:
    """A model-independent test set.

    ``variants`` contains unique inference inputs. ``memberships`` describes
    which positive/negative tissue sets contain each input. ``targets`` holds
    labels used only by the evaluator (for eQTL, effect size and effect allele).
    """

    name: str
    task: str
    data_dir: Path
    variants: tuple[VariantRecord, ...]
    memberships: pd.DataFrame
    targets: pd.DataFrame
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).resolve())
        self._require_columns(self.memberships, MEMBERSHIP_COLUMNS, "memberships")
        self._require_columns(self.targets, TARGET_COLUMNS, "targets")
        keys = [variant_key(record) for record in self.variants]
        if len(keys) != len(set(keys)):
            raise ValueError("BenchmarkDataset.variants must be unique by genomic allele")
        unknown = set(self.memberships.variant_key) - set(keys)
        if unknown:
            raise ValueError(
                f"memberships contains {len(unknown)} variants absent from inference inputs"
            )
        invalid_splits = set(self.memberships.split) - {"pos", "neg"}
        if invalid_splits:
            raise ValueError(f"Unknown dataset splits: {sorted(invalid_splits)}")

    @staticmethod
    def _require_columns(table: pd.DataFrame, required, name: str) -> None:
        missing = set(required) - set(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    @property
    def tissues(self) -> tuple[str, ...]:
        return tuple(
            tissue
            for tissue in dict.fromkeys(self.memberships.tissue.astype(str))
            if tissue
        )

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "data_dir": str(self.data_dir),
            "variants": len(self.variants),
            "memberships": len(self.memberships),
            "targets": len(self.targets),
            "tissues": list(self.tissues),
            "split_counts": {
                str(key): int(value)
                for key, value in self.memberships.split.value_counts().items()
            },
            "metadata": dict(self.metadata),
        }

    def for_model(self) -> "InferenceDataset":
        """Return a label-free view safe to pass to model code."""
        return InferenceDataset(
            name=self.name,
            task=self.task,
            data_dir=self.data_dir,
            variants=tuple(
                replace(record, split="test", matched_positive_id=None, info={})
                for record in self.variants
            ),
            tissues=self.tissues,
        )


@dataclass(frozen=True)
class InferenceDataset:
    """The subset of a benchmark dataset visible to model adapters."""

    name: str
    task: str
    data_dir: Path
    variants: tuple[VariantRecord, ...]
    tissues: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkContext:
    """Filesystem and user options available during one benchmark run."""

    output_dir: Path
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())


def normalize_prediction_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one model's tidy prediction table."""
    table = table.copy()
    defaults = {"gene_id": "", "tissue": "", "n_tracks": 1}
    for column, default in defaults.items():
        if column not in table:
            table[column] = default
    missing = {"variant_key", "score"} - set(table.columns)
    if missing:
        raise ValueError(f"prediction table is missing columns: {sorted(missing)}")
    table = table.loc[:, PREDICTION_COLUMNS]
    table["gene_id"] = table.gene_id.fillna("").astype(str).str.split(".").str[0]
    table["tissue"] = table.tissue.fillna("").astype(str)
    table["score"] = pd.to_numeric(table.score, errors="coerce")
    table["n_tracks"] = pd.to_numeric(table.n_tracks, errors="raise").astype(int)
    table = table[table.score.notna()].copy()
    if (table.n_tracks < 1).any():
        raise ValueError("n_tracks must be positive")
    if table.duplicated(["variant_key", "gene_id", "tissue"]).any():
        raise ValueError("prediction keys (variant, gene, tissue) must be unique")
    return table.reset_index(drop=True)
