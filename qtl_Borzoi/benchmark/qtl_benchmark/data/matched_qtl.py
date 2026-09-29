"""Matched positive/negative datasets for sQTL, paQTL, and iPaQTL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core import BenchmarkDataset, TARGET_COLUMNS, variant_key
from ..data_provider import TASK_SPECS, read_variants
from ..model_base import VariantRecord
from .base import DataModule


MATCHED_TASKS = frozenset({"sqtl", "paqtl", "ipaqtl"})


class MatchedQTLDataModule(DataModule):
    """Load one of the matched non-expression QTL benchmarks.

    The source VCFs encode the tested gene (``MT``), distance (``SD`` or
    ``PD``), and the positive partner of every negative (``PI``). For smoke
    runs, ``max_pairs`` selects negatives first and then their positive
    partners so truncation cannot silently destroy all matched pairs.
    """

    def __init__(
        self,
        task: str,
        data_dir: str | Path = "data",
        max_pairs: int = 0,
        name: str | None = None,
    ) -> None:
        if task not in MATCHED_TASKS:
            raise ValueError(f"task must be one of {sorted(MATCHED_TASKS)}; got {task!r}")
        if max_pairs < 0:
            raise ValueError("max_pairs must be 0 or greater")
        self.task = task
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.max_pairs = int(max_pairs)
        self.name = name or task

    def _select(self) -> tuple[tuple[VariantRecord, ...], tuple[VariantRecord, ...]]:
        positive = read_variants(self.data_dir, self.task, "pos")
        negative = read_variants(self.data_dir, self.task, "neg")
        if not self.max_pairs:
            return positive, negative

        positive_by_id = {record.variant_id: record for record in positive}
        selected_negative = tuple(negative[: self.max_pairs])
        selected_ids = {
            record.matched_positive_id
            for record in selected_negative
            if record.matched_positive_id in positive_by_id
        }
        selected_positive = tuple(
            record for record in positive if record.variant_id in selected_ids
        )
        return selected_positive, selected_negative

    def load(self) -> BenchmarkDataset:
        positive, negative = self._select()
        unique: dict[str, VariantRecord] = {}
        rows: list[dict[str, object]] = []
        for split, records in (("pos", positive), ("neg", negative)):
            for record in records:
                key = variant_key(record)
                unique.setdefault(key, record)
                rows.append(
                    {
                        "variant_key": key,
                        "variant_id": record.variant_id,
                        "split": split,
                        "tissue": "",
                        "gene_id": record.gene_id or "",
                        "distance": record.distance,
                        "matched_positive_id": record.matched_positive_id or "",
                    }
                )
        memberships = pd.DataFrame(
            rows,
            columns=[
                "variant_key",
                "variant_id",
                "split",
                "tissue",
                "gene_id",
                "distance",
                "matched_positive_id",
            ],
        ).drop_duplicates()
        targets = pd.DataFrame(columns=TARGET_COLUMNS)
        spec = TASK_SPECS[self.task]
        return BenchmarkDataset(
            name=self.name,
            task=self.task,
            data_dir=self.data_dir,
            variants=tuple(unique.values()),
            memberships=memberships,
            targets=targets,
            metadata={
                "matched": True,
                "max_pairs": self.max_pairs,
                "distance_thresholds": list(spec.distance_thresholds),
            },
        )
