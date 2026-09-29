"""Adapter for scores produced by an external model or VEP tool."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..model_base import QTLModel, TaskContext, VariantPrediction, VariantRecord


class PrecomputedModel(QTLModel):
    """Look up predictions in ``{task}_{split}.tsv``-style files.

    ``path_template`` may contain ``{task}`` and ``{split}``. Each table must
    contain one variant ID column and one scalar score column.
    """

    def __init__(
        self,
        path_template: str,
        name: str = "precomputed",
        id_column: str = "variant_id",
        score_column: str = "score",
        sep: str = "\t",
    ) -> None:
        self.path_template = path_template
        self.name = name
        self.id_column = id_column
        self.score_column = score_column
        self.sep = sep
        self._cache: dict[tuple[str, str], dict[str, float]] = {}

    def _scores(self, task: str, split: str) -> dict[str, float]:
        key = (task, split)
        if key not in self._cache:
            path = Path(self.path_template.format(task=task, split=split))
            table = pd.read_csv(path, sep=self.sep)
            missing = {self.id_column, self.score_column} - set(table.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            if table[self.id_column].duplicated().any():
                raise ValueError(f"{path} contains duplicate variant IDs")
            self._cache[key] = dict(
                zip(table[self.id_column].astype(str), table[self.score_column].astype(float))
            )
        return self._cache[key]

    def predict_batch(
        self,
        task: str,
        variants: Sequence[VariantRecord],
        context: TaskContext,
    ) -> Sequence[VariantPrediction]:
        del context
        if not variants:
            return []
        split = variants[0].split
        scores = self._scores(task, split)
        predictions = []
        for variant in variants:
            if variant.variant_id not in scores:
                raise KeyError(
                    f"No precomputed score for {task}/{split}/{variant.variant_id}"
                )
            predictions.append(
                VariantPrediction(variant_id=variant.variant_id, score=scores[variant.variant_id])
            )
        return predictions
