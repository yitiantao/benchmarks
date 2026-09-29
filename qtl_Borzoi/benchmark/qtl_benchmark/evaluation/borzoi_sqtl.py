"""Borzoi-specific sQTL evaluator.

The metric contract intentionally matches the shared matched-QTL evaluator,
but this implementation is independent so Borzoi paper-compatibility changes
cannot alter AlphaGenome, NTv3, or DNA-FM evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import BenchmarkDataset, normalize_prediction_table
from ..data_provider import TASK_SPECS
from ..interfaces import Evaluator
from ..metrics import match_positive_negative, sample_binary_metrics


class BorzoiSQTLEvaluator(Evaluator):
    """Evaluate Borzoi sQTL predictions at the paper distance cutoffs."""

    name = "borzoi-sqtl"

    def __init__(
        self,
        repeats: int = 100,
        sample_fraction: float = 0.8,
        seed: int = 44,
        variant_fallback: bool = True,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        self.repeats = int(repeats)
        self.sample_fraction = float(sample_fraction)
        self.seed = int(seed)
        self.variant_fallback = bool(variant_fallback)

    @staticmethod
    def _largest_magnitude(
        table: pd.DataFrame, keys: list[str], score_name: str
    ) -> pd.DataFrame:
        if table.empty:
            return pd.DataFrame(columns=[*keys, score_name])
        indexes = (
            table.assign(_magnitude=table.score.abs())
            .groupby(keys, sort=False)["_magnitude"]
            .idxmax()
        )
        return table.loc[indexes, [*keys, "score"]].rename(
            columns={"score": score_name}
        )

    def _attach_scores(
        self, dataset: BenchmarkDataset, predictions: pd.DataFrame
    ) -> pd.DataFrame:
        memberships = dataset.memberships.copy()
        exact = self._largest_magnitude(
            predictions[predictions.gene_id.ne("")],
            ["variant_key", "gene_id"],
            "gene_score",
        )
        joined = memberships.merge(
            exact,
            on=["variant_key", "gene_id"],
            how="left",
            validate="many_to_one",
        )
        if self.variant_fallback:
            fallback = self._largest_magnitude(
                predictions, ["variant_key"], "variant_score"
            )
            joined = joined.merge(
                fallback, on="variant_key", how="left", validate="many_to_one"
            )
            joined["score"] = joined.gene_score.fillna(joined.variant_score)
            joined["score_source"] = np.select(
                [joined.gene_score.notna(), joined.variant_score.notna()],
                ["gene", "variant"],
                default="missing",
            )
            joined = joined.drop(columns=["gene_score", "variant_score"])
        else:
            joined = joined.rename(columns={"gene_score": "score"})
            joined["score_source"] = np.where(
                joined.score.notna(), "gene", "missing"
            )
        joined["score_found"] = joined.score.notna()
        joined["distance"] = pd.to_numeric(joined.distance, errors="raise")
        return joined

    def evaluate(
        self,
        dataset: BenchmarkDataset,
        predictions: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        if dataset.task != "sqtl":
            raise ValueError(
                f"BorzoiSQTLEvaluator cannot evaluate {dataset.task!r}"
            )
        predictions = normalize_prediction_table(predictions)
        joined = self._attach_scores(dataset, predictions)
        output_dir = Path(output_dir)
        detail_dir = output_dir / "predictions"
        detail_dir.mkdir(parents=True, exist_ok=True)
        joined.to_csv(
            detail_dir / "sqtl_classification.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )

        scored = joined[joined.score_found].copy()
        positive = scored[scored.split.eq("pos")].copy()
        negative = scored[scored.split.eq("neg")].copy()
        rng = np.random.default_rng(self.seed)
        rows = []
        for max_distance in TASK_SPECS["sqtl"].distance_thresholds:
            positive_subset = positive[positive.distance <= max_distance]
            negative_subset = negative[negative.distance <= max_distance]
            positive_subset, negative_subset = match_positive_negative(
                positive_subset, negative_subset
            )
            row = {
                "task": "sqtl",
                "max_distance": max_distance,
                "matched_positive": len(positive_subset),
                "matched_negative": len(negative_subset),
                "scored_positive": int(
                    (positive.distance <= max_distance).sum()
                ),
                "scored_negative": int(
                    (negative.distance <= max_distance).sum()
                ),
            }
            row.update(
                sample_binary_metrics(
                    positive_subset,
                    negative_subset,
                    repeats=self.repeats,
                    sample_fraction=self.sample_fraction,
                    rng=rng,
                )
            )
            rows.append(row)
        return pd.DataFrame(rows)
