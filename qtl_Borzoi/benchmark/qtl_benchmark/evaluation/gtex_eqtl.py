"""Unified tissue-aware eQTL evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core import BenchmarkDataset, normalize_prediction_table
from ..interfaces import Evaluator
from ..metrics import roc_auc_score


def _safe_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores_array)
    labels_array, scores_array = labels_array[finite], scores_array[finite]
    if len(scores_array) == 0 or len(np.unique(labels_array)) < 2:
        return float("nan")
    return roc_auc_score(labels_array, scores_array)


def _correlation(left: pd.Series, right: pd.Series, rank: bool = False) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if rank:
        frame = frame.rank(method="average")
    if len(frame) < 2 or frame.left.nunique() < 2 or frame.right.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(frame.left, frame.right)[0, 1])


class GTExEQTLEvaluator(Evaluator):
    """Direction, effect correlation, and causal classification metrics."""

    name = "gtex-eqtl"

    def __init__(self, min_variants: int = 32, missing_score: float = 0.0) -> None:
        self.min_variants = int(min_variants)
        self.missing_score = float(missing_score)
        if self.min_variants < 0:
            raise ValueError("min_variants must be non-negative")

    def evaluate(
        self,
        dataset: BenchmarkDataset,
        predictions: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        if dataset.task != "eqtl":
            raise ValueError(f"GTExEQTLEvaluator cannot evaluate {dataset.task!r}")
        predictions = normalize_prediction_table(predictions)
        output_dir = Path(output_dir)
        detail_dir = output_dir / "predictions"
        detail_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for tissue in dataset.tissues:
            tissue_predictions = predictions[predictions.tissue == tissue]
            targets = dataset.targets[dataset.targets.tissue == tissue].copy()
            coefficient = targets.merge(
                tissue_predictions.loc[:, ["variant_key", "gene_id", "score"]],
                on=["variant_key", "gene_id"],
                how="left",
                validate="many_to_one",
            )
            coefficient["score_found"] = coefficient.score.notna()
            coefficient["score"] = coefficient.score.fillna(self.missing_score)
            flip = coefficient.ref.astype(str) != coefficient.effect_allele.astype(str)
            coefficient.loc[flip & coefficient.score.ne(0), "score"] *= -1

            by_variant = (
                tissue_predictions.assign(abs_score=tissue_predictions.score.abs())
                .groupby("variant_key", sort=False).abs_score.max()
            )
            memberships = dataset.memberships[dataset.memberships.tissue == tissue].copy()
            classification = memberships.merge(
                by_variant.rename("score"),
                left_on="variant_key",
                right_index=True,
                how="left",
            )
            classification["score_found"] = classification.score.notna()
            # Match Borzoi SED semantics: variants without any complete gene
            # prediction do not participate in causal classification.
            classification = classification[classification.score_found].copy()

            coefficient.to_csv(
                detail_dir / f"{tissue}_coefficients.tsv.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )
            classification.to_csv(
                detail_dir / f"{tissue}_classification.tsv.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )

            eligible = len(coefficient) > self.min_variants
            rows.append(
                {
                    "tissue": tissue,
                    "auroc_sign": _safe_auroc(
                        coefficient.effect_size > 0, coefficient.score
                    ) if eligible else float("nan"),
                    "spearmanr": _correlation(
                        coefficient.effect_size, coefficient.score, rank=True
                    ) if eligible else float("nan"),
                    "pearsonr": _correlation(
                        coefficient.effect_size, coefficient.score
                    ) if eligible else float("nan"),
                    "n": len(coefficient),
                    "n_score_found": int(coefficient.score_found.sum()),
                    "auroc_class": _safe_auroc(
                        classification.split.eq("pos"), classification.score
                    ) if eligible else float("nan"),
                    "n_class_pos": int(classification.split.eq("pos").sum()),
                    "n_class_neg": int(classification.split.eq("neg").sum()),
                    "n_class_score_found": int(classification.score_found.sum()),
                    "eligible": eligible,
                }
            )
        return pd.DataFrame(rows)
