"""Model-independent metrics for normalized QTL prediction tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_provider import QTLTaskDataset


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC via average ranks, including correct handling of ties."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary average precision with threshold-group tie handling."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    true_positive = 0
    total = 0
    average_precision = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        group_positive = int(labels[start:end].sum())
        true_positive += group_positive
        total += end - start
        average_precision += (group_positive / n_pos) * (true_positive / total)
        start = end
    return float(average_precision)


def _records_frame(dataset: QTLTaskDataset, split: str) -> pd.DataFrame:
    records = dataset.positive if split == "pos" else dataset.negative
    return pd.DataFrame(
        {
            "variant_id": [record.variant_id for record in records],
            "distance": [record.distance for record in records],
            "matched_positive_id": [record.matched_positive_id for record in records],
        }
    )


def attach_labels(predictions: pd.DataFrame, dataset: QTLTaskDataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"split", "variant_id", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    positive = predictions[predictions.split == "pos"].merge(
        _records_frame(dataset, "pos"), on="variant_id", how="inner", validate="many_to_one"
    )
    negative = predictions[predictions.split == "neg"].merge(
        _records_frame(dataset, "neg"), on="variant_id", how="inner", validate="many_to_one"
    )
    positive = positive.drop_duplicates("variant_id", keep="first").reset_index(drop=True)
    negative = negative.drop_duplicates("variant_id", keep="first").reset_index(drop=True)
    return positive, negative


def match_positive_negative(
    positive: pd.DataFrame, negative: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = positive.drop_duplicates("variant_id", keep="first").copy()
    negative = negative.drop_duplicates("matched_positive_id", keep="first").copy()
    positive = positive[positive.variant_id.isin(negative.matched_positive_id)].copy()
    negative = negative[negative.matched_positive_id.isin(positive.variant_id)].copy()
    return positive.reset_index(drop=True), negative.reset_index(drop=True)


def sample_binary_metrics(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    repeats: int,
    sample_fraction: float,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    sample_n = int(sample_fraction * min(len(positive), len(negative)))
    if sample_n < 2:
        return {
            "sample_per_class": sample_n,
            "auroc_mean": np.nan,
            "auroc_std": np.nan,
            "auprc_mean": np.nan,
            "auprc_std": np.nan,
        }

    aurocs = []
    auprcs = []
    for _ in range(repeats):
        pos_sample = positive.iloc[rng.permutation(len(positive))[:sample_n]]
        neg_sample = negative.iloc[rng.permutation(len(negative))[:sample_n]]
        scores = np.concatenate(
            [np.abs(pos_sample.score.to_numpy()), np.abs(neg_sample.score.to_numpy())]
        )
        labels = np.concatenate([np.ones(sample_n), np.zeros(sample_n)])
        aurocs.append(roc_auc_score(labels, scores))
        auprcs.append(average_precision_score(labels, scores))
    return {
        "sample_per_class": sample_n,
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    dataset: QTLTaskDataset,
    repeats: int = 100,
    sample_fraction: float = 0.8,
    seed: int = 44,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate standardized predictions and return metrics plus joined rows."""
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    positive, negative = attach_labels(predictions, dataset)
    if dataset.spec.matched:
        joined_positive, joined_negative = match_positive_negative(positive, negative)
    else:
        joined_positive, joined_negative = positive, negative

    rng = np.random.default_rng(seed)
    rows = []
    thresholds: tuple[int | None, ...]
    thresholds = dataset.spec.distance_thresholds if dataset.spec.distance_thresholds else (None,)
    for max_distance in thresholds:
        if max_distance is None:
            positive_subset = positive
            negative_subset = negative
        else:
            positive_subset = positive[positive.distance <= max_distance]
            negative_subset = negative[negative.distance <= max_distance]
        if dataset.spec.matched:
            # The paper notebooks apply the distance filter before matching, so
            # both members of every retained pair satisfy the same cutoff.
            positive_subset, negative_subset = match_positive_negative(
                positive_subset, negative_subset
            )
        row = {
            "task": dataset.task,
            "max_distance": max_distance,
            "matched_positive": len(positive_subset),
            "matched_negative": len(negative_subset),
        }
        row.update(
            sample_binary_metrics(
                positive_subset,
                negative_subset,
                repeats=repeats,
                sample_fraction=sample_fraction,
                rng=rng,
            )
        )
        rows.append(row)

    joined = pd.concat(
        [joined_positive.assign(label=1), joined_negative.assign(label=0)], ignore_index=True
    )
    return pd.DataFrame(rows), joined
