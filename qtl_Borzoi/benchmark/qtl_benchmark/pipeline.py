"""Reusable orchestration: data provider -> model adapter -> metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_provider import load_task_dataset, run_model_on_dataset
from .metrics import evaluate_predictions
from .model_base import QTLModel


def predict_model_task(
    model: QTLModel,
    task: str,
    data_dir: Path | str,
    output_dir: Path | str,
    batch_size: int = 64,
    max_variants: int | None = None,
    model_options: dict | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Prediction-only stage; no labels or metrics are computed here."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve() / model.name / task
    dataset = load_task_dataset(data_dir, task, max_variants=max_variants)
    predictions = run_model_on_dataset(
        model,
        dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        options=model_options,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / "predictions.tsv.gz"
    predictions.to_csv(predictions_file, sep="\t", index=False, compression="gzip")
    return predictions, predictions_file


def evaluate_prediction_table(
    predictions: pd.DataFrame,
    task: str,
    data_dir: Path | str,
    output_dir: Path | str,
    repeats: int = 100,
    sample_fraction: float = 0.8,
    seed: int = 44,
) -> pd.DataFrame:
    """Evaluation-only stage for any table using the common prediction schema."""
    output_dir = Path(output_dir).resolve()
    dataset = load_task_dataset(Path(data_dir).resolve(), task)
    metrics, joined = evaluate_predictions(
        predictions,
        dataset,
        repeats=repeats,
        sample_fraction=sample_fraction,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_dir / "predictions_with_labels.tsv.gz", sep="\t", index=False, compression="gzip")
    metrics.to_csv(output_dir / "metrics.tsv", sep="\t", index=False, float_format="%.6f")
    return metrics


def benchmark_model(
    model: QTLModel,
    task: str,
    data_dir: Path | str,
    output_dir: Path | str,
    batch_size: int = 64,
    max_variants: int | None = None,
    repeats: int = 100,
    sample_fraction: float = 0.8,
    seed: int = 44,
    model_options: dict | None = None,
) -> pd.DataFrame:
    """Convenience wrapper that executes the separated predict and evaluate stages."""
    predictions, predictions_file = predict_model_task(
        model=model,
        task=task,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        max_variants=max_variants,
        model_options=model_options,
    )
    return evaluate_prediction_table(
        predictions=predictions,
        task=task,
        data_dir=data_dir,
        output_dir=predictions_file.parent,
        repeats=repeats,
        sample_fraction=sample_fraction,
        seed=seed,
    )
