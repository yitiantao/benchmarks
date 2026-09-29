#!/usr/bin/env python3
"""Evaluation-only CLI for normalized prediction tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import evaluate_prediction_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate existing prediction tables without loading a model"
    )
    parser.add_argument(
        "--prediction-dir",
        required=True,
        type=Path,
        help="Directory containing <task>/predictions.tsv.gz",
    )
    parser.add_argument("--tasks", default="eqtl")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    for task in filter(None, (item.strip() for item in args.tasks.split(","))):
        task_dir = args.prediction_dir / task
        prediction_file = task_dir / "predictions.tsv.gz"
        predictions = pd.read_csv(prediction_file, sep="\t")
        metrics = evaluate_prediction_table(
            predictions=predictions,
            task=task,
            data_dir=args.data_dir,
            output_dir=task_dir,
            repeats=args.repeats,
            sample_fraction=args.sample_fraction,
            seed=args.seed,
        )
        print(f"\n[{task}] {task_dir / 'metrics.tsv'}")
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
