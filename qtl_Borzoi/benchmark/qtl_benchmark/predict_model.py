#!/usr/bin/env python3
"""Prediction-only CLI: QTL data -> model -> normalized prediction table."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model_loader import load_config, load_model
from .pipeline import predict_model_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a QTLModel without computing metrics")
    parser.add_argument("--model-class", required=True, help="python.module:ClassName")
    parser.add_argument("--model-config", help="JSON string or path to a JSON file")
    parser.add_argument("--tasks", default="eqtl")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-variants",
        type=int,
        default=0,
        help="Maximum variants per positive/negative split; 0 means all.",
    )
    args = parser.parse_args()

    model = load_model(args.model_class, load_config(args.model_config))
    for task in filter(None, (item.strip() for item in args.tasks.split(","))):
        _, predictions_file = predict_model_task(
            model=model,
            task=task,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_variants=args.max_variants,
        )
        print(f"[{model.name}/{task}] {predictions_file}")


if __name__ == "__main__":
    main()
