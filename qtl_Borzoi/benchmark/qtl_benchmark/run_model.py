#!/usr/bin/env python3
"""CLI for dynamically loading a QTLModel adapter and benchmarking it."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model_loader import load_config, load_model
from .pipeline import benchmark_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an extensible QTLModel adapter on standardized QTL datasets"
    )
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
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    model = load_model(args.model_class, load_config(args.model_config))
    for task in args.tasks.split(","):
        task = task.strip()
        if not task:
            continue
        metrics = benchmark_model(
            model=model,
            task=task,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_variants=args.max_variants,
            repeats=args.repeats,
            sample_fraction=args.sample_fraction,
            seed=args.seed,
        )
        print(f"\n[{model.name}/{task}]")
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
