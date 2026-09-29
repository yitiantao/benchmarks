#!/usr/bin/env python3
"""Send completed Borzoi artifacts through the unified QTL evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from .data import GTExEQTLDataModule, MatchedQTLDataModule
from .evaluation import (
    BorzoiSQTLEvaluator,
    GTExEQTLEvaluator,
    MatchedQTLEvaluator,
)
from .models.borzoi import BorzoiQTLModelAdapter, BorzoiSEDModelAdapter
from .runner import BenchmarkRunner


def _discard_prediction_cache(
    output_dir: Path, model_name: str, dataset_name: str
) -> None:
    """Implement ``--force`` before model-identity validation.

    Borzoi adapters consume regenerable HDF5 artifacts.  Removing only their
    unified SQLite cache lets an output-affecting adapter correction replace a
    stale identity without touching the expensive model artifacts.
    """
    cache = output_dir / model_name / dataset_name / "predictions.sqlite"
    for path in (cache, Path(f"{cache}-wal"), Path(f"{cache}-shm")):
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("eqtl", "sqtl", "paqtl", "ipaqtl"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument(
        "--limit-tag",
        help="Explicit dataset/output tag, e.g. per_tissue_max_2000",
    )
    parser.add_argument("--model-name", default="borzoi")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_variants < 0:
        parser.error("--max-variants must be 0 or greater")

    limit_tag = args.limit_tag or (
        f"max_{args.max_variants}" if args.max_variants else "all"
    )
    if args.task == "eqtl":
        if args.artifact_dir is None:
            parser.error("eQTL requires --artifact-dir")
        data = GTExEQTLDataModule(
            data_dir=args.data_dir,
            all_tissues=True,
            max_variants=0,
            name=f"gtex-eqtl-{limit_tag}",
        )
        model = BorzoiSEDModelAdapter(
            artifact_dir=args.artifact_dir,
            name=args.model_name,
        )
        evaluator = GTExEQTLEvaluator()
    else:
        if args.experiment_dir is None:
            parser.error(f"{args.task} requires --experiment-dir")
        data = MatchedQTLDataModule(
            task=args.task,
            data_dir=args.data_dir,
            max_pairs=args.max_variants,
            name=f"{args.task}-{limit_tag}",
        )
        model = BorzoiQTLModelAdapter(
            experiment_dir=args.experiment_dir,
            task=args.task,
            name=args.model_name,
        )
        evaluator = (
            BorzoiSQTLEvaluator()
            if args.task == "sqtl"
            else MatchedQTLEvaluator()
        )

    dataset_name = (
        f"gtex-eqtl-{limit_tag}"
        if args.task == "eqtl"
        else f"{args.task}-{limit_tag}"
    )
    if args.force:
        _discard_prediction_cache(args.output_dir, args.model_name, dataset_name)

    started = time.perf_counter()
    metrics = BenchmarkRunner(
        data=data,
        model=model,
        evaluator=evaluator,
        output_dir=args.output_dir,
    ).run(force=args.force)
    assert metrics is not None
    print(metrics.to_string(index=False))
    print(
        f"[result] metrics="
        f"{(args.output_dir / args.model_name / dataset_name / 'metrics.tsv').resolve()}"
    )
    print(
        f"[performance] task={args.task} elapsed_seconds="
        f"{time.perf_counter() - started:.1f}"
    )


if __name__ == "__main__":
    main()
