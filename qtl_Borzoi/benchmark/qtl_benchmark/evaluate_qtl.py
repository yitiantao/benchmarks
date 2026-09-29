#!/usr/bin/env python3
"""Convert Borzoi HDF5 artifacts to the common prediction/metric interface.

This file is deliberately an adapter, not the metric implementation. Shared
metrics live in :mod:`tools.qtl_benchmark.metrics` and are also used by custom
``QTLModel`` implementations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

try:
    from .data_provider import load_task_dataset_from_vcf_dir
    from .metrics import evaluate_predictions
except ImportError:  # Support direct execution by run_qtl_benchmark.sh.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.qtl_benchmark.data_provider import load_task_dataset_from_vcf_dir
    from tools.qtl_benchmark.metrics import evaluate_predictions


TASK_CONFIG = {
    "sqtl": {"output": "sqtl_span", "score": "nDi"},
    "paqtl": {"output": "paqtl", "score": "COVR"},
    "ipaqtl": {"output": "ipaqtl", "score": "COVR"},
}


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]
    )


def load_borzoi_predictions(
    task: str,
    experiment: Path,
    output_name: str,
    split: str,
    score_key: str,
) -> pd.DataFrame:
    """Average Borzoi replicate HDF5 scores into the common table schema."""
    files = sorted(experiment.glob(f"f0c*/{output_name}/merge_{split}/sed.h5"))
    if not files:
        raise FileNotFoundError(
            f"No completed scores matching f0c*/{output_name}/merge_{split}/sed.h5"
        )

    score_sum = None
    first_meta: dict[str, np.ndarray] | None = None
    for score_file in files:
        with h5py.File(score_file, "r") as scores:
            if score_key not in scores:
                raise KeyError(f"{score_file} has no {score_key!r}; keys={list(scores.keys())}")
            current = np.asarray(scores[score_key], dtype=np.float32)
            meta = {
                "snp": decode(scores["snp"][:]),
                "si": np.asarray(scores["si"][:], dtype=np.int64),
                "gene": decode(scores["gene"][:]),
            }
            if first_meta is None:
                first_meta = meta
                score_sum = current
            else:
                if current.shape != score_sum.shape:
                    raise ValueError(f"Score shape mismatch in {score_file}")
                for key in first_meta:
                    if not np.array_equal(meta[key], first_meta[key]):
                        raise ValueError(f"Row metadata mismatch ({key}) in {score_file}")
                score_sum += current

    assert first_meta is not None and score_sum is not None
    scalar_score = np.nanmean(score_sum / len(files), axis=1)
    row_snps = first_meta["snp"][first_meta["si"]]
    return pd.DataFrame(
        {
            "task": task,
            "split": split,
            "variant_id": row_snps,
            "score": scalar_score,
            "gene_id": first_meta["gene"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASK_CONFIG)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("vcf_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    config = TASK_CONFIG[args.task]
    positive = load_borzoi_predictions(
        args.task, args.experiment, config["output"], "pos", config["score"]
    )
    negative = load_borzoi_predictions(
        args.task, args.experiment, config["output"], "neg", config["score"]
    )
    predictions = pd.concat([positive, negative], ignore_index=True)
    dataset = load_task_dataset_from_vcf_dir(args.vcf_dir, args.task)
    metrics, joined = evaluate_predictions(
        predictions,
        dataset,
        repeats=args.repeats,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.out_dir / "metrics.tsv"
    raw_predictions_file = args.out_dir / "predictions_raw.tsv.gz"
    predictions_file = args.out_dir / "predictions.tsv.gz"
    metrics.to_csv(metrics_file, sep="\t", index=False, float_format="%.6f")
    predictions.to_csv(raw_predictions_file, sep="\t", index=False, compression="gzip")
    joined.to_csv(predictions_file, sep="\t", index=False, compression="gzip")

    print(metrics.to_string(index=False))
    print(f"\nMetrics: {metrics_file}")
    print(f"Raw predictions: {raw_predictions_file}")
    print(f"Joined predictions: {predictions_file}")


if __name__ == "__main__":
    main()
