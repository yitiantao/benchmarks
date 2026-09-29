#!/usr/bin/env python3
"""Backward-compatible AlphaGenome entry point.

The implementation now lives in the shared data/model/evaluation framework.
This module retains the old command and helper names used by downstream code;
it contains no AlphaGenome-specific evaluation logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import time
from typing import Iterable, Sequence

import pandas as pd

from .core import BenchmarkDataset, variant_key
from .data.gtex_eqtl import (
    GTExEQTLDataModule,
    _causal_targets,
    read_tissue_vcf,
    tissue_vcf_path,
)
from .evaluation.gtex_eqtl import GTExEQTLEvaluator
from .interfaces import TidyVariantModelAdapter
from .model_base import QTLModel, TaskContext, VariantRecord
from .model_loader import load_config, load_model
from .predictions import PredictionStore
from .runner import BenchmarkRunner


_variant_key = variant_key


def summarize_score_table(table: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper for AlphaGenome's raw tidy score table."""
    columns = ["gene_id", "gtex_tissue", "score", "n_tracks"]
    if table is None or table.empty:
        return pd.DataFrame(columns=columns)
    normalized = table.rename(
        columns={"raw_score": "score", "tissue": "gtex_tissue"}
    ).copy()
    required = {"gene_id", "gtex_tissue", "score"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(
            "Model score table is missing gene/tissue columns: "
            f"{sorted(missing)}"
        )
    normalized["gene_id"] = normalized.gene_id.astype(str).str.split(".").str[0]
    normalized["gtex_tissue"] = normalized.gtex_tissue.fillna("").astype(str)
    normalized["score"] = pd.to_numeric(normalized.score, errors="coerce")
    normalized = normalized[
        normalized.gene_id.ne("")
        & normalized.gtex_tissue.ne("")
        & normalized.score.notna()
    ]
    return (
        normalized.groupby(["gene_id", "gtex_tissue"], sort=False, as_index=False)
        .agg(score=("score", "mean"), n_tracks=("score", "size"))
        .loc[:, columns]
    )


class ScoreStore(PredictionStore):
    """Compatibility view over the shared :class:`PredictionStore`."""

    def ensure_identity(self, identity: str) -> None:
        try:
            super().ensure_identity(identity)
        except RuntimeError as error:
            raise RuntimeError(
                f"Score cache {self.path} belongs to a different model/config. "
                "Use a different output directory or model name."
            ) from error

    def save(
        self,
        record: VariantRecord,
        scores: pd.DataFrame,
        elapsed_seconds: float,
    ) -> None:
        table = scores.rename(columns={"gtex_tissue": "tissue"}).copy()
        super().save(record, table, elapsed_seconds)

    def tissue_scores(self, tissue: str) -> pd.DataFrame:
        return self.read(tissue).loc[
            :, ["variant_key", "gene_id", "score", "n_tracks"]
        ]

    def clear(self, records_or_keys: Iterable[VariantRecord | str]) -> None:
        with self.connection:
            for item in records_or_keys:
                key = variant_key(item) if isinstance(item, VariantRecord) else item
                self.connection.execute(
                    "DELETE FROM predictions WHERE variant_key = ?", (key,)
                )
                self.connection.execute(
                    "DELETE FROM variants WHERE variant_key = ?", (key,)
                )


def collect_inputs(
    data_dir: Path,
    tissues: Sequence[str],
    max_variants: int,
) -> tuple[
    dict[str, dict[str, tuple[VariantRecord, ...]]], dict[str, VariantRecord]
]:
    by_tissue: dict[str, dict[str, tuple[VariantRecord, ...]]] = {}
    unique: dict[str, VariantRecord] = {}
    for tissue in tissues:
        by_tissue[tissue] = {}
        for split in ("pos", "neg"):
            records = read_tissue_vcf(
                tissue_vcf_path(data_dir, tissue, split),
                split,
                tissue,
                max_variants,
            )
            by_tissue[tissue][split] = records
            for record in records:
                unique.setdefault(variant_key(record), record)
    return by_tissue, unique


def score_variants(
    model: QTLModel,
    store: ScoreStore,
    records: Sequence[VariantRecord],
    context: TaskContext,
) -> None:
    """Compatibility wrapper around per-variant tidy prediction."""
    score_method = getattr(model, "score_variant_table", None)
    if not callable(score_method):
        raise TypeError(
            f"{type(model).__name__} must implement score_variant_table() for "
            "the tissue-aware eQTL benchmark"
        )
    pending = [record for record in records if not store.completed(record)]
    print(
        f"[score] total={len(records)} cached={len(records) - len(pending)} "
        f"pending={len(pending)}"
    )
    if not pending:
        return
    model.setup_task(context)
    try:
        for index, record in enumerate(pending, 1):
            started = time.time()
            summary = summarize_score_table(score_method(record))
            elapsed = time.time() - started
            store.save(record, summary, elapsed)
            print(
                f"[score] {index}/{len(pending)} {record.variant_id} "
                f"gene_tissues={len(summary)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        model.teardown_task(context)


def evaluate_tissue(
    store: ScoreStore,
    data_dir: Path,
    output_dir: Path,
    tissue: str,
    positive: Sequence[VariantRecord],
    negative: Sequence[VariantRecord],
    pip_threshold: float,
    min_variants: int,
) -> dict[str, object]:
    """Compatibility wrapper around :class:`GTExEQTLEvaluator`."""
    records: dict[str, VariantRecord] = {}
    memberships = []
    for split, split_records in (("pos", positive), ("neg", negative)):
        for record in split_records:
            key = variant_key(record)
            records.setdefault(key, record)
            memberships.append(
                {
                    "variant_key": key,
                    "variant_id": record.variant_id,
                    "split": split,
                    "tissue": tissue,
                }
            )
    dataset = BenchmarkDataset(
        name="gtex-eqtl",
        task="eqtl",
        data_dir=data_dir,
        variants=tuple(records.values()),
        memberships=pd.DataFrame(memberships),
        targets=_causal_targets(data_dir, tissue, positive, pip_threshold),
    )
    metrics = GTExEQTLEvaluator(min_variants=min_variants).evaluate(
        dataset, store.read(), output_dir
    )
    return metrics.iloc[0].to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a model on the shared GTEx eQTL benchmark"
    )
    parser.add_argument(
        "--model-class",
        default="qtl_benchmark.models.alphagenome:AlphaGenomeModel",
    )
    parser.add_argument("--model-config", default="configs/alphagenome_smoke.json")
    parser.add_argument("--data-dir", type=Path, default=Path("data/eqtl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alphagenome_eqtl"))
    parser.add_argument("--tissue", action="append", dest="tissues")
    parser.add_argument("--all-tissues", action="store_true")
    parser.add_argument("--max-variants", type=int, default=1)
    parser.add_argument("--pip-threshold", type=float, default=0.9)
    parser.add_argument("--min-variants", type=int, default=32)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--force-rescore", action="store_true")
    return parser.parse_args()


def _migrate_legacy_cache(run_dir: Path, identity: str) -> None:
    """Copy the old ``scores.sqlite`` cache without changing or deleting it."""
    legacy_path = run_dir / "scores.sqlite"
    current_path = run_dir / "predictions.sqlite"
    if not legacy_path.is_file() or current_path.exists():
        return
    legacy = sqlite3.connect(legacy_path)
    try:
        tables = {
            row[0]
            for row in legacy.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not {"variants", "scores"}.issubset(tables):
            return
        variants = legacy.execute(
            """SELECT variant_key, variant_id, chrom, pos, ref, alt,
                      completed, elapsed_seconds FROM variants"""
        ).fetchall()
        predictions = legacy.execute(
            """SELECT variant_key, gene_id, tissue, score, n_tracks
               FROM scores"""
        ).fetchall()
    finally:
        legacy.close()
    with PredictionStore(current_path) as current:
        current.ensure_identity(identity)
        with current.connection:
            current.connection.executemany(
                """INSERT OR REPLACE INTO variants
                   (variant_key, variant_id, chrom, pos, ref, alt, completed,
                    elapsed_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                variants,
            )
            current.connection.executemany(
                """INSERT OR REPLACE INTO predictions
                   (variant_key, gene_id, tissue, score, n_tracks)
                   VALUES (?, ?, ?, ?, ?)""",
                predictions,
            )
    print(
        f"[cache] migrated {len(variants)} variants and {len(predictions)} "
        f"predictions from {legacy_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.evaluate_only and args.score_only:
        raise ValueError("--evaluate-only and --score-only are mutually exclusive")
    model_config = load_config(args.model_config)
    model = load_model(args.model_class, model_config)
    adapter = TidyVariantModelAdapter(
        model, model_class=args.model_class, model_config=model_config
    )
    _migrate_legacy_cache(
        args.output_dir.resolve() / model.name / "borzoi_eqtl",
        adapter.fingerprint(),
    )
    runner = BenchmarkRunner(
        data=GTExEQTLDataModule(
            data_dir=args.data_dir,
            tissues=args.tissues,
            all_tissues=args.all_tissues,
            max_variants=args.max_variants,
            pip_threshold=args.pip_threshold,
            name="borzoi_eqtl",
        ),
        model=adapter,
        evaluator=GTExEQTLEvaluator(min_variants=args.min_variants),
        output_dir=args.output_dir,
    )
    metrics = runner.run(
        predict=not args.evaluate_only,
        evaluate=not args.score_only,
        force=args.force_rescore,
    )
    if metrics is not None:
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
