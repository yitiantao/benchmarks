#!/usr/bin/env python3
"""Evaluate AlphaGenome and Borzoi eQTL predictions on one shared dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np
import pandas as pd

from .core import normalize_prediction_table, variant_key
from .data import GTExEQTLDataModule
from .evaluation import GTExEQTLEvaluator
from .models.borzoi import TISSUE_KEYWORDS


METRIC_COLUMNS = ("auroc_sign", "spearmanr", "pearsonr", "auroc_class")
VIEW_NAMES = (
    "alphagenome_native",
    "alphagenome_borzoi_grouped",
    "borzoi_native",
)


def group_alphagenome_to_borzoi_tissues(
    predictions: pd.DataFrame,
    tissues: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Coarsen exact AlphaGenome tissues to Borzoi's tissue-keyword groups.

    Exact-tissue scores are averaged using ``n_tracks`` as weights, then the
    grouped score is assigned back to every benchmark tissue in that group.
    The native AlphaGenome predictions are not modified.
    """
    predictions = normalize_prediction_table(predictions)
    tissue_map = pd.DataFrame(
        [
            {"tissue": tissue, "tissue_group": TISSUE_KEYWORDS[tissue]}
            for tissue in tissues
            if tissue in TISSUE_KEYWORDS
        ]
    )
    if tissue_map.empty:
        raise ValueError("No benchmark tissues have a Borzoi tissue-keyword mapping")

    selected = predictions.merge(tissue_map, on="tissue", how="inner")
    if selected.empty:
        raise ValueError("AlphaGenome predictions contain no mapped GTEx tissues")
    selected["weighted_score"] = selected.score * selected.n_tracks
    grouped = (
        selected.groupby(
            ["variant_key", "gene_id", "tissue_group"],
            sort=False,
            as_index=False,
        )
        .agg(
            weighted_score=("weighted_score", "sum"),
            n_tracks=("n_tracks", "sum"),
        )
    )
    grouped["score"] = grouped.weighted_score / grouped.n_tracks
    expanded = grouped.merge(
        tissue_map.rename(columns={"tissue": "output_tissue"}),
        on="tissue_group",
        how="inner",
    )
    return normalize_prediction_table(
        expanded.rename(columns={"output_tissue": "tissue"})[
            ["variant_key", "gene_id", "tissue", "score", "n_tracks"]
        ]
    )


def completed_variant_keys(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction store not found: {path}")
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT variant_key FROM variants WHERE completed = 1"
        ).fetchall()
    return {str(row[0]) for row in rows}


def _chunks(values: list[str], size: int = 128) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def read_prediction_pairs(
    path: Path,
    pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    """Read only requested ``(variant_key, tissue)`` pairs from a store.

    AlphaGenome eQTL stores can contain tens of gigabytes because they retain
    every gene and output tissue. Querying small variant batches uses the
    leading ``variant_key`` part of the prediction primary key, while the
    in-memory merge removes unrequested tissues before batches are combined.
    """
    columns = ["variant_key", "gene_id", "tissue", "score", "n_tracks"]
    if not pairs:
        return pd.DataFrame(columns=columns)
    requested = pd.DataFrame(sorted(pairs), columns=["variant_key", "tissue"])
    variants = requested.variant_key.drop_duplicates().tolist()
    tables: list[pd.DataFrame] = []
    with sqlite3.connect(path) as connection:
        for batch in _chunks(variants):
            placeholders = ",".join("?" for _ in batch)
            table = pd.read_sql_query(
                "SELECT variant_key, gene_id, tissue, score, n_tracks "
                f"FROM predictions WHERE variant_key IN ({placeholders})",
                connection,
                params=batch,
            )
            if table.empty:
                continue
            requested_batch = requested[requested.variant_key.isin(batch)]
            table = table.merge(
                requested_batch,
                on=["variant_key", "tissue"],
                how="inner",
                validate="many_to_one",
            )
            if not table.empty:
                tables.append(table)
    if not tables:
        return pd.DataFrame(columns=columns)
    return normalize_prediction_table(pd.concat(tables, ignore_index=True))


def _pairs_from_memberships(memberships: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        memberships.loc[:, ["variant_key", "tissue"]].itertuples(
            index=False, name=None
        )
    )


def _alphagenome_group_source_pairs(
    membership_pairs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    tissues_by_group: dict[str, set[str]] = defaultdict(set)
    for tissue, group in TISSUE_KEYWORDS.items():
        tissues_by_group[group].add(tissue)
    groups_by_variant: dict[str, set[str]] = defaultdict(set)
    for key, tissue in membership_pairs:
        if tissue not in TISSUE_KEYWORDS:
            raise KeyError(f"No Borzoi tissue-keyword mapping for {tissue!r}")
        groups_by_variant[key].add(TISSUE_KEYWORDS[tissue])
    return {
        (key, source_tissue)
        for key, groups in groups_by_variant.items()
        for group in groups
        for source_tissue in tissues_by_group[group]
    }


def _filter_pairs(
    predictions: pd.DataFrame,
    pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    requested = pd.DataFrame(sorted(pairs), columns=["variant_key", "tissue"])
    return normalize_prediction_table(
        predictions.merge(
            requested,
            on=["variant_key", "tissue"],
            how="inner",
            validate="many_to_one",
        )
    )


def _table_hash(table: pd.DataFrame, columns: list[str]) -> str:
    canonical = table.loc[:, columns].sort_values(columns, kind="stable")
    values = canonical.to_csv(sep="\t", index=False, lineterminator="\n")
    return hashlib.sha256(values.encode()).hexdigest()


def input_statistics(dataset) -> pd.DataFrame:
    """Return auditable shared input counts and hashes by tissue and overall."""
    rows: list[dict[str, object]] = []
    tissue_groups: list[tuple[str, pd.DataFrame, pd.DataFrame]] = [
        (
            tissue,
            dataset.memberships[dataset.memberships.tissue == tissue],
            dataset.targets[dataset.targets.tissue == tissue],
        )
        for tissue in dataset.tissues
    ]
    tissue_groups.append(("__ALL__", dataset.memberships, dataset.targets))
    membership_columns = ["variant_key", "variant_id", "split", "tissue"]
    target_columns = [
        "variant_key",
        "variant_id",
        "gene_id",
        "tissue",
        "effect_size",
        "effect_allele",
        "ref",
    ]
    for tissue, memberships, targets in tissue_groups:
        rows.append(
            {
                "tissue": tissue,
                "n_pos": int(memberships.split.eq("pos").sum()),
                "n_neg": int(memberships.split.eq("neg").sum()),
                "n_memberships": len(memberships),
                "n_unique_variants": memberships.variant_key.nunique(),
                "n_targets": len(targets),
                "n_target_variants": targets.variant_key.nunique(),
                "n_target_genes": targets.gene_id.nunique(),
                "membership_sha256": _table_hash(
                    memberships, membership_columns
                ),
                "target_sha256": _table_hash(targets, target_columns),
            }
        )
    return pd.DataFrame(rows)


def _validate_subset_manifest(
    data_dir: Path,
    limit_scope: str,
    max_variants: int,
) -> dict[str, object] | None:
    path = data_dir / "subset_manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    observed_scope = manifest.get("limit_scope")
    if observed_scope is None:
        observed_scope = (
            "global"
            if manifest.get("selection") == "round_robin_across_tissues"
            else None
        )
    if observed_scope != limit_scope:
        raise RuntimeError(
            f"Subset scope mismatch: requested {limit_scope!r}, "
            f"manifest has {observed_scope!r}: {path}"
        )
    limit_field = (
        "max_variants_per_tissue_per_split"
        if limit_scope == "per-tissue"
        else "max_variants_per_split"
    )
    observed_limit = manifest.get(limit_field)
    if observed_limit != max_variants:
        raise RuntimeError(
            f"Subset limit mismatch: requested {max_variants}, "
            f"manifest {limit_field}={observed_limit!r}: {path}"
        )
    return manifest


def _evaluate_view(
    name: str,
    predictions: pd.DataFrame,
    dataset,
    output_dir: Path,
    min_variants: int,
) -> pd.DataFrame:
    view_dir = output_dir / name
    view_dir.mkdir(parents=True, exist_ok=True)
    predictions = normalize_prediction_table(predictions)
    predictions.to_csv(
        view_dir / "predictions.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    metrics = GTExEQTLEvaluator(min_variants=min_variants).evaluate(
        dataset, predictions, view_dir
    )
    metrics.to_csv(view_dir / "metrics.tsv", sep="\t", index=False)
    return metrics


def _comparison_table(metrics_by_view: dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparison: pd.DataFrame | None = None
    for name, metrics in metrics_by_view.items():
        renamed = metrics.rename(
            columns={column: f"{name}_{column}" for column in metrics if column != "tissue"}
        )
        comparison = (
            renamed
            if comparison is None
            else comparison.merge(renamed, on="tissue", how="outer", validate="one_to_one")
        )
    assert comparison is not None
    return comparison.sort_values("tissue", kind="stable").reset_index(drop=True)


def _summary_table(metrics_by_view: dict[str, pd.DataFrame]) -> pd.DataFrame:
    eligible_tissues = None
    for metrics in metrics_by_view.values():
        current = set(metrics.loc[metrics.eligible, "tissue"])
        eligible_tissues = current if eligible_tissues is None else eligible_tissues & current
    eligible_tissues = eligible_tissues or set()

    rows = []
    for name, metrics in metrics_by_view.items():
        selected = metrics[metrics.tissue.isin(eligible_tissues)]
        row: dict[str, object] = {
            "view": name,
            "common_eligible_tissues": len(eligible_tissues),
            "mean_n_score_found": float(selected.n_score_found.mean()),
            "mean_n_class_score_found": float(selected.n_class_score_found.mean()),
        }
        for column in METRIC_COLUMNS:
            finite = pd.to_numeric(selected[column], errors="coerce")
            row[column] = float(finite.mean())
            row[f"{column}_n_tissues"] = int(np.isfinite(finite).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--alphagenome-store", type=Path, required=True)
    parser.add_argument("--borzoi-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-variants", type=int, default=32)
    parser.add_argument(
        "--limit-scope", choices=("global", "per-tissue"), default="global"
    )
    parser.add_argument("--max-variants", type=int, default=0)
    args = parser.parse_args()

    if args.min_variants < 0:
        parser.error("--min-variants must be non-negative")
    if args.max_variants < 0:
        parser.error("--max-variants must be non-negative")
    alpha_store = args.alphagenome_store
    borzoi_store = args.borzoi_store

    dataset = GTExEQTLDataModule(
        data_dir=args.data_dir,
        all_tissues=True,
        max_variants=0,
        name="gtex-eqtl-shared",
    ).load()
    subset_manifest = _validate_subset_manifest(
        args.data_dir, args.limit_scope, args.max_variants
    )
    shared_statistics = input_statistics(dataset)
    required_keys = {variant_key(record) for record in dataset.variants}
    alpha_completed = completed_variant_keys(alpha_store)
    borzoi_completed = completed_variant_keys(borzoi_store)
    alpha_missing = sorted(required_keys - alpha_completed)
    borzoi_missing = sorted(required_keys - borzoi_completed)
    audit = {
        "data_dir": str(args.data_dir.resolve()),
        "shared_variants": len(required_keys),
        "memberships": len(dataset.memberships),
        "targets": len(dataset.targets),
        "limit_scope": args.limit_scope,
        "max_variants": args.max_variants,
        "membership_sha256": _table_hash(
            dataset.memberships,
            ["variant_key", "variant_id", "split", "tissue"],
        ),
        "target_sha256": _table_hash(
            dataset.targets,
            [
                "variant_key",
                "variant_id",
                "gene_id",
                "tissue",
                "effect_size",
                "effect_allele",
                "ref",
            ],
        ),
        "alphagenome_completed_total": len(alpha_completed),
        "alphagenome_shared_covered": len(required_keys & alpha_completed),
        "alphagenome_missing": alpha_missing,
        "borzoi_completed_total": len(borzoi_completed),
        "borzoi_shared_covered": len(required_keys & borzoi_completed),
        "borzoi_missing": borzoi_missing,
        "same_benchmark_inputs": not alpha_missing and not borzoi_missing,
        "same_input_statistics": not alpha_missing and not borzoi_missing,
        "note": (
            "This audit proves identical benchmark variant/tissue/target inputs; "
            "model-specific sequence windows and output tracks remain native."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_statistics.to_csv(
        args.output_dir / "shared_input_statistics.tsv", sep="\t", index=False
    )
    pd.concat(
        [shared_statistics.assign(view=view) for view in VIEW_NAMES],
        ignore_index=True,
    ).loc[:, ["view", *shared_statistics.columns]].to_csv(
        args.output_dir / "view_input_statistics.tsv", sep="\t", index=False
    )
    (args.output_dir / "input_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if alpha_missing or borzoi_missing:
        raise RuntimeError(
            "Shared-input coverage failed: "
            f"AlphaGenome missing {len(alpha_missing)}, "
            f"Borzoi missing {len(borzoi_missing)}; see input_audit.json"
        )

    membership_pairs = _pairs_from_memberships(dataset.memberships)
    alpha_source_pairs = _alphagenome_group_source_pairs(membership_pairs)
    alpha_source = read_prediction_pairs(alpha_store, alpha_source_pairs)
    alpha = _filter_pairs(alpha_source, membership_pairs)
    borzoi = read_prediction_pairs(borzoi_store, membership_pairs)
    alpha_grouped = group_alphagenome_to_borzoi_tissues(
        alpha_source, list(dataset.tissues)
    )
    alpha_grouped = _filter_pairs(alpha_grouped, membership_pairs)

    audit["prediction_rows"] = {
        "alphagenome_native": len(alpha),
        "alphagenome_borzoi_grouped": len(alpha_grouped),
        "borzoi_native": len(borzoi),
    }
    (args.output_dir / "input_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )

    metrics_by_view = {
        "alphagenome_native": _evaluate_view(
            "alphagenome_native", alpha, dataset, args.output_dir, args.min_variants
        ),
        "alphagenome_borzoi_grouped": _evaluate_view(
            "alphagenome_borzoi_grouped",
            alpha_grouped,
            dataset,
            args.output_dir,
            args.min_variants,
        ),
        "borzoi_native": _evaluate_view(
            "borzoi_native", borzoi, dataset, args.output_dir, args.min_variants
        ),
    }
    comparison = _comparison_table(metrics_by_view)
    summary = _summary_table(metrics_by_view)
    common_metrics = pd.concat(
        [metrics.assign(view=name) for name, metrics in metrics_by_view.items()],
        ignore_index=True,
    ).loc[:, ["view", *next(iter(metrics_by_view.values())).columns]]
    comparison.to_csv(args.output_dir / "comparison.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    common_metrics.to_csv(
        args.output_dir / "common_metrics.tsv", sep="\t", index=False
    )
    summary.to_csv(
        args.output_dir / "common_metrics_summary.tsv", sep="\t", index=False
    )
    comparison_manifest = {
        "evaluator": "GTExEQTLEvaluator",
        "common_metric_columns": list(METRIC_COLUMNS),
        "views": list(metrics_by_view),
        "primary_common_views": [
            "alphagenome_borzoi_grouped",
            "borzoi_native",
        ],
        "secondary_native_resolution_view": "alphagenome_native",
        "limit_scope": args.limit_scope,
        "max_variants": args.max_variants,
        "same_benchmark_inputs": audit["same_benchmark_inputs"],
        "same_input_statistics": audit["same_input_statistics"],
        "membership_sha256": audit["membership_sha256"],
        "target_sha256": audit["target_sha256"],
        "subset_manifest": subset_manifest,
        "native_metrics_note": (
            "Model-native metrics remain in their original run directories; "
            "common_metrics.tsv is the cross-model comparison surface."
        ),
    }
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2))
    print(summary.to_string(index=False))
    print(f"[comparison] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
