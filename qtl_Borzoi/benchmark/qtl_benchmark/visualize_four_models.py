#!/usr/bin/env python3
"""Create publication-ready comparisons for five formal QTL result sets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .models.borzoi import TISSUE_KEYWORDS


DNA_FM_OLD = "DNA-FM step150000"
DNA_FM_NEW = "DNA-FM step340000"
MODEL_ORDER = ("AlphaGenome", "Borzoi", "NTv3", DNA_FM_OLD, DNA_FM_NEW)
MODEL_COLORS = {
    "AlphaGenome": "#4C78A8",
    "Borzoi": "#F58518",
    "NTv3": "#54A24B",
    DNA_FM_OLD: "#E45756",
    DNA_FM_NEW: "#B279A2",
}
EQTL_METRICS = {
    "auroc_sign": "Direction AUROC",
    "spearmanr": "Effect-size Spearman r",
    "pearsonr": "Effect-size Pearson r",
    "auroc_class": "Causal classification AUROC",
}
MATCHED_TASKS = ("sqtl", "paqtl", "ipaqtl")
MATCHED_METRICS = {
    "auroc_mean": "AUROC",
    "auprc_mean": "AUPRC",
}


@dataclass(frozen=True)
class ModelResultPaths:
    name: str
    eqtl: Path
    sqtl: Path
    paqtl: Path
    ipaqtl: Path

    def items(self) -> Iterable[tuple[str, Path]]:
        yield "eqtl", self.eqtl
        yield "sqtl", self.sqtl
        yield "paqtl", self.paqtl
        yield "ipaqtl", self.ipaqtl


def default_result_paths(root: Path) -> tuple[ModelResultPaths, ...]:
    """Return the pinned max-2000 result paths used by the one-click runs."""
    outputs = root / "outputs"
    alpha = outputs / "alphagenome_all_qtl/max_2000/alphagenome-all-folds"
    borzoi_eqtl = (
        outputs
        / "borzoi_replicate_0_all_qtl/unified/per_tissue_max_2000"
        / "borzoi-replicate-0/gtex-eqtl-per_tissue_max_2000/metrics.tsv"
    )
    borzoi_matched = (
        outputs / "borzoi_replicate_0_all_qtl/unified/max_2000/borzoi-replicate-0"
    )
    ntv3 = outputs / "ntv3_100m_post_all_qtl/max_2000/ntv3-100m-post"
    dna_fm_old = outputs / "dna_fm_all_qtl/max_2000/DNA-FM-100M-post-step150000"
    dna_fm_new = (
        outputs
        / "dna_fm_step_0034000_all_qtl/max_2000/DNA-FM-100M-post-step340000"
    )

    def unified_model(name: str, base: Path) -> ModelResultPaths:
        return ModelResultPaths(
            name=name,
            eqtl=base / "gtex-eqtl-max_2000/metrics.tsv",
            sqtl=base / "sqtl-max_2000/metrics.tsv",
            paqtl=base / "paqtl-max_2000/metrics.tsv",
            ipaqtl=base / "ipaqtl-max_2000/metrics.tsv",
        )

    return (
        unified_model("AlphaGenome", alpha),
        ModelResultPaths(
            name="Borzoi",
            eqtl=borzoi_eqtl,
            sqtl=borzoi_matched / "sqtl-max_2000/metrics.tsv",
            paqtl=borzoi_matched / "paqtl-max_2000/metrics.tsv",
            ipaqtl=borzoi_matched / "ipaqtl-max_2000/metrics.tsv",
        ),
        unified_model("NTv3", ntv3),
        unified_model(DNA_FM_OLD, dna_fm_old),
        unified_model(DNA_FM_NEW, dna_fm_new),
    )


def _read_metrics(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required benchmark result is missing: {path}")
    table = pd.read_csv(path, sep="\t")
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return table


def load_metric_tables(
    paths: Iterable[ModelResultPaths],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate all five result sets' eQTL and matched-QTL metrics."""
    paths = tuple(paths)
    names = tuple(item.name for item in paths)
    if names != MODEL_ORDER:
        raise ValueError(
            f"Expected exactly these models in order: {MODEL_ORDER}; got {names}"
        )

    eqtl_tables: list[pd.DataFrame] = []
    matched_tables: list[pd.DataFrame] = []
    tissue_sets: dict[str, set[str]] = {}
    distance_sets: dict[tuple[str, str], set[int]] = {}
    eqtl_required = {"tissue", "eligible", *EQTL_METRICS}
    matched_required = {
        "task",
        "max_distance",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
    }

    for model in paths:
        eqtl = _read_metrics(model.eqtl, eqtl_required).copy()
        eligible = eqtl["eligible"].astype(str).str.lower().isin({"true", "1"})
        eqtl = eqtl[eligible].copy()
        if eqtl["tissue"].duplicated().any():
            raise ValueError(f"Duplicate eQTL tissues in {model.eqtl}")
        for metric in EQTL_METRICS:
            eqtl[metric] = pd.to_numeric(eqtl[metric], errors="coerce")
        eqtl["model"] = model.name
        eqtl["organ_group"] = eqtl["tissue"].map(TISSUE_KEYWORDS)
        if eqtl["organ_group"].isna().any():
            unknown = sorted(eqtl.loc[eqtl.organ_group.isna(), "tissue"].unique())
            raise ValueError(f"No broad organ mapping for tissues: {unknown}")
        eqtl_tables.append(eqtl)
        tissue_sets[model.name] = set(eqtl["tissue"])

        for task, path in list(model.items())[1:]:
            table = _read_metrics(path, matched_required).copy()
            table = table[table["task"].astype(str).str.lower() == task].copy()
            if table.empty:
                raise ValueError(f"{path} has no rows for task={task}")
            table["max_distance"] = pd.to_numeric(
                table["max_distance"], errors="raise"
            ).astype(int)
            if table["max_distance"].duplicated().any():
                raise ValueError(f"Duplicate max_distance rows in {path}")
            for metric in (*MATCHED_METRICS, "auroc_std", "auprc_std"):
                table[metric] = pd.to_numeric(table[metric], errors="coerce")
            table["task"] = task
            table["model"] = model.name
            matched_tables.append(table)
            distance_sets[(model.name, task)] = set(table["max_distance"])

    reference_tissues = tissue_sets[MODEL_ORDER[0]]
    tissue_mismatch = {
        model: sorted(values.symmetric_difference(reference_tissues))
        for model, values in tissue_sets.items()
        if values != reference_tissues
    }
    if tissue_mismatch:
        raise ValueError(f"eQTL tissue sets differ across models: {tissue_mismatch}")
    for task in MATCHED_TASKS:
        reference_distances = distance_sets[(MODEL_ORDER[0], task)]
        mismatched = {
            model: sorted(distance_sets[(model, task)])
            for model in MODEL_ORDER
            if distance_sets[(model, task)] != reference_distances
        }
        if mismatched:
            raise ValueError(
                f"max_distance values differ across models for {task}: {mismatched}"
            )

    return (
        pd.concat(eqtl_tables, ignore_index=True),
        pd.concat(matched_tables, ignore_index=True),
    )


def build_export_tables(
    eqtl: pd.DataFrame, matched: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build tidy tissue, broad-organ macro, and matched-QTL tables."""
    eqtl_columns = ["model", "organ_group", "tissue", *EQTL_METRICS]
    eqtl_tissue = eqtl.loc[:, eqtl_columns].sort_values(
        ["organ_group", "tissue", "model"], kind="stable"
    )
    eqtl_organ = (
        eqtl_tissue.groupby(["model", "organ_group"], as_index=False, sort=False)[
            list(EQTL_METRICS)
        ]
        .mean()
        .sort_values(["organ_group", "model"], kind="stable")
    )
    matched_columns = [
        "model",
        "task",
        "max_distance",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
    ]
    matched_export = matched.loc[:, matched_columns].sort_values(
        ["task", "max_distance", "model"], kind="stable"
    )
    return eqtl_tissue, eqtl_organ, matched_export


def build_overview_table(
    eqtl_tissue: pd.DataFrame, matched: pd.DataFrame
) -> pd.DataFrame:
    """Build the four-task AUROC summary used by overview figures."""
    eqtl_summary = (
        eqtl_tissue.groupby("model", as_index=False)["auroc_class"]
        .mean()
        .assign(benchmark="eQTL")
        .rename(columns={"auroc_class": "value"})
    )
    largest_distance = matched.groupby("task")["max_distance"].max()
    selected = matched[
        matched.apply(
            lambda row: row.max_distance == largest_distance[row.task], axis=1
        )
    ].copy()
    selected["benchmark"] = selected["task"].map(
        {"sqtl": "sQTL", "paqtl": "paQTL", "ipaqtl": "iPaQTL"}
    )
    selected = selected.rename(columns={"auroc_mean": "value"})
    return pd.concat(
        [
            eqtl_summary[["model", "benchmark", "value"]],
            selected[["model", "benchmark", "value"]],
        ],
        ignore_index=True,
    )


def _paired_model_delta(
    table: pd.DataFrame,
    index_columns: list[str],
    metrics: Iterable[str],
    baseline_model: str = "NTv3",
    comparison_model: str = DNA_FM_NEW,
    baseline_prefix: str = "ntv3",
    comparison_prefix: str = "dna_fm",
) -> pd.DataFrame:
    metrics = list(metrics)
    baseline = table[table.model == baseline_model][index_columns + metrics].rename(
        columns={metric: f"{baseline_prefix}_{metric}" for metric in metrics}
    )
    comparison = table[table.model == comparison_model][
        index_columns + metrics
    ].rename(
        columns={metric: f"{comparison_prefix}_{metric}" for metric in metrics}
    )
    paired = baseline.merge(
        comparison,
        on=index_columns,
        how="inner",
        validate="one_to_one",
    )
    for metric in metrics:
        paired[f"delta_{metric}"] = (
            paired[f"{comparison_prefix}_{metric}"]
            - paired[f"{baseline_prefix}_{metric}"]
        )
    return paired.sort_values(index_columns, kind="stable")


def build_ntv3_dna_fm_deltas(
    eqtl_tissue: pd.DataFrame,
    eqtl_organ: pd.DataFrame,
    matched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return paired new-DNA-FM minus NTv3 metric differences."""
    tissue_delta = _paired_model_delta(
        eqtl_tissue, ["organ_group", "tissue"], EQTL_METRICS
    )
    organ_delta = _paired_model_delta(eqtl_organ, ["organ_group"], EQTL_METRICS)
    matched_delta = _paired_model_delta(
        matched,
        ["task", "max_distance"],
        ["auroc_mean", "auprc_mean"],
    )
    return tissue_delta, organ_delta, matched_delta


def build_dna_fm_checkpoint_deltas(
    eqtl_tissue: pd.DataFrame,
    eqtl_organ: pd.DataFrame,
    matched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return paired step340000 minus step150000 metric differences."""
    options = {
        "baseline_model": DNA_FM_OLD,
        "comparison_model": DNA_FM_NEW,
        "baseline_prefix": "step150000",
        "comparison_prefix": "step340000",
    }
    tissue_delta = _paired_model_delta(
        eqtl_tissue, ["organ_group", "tissue"], EQTL_METRICS, **options
    )
    organ_delta = _paired_model_delta(
        eqtl_organ, ["organ_group"], EQTL_METRICS, **options
    )
    matched_delta = _paired_model_delta(
        matched,
        ["task", "max_distance"],
        ["auroc_mean", "auprc_mean"],
        **options,
    )
    return tissue_delta, organ_delta, matched_delta


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )
    return plt


def _save_figure(figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def _grouped_vertical(
    ax,
    table: pd.DataFrame,
    category: str,
    value: str,
    categories: list[object],
    errors: str | None = None,
) -> None:
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / len(MODEL_ORDER)
    offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * width
    for offset, model in zip(offsets, MODEL_ORDER):
        selected = table[table.model == model].set_index(category).reindex(categories)
        error_values = selected[errors].to_numpy() if errors else None
        ax.bar(
            x + offset,
            selected[value].to_numpy(),
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            yerr=error_values,
            capsize=2 if errors else 0,
            linewidth=0.3,
            edgecolor="white",
        )
    ax.set_xticks(x, [str(item) for item in categories])
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)


def _grouped_horizontal(
    ax,
    table: pd.DataFrame,
    category: str,
    value: str,
    categories: list[str],
) -> None:
    y = np.arange(len(categories), dtype=float)
    height = 0.8 / len(MODEL_ORDER)
    offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * height
    for offset, model in zip(offsets, MODEL_ORDER):
        selected = table[table.model == model].set_index(category).reindex(categories)
        ax.barh(
            y + offset,
            selected[value].to_numpy(),
            height=height,
            label=model,
            color=MODEL_COLORS[model],
            linewidth=0.2,
            edgecolor="white",
        )
    ax.set_yticks(y, [item.replace("_", " ") for item in categories])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)


def plot_overview(
    eqtl_tissue: pd.DataFrame, matched: pd.DataFrame, output_dir: Path
) -> None:
    plt = _pyplot()
    overview = build_overview_table(eqtl_tissue, matched)
    categories = ["eQTL", "sQTL", "paQTL", "iPaQTL"]
    figure, ax = plt.subplots(figsize=(11, 5.8))
    _grouped_vertical(ax, overview, "benchmark", "value", categories)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Five-result QTL benchmark overview", pad=30)
    ax.text(
        0.5,
        1.01,
        "eQTL: tissue-macro causal AUROC; matched QTL: AUROC at 10 kb",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="center",
    )
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    figure.subplots_adjust(bottom=0.22)
    _save_figure(figure, output_dir, "all_qtl_overview")
    plt.close(figure)


def _plot_eqtl_grid(
    table: pd.DataFrame,
    category: str,
    output_dir: Path,
    stem: str,
    subtitle: str,
) -> None:
    plt = _pyplot()
    categories = sorted(table[category].unique())
    figure_height = max(24.0, len(categories) * 0.85)
    figure, axes = plt.subplots(2, 2, figsize=(20, figure_height), sharex=False)
    for ax, (metric, title) in zip(axes.flat, EQTL_METRICS.items()):
        _grouped_horizontal(ax, table, category, metric, categories)
        ax.set_xlabel(title)
        ax.set_title(title)
        if "auroc" in metric:
            ax.axvline(0.5, color="#555555", linestyle="--", linewidth=0.8)
        ax.tick_params(axis="y", labelsize=7.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    figure.suptitle(subtitle, fontsize=16, fontweight="bold", y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(figure, output_dir, stem)
    plt.close(figure)


def plot_eqtl(
    eqtl_tissue: pd.DataFrame,
    eqtl_organ: pd.DataFrame,
    output_dir: Path,
) -> None:
    _plot_eqtl_grid(
        eqtl_organ,
        "organ_group",
        output_dir,
        "eqtl_by_organ",
        "eQTL performance by broad organ group (macro mean across tissues)",
    )
    _plot_eqtl_grid(
        eqtl_tissue,
        "tissue",
        output_dir,
        "eqtl_by_tissue",
        "eQTL performance across 49 GTEx tissues",
    )


def plot_matched_qtl(matched: pd.DataFrame, output_dir: Path) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(3, 2, figsize=(15, 14), sharey=False)
    for row, task in enumerate(MATCHED_TASKS):
        task_table = matched[matched.task == task]
        distances = sorted(task_table.max_distance.unique())
        labels = [f"{distance / 1000:g} kb" for distance in distances]
        plot_table = task_table.copy()
        plot_table["distance_label"] = plot_table.max_distance.map(
            dict(zip(distances, labels))
        )
        for column, (metric, metric_title) in enumerate(MATCHED_METRICS.items()):
            ax = axes[row, column]
            std = metric.replace("mean", "std")
            _grouped_vertical(
                ax,
                plot_table,
                "distance_label",
                metric,
                labels,
                errors=std,
            )
            ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_ylim(0.45, 0.9)
            ax.set_ylabel(metric_title)
            ax.set_title(f"{task.upper()} — {metric_title}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    figure.suptitle(
        "sQTL, paQTL and iPaQTL performance by matching distance",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, output_dir, "matched_qtl_by_distance")
    plt.close(figure)


def plot_ntv3_dna_fm_overview(
    eqtl_tissue: pd.DataFrame, matched: pd.DataFrame, output_dir: Path
) -> None:
    """Plot direct NTv3/new-DNA-FM values with paired delta labels."""
    plt = _pyplot()
    overview = build_overview_table(eqtl_tissue, matched)
    overview = overview[overview.model.isin({"NTv3", DNA_FM_NEW})]
    categories = ["eQTL", "sQTL", "paQTL", "iPaQTL"]
    x = np.arange(len(categories), dtype=float)
    width = 0.36
    figure, ax = plt.subplots(figsize=(11, 6.2))
    for offset, model in zip((-width / 2, width / 2), ("NTv3", DNA_FM_NEW)):
        selected = (
            overview[overview.model == model].set_index("benchmark").reindex(categories)
        )
        ax.bar(
            x + offset,
            selected.value.to_numpy(),
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.4,
        )
    paired = _paired_model_delta(overview, ["benchmark"], ["value"])
    paired = paired.set_index("benchmark").reindex(categories)
    for index, row in enumerate(paired.itertuples()):
        ax.text(
            x[index],
            max(row.ntv3_value, row.dna_fm_value) + 0.012,
            f"Δ {row.delta_value:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(x, categories)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("AUROC")
    ax.set_title("NTv3 vs DNA-FM step340000 across four QTL benchmarks", pad=30)
    ax.text(
        0.5,
        1.01,
        "Δ = DNA-FM step340000 − NTv3; eQTL uses tissue-macro causal AUROC; matched QTL uses 10 kb",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="center",
    )
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    figure.subplots_adjust(bottom=0.22)
    _save_figure(figure, output_dir, "ntv3_vs_dna_fm_step340000_overview")
    plt.close(figure)


def _plot_two_model_eqtl_grid(
    table: pd.DataFrame,
    category: str,
    output_dir: Path,
    stem: str,
    title: str,
) -> None:
    """Plot NTv3 and new DNA-FM as adjacent bars for every eQTL category."""
    plt = _pyplot()
    categories = sorted(table[category].unique())
    y = np.arange(len(categories), dtype=float)
    bar_height = 0.40
    figure_height = max(24.0, len(categories) * 0.85)
    figure, axes = plt.subplots(2, 2, figsize=(20, figure_height), sharex=False)
    for ax, (metric, metric_title) in zip(axes.flat, EQTL_METRICS.items()):
        for offset, model in zip((-bar_height / 2, bar_height / 2), ("NTv3", DNA_FM_NEW)):
            selected = (
                table[table.model == model].set_index(category).reindex(categories)
            )
            ax.barh(
                y + offset,
                selected[metric].to_numpy(),
                height=bar_height,
                label=model,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_yticks(y, [item.replace("_", " ") for item in categories])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.22, linewidth=0.6)
        ax.set_xlabel(metric_title)
        ax.set_title(metric_title)
        if "auroc" in metric:
            ax.axvline(0.5, color="#555555", linestyle="--", linewidth=0.8)
        ax.tick_params(axis="y", labelsize=7.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    figure.suptitle(title, fontsize=16, fontweight="bold", y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(figure, output_dir, stem)
    plt.close(figure)


def plot_ntv3_dna_fm_side_by_side(
    eqtl_tissue: pd.DataFrame,
    eqtl_organ: pd.DataFrame,
    matched: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot paired NTv3/new-DNA-FM bars for every category and distance."""
    _plot_two_model_eqtl_grid(
        eqtl_organ,
        "organ_group",
        output_dir,
        "ntv3_vs_dna_fm_step340000_eqtl_by_organ",
        "NTv3 vs DNA-FM step340000 eQTL performance by broad organ group",
    )
    _plot_two_model_eqtl_grid(
        eqtl_tissue,
        "tissue",
        output_dir,
        "ntv3_vs_dna_fm_step340000_eqtl_by_tissue",
        "NTv3 vs DNA-FM step340000 eQTL performance across 49 GTEx tissues",
    )

    plt = _pyplot()
    figure, axes = plt.subplots(3, 2, figsize=(15, 14), sharey=False)
    width = 0.38
    for row, task in enumerate(MATCHED_TASKS):
        task_table = matched[matched.task == task]
        distances = sorted(task_table.max_distance.unique())
        labels = [f"{distance / 1000:g} kb" for distance in distances]
        x = np.arange(len(distances), dtype=float)
        for column, (metric, metric_title) in enumerate(MATCHED_METRICS.items()):
            ax = axes[row, column]
            std = metric.replace("mean", "std")
            for offset, model in zip((-width / 2, width / 2), ("NTv3", DNA_FM_NEW)):
                selected = (
                    task_table[task_table.model == model]
                    .set_index("max_distance")
                    .reindex(distances)
                )
                ax.bar(
                    x + offset,
                    selected[metric].to_numpy(),
                    width=width,
                    yerr=selected[std].to_numpy(),
                    capsize=2,
                    label=model,
                    color=MODEL_COLORS[model],
                    edgecolor="white",
                    linewidth=0.3,
                )
            ax.set_xticks(x, labels)
            ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_ylim(0.45, 0.9)
            ax.grid(axis="y", alpha=0.22, linewidth=0.6)
            ax.set_ylabel(metric_title)
            ax.set_title(f"{task.upper()} — {metric_title}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    figure.suptitle(
        "NTv3 vs DNA-FM step340000 matched-QTL performance by matching distance",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(
        figure, output_dir, "ntv3_vs_dna_fm_step340000_matched_qtl_by_distance"
    )
    plt.close(figure)


def _plot_eqtl_delta_grid(
    table: pd.DataFrame,
    category: str,
    output_dir: Path,
    stem: str,
    title: str,
    positive_model: str = DNA_FM_NEW,
    negative_model: str = "NTv3",
    delta_description: str = "Metric difference Δ = DNA-FM step340000 − NTv3",
) -> None:
    from matplotlib.patches import Patch

    plt = _pyplot()
    categories = sorted(table[category].unique())
    figure_height = max(24.0, len(categories) * 0.85)
    figure, axes = plt.subplots(2, 2, figsize=(20, figure_height), sharex=False)
    for ax, (metric, metric_title) in zip(axes.flat, EQTL_METRICS.items()):
        selected = table.set_index(category).reindex(categories)
        values = selected[f"delta_{metric}"].to_numpy()
        colors = [
            MODEL_COLORS[positive_model] if value >= 0 else MODEL_COLORS[negative_model]
            for value in values
        ]
        y = np.arange(len(categories), dtype=float)
        ax.barh(y, values, height=0.72, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y, [item.replace("_", " ") for item in categories])
        ax.invert_yaxis()
        ax.axvline(0.0, color="#333333", linewidth=0.9)
        limit = max(float(np.nanmax(np.abs(values))) * 1.08, 0.01)
        ax.set_xlim(-limit, limit)
        ax.grid(axis="x", alpha=0.22, linewidth=0.6)
        ax.set_xlabel(f"Δ {metric_title}")
        ax.set_title(metric_title)
        ax.tick_params(axis="y", labelsize=7.5)
    figure.legend(
        handles=[
            Patch(color=MODEL_COLORS[positive_model], label=f"positive: {positive_model} higher"),
            Patch(color=MODEL_COLORS[negative_model], label=f"negative: {negative_model} higher"),
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    figure.suptitle(title, fontsize=16, fontweight="bold", y=1.0)
    figure.text(
        0.5,
        0.975,
        delta_description,
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, output_dir, stem)
    plt.close(figure)


def plot_ntv3_dna_fm_differences(
    tissue_delta: pd.DataFrame,
    organ_delta: pd.DataFrame,
    matched_delta: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot paired differences; positive values favor DNA-FM step340000."""
    _plot_eqtl_delta_grid(
        organ_delta,
        "organ_group",
        output_dir,
        "ntv3_vs_dna_fm_step340000_eqtl_delta_by_organ",
        "NTv3 vs DNA-FM step340000 eQTL differences by broad organ group",
    )
    _plot_eqtl_delta_grid(
        tissue_delta,
        "tissue",
        output_dir,
        "ntv3_vs_dna_fm_step340000_eqtl_delta_by_tissue",
        "NTv3 vs DNA-FM step340000 eQTL differences across 49 GTEx tissues",
    )

    plt = _pyplot()
    figure, axes = plt.subplots(3, 2, figsize=(15, 14), sharey=False)
    for row, task in enumerate(MATCHED_TASKS):
        task_table = matched_delta[matched_delta.task == task]
        distances = sorted(task_table.max_distance.unique())
        labels = [f"{distance / 1000:g} kb" for distance in distances]
        selected = task_table.set_index("max_distance").reindex(distances)
        for column, (metric, metric_title) in enumerate(MATCHED_METRICS.items()):
            ax = axes[row, column]
            values = selected[f"delta_{metric}"].to_numpy()
            colors = [
                MODEL_COLORS[DNA_FM_NEW] if value >= 0 else MODEL_COLORS["NTv3"]
                for value in values
            ]
            ax.bar(labels, values, width=0.72, color=colors, edgecolor="white")
            ax.axhline(0.0, color="#333333", linewidth=0.9)
            limit = max(float(np.nanmax(np.abs(values))) * 1.18, 0.01)
            ax.set_ylim(-limit, limit)
            ax.grid(axis="y", alpha=0.22, linewidth=0.6)
            ax.set_ylabel(f"Δ {metric_title}")
            ax.set_title(f"{task.upper()} — {metric_title}")
    figure.suptitle(
        "NTv3 vs DNA-FM step340000 matched-QTL differences "
        "(Δ = DNA-FM step340000 − NTv3)",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(
        figure, output_dir, "ntv3_vs_dna_fm_step340000_matched_qtl_delta"
    )
    plt.close(figure)


def plot_dna_fm_checkpoint_overview(
    eqtl_tissue: pd.DataFrame, matched: pd.DataFrame, output_dir: Path
) -> None:
    """Plot the two DNA-FM checkpoints with new-minus-old annotations."""
    plt = _pyplot()
    overview = build_overview_table(eqtl_tissue, matched)
    overview = overview[overview.model.isin({DNA_FM_OLD, DNA_FM_NEW})]
    categories = ["eQTL", "sQTL", "paQTL", "iPaQTL"]
    x = np.arange(len(categories), dtype=float)
    width = 0.36
    figure, ax = plt.subplots(figsize=(11, 6.2))
    for offset, model in zip((-width / 2, width / 2), (DNA_FM_OLD, DNA_FM_NEW)):
        selected = (
            overview[overview.model == model].set_index("benchmark").reindex(categories)
        )
        ax.bar(
            x + offset,
            selected.value.to_numpy(),
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.4,
        )
    paired = _paired_model_delta(
        overview,
        ["benchmark"],
        ["value"],
        baseline_model=DNA_FM_OLD,
        comparison_model=DNA_FM_NEW,
        baseline_prefix="step150000",
        comparison_prefix="step340000",
    ).set_index("benchmark").reindex(categories)
    for index, row in enumerate(paired.itertuples()):
        ax.text(
            x[index],
            max(row.step150000_value, row.step340000_value) + 0.012,
            f"Δ {row.delta_value:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(x, categories)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("AUROC")
    ax.set_title("DNA-FM checkpoint comparison across four QTL benchmarks", pad=30)
    ax.text(
        0.5,
        1.01,
        "Δ = step340000 − step150000; eQTL uses tissue-macro causal AUROC; matched QTL uses 10 kb",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="center",
    )
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    figure.subplots_adjust(bottom=0.22)
    _save_figure(figure, output_dir, "dna_fm_checkpoint_overview")
    plt.close(figure)


def plot_dna_fm_checkpoint_differences(
    tissue_delta: pd.DataFrame,
    organ_delta: pd.DataFrame,
    matched_delta: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot step340000-minus-step150000 paired metric differences."""
    delta_description = "Metric difference Δ = step340000 − step150000"
    options = {
        "positive_model": DNA_FM_NEW,
        "negative_model": DNA_FM_OLD,
        "delta_description": delta_description,
    }
    _plot_eqtl_delta_grid(
        organ_delta,
        "organ_group",
        output_dir,
        "dna_fm_checkpoint_eqtl_delta_by_organ",
        "DNA-FM checkpoint eQTL differences by broad organ group",
        **options,
    )
    _plot_eqtl_delta_grid(
        tissue_delta,
        "tissue",
        output_dir,
        "dna_fm_checkpoint_eqtl_delta_by_tissue",
        "DNA-FM checkpoint eQTL differences across 49 GTEx tissues",
        **options,
    )

    plt = _pyplot()
    figure, axes = plt.subplots(3, 2, figsize=(15, 14), sharey=False)
    for row, task in enumerate(MATCHED_TASKS):
        task_table = matched_delta[matched_delta.task == task]
        distances = sorted(task_table.max_distance.unique())
        labels = [f"{distance / 1000:g} kb" for distance in distances]
        selected = task_table.set_index("max_distance").reindex(distances)
        for column, (metric, metric_title) in enumerate(MATCHED_METRICS.items()):
            ax = axes[row, column]
            values = selected[f"delta_{metric}"].to_numpy()
            colors = [
                MODEL_COLORS[DNA_FM_NEW]
                if value >= 0
                else MODEL_COLORS[DNA_FM_OLD]
                for value in values
            ]
            ax.bar(labels, values, width=0.72, color=colors, edgecolor="white")
            ax.axhline(0.0, color="#333333", linewidth=0.9)
            limit = max(float(np.nanmax(np.abs(values))) * 1.18, 0.01)
            ax.set_ylim(-limit, limit)
            ax.grid(axis="y", alpha=0.22, linewidth=0.6)
            ax.set_ylabel(f"Δ {metric_title}")
            ax.set_title(f"{task.upper()} — {metric_title}")
    figure.suptitle(
        "DNA-FM matched-QTL checkpoint differences (Δ = step340000 − step150000)",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(figure, output_dir, "dna_fm_checkpoint_matched_qtl_delta")
    plt.close(figure)


def write_outputs(root: Path, output_dir: Path) -> dict[str, object]:
    paths = default_result_paths(root)
    eqtl, matched = load_metric_tables(paths)
    eqtl_tissue, eqtl_organ, matched_export = build_export_tables(eqtl, matched)
    tissue_delta, organ_delta, matched_delta = build_ntv3_dna_fm_deltas(
        eqtl_tissue, eqtl_organ, matched_export
    )
    checkpoint_tissue_delta, checkpoint_organ_delta, checkpoint_matched_delta = (
        build_dna_fm_checkpoint_deltas(eqtl_tissue, eqtl_organ, matched_export)
    )
    overview = build_overview_table(eqtl_tissue, matched_export)
    ntv3_dna_fm_overview = _paired_model_delta(overview, ["benchmark"], ["value"])
    checkpoint_overview = _paired_model_delta(
        overview,
        ["benchmark"],
        ["value"],
        baseline_model=DNA_FM_OLD,
        comparison_model=DNA_FM_NEW,
        baseline_prefix="step150000",
        comparison_prefix="step340000",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    eqtl_tissue.to_csv(output_dir / "eqtl_by_tissue.tsv", sep="\t", index=False)
    eqtl_organ.to_csv(output_dir / "eqtl_by_organ.tsv", sep="\t", index=False)
    matched_export.to_csv(
        output_dir / "matched_qtl_by_distance.tsv", sep="\t", index=False
    )
    overview.to_csv(output_dir / "all_qtl_overview.tsv", sep="\t", index=False)
    ntv3_dna_fm_overview.to_csv(
        output_dir / "ntv3_vs_dna_fm_step340000_overview.tsv", sep="\t", index=False
    )
    organ_delta.to_csv(
        output_dir / "ntv3_vs_dna_fm_step340000_eqtl_delta_by_organ.tsv",
        sep="\t",
        index=False,
    )
    tissue_delta.to_csv(
        output_dir / "ntv3_vs_dna_fm_step340000_eqtl_delta_by_tissue.tsv",
        sep="\t",
        index=False,
    )
    matched_delta.to_csv(
        output_dir / "ntv3_vs_dna_fm_step340000_matched_qtl_delta.tsv",
        sep="\t",
        index=False,
    )
    checkpoint_overview.to_csv(
        output_dir / "dna_fm_checkpoint_overview.tsv", sep="\t", index=False
    )
    checkpoint_organ_delta.to_csv(
        output_dir / "dna_fm_checkpoint_eqtl_delta_by_organ.tsv",
        sep="\t",
        index=False,
    )
    checkpoint_tissue_delta.to_csv(
        output_dir / "dna_fm_checkpoint_eqtl_delta_by_tissue.tsv",
        sep="\t",
        index=False,
    )
    checkpoint_matched_delta.to_csv(
        output_dir / "dna_fm_checkpoint_matched_qtl_delta.tsv",
        sep="\t",
        index=False,
    )

    plot_overview(eqtl_tissue, matched_export, output_dir)
    plot_eqtl(eqtl_tissue, eqtl_organ, output_dir)
    plot_matched_qtl(matched_export, output_dir)
    plot_ntv3_dna_fm_overview(eqtl_tissue, matched_export, output_dir)
    plot_ntv3_dna_fm_side_by_side(eqtl_tissue, eqtl_organ, matched_export, output_dir)
    plot_ntv3_dna_fm_differences(tissue_delta, organ_delta, matched_delta, output_dir)
    plot_dna_fm_checkpoint_overview(eqtl_tissue, matched_export, output_dir)
    plot_dna_fm_checkpoint_differences(
        checkpoint_tissue_delta,
        checkpoint_organ_delta,
        checkpoint_matched_delta,
        output_dir,
    )

    manifest = {
        "models": list(MODEL_ORDER),
        "n_eqtl_tissues": int(eqtl_tissue.tissue.nunique()),
        "n_eqtl_organ_groups": int(eqtl_organ.organ_group.nunique()),
        "eqtl_organ_aggregation": "unweighted macro mean of tissue metrics",
        "ntv3_vs_dna_fm_delta": "DNA-FM step340000 metric minus NTv3 metric",
        "dna_fm_checkpoint_delta": "step340000 metric minus step150000 metric",
        "overview": {
            "eqtl": "macro mean of tissue causal classification AUROC",
            "sqtl_paqtl_ipaqtl": "mean AUROC at the largest (10000 bp) matching distance",
        },
        "source_metrics": {
            model.name: {task: str(path) for task, path in model.items()}
            for model in paths
        },
        "files": [
            "all_qtl_overview.png",
            "all_qtl_overview.pdf",
            "all_qtl_overview.tsv",
            "eqtl_by_organ.png",
            "eqtl_by_organ.pdf",
            "eqtl_by_tissue.png",
            "eqtl_by_tissue.pdf",
            "matched_qtl_by_distance.png",
            "matched_qtl_by_distance.pdf",
            "eqtl_by_tissue.tsv",
            "eqtl_by_organ.tsv",
            "matched_qtl_by_distance.tsv",
            "ntv3_vs_dna_fm_step340000_overview.png",
            "ntv3_vs_dna_fm_step340000_overview.pdf",
            "ntv3_vs_dna_fm_step340000_overview.tsv",
            "ntv3_vs_dna_fm_step340000_eqtl_by_organ.png",
            "ntv3_vs_dna_fm_step340000_eqtl_by_organ.pdf",
            "ntv3_vs_dna_fm_step340000_eqtl_by_tissue.png",
            "ntv3_vs_dna_fm_step340000_eqtl_by_tissue.pdf",
            "ntv3_vs_dna_fm_step340000_matched_qtl_by_distance.png",
            "ntv3_vs_dna_fm_step340000_matched_qtl_by_distance.pdf",
            "ntv3_vs_dna_fm_step340000_eqtl_delta_by_organ.png",
            "ntv3_vs_dna_fm_step340000_eqtl_delta_by_organ.pdf",
            "ntv3_vs_dna_fm_step340000_eqtl_delta_by_organ.tsv",
            "ntv3_vs_dna_fm_step340000_eqtl_delta_by_tissue.png",
            "ntv3_vs_dna_fm_step340000_eqtl_delta_by_tissue.pdf",
            "ntv3_vs_dna_fm_step340000_eqtl_delta_by_tissue.tsv",
            "ntv3_vs_dna_fm_step340000_matched_qtl_delta.png",
            "ntv3_vs_dna_fm_step340000_matched_qtl_delta.pdf",
            "ntv3_vs_dna_fm_step340000_matched_qtl_delta.tsv",
            "dna_fm_checkpoint_overview.png",
            "dna_fm_checkpoint_overview.pdf",
            "dna_fm_checkpoint_overview.tsv",
            "dna_fm_checkpoint_eqtl_delta_by_organ.png",
            "dna_fm_checkpoint_eqtl_delta_by_organ.pdf",
            "dna_fm_checkpoint_eqtl_delta_by_organ.tsv",
            "dna_fm_checkpoint_eqtl_delta_by_tissue.png",
            "dna_fm_checkpoint_eqtl_delta_by_tissue.pdf",
            "dna_fm_checkpoint_eqtl_delta_by_tissue.tsv",
            "dna_fm_checkpoint_matched_qtl_delta.png",
            "dna_fm_checkpoint_matched_qtl_delta.pdf",
            "dna_fm_checkpoint_matched_qtl_delta.tsv",
            "manifest.json",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (default: inferred from this module)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot directory (default: <root>/outputs/five_result_qtl_plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "outputs/five_result_qtl_plots"
    )
    manifest = write_outputs(root, output_dir)
    print(
        f"[done] models={','.join(manifest['models'])} "
        f"tissues={manifest['n_eqtl_tissues']} "
        f"organ_groups={manifest['n_eqtl_organ_groups']}"
    )
    print(f"[done] figures and tables: {output_dir}")


if __name__ == "__main__":
    main()
