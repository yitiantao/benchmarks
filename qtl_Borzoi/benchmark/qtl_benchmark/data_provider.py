"""Model-independent readers and batching for downloaded QTL VCF files."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from .model_base import QTLModel, TaskContext, VariantPrediction, VariantRecord


@dataclass(frozen=True)
class TaskSpec:
    name: str
    distance_info_key: str | None
    matched: bool
    distance_thresholds: tuple[int, ...]


TASK_SPECS = {
    "eqtl": TaskSpec("eqtl", None, False, ()),
    "sqtl": TaskSpec("sqtl", "SD", True, (10000, 2000, 500, 200, 100, 50)),
    "paqtl": TaskSpec("paqtl", "PD", True, (10000, 2000, 500, 300, 200, 100, 50)),
    "ipaqtl": TaskSpec("ipaqtl", "PD", True, (10000, 2000, 500, 300, 200, 100, 50)),
}


@dataclass(frozen=True)
class QTLTaskDataset:
    task: str
    data_dir: Path
    positive: tuple[VariantRecord, ...]
    negative: tuple[VariantRecord, ...]
    spec: TaskSpec


def parse_info(value: str) -> dict[str, str]:
    parsed = {}
    for field in value.split(";"):
        if "=" in field:
            key, item = field.split("=", 1)
            parsed[key] = item
    return parsed


def find_merged_vcf(data_dir: Path, task: str, split: str) -> Path:
    base = data_dir / task / f"{split}_merge.vcf"
    candidates = (base, Path(str(base) + ".gz"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {task} {split} VCF; tried: {candidates}")


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def read_variant_file(vcf_file: Path | str, task: str, split: str) -> tuple[VariantRecord, ...]:
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task {task!r}")
    if split not in {"pos", "neg"}:
        raise ValueError("split must be 'pos' or 'neg'")

    spec = TASK_SPECS[task]
    vcf_file = Path(vcf_file)
    records = []
    with _open_text(vcf_file) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            info = parse_info(fields[7]) if len(fields) >= 8 else {}
            trait = info.get("MT")
            distance = info.get(spec.distance_info_key) if spec.distance_info_key else None
            records.append(
                VariantRecord(
                    task=task,
                    split=split,
                    chrom=fields[0],
                    pos=int(fields[1]),
                    variant_id=fields[2],
                    ref=fields[3],
                    alt=fields[4],
                    gene_id=trait.split(".")[0] if trait else None,
                    distance=int(distance) if distance is not None else None,
                    matched_positive_id=info.get("PI"),
                    info=info,
                )
            )
    return tuple(records)


def read_variants(data_dir: Path | str, task: str, split: str) -> tuple[VariantRecord, ...]:
    vcf_file = find_merged_vcf(Path(data_dir), task, split)
    return read_variant_file(vcf_file, task, split)


def load_task_dataset(
    data_dir: Path | str,
    task: str,
    max_variants: int | None = None,
) -> QTLTaskDataset:
    if max_variants is not None and max_variants < 0:
        raise ValueError("max_variants must be 0 or greater")
    positive = read_variants(data_dir, task, "pos")
    negative = read_variants(data_dir, task, "neg")
    if max_variants:
        positive = positive[:max_variants]
        negative = negative[:max_variants]
    return QTLTaskDataset(
        task=task,
        data_dir=Path(data_dir).resolve(),
        positive=positive,
        negative=negative,
        spec=TASK_SPECS[task],
    )


def load_task_dataset_from_vcf_dir(vcf_dir: Path | str, task: str) -> QTLTaskDataset:
    """Load a task from a prepared directory containing uncompressed merged VCFs."""
    vcf_dir = Path(vcf_dir).resolve()
    return QTLTaskDataset(
        task=task,
        data_dir=vcf_dir,
        positive=read_variant_file(vcf_dir / "pos_merge.vcf", task, "pos"),
        negative=read_variant_file(vcf_dir / "neg_merge.vcf", task, "neg"),
        spec=TASK_SPECS[task],
    )


def iter_batches(records: Sequence[VariantRecord], batch_size: int) -> Iterator[Sequence[VariantRecord]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _validate_batch(
    variants: Sequence[VariantRecord], predictions: Sequence[VariantPrediction]
) -> None:
    if len(predictions) != len(variants):
        raise ValueError(
            f"Model returned {len(predictions)} predictions for {len(variants)} variants"
        )
    expected = [variant.variant_id for variant in variants]
    observed = [prediction.variant_id for prediction in predictions]
    if expected != observed:
        raise ValueError("Prediction IDs/order do not match the input batch")


def predict_split(
    model: QTLModel,
    records: Sequence[VariantRecord],
    context: TaskContext,
    batch_size: int,
) -> pd.DataFrame:
    rows = []
    for batch in iter_batches(records, batch_size):
        predictions = tuple(model.predict_batch(context.task, batch, context))
        _validate_batch(batch, predictions)
        for variant, prediction in zip(batch, predictions):
            row = {
                "task": context.task,
                "split": variant.split,
                "variant_id": prediction.variant_id,
                "score": float(prediction.score),
            }
            overlap = set(row).intersection(prediction.metadata)
            if overlap:
                raise ValueError(
                    f"Prediction metadata uses reserved columns: {sorted(overlap)}"
                )
            row.update(prediction.metadata)
            rows.append(row)
    return pd.DataFrame(rows)


def run_model_on_dataset(
    model: QTLModel,
    dataset: QTLTaskDataset,
    output_dir: Path | str,
    batch_size: int = 64,
    options: dict | None = None,
) -> pd.DataFrame:
    """Feed standardized QTL data to a model and return normalized predictions."""
    model.validate_task(dataset.task)
    context = TaskContext(
        task=dataset.task,
        data_dir=dataset.data_dir,
        output_dir=Path(output_dir).resolve(),
        options=options or {},
    )
    context.output_dir.mkdir(parents=True, exist_ok=True)
    model.setup_task(context)
    try:
        positive = predict_split(model, dataset.positive, context, batch_size)
        negative = predict_split(model, dataset.negative, context, batch_size)
    finally:
        model.teardown_task(context)
    return pd.concat([positive, negative], ignore_index=True)
