"""One orchestration path for every dataset/model/evaluator combination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .core import BenchmarkContext
from .data.base import DataModule
from .interfaces import Evaluator, ModelAdapter
from .predictions import PredictionStore


def _json_metric_value(value: Any) -> Any:
    """Convert pandas/numpy scalars into strict JSON values for metric logs."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _metric_records(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _json_metric_value(value) for column, value in row.items()}
        for row in metrics.to_dict(orient="records")
    ]


class BenchmarkRunner:
    """Coordinate preparation, inference, persistence, and evaluation."""

    def __init__(
        self,
        data: DataModule,
        model: ModelAdapter,
        evaluator: Evaluator,
        output_dir: str | Path,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.data = data
        self.model = model
        self.evaluator = evaluator
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.options = dict(options or {})

    def run(
        self,
        predict: bool = True,
        evaluate: bool = True,
        force: bool = False,
    ) -> pd.DataFrame | None:
        if not predict and not evaluate:
            raise ValueError("At least one of predict/evaluate must be enabled")
        dataset = self.data.load()
        run_dir = self.output_dir / self.model.name / dataset.name
        run_dir.mkdir(parents=True, exist_ok=True)
        context = BenchmarkContext(run_dir, self.options)
        manifest = {
            "dataset": dataset.summary(),
            "model": self.model.name,
            "model_identity": self.model.fingerprint(),
            "evaluator": self.evaluator.name,
            "predict": predict,
            "evaluate": evaluate,
        }
        (run_dir / "run.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        with PredictionStore(run_dir / "predictions.sqlite") as store:
            store.ensure_identity(self.model.fingerprint())
            if force:
                store.clear(list(dataset.variants))
            if predict:
                self.model.predict(dataset.for_model(), store, context)
            predictions = store.read()
            # Parallel prediction workers share the WAL database. Only the
            # final evaluation process exports the complete snapshot.
            if evaluate:
                store.export(run_dir / "predictions.tsv.gz")

        if not evaluate:
            return None
        metrics = self.evaluator.evaluate(dataset, predictions, run_dir)
        metrics_tsv = run_dir / "metrics.tsv"
        metrics_json = run_dir / "metrics.json"
        metrics.to_csv(metrics_tsv, sep="\t", index=False, float_format="%.6f")
        records = _metric_records(metrics)
        metrics_json.write_text(
            json.dumps(
                {
                    "model": self.model.name,
                    "dataset": dataset.name,
                    "task": dataset.task,
                    "evaluator": self.evaluator.name,
                    "columns": list(metrics.columns),
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        print(
            f"[metrics] model={self.model.name} dataset={dataset.name} "
            f"task={dataset.task} evaluator={self.evaluator.name} rows={len(records)} "
            f"columns={','.join(map(str, metrics.columns))}",
            flush=True,
        )
        for row_index, record in enumerate(records, start=1):
            # One complete JSON object per row is grep-friendly and avoids
            # pandas display truncation for many eQTL tissues/metric columns.
            print(
                "[metric] "
                + json.dumps(
                    {
                        "model": self.model.name,
                        "dataset": dataset.name,
                        "task": dataset.task,
                        "evaluator": self.evaluator.name,
                        "row": row_index,
                        "values": record,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                flush=True,
            )
        print(
            f"[metrics] tsv={metrics_tsv} json={metrics_json}",
            flush=True,
        )
        return metrics
