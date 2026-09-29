"""Extension points for models and evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import pandas as pd

from .core import BenchmarkContext, BenchmarkDataset, InferenceDataset, variant_key
from .model_base import QTLModel, TaskContext, VariantRecord
from .predictions import PredictionStore


class ModelAdapter(ABC):
    """Model-facing interface consumed by :class:`BenchmarkRunner`."""

    name = "unnamed-model"

    def identity_payload(self) -> Mapping[str, Any]:
        """Stable configuration used to prevent cache/model mixing."""
        return {"class": f"{type(self).__module__}:{type(self).__qualname__}"}

    def fingerprint(self) -> str:
        payload = json.dumps(self.identity_payload(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @abstractmethod
    def predict(
        self,
        dataset: InferenceDataset,
        store: PredictionStore,
        context: BenchmarkContext,
    ) -> None:
        """Write standardized predictions into ``store``."""


class Evaluator(ABC):
    """Evaluation interface; implementations must not import model runtimes."""

    name = "unnamed-evaluator"

    @abstractmethod
    def evaluate(
        self,
        dataset: InferenceDataset,
        predictions: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        """Evaluate predictions and persist useful detail tables."""


class TidyVariantModelAdapter(ModelAdapter):
    """Adapt models that emit a tidy gene/tissue table for each variant.

    This is the shortest extension path for a new sequence model: implement
    ``score_variant_table(VariantRecord)`` on a ``QTLModel`` and let this class
    own setup, deduplication, caching, and schema validation.
    """

    def __init__(
        self,
        model: QTLModel,
        model_class: str | None = None,
        model_config: Mapping[str, Any] | None = None,
        batch_size: int = 1,
    ) -> None:
        score_method = getattr(model, "score_variant_table", None)
        if not callable(score_method):
            raise TypeError(
                f"{type(model).__name__} must implement score_variant_table(record)"
            )
        self.model = model
        self.name = model.name
        self.model_class = model_class or f"{type(model).__module__}:{type(model).__qualname__}"
        self.model_config = dict(model_config or {})
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def identity_payload(self) -> Mapping[str, Any]:
        return {"model_class": self.model_class, "model_config": self.model_config}

    @staticmethod
    def _normalize_model_output(table: pd.DataFrame, record: VariantRecord) -> pd.DataFrame:
        if table is None or table.empty:
            return pd.DataFrame(
                columns=["variant_key", "gene_id", "tissue", "score", "n_tracks"]
            )
        table = table.copy()
        if "score" not in table and "raw_score" in table:
            table = table.rename(columns={"raw_score": "score"})
        if "tissue" not in table and "gtex_tissue" in table:
            table = table.rename(columns={"gtex_tissue": "tissue"})
        if "gene_id" not in table:
            table["gene_id"] = record.gene_id or ""
        if "tissue" not in table:
            table["tissue"] = ""
        if "score" not in table:
            raise ValueError("Model output must contain score or raw_score")
        table["gene_id"] = table.gene_id.astype(str).str.split(".").str[0]
        table["tissue"] = table.tissue.fillna("").astype(str)
        table["score"] = pd.to_numeric(table.score, errors="coerce")
        table = table[table.score.notna() & table.gene_id.ne("")]
        if table.empty:
            return pd.DataFrame(
                columns=["variant_key", "gene_id", "tissue", "score", "n_tracks"]
            )
        return (
            table.groupby(["gene_id", "tissue"], sort=False, as_index=False)
            .agg(score=("score", "mean"), n_tracks=("score", "size"))
            .assign(variant_key=variant_key(record))
            .loc[:, ["variant_key", "gene_id", "tissue", "score", "n_tracks"]]
        )

    def predict(
        self,
        dataset: BenchmarkDataset,
        store: PredictionStore,
        context: BenchmarkContext,
    ) -> None:
        num_shards = int(context.options.get("num_shards", 1))
        shard_index = int(context.options.get("shard_index", 0))
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError(
                f"Invalid prediction shard {shard_index}/{num_shards}"
            )
        assigned = tuple(
            record
            for index, record in enumerate(dataset.variants)
            if index % num_shards == shard_index
        )
        pending = [record for record in assigned if not store.completed(record)]
        print(
            f"[predict/{self.name}] shard={shard_index + 1}/{num_shards} "
            f"variants={len(assigned)} cached={len(assigned) - len(pending)} "
            f"pending={len(pending)} batch_size={self.batch_size}",
            flush=True,
        )
        if not pending:
            return
        task_context = TaskContext(
            task=dataset.task,
            data_dir=dataset.data_dir,
            output_dir=context.output_dir,
            options=context.options,
        )
        self.model.validate_task(dataset.task)
        self.model.setup_task(task_context)
        score_method = getattr(self.model, "score_variant_table")
        batch_score_method = getattr(self.model, "score_variant_tables", None)
        try:
            for batch_start in range(0, len(pending), self.batch_size):
                batch = pending[batch_start : batch_start + self.batch_size]
                started = time.time()
                if callable(batch_score_method):
                    raw_outputs = list(batch_score_method(batch))
                else:
                    raw_outputs = [score_method(record) for record in batch]
                if len(raw_outputs) != len(batch):
                    raise RuntimeError(
                        f"{type(self.model).__name__}.score_variant_tables returned "
                        f"{len(raw_outputs)} tables for {len(batch)} variants"
                    )
                elapsed = time.time() - started
                row_count = 0
                for record, raw_output in zip(batch, raw_outputs, strict=True):
                    output = self._normalize_model_output(raw_output, record)
                    row_count += len(output)
                    store.save(record, output, elapsed / len(batch))
                completed = batch_start + len(batch)
                print(
                    f"[predict/{self.name}] {completed}/{len(pending)} "
                    f"rows={row_count} batch_elapsed={elapsed:.1f}s",
                    flush=True,
                )
        finally:
            self.model.teardown_task(task_context)
