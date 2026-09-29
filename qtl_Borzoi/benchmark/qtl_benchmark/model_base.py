"""Public model interface for extensible QTL benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_TASKS = frozenset({"eqtl", "sqtl", "paqtl", "ipaqtl"})


@dataclass(frozen=True)
class VariantRecord:
    """One standardized variant passed to a model adapter."""

    task: str
    split: str
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: str
    gene_id: str | None = None
    tissue: str | None = None
    distance: int | None = None
    matched_positive_id: str | None = None
    info: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskContext:
    """Paths and free-form options available to a task-specific adapter."""

    task: str
    data_dir: Path
    output_dir: Path
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantPrediction:
    """Normalized scalar prediction consumed by the metric layer.

    ``score`` should be signed when the model has a meaningful direction. The
    causal/non-causal metrics use its magnitude. Task-specific values (for
    example, per-tissue scores) can be placed in ``metadata`` and summarized by
    the adapter before returning the primary scalar score.
    """

    variant_id: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class QTLModel(ABC):
    """Base class that every model adapter implements.

    The benchmark owns VCF parsing, batching, validation and metrics. A model
    adapter only converts a batch of standardized variants into one scalar
    prediction for each input variant.
    """

    name = "unnamed-model"
    supported_tasks = SUPPORTED_TASKS

    def setup_task(self, context: TaskContext) -> None:
        """Optional hook called once before predictions for a task."""

    @abstractmethod
    def predict_batch(
        self,
        task: str,
        variants: Sequence[VariantRecord],
        context: TaskContext,
    ) -> Sequence[VariantPrediction]:
        """Return exactly one prediction for every variant in ``variants``."""

    def teardown_task(self, context: TaskContext) -> None:
        """Optional hook called once after predictions for a task."""

    def validate_task(self, task: str) -> None:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unknown QTL task: {task}")
        if task not in self.supported_tasks:
            supported = ", ".join(sorted(self.supported_tasks))
            raise ValueError(f"{self.name} does not support {task}; supported: {supported}")


class TaskSpecificQTLModel(QTLModel):
    """Convenience base that dispatches to ``predict_<task>`` methods.

    This lets adapters expose different internal logic for expression,
    splicing and polyadenylation while retaining one normalized output schema.
    """

    def predict_batch(
        self,
        task: str,
        variants: Sequence[VariantRecord],
        context: TaskContext,
    ) -> Sequence[VariantPrediction]:
        method = getattr(self, f"predict_{task}", None)
        if method is None:
            raise NotImplementedError(f"{self.name} does not implement predict_{task}()")
        return method(variants, context)
