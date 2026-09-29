"""Extensible QTL benchmark interfaces.

New integrations should use ``DataModule`` + ``ModelAdapter`` + ``Evaluator``
with ``BenchmarkRunner``.  The scalar ``QTLModel`` API remains available for
older QTL tasks and simple precomputed models.
"""

from .core import BenchmarkContext, BenchmarkDataset, InferenceDataset
from .data import DataModule, GTExEQTLDataModule, MatchedQTLDataModule
from .evaluation import GTExEQTLEvaluator, MatchedQTLEvaluator
from .interfaces import Evaluator, ModelAdapter, TidyVariantModelAdapter
from .model_base import (
    QTLModel,
    TaskContext,
    TaskSpecificQTLModel,
    VariantPrediction,
    VariantRecord,
)
from .pipeline import benchmark_model, evaluate_prediction_table, predict_model_task
from .predictions import PredictionStore
from .runner import BenchmarkRunner

__all__ = [
    "BenchmarkContext",
    "BenchmarkDataset",
    "BenchmarkRunner",
    "DataModule",
    "Evaluator",
    "GTExEQTLDataModule",
    "GTExEQTLEvaluator",
    "InferenceDataset",
    "ModelAdapter",
    "MatchedQTLDataModule",
    "MatchedQTLEvaluator",
    "PredictionStore",
    "QTLModel",
    "TaskContext",
    "TaskSpecificQTLModel",
    "VariantPrediction",
    "VariantRecord",
    "TidyVariantModelAdapter",
    "benchmark_model",
    "evaluate_prediction_table",
    "predict_model_task",
]
