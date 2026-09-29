"""Declarative component loading for the unified benchmark CLI."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from .interfaces import Evaluator, ModelAdapter, TidyVariantModelAdapter
from .model_base import QTLModel
from .runner import BenchmarkRunner


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def import_class(specification: str):
    if ":" not in specification:
        raise ValueError(f"Class specification must be module:Class, got {specification!r}")
    module, name = specification.split(":", 1)
    return getattr(importlib.import_module(module), name)


def _component(specification: Mapping[str, Any]):
    allowed = {"class", "config"}
    unknown = set(specification) - allowed
    if unknown:
        raise ValueError(f"Unknown component keys: {sorted(unknown)}")
    return import_class(str(specification["class"]))(**dict(specification.get("config", {})))


def build_runner(config: Mapping[str, Any]) -> BenchmarkRunner:
    data = _component(config["data"])
    model_spec = config["model"]
    model = _component(model_spec)
    if isinstance(model, QTLModel):
        model = TidyVariantModelAdapter(
            model,
            model_class=str(model_spec["class"]),
            model_config=dict(model_spec.get("config", {})),
        )
    if not isinstance(model, ModelAdapter):
        raise TypeError("model must be ModelAdapter or a tidy-table QTLModel")
    evaluator = _component(config["evaluator"])
    if not isinstance(evaluator, Evaluator):
        raise TypeError("evaluator must subclass Evaluator")
    run = dict(config.get("run", {}))
    return BenchmarkRunner(
        data=data,
        model=model,
        evaluator=evaluator,
        output_dir=run.pop("output_dir", "outputs/benchmark"),
        options=run.pop("options", {}),
    )
