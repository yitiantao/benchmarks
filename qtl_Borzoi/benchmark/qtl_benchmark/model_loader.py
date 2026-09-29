"""Dynamic loading helpers shared by prediction and combined CLIs."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .model_base import QTLModel


def load_config(value: str | None) -> dict:
    if value is None:
        return {}
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads(value)


def load_model(specification: str, config: dict) -> QTLModel:
    if ":" not in specification:
        raise ValueError("--model-class must have the form python.module:ClassName")
    module_name, class_name = specification.split(":", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class(**config)
    if not isinstance(model, QTLModel):
        raise TypeError(f"{specification} must subclass QTLModel")
    return model
