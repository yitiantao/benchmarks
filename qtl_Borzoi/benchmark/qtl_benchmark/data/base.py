"""Data-module interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core import BenchmarkDataset


class DataModule(ABC):
    """Build a frozen, model-independent benchmark dataset."""

    @abstractmethod
    def load(self) -> BenchmarkDataset:
        """Read, validate, split, and return the test data."""
