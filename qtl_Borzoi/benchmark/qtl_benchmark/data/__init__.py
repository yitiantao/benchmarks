"""Benchmark data modules."""

from .base import DataModule
from .gtex_eqtl import GTExEQTLDataModule
from .matched_qtl import MatchedQTLDataModule

__all__ = ["DataModule", "GTExEQTLDataModule", "MatchedQTLDataModule"]
