"""Compatibility module for the decomposed semantic alignment runner."""

from exact.core.entities.configs.dataset import DatasetMask
from exact.impl.trainer.runner import SemanticAlignmentRunner

__all__ = ["DatasetMask", "SemanticAlignmentRunner"]
