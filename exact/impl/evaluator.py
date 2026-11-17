"""Compatibility imports for the evaluator module moved in Exact 2.0."""

from .evaluators.builtin import BuiltinEvaluator

Evaluator = BuiltinEvaluator

__all__ = ["BuiltinEvaluator", "Evaluator"]
