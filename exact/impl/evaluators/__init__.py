"""Built-in evaluator backends.

Importing this package registers the in-tree backends.  The optional BioML
dependency itself is still imported only when that backend is selected.
"""

from .bioml import BioMLEvaluator
from .builtin import BuiltinEvaluator

__all__ = ["BioMLEvaluator", "BuiltinEvaluator"]
