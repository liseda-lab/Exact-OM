"""Deprecated import shim for the source-neutral I/O hierarchy utility.

OWL hierarchy queries use shared :mod:`pyowl_core` views in ``store.py``.
"""

from exact.io._hierarchy import HierarchyIndex

__all__ = ["HierarchyIndex"]
