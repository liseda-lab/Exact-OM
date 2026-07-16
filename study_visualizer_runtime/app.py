"""Deprecated compatibility facade for :mod:`exact_inspect`."""

from __future__ import annotations

import warnings

warnings.warn(
    "study_visualizer_runtime was renamed to exact_inspect and will be removed in Exact-OM 2.1.",
    DeprecationWarning,
    stacklevel=2,
)

from exact_inspect.app import create_app, create_study_visualizer_app  # noqa: E402
from exact_inspect.bundles import (  # noqa: E402
    PrecomputedOntologyLookup,
    StudyVisualizerService,
)

__all__ = [
    "PrecomputedOntologyLookup",
    "StudyVisualizerService",
    "create_app",
    "create_study_visualizer_app",
]
