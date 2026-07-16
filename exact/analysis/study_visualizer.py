"""Deprecated compatibility imports for the renamed inspection package."""

from __future__ import annotations

import logging
import warnings
from typing import Any

warnings.warn(
    "exact.analysis.study_visualizer moved to exact_inspect and will be removed in Exact-OM 2.1.",
    DeprecationWarning,
    stacklevel=2,
)

from exact_inspect.app import create_app, create_study_visualizer_app  # noqa: E402
from exact_inspect.bundles import (  # noqa: E402
    InspectionService,
    PrecomputedOntologyLookup,
    StudyOntologyLookup,
    StudyVisualizerService,
    export_bundle,
)
from exact_inspect.settings import InspectSettings  # noqa: E402


def create_app_from_env(logger: logging.Logger | None = None) -> Any:
    if logger is not None:
        logging.getLogger("exact_inspect").setLevel(logger.level)
    return create_app(InspectSettings())


__all__ = [
    "InspectionService",
    "PrecomputedOntologyLookup",
    "StudyOntologyLookup",
    "StudyVisualizerService",
    "create_app",
    "create_app_from_env",
    "create_study_visualizer_app",
    "export_bundle",
]
