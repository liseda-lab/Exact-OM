"""Deprecated compatibility alias for :mod:`exact.llm.routing`."""

from __future__ import annotations

import sys
import warnings

from exact.llm import routing as _routing

warnings.warn(
    "exact.utils.llm_routing is deprecated; import exact.llm.routing instead",
    DeprecationWarning,
    stacklevel=2,
)

# A module alias, rather than copied attributes, preserves monkeypatching and
# identity for callers that still use the legacy import path.
sys.modules[__name__] = _routing
