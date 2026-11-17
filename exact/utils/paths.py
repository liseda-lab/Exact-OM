"""Deprecated compatibility exports for :mod:`exact.utils.graph_search`."""

import warnings

warnings.warn(
    "exact.utils.paths was renamed to exact.utils.graph_search; update imports before Exact-OM 2.1",
    DeprecationWarning,
    stacklevel=2,
)

from exact.utils.graph_search import *  # noqa: E402,F401,F403
