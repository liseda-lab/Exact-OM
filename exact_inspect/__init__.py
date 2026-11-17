"""Optional alignment inspection service for Exact-OM."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exact-om")
except PackageNotFoundError:  # Source checkout.
    __version__ = "0+unknown"

__all__ = ["__version__"]
