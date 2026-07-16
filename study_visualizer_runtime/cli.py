"""Deprecated ``exact-study-viz`` runtime entry point."""

from __future__ import annotations

import sys
import warnings
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    warnings.warn(
        "study_visualizer_runtime was renamed to exact-inspect; use `exact-inspect serve`.",
        DeprecationWarning,
        stacklevel=2,
    )
    from exact_inspect.cli import main as inspect_main

    return inspect_main(["serve", *(list(argv) if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
