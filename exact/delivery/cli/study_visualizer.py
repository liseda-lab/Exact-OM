"""Deprecated ``exact-study-viz`` console-script shim."""

from __future__ import annotations

import sys
import warnings
from typing import Sequence


def _without_jvm_option(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    iterator = iter(argv)
    for item in iterator:
        if item == "--jvm-heap-size":
            next(iterator, None)
            warnings.warn(
                "--jvm-heap-size is ignored because Exact-OM no longer needs Java.",
                DeprecationWarning,
                stacklevel=3,
            )
            continue
        cleaned.append(item)
    return cleaned


def main(argv: Sequence[str] | None = None) -> int:
    warnings.warn(
        "exact-study-viz was renamed to exact-inspect; delegating to `exact-inspect serve`.",
        DeprecationWarning,
        stacklevel=2,
    )
    from exact_inspect.cli import main as inspect_main

    arguments = list(argv) if argv is not None else sys.argv[1:]
    return inspect_main(["serve", *_without_jvm_option(arguments)])


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
