"""Backward-compatible dispatcher for Exact command groups."""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None):
    """Dispatch an optional leading command while preserving flat alignment flags."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "data":
        from exact.delivery.cli.data import main as data_main

        return data_main(arguments[1:])

    from exact.delivery.cli.align import main as align_main

    return align_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
