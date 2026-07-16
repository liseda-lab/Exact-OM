"""Backward-compatible dispatcher for Exact command groups."""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None):
    """Dispatch an optional leading command while preserving flat alignment flags."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        command = arguments[0]
        if command == "data":
            from exact.delivery.cli.data import main as data_main

            return data_main(arguments[1:])
        if command == "config":
            from exact.delivery.cli.config import main as config_main

            return config_main(arguments[1:])
        if command == "run":
            from exact.delivery.cli.run import main as run_main

            return run_main(arguments[1:])
        if command == "align":
            arguments = arguments[1:]

    from exact.delivery.cli.align import main as align_main

    return align_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
