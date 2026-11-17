"""Private verified-wire process entry point for hierarchy classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyowl_core

from exact.ontology.reasoning import ReasonerSettings, _worker_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wire", required=True, type=Path)
    parser.add_argument("--reasoner", required=True, choices=("elk", "hermit"))
    parser.add_argument("--backend", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--iri", required=True)
    parser.add_argument("--direction", required=True, choices=("up", "down"))
    parser.add_argument("--direct", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    # This is deliberately the only ontology acquisition call in the worker.
    # ``load_snapshot`` and original source paths never enter this process.
    snapshot = pyowl_core.open_snapshot(args.wire, mmap=True, verify=True)
    try:
        settings = ReasonerSettings(
            backend=args.backend,
            workers=args.workers,
            timeout_seconds=args.timeout,
        )
        payload = _worker_payload(
            args.reasoner,
            snapshot,
            settings,
            iri=args.iri,
            upward=args.direction == "up",
            direct=args.direct,
        )
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    finally:
        close = getattr(snapshot, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess.
    raise SystemExit(main())
