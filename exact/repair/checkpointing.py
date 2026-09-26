"""Small durable time ledger shared by repair stages and replacement processes."""

from __future__ import annotations

import fcntl
import json
import math
import time
from pathlib import Path
from typing import Any

from .api import write_artifact


class CumulativeBudget:
    """Debit actual elapsed time, reserving an upper bound before blocking work.

    A lost process is charged its outstanding reservation, never a fresh budget.
    Use small reservations (one teacher case or study arm) where possible. An
    ungracefully lost training stage conservatively consumes its remaining time.
    The enclosing Slurm worker enforces the reserved wall-time limit externally.
    """

    def __init__(self, path: Path, identity: str, seconds: float):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("A positive finite cumulative budget is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = path.with_suffix(".lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state: dict[str, Any] = (
            json.loads(path.read_text())
            if path.exists()
            else dict(identity=identity, limit_seconds=seconds, spent_seconds=0.0, attempts=[])
        )
        if self.state["identity"] != identity or self.state["limit_seconds"] != seconds:
            self.close()
            raise ValueError("Cumulative budget identity/limit changed; an amendment is required")
        if self.state.get("active"):
            reservation = self.state.pop("active")
            self.state["spent_seconds"] += reservation["reserved_seconds"]
            self.state["attempts"].append({**reservation, "status": "lost_reserved_cost"})
        self.started: float | None = None
        self._save()

    def _save(self):
        write_artifact(self.path, self.state)

    @property
    def remaining(self) -> float:
        active = 0.0 if self.started is None else time.monotonic() - self.started
        return max(
            0.0, float(self.state["limit_seconds"]) - float(self.state["spent_seconds"]) - active
        )

    def begin(self, seconds: float | None = None) -> float:
        if self.started is not None:
            raise RuntimeError("Budget already has active work")
        reserved = min(self.remaining, seconds if seconds is not None else self.remaining)
        if reserved <= 0:
            raise TimeoutError("Cumulative resource budget exhausted")
        self.state["active"] = dict(reserved_seconds=reserved, started_epoch=time.time())
        self._save()
        self.started = time.monotonic()
        return reserved

    def finish(self, status="complete"):
        if self.started is not None:
            elapsed = time.monotonic() - self.started
            self.started = None
            self.state["spent_seconds"] += elapsed
            self.state["attempts"].append(
                {**self.state.pop("active"), "elapsed_seconds": elapsed, "status": status}
            )
            self._save()

    def close(self):
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.finish("complete" if kind is None else "interrupted")
        self.close()
