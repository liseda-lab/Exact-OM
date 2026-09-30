"""Small durable resource ledger shared by repair stages and replacement processes."""

from __future__ import annotations

import fcntl
import json
import math
import time
from pathlib import Path
from typing import Any

from .api import write_artifact


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return float(value)


class CumulativeBudget:
    """Debit observed wall/CPU and allocated-GPU time, reserving before blocking work.

    A lost process retains its entire outstanding reservation. Unmeasured CPU is
    charged its declared reservation; historical wall-only ledgers remain readable.
    Whole-worker limits are enforced by the existing independent worker supervisor.
    """

    def __init__(
        self,
        path: Path,
        identity: str,
        seconds: float,
        *,
        cpu_seconds: float | None = None,
        gpu_hours: float | None = None,
        allocated_gpus: int = 0,
    ):
        _finite(seconds, "Cumulative wall budget", positive=True)
        if cpu_seconds is not None:
            _finite(cpu_seconds, "Cumulative CPU budget", positive=True)
        if gpu_hours is not None:
            _finite(gpu_hours, "Cumulative GPU budget")
        if type(allocated_gpus) is not int or allocated_gpus < 0:
            raise ValueError("allocated_gpus must be a nonnegative integer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = path.with_suffix(".lock").open("a")
        self.started: float | None = None
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state: dict[str, Any] = (
                json.loads(path.read_text())
                if path.exists()
                else dict(identity=identity, limit_seconds=seconds, spent_seconds=0.0, attempts=[])
            )
            if not path.exists() and (
                cpu_seconds is not None or gpu_hours is not None or allocated_gpus
            ):
                self.state.update(
                    limit_cpu_seconds=cpu_seconds,
                    spent_cpu_seconds=0.0,
                    limit_gpu_hours=gpu_hours,
                    spent_gpu_hours=0.0,
                    allocated_gpus=allocated_gpus,
                )
            if (
                self.state.get("limit_cpu_seconds") != cpu_seconds
                or self.state.get("limit_gpu_hours") != gpu_hours
                or self.state.get("allocated_gpus", 0) != allocated_gpus
            ):
                raise ValueError("Cumulative resource policy changed; an amendment is required")
            if self.state["identity"] != identity or self.state["limit_seconds"] != seconds:
                raise ValueError(
                    "Cumulative budget identity/limit changed; an amendment is required"
                )
            for key in ("spent_seconds", "spent_cpu_seconds", "spent_gpu_hours"):
                if key in self.state:
                    _finite(self.state[key], key)
            if not isinstance(self.state["attempts"], list):
                raise ValueError("Budget attempts must be a retained list")
            reservation = self.state.get("active")
            if reservation is not None:
                _finite(reservation["reserved_seconds"], "Reserved wall seconds", positive=True)
                if "reserved_cpu_seconds" in reservation:
                    _finite(
                        reservation["reserved_cpu_seconds"], "Reserved CPU seconds", positive=True
                    )
                elif cpu_seconds is not None:
                    raise ValueError(
                        "An active CPU-budgeted attempt lacks its retained CPU reservation"
                    )
                self.state["spent_seconds"] += reservation["reserved_seconds"]
                if "reserved_cpu_seconds" in reservation:
                    self.state["spent_cpu_seconds"] = (
                        self.state.get("spent_cpu_seconds", 0.0)
                        + reservation["reserved_cpu_seconds"]
                    )
                if "spent_gpu_hours" in self.state:
                    self.state["spent_gpu_hours"] += (
                        reservation["reserved_seconds"] * allocated_gpus / 3600
                    )
                self.state["attempts"].append(
                    {
                        **reservation,
                        "status": "lost_reserved_cost",
                        "cpu_accounting": (
                            "reserved_upper_bound"
                            if "reserved_cpu_seconds" in reservation
                            else "unavailable"
                        ),
                    }
                )
                del self.state["active"]
            self._save()
        except BaseException:
            self.close()
            raise

    def _save(self):
        write_artifact(self.path, self.state)

    @property
    def remaining(self) -> float:
        active = 0.0 if self.started is None else max(0.0, time.monotonic() - self.started)
        remaining = max(
            0.0, float(self.state["limit_seconds"]) - float(self.state["spent_seconds"]) - active
        )
        if self.state.get("limit_gpu_hours") is not None and self.state.get("allocated_gpus", 0):
            remaining = min(
                remaining,
                max(
                    0.0,
                    (self.state["limit_gpu_hours"] - self.state["spent_gpu_hours"])
                    * 3600
                    / self.state["allocated_gpus"]
                    - active,
                ),
            )
        if self.remaining_cpu is not None and self.remaining_cpu <= 0:
            return 0.0
        return remaining

    @property
    def remaining_cpu(self) -> float | None:
        limit = self.state.get("limit_cpu_seconds")
        if limit is None:
            return None
        available = max(0.0, float(limit) - float(self.state["spent_cpu_seconds"]))
        if self.started is not None and "reserved_cpu_seconds" in self.state.get("active", {}):
            return min(available, float(self.state["active"]["reserved_cpu_seconds"]))
        return available

    def begin(self, seconds: float | None = None, *, cpu_seconds: float | None = None) -> float:
        if seconds is not None:
            _finite(seconds, "Requested wall reservation", positive=True)
        if cpu_seconds is not None:
            _finite(cpu_seconds, "Requested CPU reservation", positive=True)
        if self.started is not None or "active" in self.state:
            raise RuntimeError("Budget already has active work")
        reserved = min(self.remaining, seconds if seconds is not None else self.remaining)
        if reserved <= 0:
            raise TimeoutError("Cumulative resource budget exhausted")
        reservation = dict(reserved_seconds=reserved, started_epoch=time.time())
        cpu_cap = self.remaining_cpu
        if cpu_cap is not None:
            reservation["reserved_cpu_seconds"] = (
                min(cpu_cap, cpu_seconds) if cpu_seconds is not None else cpu_cap
            )
        elif cpu_seconds is not None:
            reservation["reserved_cpu_seconds"] = cpu_seconds
        updated = {**self.state, "active": reservation}
        write_artifact(self.path, updated)
        self.state, self.started = updated, time.monotonic()
        return reserved

    def finish(self, status="complete", *, cpu_seconds: float | None = None):
        # Validate before changing active ownership or any accumulated counter.
        if cpu_seconds is not None:
            _finite(cpu_seconds, "Observed CPU usage")
        if not isinstance(status, str) or not status:
            raise ValueError("Attempt status must be a nonempty string")
        if self.started is None:
            return
        elapsed = _finite(time.monotonic() - self.started, "Observed elapsed wall time")
        reservation = dict(self.state["active"])
        used_cpu = reservation.get("reserved_cpu_seconds") if cpu_seconds is None else cpu_seconds
        if used_cpu is not None:
            _finite(used_cpu, "Observed/reserved CPU usage")
        updated = {
            **self.state,
            "spent_seconds": self.state["spent_seconds"] + elapsed,
            "attempts": list(self.state["attempts"]),
        }
        del updated["active"]
        if used_cpu is not None:
            updated["spent_cpu_seconds"] = self.state.get("spent_cpu_seconds", 0.0) + used_cpu
            reservation["observed_cpu_seconds"] = cpu_seconds
            reservation["charged_cpu_seconds"] = used_cpu
            reservation["cpu_accounting"] = (
                "observed" if cpu_seconds is not None else "reserved_upper_bound"
            )
        else:
            reservation["cpu_accounting"] = "unavailable"
        if "spent_gpu_hours" in self.state:
            updated["spent_gpu_hours"] = (
                self.state["spent_gpu_hours"] + elapsed * self.state.get("allocated_gpus", 0) / 3600
            )
        updated["attempts"].append({**reservation, "elapsed_seconds": elapsed, "status": status})
        # If publication fails, retain the active reservation in memory and on disk.
        write_artifact(self.path, updated)
        self.state, self.started = updated, None

    def close(self):
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.finish("complete" if kind is None else "interrupted")
        finally:
            self.close()
