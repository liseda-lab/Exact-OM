"""Single-node admission and cumulative accounting across experiment attempts."""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, cast


def interval_seconds(intervals: list[list[float]]) -> float:
    """Measure the union of overlapping node allocation intervals."""
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        if not (math.isfinite(start) and math.isfinite(stop)) or stop < start:
            raise ValueError("invalid node allocation interval")
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


class BudgetLedger:
    """Track time forecasts and enforce declared hosted spending limits.

    The 2026-09-29 amendment makes node-hour and family envelopes advisory.
    Historical limits and actual charges remain intact; an underestimated
    duration must never prevent checkpoint continuation.
    """

    def __init__(self, path: Path, limits: Mapping[str, Any]):
        self.path = Path(path)
        self.limits = dict(limits)

    @contextmanager
    def _transaction(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {"schema_version": 2, "limits": self.limits, "work": {}, "intervals": []}
            )
            if state["limits"] != self.limits:
                raise ValueError("budget limits changed without a recorded amendment")
            yield state
            descriptor, name = tempfile.mkstemp(prefix=".budget-", dir=self.path.parent)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(state, stream, sort_keys=True, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, self.path)
            finally:
                Path(name).unlink(missing_ok=True)

    def admit(
        self,
        work_id: str,
        *,
        group: str,
        seconds: float,
        requests: int = 0,
        tokens: int = 0,
        projected_usd: float = 0,
        forecast_known: bool = True,
    ) -> None:
        """Record a forecast before scheduling; time overruns are warnings only."""
        if (
            not math.isfinite(seconds)
            or seconds < 0
            or requests < 0
            or tokens < 0
            or not math.isfinite(projected_usd)
            or projected_usd < 0
        ):
            raise ValueError("invalid work estimate")
        with self._transaction() as state:
            if work_id in state["work"]:
                raise ValueError("work ID already admitted; a retry needs a new attempt ID")
            if group not in self.limits["envelopes_hours"]:
                raise ValueError(f"unknown budget envelope {group}")
            items = list(state["work"].values())
            group_seconds = interval_seconds(
                [
                    [item["start"], item["end"]]
                    for item in items
                    if item["group"] == group and item["status"] != "reserved"
                ]
            ) + sum(
                item["seconds"]
                for item in items
                if item["group"] == group and item["status"] == "reserved"
            )
            warnings = []
            if group_seconds + seconds > self.limits["envelopes_hours"][group] * 3600:
                warnings.append(f"{group} forecast exceeds historical time envelope")
            node_seconds = interval_seconds(state["intervals"]) + sum(
                item["seconds"] for item in items if item["status"] == "reserved"
            )
            if node_seconds + seconds > self.limits.get("node_hours_cap", math.inf) * 3600:
                warnings.append("forecast exceeds historical node-hour envelope")
            for field, amount in (("requests", requests), ("tokens", tokens)):
                reserve = self.limits.get(f"final_{field}_reserved", 0) if group != "final" else 0
                spent = sum(
                    item[field] for item in items if group == "final" or item["group"] != "final"
                )
                if spent + amount + reserve > self.limits[f"{field}_cap"]:
                    raise ValueError(f"deferred_budget: {field} would consume protected final work")
            state["work"][work_id] = {
                "group": group,
                "status": "reserved",
                "seconds": seconds,
                "requests": requests,
                "tokens": tokens,
                "projected_usd": projected_usd,
                "actual_usd": None,
                "forecast_seconds": seconds,
                "forecast_known": forecast_known,
                "time_warnings": warnings,
            }
            state["time_policy"] = "advisory_2026-09-29"

    def finish(
        self,
        work_id: str,
        *,
        start: float,
        end: float,
        status: str,
        requests: int,
        tokens: int,
        actual_usd: float | None,
    ) -> None:
        """Record actual usage even on failure; retries never erase previous spend."""
        if status not in {"complete", "failed", "interrupted", "unknown"}:
            raise ValueError("invalid work status")
        elapsed = interval_seconds([[start, end]])
        if (
            requests < 0
            or tokens < 0
            or (actual_usd is not None and (not math.isfinite(actual_usd) or actual_usd < 0))
        ):
            raise ValueError("invalid actual usage")
        with self._transaction() as state:
            item = state["work"][work_id]
            if item["status"] != "reserved":
                raise ValueError("completed accounting is immutable")
            item.update(
                start=start,
                end=end,
                status=status,
                seconds=elapsed,
                requests=requests,
                tokens=tokens,
                actual_usd=actual_usd,
            )
            if item.get("forecast_known", True) and elapsed > item.get("forecast_seconds", elapsed):
                item.setdefault("time_warnings", []).append("actual duration exceeds forecast")
            state["intervals"].append([start, end])
            state["node_seconds"] = interval_seconds(state["intervals"])
            state["elapsed_seconds"] = max(x[1] for x in state["intervals"]) - min(
                x[0] for x in state["intervals"]
            )

    def record_allocation(self, allocation: str, *, start: float, end: float) -> None:
        """Track scheduler-confirmed retained time, separately from active work."""
        interval_seconds([[start, end]])
        if not allocation:
            raise ValueError("allocation identity is required")
        with self._transaction() as state:
            allocations = state.setdefault("allocations", {})
            previous = allocations.get(allocation)
            if previous and (previous["start"] != start or end < previous["end"]):
                raise ValueError("allocation observation cannot rewrite prior accounting")
            allocations[allocation] = {"start": start, "end": end}
            state["allocation_seconds"] = interval_seconds(
                [[row["start"], row["end"]] for row in allocations.values()]
            )
            state["time_policy"] = "advisory_2026-09-29"

    def snapshot(self) -> dict[str, Any]:
        """Return the consistent retained account, including incomplete reservations."""
        with self._transaction() as state:
            return cast(dict[str, Any], json.loads(json.dumps(state)))
