"""Crash-safe timing records for alignment and evaluation runs.

``timings.json`` intentionally uses newline-delimited JSON despite its historical
suffix.  Each line is an immutable event, which means a completed stage is durable
without rewriting (or risking) timings from an earlier invocation.  The public
objects below materialize the session-oriented schema described in the project
contracts.

``times.txt`` is a deprecated, derived view retained for compatibility.  New code
should write through :class:`TimingLedger`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from exact.core.values import TIMING_STEP_ORDER
from exact.utils.run_context import activate_run_session

TIMING_SCHEMA_VERSION = 1
TIMING_LEDGER_FILENAME = "timings.json"
LEGACY_TIMING_FILENAME = "times.txt"


class CacheStatus(str, Enum):
    """How a stage's work relates to previously persisted artifacts."""

    FRESH = "fresh"
    RESUMED = "resumed"
    CACHE_HIT = "cache_hit"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StageRecord:
    """One stage measurement from one run session, always expressed in seconds."""

    stage: str
    seconds: float
    cache_status: CacheStatus = CacheStatus.FRESH
    work_done: Optional[int] = None
    work_total: Optional[int] = None
    unit: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seconds", max(0.0, float(self.seconds)))
        if not isinstance(self.cache_status, CacheStatus):
            object.__setattr__(self, "cache_status", CacheStatus(self.cache_status))
        if self.work_done is not None:
            object.__setattr__(self, "work_done", max(0, int(self.work_done)))
        if self.work_total is not None:
            object.__setattr__(self, "work_total", max(0, int(self.work_total)))


@dataclass(frozen=True)
class StageTotal:
    """Cumulative stage cost split between artifact compute and run overhead."""

    compute_seconds: float = 0.0
    overhead_seconds: float = 0.0
    sessions: int = 0
    work_done: Optional[int] = None
    work_total: Optional[int] = None


@dataclass(frozen=True)
class SessionRecord:
    """Materialized view of the append-only events for a run invocation."""

    run_id: str
    command: str
    started_at: str
    ended_at: Optional[str]
    config_fingerprint: str
    dataset_signature: Optional[str]
    exact_version: str
    stages: tuple[StageRecord, ...] = field(default_factory=tuple)
    crashed: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exact_version() -> str:
    try:
        return version("exact-om")
    except PackageNotFoundError:
        return "1.0.0"


class _ExclusiveFileLock:
    """Small cross-process lock based on atomic ``O_EXCL`` creation."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 10.0,
        stale_seconds: float = 60.0,
    ) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self._owned = False

    def __enter__(self) -> "_ExclusiveFileLock":
        deadline = time.monotonic() + self.timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for timing ledger lock {self.path}")
                time.sleep(0.01)
                continue
            try:
                os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            finally:
                os.close(descriptor)
            self._owned = True
            return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._owned:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._owned = False


class TimingLedger:
    """Append-only timing event ledger rooted in an output directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / TIMING_LEDGER_FILENAME
        self.times_path = self.run_dir / LEGACY_TIMING_FILENAME
        self.lock_path = self.run_dir / f"{TIMING_LEDGER_FILENAME}.lock"

    @classmethod
    def open(cls, run_dir: Path) -> "TimingLedger":
        ledger = cls(Path(run_dir))
        ledger._initialize()
        return ledger

    @contextmanager
    def session(
        self,
        *,
        command: str,
        config_fingerprint: str,
        dataset_signature: Optional[str] = None,
    ) -> Iterator["RunSession"]:
        """Open a session and leave it unfinished when the body raises.

        Stage records written before an exception remain readable and count toward
        cumulative compute.  A normal exit appends the end event and refreshes the
        deprecated ``times.txt`` projection.
        """

        run_session = self.start_session(
            command=command,
            config_fingerprint=config_fingerprint,
            dataset_signature=dataset_signature,
        )
        with run_session:
            yield run_session

    def start_session(
        self,
        *,
        command: str,
        config_fingerprint: str,
        dataset_signature: Optional[str] = None,
    ) -> "RunSession":
        """Start a session for call sites that cannot wrap their whole body."""

        run_session = RunSession(
            ledger=self,
            run_id=str(uuid.uuid4()),
            command=str(command),
            config_fingerprint=str(config_fingerprint),
            dataset_signature=dataset_signature,
            started_at=_utc_now(),
        )
        self._append_event(
            "session_started",
            run_id=run_session.run_id,
            command=run_session.command,
            started_at=run_session.started_at,
            config_fingerprint=run_session.config_fingerprint,
            dataset_signature=run_session.dataset_signature,
            exact_version=_exact_version(),
        )
        return run_session

    def sessions(self) -> list[SessionRecord]:
        """Return the materialized sessions in append order."""

        events = self._read_events()
        mutable: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for event in events:
            event_name = event.get("event")
            run_id = event.get("run_id")
            if event_name == "session_started" and run_id:
                if run_id not in mutable:
                    order.append(run_id)
                mutable[run_id] = {
                    "run_id": run_id,
                    "command": str(event.get("command", "unknown")),
                    "started_at": str(event.get("started_at", "")),
                    "ended_at": None,
                    "config_fingerprint": str(event.get("config_fingerprint", "legacy")),
                    "dataset_signature": event.get("dataset_signature"),
                    "exact_version": str(event.get("exact_version", "unknown")),
                    "stages": [],
                    "crashed": False,
                }
                continue
            if not run_id or run_id not in mutable:
                continue
            session = mutable[run_id]
            if event_name == "session_metadata":
                if "dataset_signature" in event:
                    session["dataset_signature"] = event.get("dataset_signature")
            elif event_name in {"stage_recorded", "span_finished"}:
                try:
                    session["stages"].append(
                        StageRecord(
                            stage=str(event["stage"]),
                            seconds=float(event.get("seconds", 0.0)),
                            cache_status=CacheStatus(event.get("cache_status", "fresh")),
                            work_done=event.get("work_done"),
                            work_total=event.get("work_total"),
                            unit=event.get("unit"),
                            span_id=event.get("span_id"),
                            parent_span_id=event.get("parent_span_id"),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            elif event_name == "session_finished":
                session["ended_at"] = event.get("ended_at")
            elif event_name == "session_failed":
                session["crashed"] = True

        records: list[SessionRecord] = []
        for run_id in order:
            item = mutable[run_id]
            records.append(
                SessionRecord(
                    run_id=item["run_id"],
                    command=item["command"],
                    started_at=item["started_at"],
                    ended_at=item["ended_at"],
                    config_fingerprint=item["config_fingerprint"],
                    dataset_signature=item["dataset_signature"],
                    exact_version=item["exact_version"],
                    stages=tuple(item["stages"]),
                    crashed=bool(item["crashed"] or item["ended_at"] is None),
                )
            )
        return records

    def stage_totals(
        self,
        *,
        config_fingerprint: Optional[str] = None,
    ) -> dict[str, StageTotal]:
        """Aggregate compute and overhead without allowing cache hits to erase work."""

        accumulators: dict[str, dict[str, Any]] = {}
        for session in self.sessions():
            if config_fingerprint is not None and session.config_fingerprint != config_fingerprint:
                continue
            for record in session.stages:
                item = accumulators.setdefault(
                    record.stage,
                    {
                        "compute_seconds": 0.0,
                        "overhead_seconds": 0.0,
                        "session_ids": set(),
                        "work_done": 0,
                        "has_work_done": False,
                        "work_total": 0,
                        "has_work_total": False,
                    },
                )
                if record.cache_status in {CacheStatus.FRESH, CacheStatus.RESUMED}:
                    item["compute_seconds"] += record.seconds
                else:
                    item["overhead_seconds"] += record.seconds
                item["session_ids"].add(session.run_id)
                if record.work_done is not None:
                    item["work_done"] += record.work_done
                    item["has_work_done"] = True
                if record.work_total is not None:
                    item["work_total"] += record.work_total
                    item["has_work_total"] = True

        return {
            stage: StageTotal(
                compute_seconds=item["compute_seconds"],
                overhead_seconds=item["overhead_seconds"],
                sessions=len(item["session_ids"]),
                work_done=item["work_done"] if item["has_work_done"] else None,
                work_total=item["work_total"] if item["has_work_total"] else None,
            )
            for stage, item in accumulators.items()
        }

    def estimates(
        self,
        *,
        config_fingerprint: Optional[str] = None,
    ) -> dict[str, float]:
        """Estimate fresh stage seconds, using per-unit rates when available.

        Fresh measurements are preferred over resumed fragments.  When a requested
        configuration has no usable record for a stage, history from other
        fingerprints is used as a best-effort fallback.
        """

        sessions = self.sessions()
        all_records: dict[str, list[StageRecord]] = {}
        matching_records: dict[str, list[StageRecord]] = {}
        for session in sessions:
            for record in session.stages:
                if record.cache_status not in {CacheStatus.FRESH, CacheStatus.RESUMED}:
                    continue
                all_records.setdefault(record.stage, []).append(record)
                if config_fingerprint is None or session.config_fingerprint == config_fingerprint:
                    matching_records.setdefault(record.stage, []).append(record)

        estimates: dict[str, float] = {}
        for stage in all_records.keys() | matching_records.keys():
            candidates = matching_records.get(stage) or all_records.get(stage, [])
            fresh = [record for record in candidates if record.cache_status is CacheStatus.FRESH]
            selected = fresh or candidates
            if not selected:
                continue
            rate_records = [
                record
                for record in selected
                if record.work_done is not None
                and record.work_done > 0
                and record.work_total is not None
                and record.work_total > 0
            ]
            if rate_records:
                seconds = sum(record.seconds for record in rate_records)
                work_done = sum(record.work_done or 0 for record in rate_records)
                work_total = max(record.work_total or 0 for record in rate_records)
                estimates[stage] = seconds / work_done * work_total
            else:
                estimates[stage] = sum(record.seconds for record in selected) / len(selected)
        return estimates

    def render_legacy_times(self, *, config_fingerprint: Optional[str] = None) -> Path:
        """Refresh the deprecated minute-based view from cumulative ledger totals."""

        totals = self.stage_totals(config_fingerprint=config_fingerprint)
        timings: dict[str, float] = {}
        for stage, total in totals.items():
            seconds = (
                total.compute_seconds if total.compute_seconds > 0.0 else total.overhead_seconds
            )
            timings[stage] = seconds / 60.0
        write_recorded_timings(self.times_path, timings)
        return self.times_path

    def _initialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with _ExclusiveFileLock(self.lock_path):
            imported_events: list[dict[str, Any]] = []
            if self.path.exists() and self.path.stat().st_size:
                try:
                    parsed, monolithic = self._parse_existing_file()
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    backup_path = self._backup_path("corrupt")
                    os.replace(self.path, backup_path)
                    warnings.warn(
                        f"Corrupt timing ledger moved to {backup_path}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    if not monolithic:
                        return
                    backup_path = self._backup_path("legacy-json")
                    os.replace(self.path, backup_path)
                    imported_events = self._events_from_monolithic(parsed)

            self._append_lines_unlocked(
                [self._event("ledger_started", created_at=_utc_now()), *imported_events]
            )
            if not imported_events and self.times_path.exists():
                legacy_timings = load_recorded_timings(self.times_path)
                if legacy_timings:
                    self._append_lines_unlocked(self._legacy_events(legacy_timings))

    def _parse_existing_file(self) -> tuple[Any, bool]:
        raw = self.path.read_text(encoding="utf-8")
        try:
            parsed_whole = json.loads(raw)
        except json.JSONDecodeError:
            parsed_whole = None
        if isinstance(parsed_whole, dict) and "sessions" in parsed_whole:
            return parsed_whole, True

        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"timing event on line {line_number} is not an object")
            if event.get("schema_version") != TIMING_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported timing schema {event.get('schema_version')!r} on line {line_number}"
                )
            events.append(event)
        if not events:
            raise ValueError("timing ledger contains no events")
        return events, False

    def _read_events(self) -> list[dict[str, Any]]:
        with _ExclusiveFileLock(self.lock_path):
            parsed, monolithic = self._parse_existing_file()
            if monolithic:
                return self._events_from_monolithic(parsed)
            return parsed

    def _append_event(self, event_name: str, **payload: Any) -> None:
        with _ExclusiveFileLock(self.lock_path):
            if not self.path.exists() or not self.path.stat().st_size:
                self._append_lines_unlocked([self._event("ledger_started", created_at=_utc_now())])
            self._append_lines_unlocked([self._event(event_name, **payload)])

    def _append_lines_unlocked(self, events: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _event(event_name: str, **payload: Any) -> dict[str, Any]:
        return {
            "schema_version": TIMING_SCHEMA_VERSION,
            "event": event_name,
            **payload,
        }

    def _backup_path(self, label: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return self.path.with_name(f"{self.path.name}.{label}-{stamp}-{uuid.uuid4().hex[:8]}")

    def _legacy_events(self, timings: Mapping[str, float]) -> list[dict[str, Any]]:
        run_id = str(uuid.uuid4())
        now = _utc_now()
        events = [
            self._event(
                "session_started",
                run_id=run_id,
                command="legacy",
                started_at=now,
                config_fingerprint="legacy",
                dataset_signature=None,
                exact_version="unknown",
            )
        ]
        for stage, minutes in timings.items():
            events.append(
                self._event(
                    "stage_recorded",
                    run_id=run_id,
                    stage=stage,
                    seconds=max(0.0, float(minutes) * 60.0),
                    cache_status=CacheStatus.FRESH.value,
                )
            )
        events.append(self._event("session_finished", run_id=run_id, ended_at=now))
        return events

    def _events_from_monolithic(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for raw_session in payload.get("sessions", []):
            run_id = str(raw_session.get("run_id") or uuid.uuid4())
            events.append(
                self._event(
                    "session_started",
                    run_id=run_id,
                    command=raw_session.get("command", "unknown"),
                    started_at=raw_session.get("started_at") or _utc_now(),
                    config_fingerprint=raw_session.get("config_fingerprint", "legacy"),
                    dataset_signature=raw_session.get("dataset_signature"),
                    exact_version=raw_session.get("exact_version", "unknown"),
                )
            )
            for raw_stage in raw_session.get("stages", []):
                events.append(
                    self._event(
                        "stage_recorded",
                        run_id=run_id,
                        stage=raw_stage.get("stage", "unknown"),
                        seconds=max(0.0, float(raw_stage.get("seconds", 0.0))),
                        cache_status=raw_stage.get("cache_status", "fresh"),
                        work_done=raw_stage.get("work_done"),
                        work_total=raw_stage.get("work_total"),
                        unit=raw_stage.get("unit"),
                    )
                )
            if raw_session.get("ended_at") is not None:
                events.append(
                    self._event(
                        "session_finished",
                        run_id=run_id,
                        ended_at=raw_session.get("ended_at"),
                    )
                )
        return events


class RunSession:
    """A single invocation within a :class:`TimingLedger`."""

    def __init__(
        self,
        *,
        ledger: TimingLedger,
        run_id: str,
        command: str,
        config_fingerprint: str,
        dataset_signature: Optional[str],
        started_at: str,
    ) -> None:
        self.ledger = ledger
        self.run_id = run_id
        self.command = command
        self.config_fingerprint = config_fingerprint
        self.dataset_signature = dataset_signature
        self.started_at = started_at
        self._closed = False
        self._span_stack: list[str] = []
        self._activation = None

    def __enter__(self) -> "RunSession":
        if self._activation is None:
            self._activation = activate_run_session(self)
            self._activation.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None:
                self.finish()
            else:
                self.fail(exc)
        finally:
            if self._activation is not None:
                self._activation.__exit__(exc_type, exc, traceback)
                self._activation = None

    def stage(
        self,
        name: str,
        *,
        cache_status: CacheStatus = CacheStatus.FRESH,
        work_total: Optional[int] = None,
        unit: Optional[str] = None,
    ) -> "StageSpan":
        return StageSpan(
            session=self,
            name=name,
            cache_status=cache_status,
            work_total=work_total,
            unit=unit,
        )

    def record(
        self,
        name: str | StageRecord,
        *,
        seconds: Optional[float] = None,
        cache_status: CacheStatus = CacheStatus.FRESH,
        work_done: Optional[int] = None,
        work_total: Optional[int] = None,
        unit: Optional[str] = None,
    ) -> StageRecord:
        """Durably append a direct stage measurement."""

        self._ensure_open()
        if isinstance(name, StageRecord):
            record = name
        else:
            if seconds is None:
                raise TypeError("seconds is required when recording a stage by name")
            record = StageRecord(
                stage=name,
                seconds=seconds,
                cache_status=cache_status,
                work_done=work_done,
                work_total=work_total,
                unit=unit,
            )
        if record.parent_span_id is None and self._span_stack:
            record = replace(record, parent_span_id=self._span_stack[-1])
        payload = asdict(record)
        payload["cache_status"] = record.cache_status.value
        self.ledger._append_event("stage_recorded", run_id=self.run_id, **payload)
        return record

    def set_dataset_signature(self, dataset_signature: Optional[str]) -> None:
        self._ensure_open()
        self.dataset_signature = dataset_signature
        self.ledger._append_event(
            "session_metadata",
            run_id=self.run_id,
            dataset_signature=dataset_signature,
        )

    def finish(self) -> None:
        if self._closed:
            return
        self.ledger._append_event(
            "session_finished",
            run_id=self.run_id,
            ended_at=_utc_now(),
        )
        self._closed = True
        self.ledger.render_legacy_times(config_fingerprint=self.config_fingerprint)

    def fail(self, error: Optional[BaseException] = None) -> None:
        if self._closed:
            return
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "failed_at": _utc_now(),
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
        self.ledger._append_event("session_failed", **payload)
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Timing session {self.run_id} is already closed")


class StageSpan:
    """Monotonic, nestable timing span attached to a run session."""

    def __init__(
        self,
        *,
        session: RunSession,
        name: str,
        cache_status: CacheStatus,
        work_total: Optional[int],
        unit: Optional[str],
    ) -> None:
        self.session = session
        self.name = name
        self.cache_status = CacheStatus(cache_status)
        self.work_total = work_total
        self.work_done: Optional[int] = None
        self.unit = unit
        self.span_id = str(uuid.uuid4())
        self.parent_span_id: Optional[str] = None
        self.seconds: Optional[float] = None
        self._started: Optional[float] = None

    def __enter__(self) -> "StageSpan":
        self.session._ensure_open()
        self.parent_span_id = self.session._span_stack[-1] if self.session._span_stack else None
        self.session._span_stack.append(self.span_id)
        self._started = time.perf_counter()
        self.session.ledger._append_event(
            "span_started",
            run_id=self.session.run_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            stage=self.name,
            cache_status=self.cache_status.value,
            work_total=self.work_total,
            unit=self.unit,
            started_at=_utc_now(),
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._started is None:
            return
        self.seconds = max(0.0, time.perf_counter() - self._started)
        if self.session._span_stack and self.session._span_stack[-1] == self.span_id:
            self.session._span_stack.pop()
        else:
            try:
                self.session._span_stack.remove(self.span_id)
            except ValueError:
                pass
        self.session.ledger._append_event(
            "span_finished",
            run_id=self.session.run_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            stage=self.name,
            seconds=self.seconds,
            cache_status=self.cache_status.value,
            work_done=self.work_done,
            work_total=self.work_total,
            unit=self.unit,
            failed=exc_type is not None,
            finished_at=_utc_now(),
        )

    def set_work_done(self, work_done: int) -> None:
        self.work_done = max(0, int(work_done))


_OMIT = object()
_VOLATILE_CONFIG_KEYS = {
    "device",
    "log_file_path",
    "logging_level",
    "output_dir",
    "output_dir_path",
}


def config_fingerprint(config: Any, *, run_dir: Optional[Path] = None) -> str:
    """Return a stable SHA-1 of a resolved config model or mapping.

    Runtime-only logging/device fields are excluded.  Paths within ``run_dir`` are
    made relative; paths outside it are omitted so moving a run does not change the
    identity of the computation.
    """

    if hasattr(config, "model_dump"):
        try:
            raw = config.model_dump(mode="python")
        except TypeError:
            raw = config.model_dump()
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = vars(config)
    canonical = _canonicalize_config(raw, run_dir=run_dir, key=None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _canonicalize_config(value: Any, *, run_dir: Optional[Path], key: Optional[str]) -> Any:
    if key in _VOLATILE_CONFIG_KEYS:
        return _OMIT
    if isinstance(value, Mapping):
        result = {}
        for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0])):
            canonical = _canonicalize_config(
                child_value,
                run_dir=run_dir,
                key=str(child_key),
            )
            if canonical is not _OMIT:
                result[str(child_key)] = canonical
        return result
    if isinstance(value, set):
        result = [_canonicalize_config(item, run_dir=run_dir, key=None) for item in value]
        return sorted(
            (item for item in result if item is not _OMIT),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if isinstance(value, (list, tuple)):
        result = [_canonicalize_config(item, run_dir=run_dir, key=None) for item in value]
        return [item for item in result if item is not _OMIT]
    if isinstance(value, Path):
        resolved = value.expanduser().resolve()
        if run_dir is None:
            return str(resolved)
        try:
            return str(resolved.relative_to(Path(run_dir).expanduser().resolve()))
        except ValueError:
            return _OMIT
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _canonicalize_config(value.model_dump(), run_dir=run_dir, key=key)
    return f"{type(value).__module__}.{type(value).__qualname__}:{value}"


def load_recorded_timings(times_file_path: Path) -> dict[str, float]:
    """Read deprecated ``times.txt`` minute values.

    This compatibility parser tolerates the derived-file header.  New code should
    use :meth:`TimingLedger.stage_totals` or :meth:`TimingLedger.estimates`.
    """

    times_file_path = Path(times_file_path)
    if not times_file_path.exists():
        return {}

    timings: dict[str, float] = {}
    for line in times_file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        step, separator, value_text = line.partition(":")
        if not separator:
            continue
        cleaned_value = value_text.strip().removesuffix(" minutes").strip()
        try:
            timings[step.strip()] = float(cleaned_value)
        except ValueError:
            continue
    return timings


def write_recorded_timings(times_file_path: Path, timings: Mapping[str, float]) -> None:
    """Write the deprecated derived minute view (compatibility shim)."""

    times_file_path = Path(times_file_path)
    times_file_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_steps = [step for step in TIMING_STEP_ORDER if step in timings]
    remaining_steps = sorted(step for step in timings if step not in TIMING_STEP_ORDER)
    lines = ["# derived from timings.json — do not edit"]
    lines.extend(
        f"{step}: {float(timings[step]):.1f} minutes" for step in [*ordered_steps, *remaining_steps]
    )
    times_file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_recorded_timings(
    times_file_path: Path,
    updates: Mapping[str, float],
) -> dict[str, float]:
    """Merge deprecated minute values (compatibility shim for pre-ledger callers)."""

    timings = load_recorded_timings(Path(times_file_path))
    timings.update({step: float(value) for step, value in updates.items() if value is not None})
    write_recorded_timings(Path(times_file_path), timings)
    return timings
