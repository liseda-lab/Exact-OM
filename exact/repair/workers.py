"""Bound the full worker lifetime, including spawn serialization and event transport."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import io
import multiprocessing
import os
import pickle
import resource
import signal
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from math import isfinite
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Callable

_SUPERVISED_GROUP = False
_EVENT_CHANNEL: Any = None
_MAX_FRAME = 16 * 1024 * 1024
SUPERVISION_GRACE_SECONDS = 0.22  # process cleanup plus bounded receiver join


@dataclass(frozen=True)
class CallResult:
    """Operational outcome plus acknowledged events, even after interruption."""

    status: str
    value: Any = None
    detail: str = ""
    events: Sequence[Any] = ()
    cleanup_complete: bool = True
    resource_usage: tuple[tuple[str, float], ...] = ()
    event_failure: Any = None
    transport_identity: str = "durable-event-stream/r1"
    event_journal: str | None = None


class _ResultReader(pickle.Unpickler):
    """Only inert owned data classes cross back into the coordinator.

    Arbitrary worker return reducers/constructors (including user __setstate__)
    cannot execute in the coordinator. Frames are bounded before decoding.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in {"set", "frozenset", "complex", "slice", "bytearray"}:
            return super().find_class(module, name)
        if module == "collections" and name == "OrderedDict":
            return super().find_class(module, name)
        if module.startswith(("exact.repair.", "tools.repair.", "pyowl_core.model.")):
            value = getattr(importlib.import_module(module), name)
            if isinstance(value, type) and (
                dataclasses.is_dataclass(value)
                or issubclass(value, enum.Enum)
                or name in {"FrozenMapping", "CanonicalSet"}
            ):
                return value
        raise pickle.UnpicklingError(f"unsupported worker result constructor: {module}.{name}")


@dataclass(frozen=True)
class CommittedEvents(Sequence):
    """Immutable committed journal prefix; iteration checks every chained batch.

    Reading a journal is explicit I/O. Kernel replay performs it in a supervised
    worker. The deadline controller transports this fixed-size receipt only.
    """

    directory: str
    batches: int
    event_count: int
    digest: str

    def __len__(self):
        return self.event_count

    def __iter__(self):
        previous, count = "0" * 64, 0
        connection = sqlite3.connect(
            f"file:{Path(self.directory) / 'events.sqlite'}?mode=ro", uri=True, timeout=0
        )
        try:
            cursor = connection.execute(
                "SELECT ordinal,payload FROM events WHERE ordinal < ? ORDER BY ordinal",
                (self.batches,),
            )
            rows = 0
            for index, payload in cursor:
                if index != rows or len(payload) > _MAX_FRAME:
                    raise ValueError("invalid committed event batch")
                ordinal, parent_hash, values = _ResultReader(io.BytesIO(payload)).load()
                if ordinal != index or parent_hash != previous or not isinstance(values, tuple):
                    raise ValueError("invalid event journal coverage chain")
                previous = hashlib.sha256(payload).hexdigest()
                count += len(values)
                rows += 1
                yield from values
            if rows != self.batches or count != self.event_count or previous != self.digest:
                raise ValueError("event journal receipt mismatch")
        finally:
            connection.close()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        if index < 0:
            index += self.event_count
        for position, event in enumerate(self):
            if position == index:
                return event
        raise IndexError(index)

    def __eq__(self, other):
        if isinstance(other, CommittedEvents):
            return dataclasses.astuple(self) == dataclasses.astuple(other)
        if isinstance(other, (tuple, list)):
            return tuple(self) == tuple(other)
        return NotImplemented


@lru_cache(maxsize=4)
def _journal_connection(directory: str, owner_pid: int) -> Any:
    connection = sqlite3.connect(Path(directory) / "events.sqlite", timeout=0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS events (ordinal INTEGER PRIMARY KEY, payload BLOB NOT NULL, digest TEXT NOT NULL, total_count INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS completion (id INTEGER PRIMARY KEY CHECK(id=1), payload BLOB NOT NULL, digest TEXT NOT NULL)"
    )
    connection.commit()
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return connection


def _commit_event_batch(directory: str, prior: CommittedEvents, values: tuple) -> CommittedEvents:
    payload = pickle.dumps((prior.batches, prior.digest, values), protocol=5)
    if len(payload) > _MAX_FRAME:
        raise ValueError("committed event batch exceeds frame budget")
    receipt = CommittedEvents(
        directory,
        prior.batches + 1,
        prior.event_count + len(values),
        hashlib.sha256(payload).hexdigest(),
    )
    if prior.batches == 0:
        Path(directory).mkdir(parents=True, exist_ok=False)
    connection = _journal_connection(directory, os.getpid())
    with connection:
        connection.execute(
            "INSERT INTO events VALUES (?,?,?,?)",
            (prior.batches, payload, receipt.digest, receipt.event_count),
        )
    return receipt


def committed_events(directory: str) -> CommittedEvents:
    """Resolve committed SQLite rows only; unfinished transactions remain invisible."""
    connection = sqlite3.connect(
        f"file:{Path(directory) / 'events.sqlite'}?mode=ro", uri=True, timeout=0
    )
    try:
        row = connection.execute(
            "SELECT ordinal,total_count,digest FROM events ORDER BY ordinal DESC LIMIT 1"
        ).fetchone()
        return (
            CommittedEvents(directory, row[0] + 1, row[1], row[2])
            if row
            else CommittedEvents(directory, 0, 0, "0" * 64)
        )
    finally:
        connection.close()


def _commit_result(directory: str, value: Any) -> None:
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > _MAX_FRAME:
        raise ValueError("completion requires a bounded artifact representation")
    connection = _journal_connection(directory, os.getpid())
    with connection:
        connection.execute(
            "INSERT INTO completion VALUES (1,?,?)", (payload, hashlib.sha256(payload).hexdigest())
        )


def committed_result(directory: str) -> Any:
    connection = sqlite3.connect(
        f"file:{Path(directory) / 'events.sqlite'}?mode=ro", uri=True, timeout=0
    )
    try:
        row = connection.execute("SELECT payload,digest FROM completion WHERE id=1").fetchone()
        if row is None:
            return None
        payload, digest = row
        if len(payload) > _MAX_FRAME or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("invalid committed final result")
        return _ResultReader(io.BytesIO(payload)).load()
    finally:
        connection.close()


def emit_events(events: Sequence[Any]) -> None:
    """Commit a bounded batch before the producer proceeds; no total-event ceiling."""
    if _EVENT_CHANNEL is None or not events:
        return
    payload = pickle.dumps(("events", tuple(events)), protocol=5)
    if len(payload) > _MAX_FRAME:
        raise ValueError("verification event batch exceeds the transport frame limit")
    _EVENT_CHANNEL.send_bytes(payload)
    if _EVENT_CHANNEL.recv_bytes(16) != b"ack":
        raise RuntimeError("verification event batch was not acknowledged")


def emit_event(event: Any) -> None:
    emit_events((event,))


def _execute(
    connection: Any, function: Callable[..., Any], args: tuple, kwargs: dict, own_group: bool
) -> None:
    global _SUPERVISED_GROUP, _EVENT_CHANNEL
    try:
        if own_group:
            os.setsid()
        _SUPERVISED_GROUP = os.name == "posix"
        _EVENT_CHANNEL = connection
        value = function(*args, **kwargs)
        from .records import compact_verification_report

        value = compact_verification_report(value)
        result = CallResult(
            "complete", value, resource_usage=(("cpu_seconds", _process_cpu_seconds()),)
        )
        payload = pickle.dumps(("result", result), protocol=5)
        if len(payload) > _MAX_FRAME:
            raise ValueError("worker result exceeds the transport frame limit")
        connection.send_bytes(payload)
    except BaseException as error:
        connection.send_bytes(
            pickle.dumps(
                (
                    "result",
                    CallResult(
                        "error",
                        detail=f"{type(error).__name__}: {error}",
                        resource_usage=(("cpu_seconds", _process_cpu_seconds()),),
                    ),
                ),
                protocol=5,
            )
        )
    finally:
        connection.close()


def _process_cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _tree_pids(pid: int) -> list[int]:
    pending, result, seen = [pid], [], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        result.append(current)
        try:
            for task in Path(f"/proc/{current}/task").iterdir():
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except (OSError, ValueError):
                    continue
        except OSError:
            continue
    return result


def _stop(pid: int, own_group: bool) -> bool:
    """Kill only this broker's descendants; cleanup is bounded and explicit."""
    descendants = _tree_pids(pid)
    if own_group:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in reversed(descendants):
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 0.2
    reaped = False
    while time.monotonic() < deadline:
        if not reaped:
            try:
                reaped = bool(os.waitpid(pid, os.WNOHANG)[0])
            except ChildProcessError:
                reaped = True
        # A killed child in uninterruptible I/O is still running. Zombies have
        # stopped executing and merely await their owning process's reaper.
        alive = False
        for child in descendants:
            try:
                state = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0]
                alive |= state not in {"Z", "X"}
            except (OSError, IndexError):
                continue
        if reaped and not alive:
            return True
        time.sleep(0.005)
    return False


def _coordinate(
    control: Any,
    function: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    handler: Callable[[Any], bool] | None,
    directory: str,
) -> None:
    """Killable event validation/storage owner; controller never runs callbacks."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(
        target=_execute, args=(child, function, args, kwargs, False), daemon=False
    )
    receipt = CommittedEvents(directory, 0, 0, "0" * 64)
    failure, peak_buffer = None, 0
    try:
        process.start()
        child.close()
        while True:
            payload = parent.recv_bytes(_MAX_FRAME)
            peak_buffer = max(peak_buffer, len(payload))
            kind, value = _ResultReader(io.BytesIO(payload)).load()
            if kind == "events":
                if not isinstance(value, tuple) or not value:
                    raise ValueError("invalid event batch")
                for event in value:
                    if handler is not None and not handler(event):
                        raise ValueError("invalid worker event")
                receipt = _commit_event_batch(directory, receipt, value)
                for event in value:
                    if (
                        failure is None
                        and getattr(getattr(event, "obligation", None), "verdict", None) == "fail"
                    ):
                        failure = event
                control.send_bytes(
                    pickle.dumps(("receipt", (receipt, failure, peak_buffer)), protocol=5)
                )
                # The producer ACK follows both durable commit and controller
                # receipt. An acknowledged failure is therefore always retained
                # by the deadline controller even if the next frame never arrives.
                if control.recv_bytes(16) != b"recorded":
                    raise RuntimeError("controller did not retain the committed prefix")
                parent.send_bytes(b"ack")
            elif kind == "result" and isinstance(value, CallResult):
                if (
                    value.status == "complete"
                    and handler is not None
                    and hasattr(handler, "finish")
                ):
                    value = replace(value, value=handler.finish(value.value))
                from .records import compact_verification_report

                value = replace(
                    value,
                    value=compact_verification_report(value.value),
                    events=receipt if receipt.event_count else (),
                    event_failure=failure,
                )
                payload = pickle.dumps(("result", value), protocol=5)
                if len(payload) > _MAX_FRAME:
                    raise ValueError(
                        "final worker result requires a bounded artifact representation"
                    )
                if value.status == "complete" and receipt.event_count:
                    _commit_result(directory, value.value)
                control.send_bytes(payload)
                break
            else:
                raise ValueError("invalid worker envelope")
    except BaseException as error:
        try:
            control.send_bytes(
                pickle.dumps(
                    (
                        "result",
                        CallResult(
                            "error",
                            detail=f"{type(error).__name__}: {error}",
                            events=receipt if receipt.event_count else (),
                            event_failure=failure,
                        ),
                    ),
                    protocol=5,
                )
            )
        except BaseException:
            pass
    finally:
        # Retain ownership while the controller cancels descendants; a blocked
        # producer waiting for an uncommitted ACK must not become an orphan.
        if process.pid is not None:
            process.join()
        parent.close()
        control.close()


def bounded_call(
    function: Callable[..., Any],
    *args: Any,
    timeout: float,
    memory_mb: float | None = None,
    cpu_seconds: float | None = None,
    event_handler: Callable[[Any], bool] | None = None,
    event_directory: str | None = None,
    **kwargs: Any,
) -> CallResult:
    """Monitor startup, native work, event validation/storage and bounded transport.

    A POSIX broker starts a clean spawned native worker and owns event callbacks.
    Both are killable while the caller monitors deadlines and sampled tree CPU/RSS.
    The controller receives only bounded messages and durable-prefix receipts.
    Cleanup has a 0.2-second grace; explicit journal replay needs its own budget.
    """
    if not isfinite(timeout):
        raise ValueError("timeout must be finite")
    if cpu_seconds is not None and (
        type(cpu_seconds) not in (int, float) or not isfinite(cpu_seconds) or cpu_seconds <= 0
    ):
        raise ValueError("cpu_seconds must be positive and finite")
    if memory_mb is not None and (not isfinite(memory_mb) or memory_mb <= 0):
        raise ValueError("memory_mb must be positive and finite")
    if os.name != "posix" or not hasattr(os, "fork"):
        return CallResult("unsupported", detail="independent supervision requires POSIX")
    if (cpu_seconds is not None or memory_mb is not None) and not Path("/proc/self/stat").exists():
        return CallResult("unsupported", detail="worker CPU/RSS supervision requires Linux procfs")
    if timeout <= 0:
        return CallResult("timeout", detail="stage deadline exhausted")
    started = time.monotonic()
    deadline = started + timeout
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    directory = str(
        Path(event_directory)
        if event_directory
        else Path(tempfile.gettempdir()) / ("exact-repair-events-" + uuid.uuid4().hex)
    )
    own_group = not _SUPERVISED_GROUP
    pid = os.fork()
    if pid == 0:
        try:
            parent.close()
            if own_group:
                os.setsid()
            _coordinate(child, function, args, kwargs, event_handler, directory)
        finally:
            os._exit(0)
    child.close()
    queue: Queue[Any] = Queue(maxsize=1)
    stopped = Event()

    def enqueue(message):
        while not stopped.is_set():
            try:
                queue.put(message, timeout=0.02)
                return
            except Full:
                continue

    def receive():
        try:
            while True:
                message = _ResultReader(io.BytesIO(parent.recv_bytes(_MAX_FRAME))).load()
                enqueue(message)
                if message[0] == "result":
                    return
        except BaseException as error:
            enqueue(("result", CallResult("error", detail=f"worker transport failed: {error}")))

    receiver = Thread(target=receive, daemon=True)
    receiver.start()
    events: Sequence[Any] = ()
    failure = None
    peak_rss = measured_cpu = peak_buffer = 0.0
    sampled_at = 0.0
    result = CallResult("timeout", detail="stage deadline exhausted")
    try:
        while time.monotonic() < deadline:
            if time.monotonic() - sampled_at >= 0.01:
                sampled_at = time.monotonic()
                measured_cpu = max(measured_cpu, _cpu_tree_seconds(pid))
                resident = _resident_tree_bytes(pid)
                peak_rss = max(peak_rss, resident)
                if cpu_seconds is not None and measured_cpu > cpu_seconds:
                    result = CallResult(
                        "cpu_limit", detail=f"worker tree CPU exceeded {cpu_seconds:g} seconds"
                    )
                    break
                if memory_mb is not None and resident > memory_mb * 1024 * 1024:
                    result = CallResult(
                        "memory_limit", detail=f"worker tree RSS exceeded {memory_mb:g} MiB"
                    )
                    break
            try:
                kind, value = queue.get(timeout=min(0.02, max(0.0, deadline - time.monotonic())))
            except Empty:
                continue
            if kind == "receipt":
                events, failure, buffered = value
                peak_buffer = max(peak_buffer, buffered)
                parent.send_bytes(b"recorded")
            elif kind == "result" and isinstance(value, CallResult):
                result = value
                if value.events:
                    events, failure = value.events, value.event_failure
                break
            else:
                result = CallResult("error", detail="invalid supervisor envelope")
                break
    finally:
        stopped.set()
        cleanup = _stop(pid, own_group)
        parent.close()
        receiver.join(timeout=0.02)
    measured_cpu = max(measured_cpu, dict(result.resource_usage).get("cpu_seconds", 0.0))
    if result.status == "complete" and cpu_seconds is not None and measured_cpu > cpu_seconds:
        result = CallResult("cpu_limit", detail=f"worker tree CPU exceeded {cpu_seconds:g} seconds")
    return replace(
        result,
        events=events,
        event_failure=failure,
        cleanup_complete=cleanup,
        event_journal=directory,
        resource_usage=(
            ("cpu_seconds", measured_cpu),
            ("wall_seconds", time.monotonic() - started),
            ("peak_sampled_tree_rss_bytes", peak_rss),
            ("peak_buffered_event_bytes", peak_buffer),
        ),
    )


def _resident_tree_bytes(pid: int) -> int:
    pages = 0
    for current in _tree_pids(pid):
        try:
            pages += int(Path(f"/proc/{current}/statm").read_text().split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return pages * os.sysconf("SC_PAGE_SIZE")


def _cpu_tree_seconds(pid: int) -> float:
    ticks = 0
    for current in _tree_pids(pid):
        try:
            fields = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
            ticks += sum(int(fields[index]) for index in (11, 12, 13, 14))
        except (OSError, ValueError, IndexError):
            continue
    return ticks / os.sysconf("SC_CLK_TCK")
