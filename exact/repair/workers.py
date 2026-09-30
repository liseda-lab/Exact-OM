"""Bound the full worker lifetime, including spawn serialization and event transport."""

from __future__ import annotations

import dataclasses
import enum
import importlib
import io
import multiprocessing
import os
import pickle
import resource
import signal
import time
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable

_SUPERVISED_GROUP = False
_EVENT_CHANNEL: Any = None
_MAX_FRAME = 16 * 1024 * 1024


@dataclass(frozen=True)
class CallResult:
    """Operational outcome plus acknowledged events, even after interruption."""

    status: str
    value: Any = None
    detail: str = ""
    events: tuple[Any, ...] = ()
    cleanup_complete: bool = True
    resource_usage: tuple[tuple[str, float], ...] = ()


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


def emit_event(event: Any) -> None:
    """Send one bounded proof event and wait for coordinator acknowledgement."""
    if _EVENT_CHANNEL is None:
        return
    payload = pickle.dumps(("event", event), protocol=5)
    if len(payload) > _MAX_FRAME:
        raise ValueError("verification event exceeds the transport frame limit")
    _EVENT_CHANNEL.send_bytes(payload)
    if _EVENT_CHANNEL.recv_bytes(16) != b"ack":
        raise RuntimeError("verification event was not acknowledged")


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
    while time.monotonic() < deadline:
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
            if waited:
                return True
        except ChildProcessError:
            return True
        time.sleep(0.005)
    return False


def bounded_call(
    function: Callable[..., Any],
    *args: Any,
    timeout: float,
    memory_mb: float | None = None,
    cpu_seconds: float | None = None,
    event_handler: Callable[[Any], bool] | None = None,
    **kwargs: Any,
) -> CallResult:
    """Supervise startup, work and bounded transport independently of serialization.

    A tiny POSIX fork broker only starts a clean spawned worker. All user argument
    serialization executes inside that killable domain, never in the coordinator.
    The broker does not run native reasoning or Torch code after fork. Nested calls
    retain the outer group while their own cancellation targets their process tree.
    RSS is sampled worker-tree RSS (shared pages counted per process); it excludes
    caller inputs and device memory. Cleanup may take up to 0.2 s after the deadline.
    """
    if not isfinite(timeout):
        raise ValueError("timeout must be finite")
    if cpu_seconds is not None and (
        type(cpu_seconds) not in (int, float) or not isfinite(cpu_seconds) or cpu_seconds <= 0
    ):
        raise ValueError("cpu_seconds must be positive and finite")
    if cpu_seconds is not None and not Path("/proc/self/stat").exists():
        return CallResult("unsupported", detail="worker CPU supervision requires Linux procfs")
    if memory_mb is not None and (not isfinite(memory_mb) or memory_mb <= 0):
        raise ValueError("memory_mb must be positive and finite")
    if os.name != "posix" or not hasattr(os, "fork"):
        return CallResult("unsupported", detail="independent startup supervision requires POSIX")
    if memory_mb is not None and not Path("/proc/self/statm").exists():
        return CallResult(
            "error", detail="worker RSS memory supervision is unavailable on this platform"
        )
    if timeout <= 0:
        return CallResult("timeout", detail="stage deadline exhausted")
    deadline = time.monotonic() + timeout
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    own_group = not _SUPERVISED_GROUP
    # No pickle/Process.start executes before this independently monitored fork.
    pid = os.fork()
    if pid == 0:
        try:
            parent.close()
            if own_group:
                os.setsid()
            process = context.Process(
                target=_execute, args=(child, function, args, kwargs, False), daemon=False
            )
            process.start()
            child.close()
            process.join()
        except BaseException as error:
            try:
                child.send_bytes(
                    pickle.dumps(
                        ("result", CallResult("error", detail=f"worker startup failed: {error}")),
                        protocol=5,
                    )
                )
            except BaseException:
                pass
        finally:
            os._exit(0)
    child.close()
    queue: Queue[Any] = Queue(maxsize=1)

    def receive() -> None:
        try:
            while True:
                message = _ResultReader(io.BytesIO(parent.recv_bytes(_MAX_FRAME))).load()
                queue.put(message)
                if not isinstance(message, tuple) or message[0] == "result":
                    return
        except BaseException as error:
            queue.put(
                (
                    "result",
                    CallResult(
                        "error",
                        detail=f"worker exited without a result or transport failed: {error}",
                    ),
                )
            )

    reader = Thread(target=receive, daemon=True)
    reader.start()
    events: list[Any] = []
    peak_rss = measured_cpu = 0.0
    result = CallResult("timeout", detail="stage deadline exhausted")
    try:
        while time.monotonic() < deadline:
            measured_cpu = max(measured_cpu, _cpu_tree_seconds(pid))
            resident_bytes = _resident_tree_bytes(pid)
            peak_rss = max(peak_rss, resident_bytes)
            if cpu_seconds is not None and measured_cpu > cpu_seconds:
                result = CallResult(
                    "cpu_limit", detail=f"worker tree CPU exceeded {cpu_seconds:g} seconds"
                )
                break
            if memory_mb is not None and resident_bytes > memory_mb * 1024 * 1024:
                result = CallResult(
                    "memory_limit", detail=f"worker tree RSS exceeded {memory_mb:g} MiB"
                )
                break
            try:
                kind, value = queue.get(timeout=min(0.05, max(0, deadline - time.monotonic())))
            except Empty:
                continue
            if kind == "event":
                if len(events) >= 10000 or (event_handler is not None and not event_handler(value)):
                    result = CallResult("error", detail="invalid or excessive worker events")
                    break
                events.append(value)
                parent.send_bytes(b"ack")
            elif kind == "result" and isinstance(value, CallResult):
                result = value
                break
            else:
                result = CallResult("error", detail="worker returned an invalid envelope")
                break
    except BaseException:
        _stop(pid, own_group)
        raise
    finally:
        cleanup = _stop(pid, own_group)
        parent.close()
    measured_cpu = max(measured_cpu, dict(result.resource_usage).get("cpu_seconds", 0.0))
    if result.status == "complete" and cpu_seconds is not None and measured_cpu > cpu_seconds:
        result = CallResult("cpu_limit", detail=f"worker tree CPU exceeded {cpu_seconds:g} seconds")
    return replace(
        result,
        events=tuple(events),
        cleanup_complete=cleanup,
        resource_usage=(
            ("cpu_seconds", measured_cpu),
            ("wall_seconds", time.monotonic() - (deadline - timeout)),
            ("peak_sampled_tree_rss_bytes", peak_rss),
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
