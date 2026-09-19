"""Bounded process supervision for blocking native solver and reasoner calls."""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable

_SUPERVISED_GROUP = False


@dataclass(frozen=True)
class CallResult:
    """A completed value or explicit timeout/crash, never an inferred verdict."""

    status: str
    value: Any = None
    detail: str = ""


def _execute(
    connection: Any,
    function: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    own_group: bool,
) -> None:
    global _SUPERVISED_GROUP
    try:
        if own_group:
            os.setsid()
        _SUPERVISED_GROUP = os.name == "posix"
        connection.send(CallResult("complete", function(*args, **kwargs)))
    except BaseException as error:
        connection.send(CallResult("error", detail=f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def bounded_call(
    function: Callable[..., Any],
    *args: Any,
    timeout: float,
    memory_mb: float | None = None,
    **kwargs: Any,
) -> CallResult:
    """Kill an unfinished spawned worker, retaining all parent-owned evidence.

    Spawn avoids inheriting native reasoner/Torch thread state. The deadline includes
    process startup and result transport. Callables and arguments must be pickleable.
    On POSIX a top-level worker owns a process group: nested stage workers can be
    supervised independently, and cancellation kills the entire descendant group.
    """
    if not isfinite(timeout):
        raise ValueError("timeout must be finite")
    if memory_mb is not None and (not isfinite(memory_mb) or memory_mb <= 0):
        raise ValueError("memory_mb must be positive and finite")
    if memory_mb is not None and not Path("/proc/self/statm").exists():
        return CallResult(
            "error", detail="worker RSS memory supervision is unavailable on this platform"
        )
    if timeout <= 0:
        return CallResult("timeout", detail="stage deadline exhausted")
    deadline = time.monotonic() + timeout
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    own_group = os.name == "posix" and not _SUPERVISED_GROUP
    process = context.Process(
        target=_execute, args=(child, function, args, kwargs, own_group), daemon=False
    )
    started = False
    try:
        process.start()
        started = True
        child.close()
        received: list[CallResult] = []
        finished = Event()

        def receive() -> None:
            # poll() can become ready after only a frame header is written; recv()
            # must therefore run outside the supervising deadline thread.
            try:
                value = parent.recv()
                received.append(
                    value
                    if isinstance(value, CallResult)
                    else CallResult("error", detail="worker returned an invalid envelope")
                )
            except Exception as error:
                received.append(
                    CallResult("error", detail=f"worker exited without a result: {error}")
                )
            finally:
                finished.set()

        reader = Thread(target=receive, daemon=True)
        reader.start()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return CallResult("timeout", detail="stage deadline exhausted")
            if finished.wait(min(remaining, 0.05) if memory_mb is not None else remaining):
                return received[0]
            if memory_mb is not None and process.pid is not None:
                if _resident_tree_bytes(process.pid) > memory_mb * 1024 * 1024:
                    return CallResult(
                        "memory_limit", detail=f"worker tree RSS exceeded {memory_mb:g} MiB"
                    )
    except Exception as error:
        return CallResult("error", detail=f"worker startup failed: {error}")
    finally:
        child.close()
        if started:
            # The group ID is the child's unique PID, never the caller's group.
            # Signal it even if its leader has exited but descendants remain.
            if own_group:
                assert process.pid is not None
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.1)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.1)
            if not process.is_alive():
                process.close()
        parent.close()


def _resident_tree_bytes(pid: int) -> int:
    """Read Linux worker-tree RSS (shared resident pages counted per process).

    Sampled enforcement can overshoot between checks. It excludes parent-owned
    inputs and device memory; callers record that scope rather than claim a hard
    operating-system allocation quota.
    """
    pending, seen, pages = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        root = Path(f"/proc/{current}")
        try:
            pages += int((root / "statm").read_text().split()[1])
            for task in (root / "task").iterdir():
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except (OSError, ValueError):
                    continue
        except (OSError, ValueError, IndexError):
            continue
    return pages * os.sysconf("SC_PAGE_SIZE")
