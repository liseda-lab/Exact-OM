"""Bounded process supervision for blocking native solver and reasoner calls."""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from math import isfinite
from threading import Event, Thread
from typing import Any, Callable


@dataclass(frozen=True)
class CallResult:
    """A completed value or explicit timeout/crash, never an inferred verdict."""

    status: str
    value: Any = None
    detail: str = ""


def _execute(connection: Any, function: Callable[..., Any], args: tuple, kwargs: dict) -> None:
    try:
        connection.send(CallResult("complete", function(*args, **kwargs)))
    except BaseException as error:
        connection.send(CallResult("error", detail=f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def bounded_call(
    function: Callable[..., Any], *args: Any, timeout: float, **kwargs: Any
) -> CallResult:
    """Kill an unfinished spawned worker, retaining all parent-owned evidence.

    Spawn avoids inheriting native reasoner/Torch thread state. The deadline includes
    process startup and result transport. Callables and arguments must be pickleable.
    """
    if not isfinite(timeout):
        raise ValueError("timeout must be finite")
    if timeout <= 0:
        return CallResult("timeout", detail="stage deadline exhausted")
    deadline = time.monotonic() + timeout
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_execute, args=(child, function, args, kwargs), daemon=True)
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
        if finished.wait(max(0.0, deadline - time.monotonic())):
            return received[0]
        return CallResult("timeout", detail="stage deadline exhausted")
    except Exception as error:
        return CallResult("error", detail=f"worker startup failed: {error}")
    finally:
        child.close()
        if started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.1)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.1)
            if not process.is_alive():
                process.close()
        parent.close()
