"""Opt-in engineering timing; never part of a model or resume identity.

Rows are nested inclusive spans, not additive costs. A killed span has a recorded
start without an end and is not silently called complete on the next invocation.
The campaign ledger remains the cost authority.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import json
import os
import resource
import time
import uuid
from pathlib import Path

_recorder = contextvars.ContextVar("repair_phase_recorder", default=None)


def _usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(
        epoch=time.time(),
        monotonic=time.monotonic(),
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        max_rss_kib=usage.ru_maxrss,
        input_blocks=usage.ru_inblock,
        output_blocks=usage.ru_oublock,
    )


@contextlib.contextmanager
def recording(path):
    """Append a new invocation; never overwrite a predecessor trace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as stream:
        token = _recorder.set(
            dict(
                stream=stream,
                invocation=uuid.uuid4().hex,
                sequence=0,
                stack=[],
                source={
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("phase_timing.py", "train.py")
                },
            )
        )
        try:
            yield
        finally:
            _recorder.reset(token)


@contextlib.contextmanager
def phase(name, **identity):
    recorder = _recorder.get()
    details = dict(identity)
    if recorder is None:
        yield details
        return
    recorder["sequence"] += 1
    span = recorder["sequence"]
    parent = recorder["stack"][-1] if recorder["stack"] else None
    start = _usage()
    base = dict(
        schema="exact-repair/engineering-phase/v1",
        invocation=recorder["invocation"],
        span=span,
        parent=parent,
        phase=name,
        pid=os.getpid(),
        source=recorder["source"],
        inclusive=True,
    )

    def emit(value):
        recorder["stream"].write(json.dumps({**base, **value}, sort_keys=True) + "\n")
        recorder["stream"].flush()

    emit(dict(event="start", usage=start, identity=details))
    recorder["stack"].append(span)
    status = "complete"
    try:
        yield details
    except BaseException as error:
        status = type(error).__name__
        raise
    finally:
        end = _usage()
        recorder["stack"].pop()
        emit(
            dict(
                event="end",
                status=status,
                usage=end,
                elapsed_seconds=end["monotonic"] - start["monotonic"],
                cpu_seconds=end["cpu_seconds"] - start["cpu_seconds"],
                input_blocks=end["input_blocks"] - start["input_blocks"],
                output_blocks=end["output_blocks"] - start["output_blocks"],
                identity=details,
            )
        )


def traced_training(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        path = os.environ.get("EXACT_REPAIR_PHASE_LOG")
        context = recording(path) if path and _recorder.get() is None else contextlib.nullcontext()
        with context, phase("training_invocation"):
            return function(*args, **kwargs)

    return wrapped


def load_training_state(path, *, map_location):
    import torch

    with phase(
        "checkpoint_read_deserialize", path=str(path), bytes=Path(path).stat().st_size
    ) as row:
        state = torch.load(path, weights_only=True, map_location=map_location)
        row.update(checkpoint_identity(state))
    return state


def checkpoint_identity(state):
    return {
        key: state.get(key)
        for key in (
            "identity",
            "phase",
            "next_epoch",
            "next_offset",
            "optimizer_updates",
            "execution_count",
        )
    }
