"""Context-local metadata shared by nested actions within one run."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_CURRENT_RUN_SESSION: ContextVar[Optional[Any]] = ContextVar(
    "exact_current_run_session",
    default=None,
)


def current_run_session() -> Optional[Any]:
    """Return the active timing session, if the caller is inside one."""

    return _CURRENT_RUN_SESSION.get()


@contextmanager
def activate_run_session(session: Any) -> Iterator[Any]:
    """Make ``session`` visible to nested actions for this execution context."""

    token = _CURRENT_RUN_SESSION.set(session)
    try:
        yield session
    finally:
        _CURRENT_RUN_SESSION.reset(token)
