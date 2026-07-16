"""Boundary-safe hooks for alignment relation typing and serialization.

The core trainer owns orchestration but must not depend on concrete I/O
adapters.  The implementation layer binds these hooks to :mod:`exact.io`
when the built-in trainer is imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

RelationTyper = Callable[..., Any]
WriterDispatch = Callable[..., Path]

_relation_typer: Optional[RelationTyper] = None
_writer_dispatch: Optional[WriterDispatch] = None


def bind_alignment_io(
    *,
    relation_typer: RelationTyper,
    writer_dispatch: WriterDispatch,
) -> None:
    """Bind concrete alignment I/O callables during implementation bootstrap."""

    global _relation_typer, _writer_dispatch
    _relation_typer = relation_typer
    _writer_dispatch = writer_dispatch


def type_alignment_relations(
    candidates: Any,
    source: Any,
    target: Any,
    *,
    mode: str,
) -> Any:
    """Run the configured relation typer over a canonical mapping table."""

    if _relation_typer is None:
        raise RuntimeError(
            "Alignment I/O is not initialized; import the built-in trainer or call "
            "exact.impl.bootstrap_components() first."
        )
    return _relation_typer(candidates, source, target, mode=mode)


def write_alignment_format(
    name: str,
    mappings: Any,
    output_dir: Path,
    *,
    options: Optional[Mapping[str, Any]] = None,
    filename: Optional[str] = None,
) -> Path:
    """Serialize mappings through the configured writer registry."""

    if _writer_dispatch is None:
        raise RuntimeError(
            "Alignment I/O is not initialized; import the built-in trainer or call "
            "exact.impl.bootstrap_components() first."
        )
    return _writer_dispatch(
        name,
        mappings,
        output_dir,
        options=options,
        filename=filename,
    )


__all__ = [
    "RelationTyper",
    "WriterDispatch",
    "bind_alignment_io",
    "type_alignment_relations",
    "write_alignment_format",
]
