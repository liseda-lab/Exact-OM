"""Small dependency-free helpers shared by Exact-OM I/O backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote, urlsplit


def short_form(identifier: str) -> str:
    """Return the final human-readable component of an IRI-like identifier."""

    text = str(identifier)
    parsed = urlsplit(text)
    if parsed.fragment:
        return unquote(parsed.fragment)
    path = parsed.path.rstrip("/")
    if path:
        return unquote(path.rsplit("/", 1)[-1])
    if ":" in text:
        return text.rsplit(":", 1)[-1]
    return text


def string_tuple(value: Any, *, option: str, default: Sequence[str] = ()) -> tuple[str, ...]:
    """Normalize a string or a sequence of strings used in source options."""

    if value is None:
        return tuple(str(item) for item in default)
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{option} must be a string or a sequence of strings")
    normalized = tuple(str(item) for item in value)
    if any(not item for item in normalized):
        raise ValueError(f"{option} cannot contain empty values")
    return normalized
