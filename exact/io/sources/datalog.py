"""Strict parser for pre-materialized binary datalog facts."""

from __future__ import annotations

import re
from pathlib import Path

from exact.core.entities.graph import Edge
from exact.io.sources import SourceOptionsError

_FACT = re.compile(r"^([A-Za-z_][\w:./#-]*|'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")\s*\((.*)\)\s*\.$")


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote is not None:
            escaped = True
            continue
        if character in {"'", '"'}:
            quote = None if quote == character else character if quote is None else quote
            continue
        if quote is None and (
            character == "%" or (character == "#" and (index == 0 or line[index - 1].isspace()))
        ):
            return line[:index]
    return line


def _split_arguments(value: str, *, location: str) -> tuple[str, str]:
    pieces: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote is not None:
            escaped = True
            continue
        if character in {"'", '"'}:
            quote = None if quote == character else character if quote is None else quote
            continue
        if character == "," and quote is None:
            pieces.append(value[start:index].strip())
            start = index + 1
    pieces.append(value[start:].strip())
    if quote is not None:
        raise SourceOptionsError(f"Unterminated quoted atom at {location}")
    if len(pieces) != 2 or any(not piece for piece in pieces):
        raise SourceOptionsError(
            f"Only binary facts are supported at {location}; found {len(pieces)} argument(s)"
        )
    return _atom(pieces[0], location=location), _atom(pieces[1], location=location)


def _atom(value: str, *, location: str) -> str:
    if value[0:1] in {"'", '"'}:
        if len(value) < 2 or value[-1] != value[0]:
            raise SourceOptionsError(f"Malformed quoted atom at {location}")
        quote = value[0]
        body = value[1:-1]
        return body.replace(f"\\{quote}", quote).replace("\\\\", "\\")
    if not re.fullmatch(r"[A-Za-z0-9_][\w:./#?=&%+~-]*", value):
        raise SourceOptionsError(f"Unsupported datalog atom {value!r} at {location}")
    return value


def parse_fact(line: str, *, location: str = "fact") -> Edge | None:
    """Parse one binary fact, returning ``None`` for a blank/comment line."""

    normalized = _strip_comment(line).strip()
    if not normalized:
        return None
    if ":-" in normalized:
        raise SourceOptionsError(
            f"Datalog rules are not supported at {location}; pre-materialize facts first"
        )
    match = _FACT.fullmatch(normalized)
    if match is None:
        raise SourceOptionsError(f"Invalid datalog fact at {location}: {normalized!r}")
    relation = _atom(match.group(1), location=location)
    source, target = _split_arguments(match.group(2), location=location)
    return Edge(source, relation, target)


def read_facts(path: Path) -> list[Edge]:
    """Read deterministic unique edges from a datalog fact file."""

    fact_path = Path(path)
    edges: set[Edge] = set()
    with fact_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            edge = parse_fact(line, location=f"{fact_path}:{line_number}")
            if edge is not None:
                edges.add(edge)
    return sorted(edges, key=Edge.astuple)


__all__ = ["parse_fact", "read_facts"]
