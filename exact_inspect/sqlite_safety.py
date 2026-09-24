"""Admission and execution bounds for the two prepared SQLite package formats."""

from __future__ import annotations

import sqlite3
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from .contracts import DomainError

CONTEXT_TABLES = {
    "metadata": ("key", "value"),
    "entities": ("iri", "kind", "eligible"),
    "terms": ("iri", "kind", "term", "normalized", "language", "category", "fact_id"),
    "axioms": (
        "id",
        "category",
        "predicate",
        "lexical",
        "datatype",
        "language",
        "payload",
        "payload_bytes",
        "fact",
        "fact_bytes",
        "original_digest",
    ),
    "refs": ("iri", "kind", "axiom_id"),
    "edges": ("id", "child", "parent", "kind", "basis", "axiom_id", "payload"),
}
DECISION_TABLES = {
    "pairs": (
        "id",
        "source",
        "source_kind",
        "target",
        "target_kind",
        "candidate",
        "trace",
        "evidence",
    ),
    "sources": ("id", "iri", "kind", "payload"),
}
_MAX_SCHEMA_OBJECTS = 64
_MAX_ROW_BYTES = 64 * 1024 * 1024
_MAX_RESULT_BYTES = 16 * 1024 * 1024
_PROGRESS_INTERVAL = 1000
_SCHEMA_CACHE: OrderedDict[tuple[object, ...], None] = OrderedDict()
_SCHEMA_CACHE_LOCK = threading.Lock()


class _ReadCursor(sqlite3.Cursor):
    """Bound materialized results while permitting explicit row/blob streaming."""

    @staticmethod
    def _row_bytes(row):
        return sum(sys.getsizeof(value) for value in row) if row is not None else 0

    @staticmethod
    def _check_bytes(size):
        if size > _MAX_RESULT_BYTES:
            raise DomainError(
                "sqlite_result_budget_exceeded",
                "Prepared database result exceeds the browsing budget.",
                413,
            )

    def fetchone(self):
        row = super().fetchone()
        self._check_bytes(self._row_bytes(row))
        return row

    def _materialize(self, size=None):
        result = []
        returned_bytes = 0
        while size is None or len(result) < size:
            row = super().fetchone()
            if row is None:
                break
            returned_bytes += self._row_bytes(row)
            self._check_bytes(returned_bytes)
            result.append(row)
        return result

    def fetchmany(self, size=None):
        return self._materialize(self.arraysize if size is None else size)

    def fetchall(self):
        return self._materialize()

    def __next__(self):
        row = super().__next__()
        self._check_bytes(self._row_bytes(row))
        return row


class _ReadConnection(sqlite3.Connection):
    def cursor(self, factory=_ReadCursor):
        return super().cursor(factory=factory)

    def execute(self, sql, parameters=()):
        return self.cursor().execute(sql, parameters)


def _reject_schema() -> None:
    raise DomainError("unsafe_sqlite_schema", "Prepared database schema is not supported.", 422)


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _validate_schema(
    connection: sqlite3.Connection, expected: Mapping[str, tuple[str, ...]]
) -> None:
    objects = connection.execute(
        "SELECT type,name,tbl_name FROM sqlite_schema LIMIT ?", (_MAX_SCHEMA_OBJECTS + 1,)
    ).fetchall()
    if len(objects) > _MAX_SCHEMA_OBJECTS:
        _reject_schema()
    tables = {}
    for kind, name, owner in objects:
        if kind not in {"table", "index"}:
            _reject_schema()
        if kind == "table":
            if name not in expected and not name.startswith("sqlite_stat"):
                _reject_schema()
            tables[name] = owner
        elif owner not in expected and not owner.startswith("sqlite_stat"):
            _reject_schema()
    if set(expected) != {name for name in tables if not name.startswith("sqlite_stat")}:
        _reject_schema()
    # table_list distinguishes virtual tables even if their visible columns match.
    table_list = connection.execute("PRAGMA table_list").fetchmany(_MAX_SCHEMA_OBJECTS + 1)
    if not table_list or len(table_list) > _MAX_SCHEMA_OBJECTS:
        _reject_schema()
    for database, name, kind, *_ in table_list:
        if name in {"sqlite_schema", "sqlite_temp_schema"}:
            continue
        if database != "main" or name not in tables or kind != "table":
            _reject_schema()
    for name in tables:
        columns = connection.execute(f"PRAGMA table_xinfo({_identifier(name)})").fetchall()
        if any(column[6] != 0 for column in columns):
            _reject_schema()
        if name in expected and tuple(column[1] for column in columns) != expected[name]:
            _reject_schema()
        for index in connection.execute(f"PRAGMA index_list({_identifier(name)})"):
            if index[4]:  # Partial index predicates can contain executable expressions.
                _reject_schema()
            entries = connection.execute(f"PRAGMA index_xinfo({_identifier(index[1])})")
            if any(entry[1] < -1 for entry in entries):  # -2 denotes an expression.
                _reject_schema()


def _admit_schema(connection, path, schema):
    def signature():
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    before = signature()
    key = (str(path.resolve()), before, tuple(sorted(schema.items())))
    with _SCHEMA_CACHE_LOCK:
        if key in _SCHEMA_CACHE:
            _SCHEMA_CACHE.move_to_end(key)
            return
    _validate_schema(connection, schema)
    if signature() != before:
        raise DomainError("artifact_conflict", "Prepared database changed during admission.", 409)
    with _SCHEMA_CACHE_LOCK:
        _SCHEMA_CACHE[key] = None
        while len(_SCHEMA_CACHE) > 128:
            _SCHEMA_CACHE.popitem(last=False)


@contextmanager
def readonly_connection(
    path: Path,
    schema: Mapping[str, tuple[str, ...]],
    *,
    vm_step_budget: int = 50_000_000,
) -> Iterator[sqlite3.Connection]:
    """Admit only ordinary package tables and bound each reader connection's SQL work.

    The VM budget covers schema admission and all statements until the connection
    closes. Row/SQL limits also bound single-operation allocation before a progress
    callback can fire. These limits apply to browsing, not offline preparation.
    """
    connection = sqlite3.connect(
        Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True, factory=_ReadConnection
    )
    interrupted = False
    remaining = vm_step_budget

    def progress() -> int:
        nonlocal interrupted, remaining
        remaining -= _PROGRESS_INTERVAL
        interrupted = remaining < 0
        return int(interrupted)

    try:
        connection.row_factory = sqlite3.Row
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_ROW_BYTES)
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 128 * 1024)
        connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 64)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        connection.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 10)
        connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        connection.setlimit(sqlite3.SQLITE_LIMIT_WORKER_THREADS, 0)
        connection.set_progress_handler(progress, _PROGRESS_INTERVAL)
        # Extensions are disabled by default; explicitly retain that boundary.
        if hasattr(connection, "enable_load_extension"):
            connection.enable_load_extension(False)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("PRAGMA mmap_size=0")
        connection.execute("PRAGMA temp_store=FILE")
        _admit_schema(connection, Path(path), schema)
        yield connection
    except sqlite3.DatabaseError as error:
        if interrupted:
            raise DomainError(
                "sqlite_query_budget_exceeded",
                "Prepared database query exceeded its execution budget.",
                422,
            ) from error
        if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_TOOBIG:
            raise DomainError(
                "sqlite_value_budget_exceeded",
                "Prepared database value exceeds the browsing budget.",
                413,
            ) from error
        raise DomainError(
            "invalid_sqlite_package", "Prepared database cannot be read safely.", 422
        ) from error
    finally:
        connection.close()
