"""Untrusted prepared bytes cannot introduce executable or unbounded SQL reads."""

import json
import sqlite3

import pytest

from exact_inspect.contracts import DomainError, file_hash
from exact_inspect.decisions import DecisionStore
from exact_inspect.sqlite_safety import readonly_connection

_TABLES = {"items": ("id", "payload")}


def _database(tmp_path, sql="CREATE TABLE items(id TEXT PRIMARY KEY,payload TEXT)"):
    path = tmp_path / "index.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(sql)
        connection.commit()
    finally:
        connection.close()
    return path


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE VIEW items AS SELECT 1 id,zeroblob(2000000000) payload",
        "CREATE TABLE items(id TEXT,payload TEXT);"
        "CREATE TRIGGER hazard AFTER INSERT ON items BEGIN SELECT zeroblob(2000000000); END",
        "CREATE VIRTUAL TABLE items USING fts5(id,payload)",
        "CREATE TABLE items(id TEXT,payload TEXT GENERATED ALWAYS AS (zeroblob(2000000000)))",
        "CREATE TABLE items(id TEXT,payload TEXT);CREATE INDEX hazard ON items(lower(payload))",
        "CREATE TABLE items(id TEXT,payload TEXT);"
        "CREATE INDEX hazard ON items(id) WHERE length(payload)>0",
        "CREATE TABLE items(id TEXT,unexpected TEXT)",
    ],
)
def test_schema_admission_rejects_executable_and_unexpected_objects(tmp_path, sql):
    path = _database(tmp_path, sql)
    with pytest.raises(DomainError) as error:
        with readonly_connection(path, _TABLES):
            pytest.fail("Unsafe database was admitted")
    assert error.value.envelope.code == "unsafe_sqlite_schema"


def test_physical_indexed_tables_and_analyze_statistics_are_admitted(tmp_path):
    path = _database(
        tmp_path,
        "CREATE TABLE items(id TEXT PRIMARY KEY,payload TEXT);"
        "INSERT INTO items VALUES ('one','safe');ANALYZE",
    )
    with readonly_connection(path, _TABLES) as connection:
        assert tuple(connection.execute("SELECT * FROM items").fetchone()) == ("one", "safe")
        assert connection.execute("PRAGMA trusted_schema").fetchone()[0] == 0
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1


def test_query_vm_budget_stops_work_even_when_result_is_one_row(tmp_path):
    path = _database(tmp_path)
    with pytest.raises(DomainError) as error:
        with readonly_connection(path, _TABLES, vm_step_budget=50_000) as connection:
            connection.execute(
                "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<1000000) "
                "SELECT sum(x) FROM n"
            ).fetchone()
    assert error.value.envelope.code == "sqlite_query_budget_exceeded"


def test_single_value_and_total_result_allocations_are_bounded(tmp_path):
    path = _database(tmp_path, "CREATE TABLE items(id TEXT,payload TEXT)")
    with pytest.raises(DomainError) as error:
        with readonly_connection(path, _TABLES) as connection:
            connection.execute("SELECT zeroblob(2000000000)").fetchone()
    assert error.value.envelope.code == "sqlite_value_budget_exceeded"
    with sqlite3.connect(path) as connection:
        connection.executemany("INSERT INTO items VALUES (?, '')", [(str(i),) for i in range(20)])
    with pytest.raises(DomainError) as error:
        with readonly_connection(path, _TABLES) as connection:
            connection.execute("SELECT zeroblob(1048576) FROM items").fetchall()
    assert error.value.envelope.code == "sqlite_result_budget_exceeded"
    # Offline exporters stream one row at a time; cumulative bytes are not retained.
    with readonly_connection(path, _TABLES) as connection:
        count = 0
        for row in connection.execute("SELECT zeroblob(1048576) FROM items"):
            assert len(row[0]) == 1048576
            count += 1
        assert count == 20


def test_decision_store_rejects_hash_valid_malicious_view_before_any_get(tmp_path):
    database = tmp_path / "decisions.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE sources(id TEXT PRIMARY KEY,iri TEXT,kind TEXT,payload TEXT);"
            "CREATE VIEW pairs AS SELECT 1 id, 's' source,'class' source_kind,"
            "'t' target,'class' target_kind,zeroblob(2000000000) candidate,'{}' trace,'[]' evidence"
        )
    (tmp_path / "decisions.manifest.json").write_text(
        json.dumps({"database": database.name, "database_hash": file_hash(database)})
    )
    with pytest.raises(DomainError) as error:
        DecisionStore(tmp_path)
    assert error.value.envelope.code == "unsafe_sqlite_schema"


def test_cached_admission_rechecks_changed_bytes_and_expected_schema(tmp_path):
    path = _database(tmp_path)
    with readonly_connection(path, _TABLES):
        pass
    with pytest.raises(DomainError, match="schema"):
        with readonly_connection(path, {"items": ("different", "payload")}):
            pass
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "DROP TABLE items;CREATE VIEW items AS SELECT 'id' id, 'bad' payload"
        )
    with pytest.raises(DomainError, match="schema"):
        with readonly_connection(path, _TABLES):
            pass


@pytest.mark.parametrize("family", ["context", "decisions"])
def test_nested_manifests_have_a_byte_budget_before_database_access(tmp_path, family):
    from exact_inspect.context import OntologyContext

    name = "manifest.json" if family == "context" else "decisions.manifest.json"
    with (tmp_path / name).open("wb") as stream:
        stream.truncate(8 * 1024**2 + 1)
    with pytest.raises(DomainError) as error:
        (OntologyContext if family == "context" else DecisionStore)(tmp_path)
    assert error.value.envelope.code == "invalid_manifest"
    assert error.value.status_code == 413


@pytest.mark.parametrize("locator", ["../outside.sqlite", "/tmp/outside.sqlite", "other.sqlite"])
def test_decision_metadata_cannot_redirect_database_reads(tmp_path, locator):
    (tmp_path / "decisions.manifest.json").write_text(json.dumps({"database": locator}))
    with pytest.raises(DomainError, match="locator"):
        DecisionStore(tmp_path)
