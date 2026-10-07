"""Corrective portability, candidate continuation and owned-copy removal contracts."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import (
    BundleLibrary,
    atomic_json,
    export_archive,
    publish_bundle,
)
from exact_inspect.contracts import DomainError, canonical_hash, file_hash
from exact_inspect.decisions import DecisionStore
from exact_inspect.service import create_prepared_app


def package(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    atomic_json(root / "data.json", {"name": name})
    manifest = publish_bundle(
        root, audience="development_demo", capabilities={"context": "not_exported"}
    )
    archive = tmp_path / (name + ".zip")
    export_archive(manifest, archive)
    return manifest, archive


def test_owned_copy_metadata_delete_and_active_selection(tmp_path, monkeypatch):
    original, archive = package(tmp_path, "original")
    original_bytes = original.read_bytes()
    library = BundleLibrary(tmp_path / "library")
    imported = library.import_archive(archive)
    _, second_archive = package(tmp_path, "second")
    second = library.import_archive(second_archive)
    client = TestClient(create_prepared_app(None, library_dir=library.root))
    endpoint = "/api/v1/bundles/" + imported.package_id

    # Metadata reads the root manifest only; no artifact checksums or DB are opened.
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata must not load artifact data")

    with monkeypatch.context() as isolated:
        isolated.setattr("exact_inspect.artifacts.validate_bundle", forbidden)
        metadata = client.get(endpoint).json()
        assert metadata["counts"]["artifacts"] == 1
        assert metadata["artifact_bytes"] > 0
        assert metadata["owned_library_copy"] is True
        assert "provenance" not in metadata and str(tmp_path) not in json.dumps(metadata)
    assert client.post(endpoint + "/select").status_code == 200
    assert client.delete(endpoint).json()["code"] == "package_in_use"
    assert client.get("/api/v1/health").json()["package_id"] == imported.package_id
    assert client.post("/api/v1/bundles/" + second.package_id + "/select").status_code == 200
    assert client.delete(endpoint).status_code == 200
    assert not library.package_path(second.package_id).parent.joinpath("missing").exists()
    assert original.read_bytes() == original_bytes and archive.exists()
    assert client.get("/api/v1/health").json()["package_id"] == second.package_id
    assert client.delete(endpoint).status_code == 404


def test_deletion_denies_upload_read_lease_links_mounts_and_external_copies(tmp_path, monkeypatch):
    original, archive = package(tmp_path, "original")
    library = BundleLibrary(tmp_path / "library")
    imported = library.import_archive(archive)
    root = library.package_path(imported.package_id).parent
    with library.operation(), pytest.raises(DomainError) as blocked:
        library.remove(imported.package_id)
    assert blocked.value.envelope.code == "library_busy"
    descriptor = library.lease(imported.package_id)
    try:
        with pytest.raises(DomainError) as blocked:
            library.remove(imported.package_id)
        assert blocked.value.envelope.code == "package_in_use"
    finally:
        os.close(descriptor)
    (root / "external").symlink_to(original.parent, target_is_directory=True)
    with pytest.raises(DomainError, match="Linked"):
        library.remove(imported.package_id)
    (root / "external").unlink()
    with monkeypatch.context() as isolated:
        isolated.setattr("exact_inspect.artifacts.os.path.ismount", lambda path: Path(path) == root)
        with pytest.raises(DomainError, match="mounted"):
            library.remove(imported.package_id)
    # A package manually placed in the library is not evidence of importer ownership.
    (root / ".library-copy.json").unlink()
    library.import_archive(archive)
    with pytest.raises(DomainError) as blocked:
        library.remove(imported.package_id)
    assert blocked.value.envelope.code == "unowned_package"
    assert original.is_file() and (root / "data.json").is_file()
    shutil.rmtree(root)
    root.symlink_to(original.parent, target_is_directory=True)
    with pytest.raises(DomainError):
        library.remove(imported.package_id)
    assert original.is_file()
    for malformed in ("../original", "sha256:" + "x" * 64, str(original)):
        with pytest.raises(DomainError):
            library.remove(malformed)


def test_hosted_library_denied_and_library_pages_continue(tmp_path):
    original, first_archive = package(tmp_path, "original")
    library = BundleLibrary(tmp_path / "library")
    first = library.import_archive(first_archive)
    _, second_archive = package(tmp_path, "second")
    second = library.import_archive(second_archive)
    demo = TestClient(
        create_prepared_app(original, profile="public_demo", library_dir=library.root)
    )
    for method, suffix in (
        ("get", "/page"),
        ("get", "/" + first.package_id),
        ("delete", "/" + first.package_id),
    ):
        assert getattr(demo, method)("/api/v1/bundles" + suffix).status_code == 404
    local = TestClient(create_prepared_app(original, library_dir=library.root))
    a = local.get("/api/v1/bundles/page", params={"limit": 1}).json()
    b = local.get("/api/v1/bundles/page", params={"limit": 1, "cursor": a["next_cursor"]}).json()
    assert a["order"] == "package_id_ascending"
    assert {a["items"][0]["package_id"], b["items"][0]["package_id"]} == {
        first.package_id,
        second.package_id,
    }
    assert b["next_cursor"] is None
    assert len(local.get("/api/v1/bundles", params={"limit": 1}).json()) == 1


def test_selection_retains_old_copy_until_inflight_request_finishes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    _, archive = package(tmp_path, "first")
    _, second_archive = package(tmp_path, "second")
    library = BundleLibrary(tmp_path / "library")
    first = library.import_archive(archive)
    second = library.import_archive(second_archive)
    library.select(first.package_id)
    app = create_prepared_app(None, library_dir=library.root)
    entered, release = Event(), Event()

    @app.get("/hold-library-read")
    def hold_read():
        entered.set()
        assert release.wait(timeout=10)
        return {"status": "done"}

    client = TestClient(app)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reading = pool.submit(client.get, "/hold-library-read")
        assert entered.wait(timeout=10)
        try:
            assert (
                client.post("/api/v1/bundles/" + second.package_id + "/select").status_code == 200
            )
            assert (
                client.delete("/api/v1/bundles/" + first.package_id).json()["code"]
                == "package_in_use"
            )
        finally:
            release.set()
        assert reading.result().status_code == 200
    assert client.delete("/api/v1/bundles/" + first.package_id).status_code == 200


def test_more_than_500_candidates_keep_ranks_and_honest_pair_id_order(tmp_path):
    database = tmp_path / "decisions.sqlite"
    records = []
    for i in range(607):
        records.append(
            {
                "pair_id": canonical_hash(["pair", i]),
                "target": {"iri": f"urn:{i}"},
                "ordinal_ranks": {"candidate_joint_rank": (i // 3) + 1},
                "scores": [{"value": (i % 7) / 10}],
            }
        )
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE pairs(id TEXT PRIMARY KEY,source TEXT,source_kind TEXT,target TEXT,target_kind TEXT,candidate TEXT,trace TEXT,evidence TEXT); CREATE INDEX pair_source ON pairs(source,source_kind,id); CREATE TABLE sources(id TEXT PRIMARY KEY,iri TEXT,kind TEXT,payload TEXT);"
        )
        connection.executemany(
            "INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    item["pair_id"],
                    "urn:source",
                    "class",
                    item["target"]["iri"],
                    "class",
                    json.dumps(item),
                    "{}",
                    "[]",
                )
                for item in records
            ],
        )
    atomic_json(
        tmp_path / "decisions.manifest.json",
        {
            "database": "decisions.sqlite",
            "database_hash": file_hash(database),
            "source_ontology_version_id": "source-v1",
            "revision": "r1",
        },
    )
    store = DecisionStore(tmp_path)
    seen = []
    cursor = None
    first_cursor = None
    while True:
        page = store.candidates("urn:source", limit=100, cursor=cursor)
        assert page["total_count"] == 607 and page["order"] == "pair_id_ascending"
        seen.extend(page["items"])
        cursor = page["next_cursor"]
        first_cursor = first_cursor or cursor
        if cursor is None:
            break
    assert seen == sorted(records, key=lambda item: item["pair_id"])
    assert [item["ordinal_ranks"] for item in seen] != sorted(
        [item["ordinal_ranks"] for item in seen], key=lambda rank: rank["candidate_joint_rank"]
    )
    with pytest.raises(DomainError, match="another query"):
        store.candidates("urn:other", cursor=first_cursor)
    with pytest.raises(DomainError, match="another query"):
        store.candidates("urn:source", source_kind="individual", cursor=first_cursor)


@pytest.mark.parametrize("uri_default", [0, 1])
def test_portable_export_real_sqlite_uri_defaults_and_read_only_source(tmp_path, uri_default):
    # Change the SQLite library default in a fresh process, before creating any
    # connections. This detects the actual default-off ATTACH failure, not a mock.
    script = r"""
import ctypes, json, os, shutil, sqlite3, sys
from pathlib import Path
import _sqlite3
try:
    lib = ctypes.CDLL(_sqlite3.__file__)
except AttributeError:
    lib = ctypes.CDLL(None)
assert lib.sqlite3_shutdown() == 0
assert lib.sqlite3_config(17, int(sys.argv[2])) == 0
assert lib.sqlite3_initialize() == 0
import pyowl_core as core
from exact_inspect.context import build_context_package, OntologyContext
from exact_inspect.context_export import export_policy_context
from exact_inspect.contracts import VisibilityPolicy, file_hash, EntityRef, DomainError
import exact_inspect.context_export as exporter
root = Path(sys.argv[1]) / 'space café #?'
context = build_context_package(core.load_snapshot(b'Ontology(<urn:o> Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>)) SubClassOf(<urn:A> <urn:B>))'), root / 'source')
before = file_hash(context.database)
connect = sqlite3.connect
class Output(sqlite3.Connection):
    def execute(self, sql, *args):
        result = super().execute(sql, *args)
        if sql.startswith('ATTACH DATABASE'):
            try:
                super().execute('CREATE TABLE original.must_not_write (v TEXT)')
            except sqlite3.OperationalError as error:
                assert 'readonly' in str(error)
            else:
                raise AssertionError('ATTACH permitted writes to original')
        return result
def observed(database, *args, **kwargs):
    if isinstance(database, Path):
        assert kwargs.get('uri') is True
        kwargs['factory'] = Output
    return connect(database, *args, **kwargs)
exporter.sqlite3.connect = observed
portable = export_policy_context(context, root / 'portable', VisibilityPolicy())
assert file_hash(context.database) == before
shutil.move(portable.path, root / 'relocated')
relocated = OntologyContext(root / 'relocated')
assert relocated.hierarchy(EntityRef(ontology_version_id=context.ontology_version_id, iri='urn:A', kind='class'))['items'][0]['parent']['iri'] == 'urn:B'
class Failed(Output):
    def execute(self, sql, *args):
        if sql.startswith('ATTACH DATABASE'):
            raise sqlite3.OperationalError('injected attachment failure')
        return super().execute(sql, *args)
def failing(database, *args, **kwargs):
    if isinstance(database, Path):
        kwargs['factory'] = Failed
    return connect(database, *args, **kwargs)
exporter.sqlite3.connect = failing
try:
    export_policy_context(context, root / 'failed', VisibilityPolicy())
except DomainError:
    pass
else:
    raise AssertionError('Expected export failure')
assert not (root / 'failed').exists() and not list(root.glob('.policy-context-*'))
assert file_hash(context.database) == before
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
barrier = Barrier(2)
class Racing(Output):
    def execute(self, sql, *args):
        if sql.startswith('ATTACH DATABASE'):
            barrier.wait(timeout=10)
        return super().execute(sql, *args)
def racing(database, *args, **kwargs):
    if isinstance(database, Path):
        kwargs['factory'] = Racing
    return connect(database, *args, **kwargs)
exporter.sqlite3.connect = racing
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(export_policy_context, context, root / 'concurrent', VisibilityPolicy()) for _ in range(2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except OSError:
            outcomes.append(None)
assert sum(value is not None for value in outcomes) == 1
assert OntologyContext(root / 'concurrent').manifest['policy_filter']['policy_hash'] == VisibilityPolicy().policy_hash
assert not list(root.glob('.policy-context-*')) and file_hash(context.database) == before
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path), str(uri_default)], check=True)
