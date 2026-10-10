"""Durable review lifecycle, immutable inputs, TLS and owned-process boundaries."""

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from tools import review_environment as review


@pytest.fixture
def backend_checkout(tmp_path):
    """A small real Git checkout isolates source identity from ontology data."""
    root = tmp_path / "backend"
    for name in (
        "exact_inspect/service.py",
        "exact_inspect/study/api.py",
        "exact_inspect/study/owl_ast_schema.json",
        "exact_inspect/study/migrations/001_initial.sql",
        "tools/review_environment.py",
        "pyproject.toml",
        "deploy/render/exact_study_requirements.txt",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen " + name)
    review.command(["git", "init", "--quiet", str(root)])
    review.command(["git", "-C", str(root), "add", "."])
    review.command(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Review Test",
            "-c",
            "user.email=review@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "Frozen runtime fixture",
        ]
    )
    return root


def test_backend_identity_is_path_independent_and_ignores_private_data(backend_checkout, tmp_path):
    import shutil

    first = review.backend_identity(backend_checkout)
    relocated = tmp_path / "relocated"
    shutil.copytree(backend_checkout, relocated)
    assert review.backend_identity(relocated) == first
    for name in ("data/secret.json", "environment.json", "exact_inspect/static/app.json"):
        path = relocated / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private-marker-must-not-affect-runtime")
    assert review.backend_identity(relocated) == first
    assert first["file_count"] == 7
    assert set(first) == {"schema", "git_commit", "source_sha256", "file_count"}


def test_documentation_commit_keeps_recorded_backend_provenance(backend_checkout):
    config = {"repo": str(backend_checkout), "backend": review.backend_identity(backend_checkout)}
    recorded = copy.deepcopy(config)
    (backend_checkout / "README.md").write_text("Documentation-only change")
    review.command(["git", "-C", str(backend_checkout), "add", "README.md"])
    review.command(
        [
            "git",
            "-C",
            str(backend_checkout),
            "-c",
            "user.name=Review Test",
            "-c",
            "user.email=review@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "Documentation-only fixture",
        ]
    )
    current = review.verify_backend(config)
    assert current["git_commit"] != config["backend"]["git_commit"]
    assert current["source_sha256"] == config["backend"]["source_sha256"]
    assert config == recorded


@pytest.mark.parametrize(
    "name",
    [
        "exact_inspect/service.py",
        "exact_inspect/study/api.py",
        "exact_inspect/study/owl_ast_schema.json",
        "exact_inspect/study/migrations/001_initial.sql",
        "exact_inspect/study/new_runtime.py",
        "tools/review_environment.py",
        "pyproject.toml",
        "deploy/render/exact_study_requirements.txt",
    ],
)
def test_backend_source_drift_refuses_restart_without_rewriting_identity(backend_checkout, name):
    config = {"repo": str(backend_checkout), "backend": review.backend_identity(backend_checkout)}
    recorded = copy.deepcopy(config)
    (backend_checkout / name).write_text("changed runtime bytes")
    with pytest.raises(
        review.ReviewError, match="Backend source bytes changed.*separate durable root"
    ):
        review.preflight(config)
    assert config == recorded
    with pytest.raises(review.ReviewError, match="Backend source bytes changed"):
        review.serve(config, "study", "a" * 32)


def test_legacy_backend_identity_is_actionable_and_never_migrated():
    legacy = {"repo": "/private-legacy-checkout"}
    with pytest.raises(review.ReviewError, match="no backend fingerprint.*separate durable root"):
        review.preflight(legacy)
    assert legacy == {"repo": "/private-legacy-checkout"}


def test_backend_identity_refuses_missing_sources_or_outside_symlinks(backend_checkout, tmp_path):
    service = backend_checkout / "exact_inspect/service.py"
    service.unlink()
    with pytest.raises(review.ReviewError, match="Backend source files are missing"):
        review.backend_identity(backend_checkout)
    outside = tmp_path / "outside-service.py"
    outside.write_text("not part of the checkout")
    service.symlink_to(outside)
    with pytest.raises(review.ReviewError, match="outside its checkout"):
        review.backend_identity(backend_checkout)


def publication():
    return {
        "definition": {
            "purpose": "development_review",
            "synthetic": False,
            "correctness_metrics": "disabled",
            "contract_version": "exact-study/2.0",
            "study_revision": "source-review",
            "closes_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            "protocol_versions": {
                "forms": "exact-study-forms/3",
                "setup": "setup/3",
                "consultation": "consultation/3",
                "methods": "study-methods/3",
                "export": "exact-study-analysis/4",
            },
            "assets": [],
        }
    }


def test_refuses_research_expired_or_old_protocol_publications():
    raw = publication()
    assert review.validate_publication(raw)["purpose"] == "development_review"
    for patch in (
        {"purpose": "research"},
        {"synthetic": True},
        {"correctness_metrics": "enabled"},
        {"protocol_versions": {"forms": "exact-study-forms/2"}},
    ):
        changed = copy.deepcopy(raw)
        changed["definition"].update(patch)
        with pytest.raises(review.ReviewError):
            review.validate_publication(changed)
    raw["definition"]["closes_at"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(review.ReviewError, match="new-revision"):
        review.validate_publication(raw)
    review.validate_publication(raw, allow_expired=True)


def test_fresh_revision_preserves_source_and_previous_publication_bytes(tmp_path):
    source = tmp_path / "source.json"
    review.private_json(source, publication())
    config = {"root": str(tmp_path), "source_publication": str(source), "hashes": {}}
    original = source.read_bytes()
    review.new_publication(config)
    first = Path(config["publication"])
    frozen = first.read_bytes()
    first_revision = config["study_revision"]
    review.new_publication(config)
    assert config["study_revision"] != first_revision
    assert first.read_bytes() == frozen and source.read_bytes() == original
    assert Path(config["publication"]).stat().st_mode & 0o777 == 0o600
    assert review.file_hash(config["publication"]) == config["hashes"]["publication"]
    with pytest.raises(review.ReviewError):
        review.new_publication(config, days=0)


def test_asset_hash_size_and_path_admission(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    value = assets / "resource.json"
    value.write_text("immutable")
    raw = publication()
    raw["definition"]["assets"] = [
        {"path": value.name, "sha256": review.file_hash(value), "size_bytes": 9}
    ]
    review.verify_assets(raw, assets)
    value.write_text("corrupted")
    with pytest.raises(review.ReviewError, match="changed"):
        review.verify_assets(raw, assets)
    raw["definition"]["assets"][0]["path"] = "../outside.json"
    with pytest.raises(review.ReviewError, match="outside"):
        review.verify_assets(raw, assets)


def test_durable_paths_and_port_collisions():
    with pytest.raises(review.ReviewError, match="durable"):
        review.durable("/tmp/review-only-copy")
    with pytest.raises(review.ReviewError, match="distinct"):
        review.validate_ports({"postgres": 19183, "study": 19183, "exploration": 19185})
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        assert not review.port_available(listener.getsockname()[1])


def test_prepare_refuses_to_reinitialize_existing_database(tmp_path, monkeypatch):
    root = tmp_path / "existing"
    pgdata = root / "pgdata"
    pgdata.mkdir(parents=True)
    sentinel = pgdata / "saved-participation"
    sentinel.write_bytes(b"do not change")
    monkeypatch.setattr(review, "durable", lambda value: Path(value).resolve())
    args = argparse.Namespace(
        root=root,
        publication=tmp_path / "publication",
        assets=tmp_path,
        package=tmp_path / "package",
        frontend=tmp_path,
        python=Path(sys.executable),
        pgbin=tmp_path,
    )
    with pytest.raises(review.ReviewError, match="never be initialized"):
        review.prepare(args)
    assert sentinel.read_bytes() == b"do not change"
    assert list(pgdata.iterdir()) == [sentinel]


def test_owned_child_can_stop_but_reused_identity_cannot(tmp_path):
    config = {"root": str(tmp_path)}
    nonce = uuid4().hex
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(tmp_path), nonce]
    )
    try:
        review.wait_for(lambda: nonce in (review.process_identity(child.pid) or ""), timeout=5)
        record = {"pid": child.pid, "nonce": nonce, "identity": review.process_identity(child.pid)}
        assert review.owned(config, "study", record)
        forged = {**record, "nonce": "not-owned"}
        review.private_json(tmp_path / "processes.json", {"study": forged})
        with pytest.raises(review.ReviewError, match="ownership changed"):
            review.stop_role(config, "study")
        assert child.poll() is None
        review.private_json(tmp_path / "processes.json", {"study": record})
        review.stop_role(config, "study")
        assert child.wait(timeout=5) != 0
        assert review.processes(config) == {}
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_tls_validates_san_expiry_and_matching_private_key(tmp_path):
    cert, key = tmp_path / "localhost.pem", tmp_path / "localhost-key.pem"
    config = tmp_path / "openssl.cnf"
    config.write_text(
        "[req]\ndistinguished_name=dn\nx509_extensions=local\nprompt=no\n[dn]\nCN=Review test\n[local]\nsubjectAltName=DNS:localhost,IP:127.0.0.1\nbasicConstraints=critical,CA:TRUE\n"
    )
    review.command(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-config",
            str(config),
        ]
    )
    review.validate_tls(cert, key)
    with pytest.raises(review.ReviewError, match="do not match"):
        review.validate_tls(cert, config)
    # Wrong SAN is rejected even though the certificate is self-consistent and trusted locally.
    config.write_text(config.read_text().replace("127.0.0.1", "127.0.0.10"))
    review.command(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-config",
            str(config),
        ]
    )
    with pytest.raises(review.ReviewError, match="cover"):
        review.validate_tls(cert, key)


def test_new_session_is_test_only_with_private_token_receipt(tmp_path, monkeypatch):
    source = tmp_path / "publication.json"
    review.private_json(source, publication())
    config = {
        "root": str(tmp_path),
        "study_revision": "review-example",
        "publication": str(source),
        "ports": {"study": 19284},
    }
    monkeypatch.setattr(review, "preflight", lambda *a, **kw: None)
    monkeypatch.setattr(
        review,
        "status",
        lambda c, **kwargs: {
            "services": {r: {"state": "owned", "ready": True} for r in review.ROLES}
        },
    )
    calls = []

    def request(config, role, path, **kwargs):
        calls.append((role, path, kwargs))
        return {
            "invitations": [
                {
                    "session_id": "test-session",
                    "invitation": "https://127.0.0.1/?secret=private-test-token",
                }
            ]
        }

    monkeypatch.setattr(review, "http", request)
    result = review.fresh_session(config)
    assert calls[0][2]["timeout"] == 600
    assert calls[-1][2] == {"body": {"count": 1, "test": True}, "authenticated": True}
    assert "private-test-token" not in json.dumps(result)
    receipt = Path(result["private_invitation_file"])
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert "private-test-token" in receipt.read_text()
    assert review.read_json(receipt)["client_contract"] == "study-review/1"


def test_start_failure_rolls_back_only_newly_started_owned_processes(tmp_path, monkeypatch):
    config = {"root": str(tmp_path), "database_name": "review"}
    old = {"pid": 3, "nonce": "old", "identity": "previously-owned"}
    monkeypatch.setattr(review, "preflight", lambda config: None)
    monkeypatch.setattr(review, "require_open_publication", lambda config: None)
    monkeypatch.setattr(review, "processes", lambda config: {"postgres": old})
    monkeypatch.setattr(review, "owned", lambda *args: True)
    monkeypatch.setattr(review, "pg_ready", lambda *args: True)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, *args):
            return self

        def fetchone(self):
            return (1,)

    monkeypatch.setattr(review, "database", lambda *args: Connection())
    monkeypatch.setattr(review, "launch", lambda *args: {"pid": 4})
    monkeypatch.setattr(
        review, "http", lambda *args: (_ for _ in ()).throw(review.ReviewError("unavailable"))
    )
    stopped = []
    monkeypatch.setattr(review, "stop_role", lambda c, r: stopped.append(r))
    with pytest.raises(review.ReviewError):
        review.start(config)
    assert stopped == ["study"]


def test_real_postgresql_restart_preserves_database_and_owned_identity(tmp_path):
    pgbin = os.environ.get("EXACT_REVIEW_TEST_PGBIN")
    if not pgbin:
        pytest.skip("Set a dedicated PGBIN to exercise process-level PostgreSQL preservation")
    port = int(os.environ.get("EXACT_REVIEW_TEST_PGPORT", "19283"))
    if not review.port_available(port):
        pytest.skip("Dedicated review-test PostgreSQL port is occupied; leave it untouched")
    password_file = tmp_path / "password"
    password_file.write_text("test-only-password")
    review.command(
        [
            str(Path(pgbin) / "initdb"),
            "-D",
            str(tmp_path / "pgdata"),
            "-U",
            "review",
            "--pwfile=" + str(password_file),
            "--auth-host=scram-sha-256",
            "--auth-local=scram-sha-256",
            "--no-locale",
            "--encoding=UTF8",
        ],
        timeout=90,
    )
    password_file.unlink()
    config = {
        "root": str(tmp_path),
        "repo": str(Path(__file__).resolve().parents[1]),
        "pgbin": pgbin,
        "ports": {"postgres": port},
        "database_password": "test-only-password",
        "database_name": "postgres",
    }
    before = (tmp_path / "pgdata" / "PG_VERSION").read_bytes()
    try:
        first = review.launch(config, "postgres")
        review.wait_for(lambda: review.pg_ready(config, first))
        with review.database(config) as db:
            db.execute("CREATE TABLE durable_review (value text)")
            db.execute("INSERT INTO durable_review VALUES ('preserved')")
        review.stop_role(config, "postgres")
        second = review.launch(config, "postgres")
        assert second["nonce"] != first["nonce"]
        review.wait_for(lambda: review.pg_ready(config, second))
        with review.database(config) as db:
            assert db.execute("SELECT value FROM durable_review").fetchone() == ("preserved",)
        assert (tmp_path / "pgdata" / "PG_VERSION").read_bytes() == before
        # Exercise closure through the real authenticated API and owned PostgreSQL,
        # using invented ontology content solely for publication mechanics.
        from fastapi.testclient import TestClient
        from exact_inspect.study.api import create_study_app
        from tests.explanation_review_publication_test import review_fixture

        assets = tmp_path / "closure-assets"
        raw = review_fixture(assets)
        source = tmp_path / "source-publication.json"
        review.private_json(source, raw)
        frozen_bytes = source.read_bytes()
        config.update(
            publication=str(source),
            source_publication=str(source),
            study_revision=raw["definition"]["study_revision"],
            hashes={},
        )
        app = create_study_app(
            f"postgresql://review:test-only-password@127.0.0.1:{port}/postgres",
            "s" * 48,
            "r" * 48,
            "https://localhost",
            assets,
        )
        with TestClient(app, base_url="https://localhost") as client:
            headers = {"Authorization": "Bearer " + "r" * 48}
            assert (
                client.post("/api/v1/admin/studies", json=raw, headers=headers).status_code == 200
            )
            assert review.require_open_publication(config)["available_for_test_sessions"]
            old_revision = config["study_revision"]
            assert (
                client.post(
                    f"/api/v1/admin/studies/{old_revision}/close", headers=headers
                ).status_code
                == 200
            )
            state = review.publication_state(config)
            assert state["closed"] is True and not state["available_for_test_sessions"]
            assert state["closes_at"] == raw["definition"]["closes_at"]
            with pytest.raises(review.ReviewError, match="fresh-session --new-revision"):
                review.require_open_publication(config)
            new = review.new_publication(config)
            assert (
                client.post("/api/v1/admin/studies", json=new, headers=headers).status_code == 200
            )
            assert review.require_open_publication(config)["available_for_test_sessions"]
            with review.database(config) as db:
                assert db.execute(
                    "SELECT closed FROM studies WHERE revision=%s", (old_revision,)
                ).fetchone() == (1,)
            assert source.read_bytes() == frozen_bytes
    finally:
        review.stop_role(config, "postgres")


def test_workspace_hash_preflight_is_independent_of_exploration_bundle(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    for name in ("index.sqlite", "context/manifest.json", "context/context.sqlite"):
        (tmp_path / name).write_text("frozen " + name)
    raw = publication()
    raw["definition"]["workspace_scopes"] = [
        {
            "index_path": "index.sqlite",
            "index_sha256": review.file_hash(tmp_path / "index.sqlite"),
            "contexts": [
                {
                    "path": "context",
                    "manifest_sha256": review.file_hash(context / "manifest.json"),
                    "database_sha256": review.file_hash(context / "context.sqlite"),
                }
            ],
        }
    ]
    review.verify_workspaces(raw, tmp_path)
    for name in ("index.sqlite", "context/manifest.json", "context/context.sqlite"):
        target = tmp_path / name
        original = target.read_bytes()
        target.write_text("changed")
        with pytest.raises(review.ReviewError, match="workspace"):
            review.verify_workspaces(raw, tmp_path)
        target.write_bytes(original)
    raw["definition"]["workspace_scopes"][0]["index_path"] = "../outside.sqlite"
    with pytest.raises(review.ReviewError, match="outside"):
        review.verify_workspaces(raw, tmp_path)


def test_readiness_does_not_accept_a_replacement_listener(monkeypatch):
    ownership = iter((True, False))
    monkeypatch.setattr(review, "owned", lambda *args: next(ownership))
    monkeypatch.setattr(review, "http", lambda *args: {"ready": True})
    assert review.service_ready({}, "study", {"pid": 10}) is False
    monkeypatch.setattr(review, "owned", lambda *args: False)
    monkeypatch.setattr(review, "http", lambda *args: pytest.fail("Never probe an unowned worker"))
    assert review.service_ready({}, "study", {"pid": 10}) is False


def test_certificate_renewal_preserves_old_files_and_database(tmp_path, monkeypatch):
    cert, key = review.generate_tls(tmp_path / "tls")
    previous = (cert.read_bytes(), key.read_bytes())
    database = tmp_path / "pgdata" / "participation"
    database.parent.mkdir()
    database.write_bytes(b"saved session")
    config = {
        "root": str(tmp_path),
        "certificate": str(cert),
        "private_key": str(key),
        "hashes": {"certificate": review.file_hash(cert)},
    }
    calls = []
    monkeypatch.setattr(review, "preflight", lambda config, **kwargs: calls.append(kwargs))
    result = review.renew_certificate(config)
    assert calls == [{"allow_expired": True, "allow_closed": True, "check_tls": False}]
    assert result["database_preserved"] is True
    assert (cert.read_bytes(), key.read_bytes()) == previous
    assert database.read_bytes() == b"saved session"
    assert config["certificate"] != str(cert)
    assert review.file_hash(config["certificate"]) == config["hashes"]["certificate"]
    review.validate_tls(config["certificate"], config["private_key"])


def test_new_revision_recovers_after_interrupted_admission(tmp_path, monkeypatch):
    source = tmp_path / "publication.json"
    review.private_json(source, publication())
    config = {
        "root": str(tmp_path),
        "publication": str(source),
        "source_publication": str(source),
        "study_revision": "source-review",
        "ports": {"study": 19284},
        "hashes": {},
    }
    monkeypatch.setattr(review, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        review,
        "status",
        lambda config, **kwargs: {
            "services": {r: {"state": "owned", "ready": True} for r in review.ROLES}
        },
    )

    def interrupted(*args, **kwargs):
        raise review.ReviewError("interrupted admission")

    monkeypatch.setattr(review, "http", interrupted)
    with pytest.raises(review.ReviewError, match="interrupted"):
        review.fresh_session(config, renew=True)
    saved = review.read_json(tmp_path / "environment.json")
    assert saved["study_revision"] != "source-review"
    calls = []

    def retried(config, role, path, **kwargs):
        calls.append((path, kwargs))
        return {"invitations": [{"invitation": "private-token"}]}

    monkeypatch.setattr(review, "http", retried)
    result = review.fresh_session(saved)
    assert result["study_revision"] == saved["study_revision"]
    assert calls[0][0] == "/api/v1/admin/studies"
    assert calls[0][1]["body"]["definition"]["study_revision"] == saved["study_revision"]
    assert calls[1][1]["body"] == {"test": True, "count": 1}


def test_exploration_readiness_uses_actual_registered_route(monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from exact_inspect.service import create_prepared_app

    app = create_prepared_app(None, profile="local_app")
    app.state.service = SimpleNamespace(
        manifest=SimpleNamespace(package_id="test-package", capabilities={})
    )
    client = TestClient(app)
    requests = []

    def actual_http(config, role, path):
        requests.append(path)
        response = client.get(path)
        assert response.status_code == 200, response.text
        return response.json()

    monkeypatch.setattr(review, "http", actual_http)
    monkeypatch.setattr(review, "owned", lambda *args: True)
    assert review.service_ready({}, "exploration", {"pid": 2})
    assert requests == ["/api/v1/health"]


def test_macos_python_exec_keeps_start_time_and_complete_owned_arguments():
    suffix = " -m tools.review_environment _serve --root /durable/review --role study --nonce random-owned-nonce"
    before = "Sat Oct 10 13:58:26 2026 /durable/venv/bin/python" + suffix
    after = "Sat Oct 10 13:58:26 2026 /Library/Python.app/Contents/MacOS/Python" + suffix
    assert review.stable_worker_identity(before) == review.stable_worker_identity(after)
    assert review.stable_worker_identity(before) != review.stable_worker_identity(
        after.replace("13:58:26", "13:58:27")
    )
    assert review.stable_worker_identity(before) != review.stable_worker_identity(
        after.replace("random-owned-nonce", "other-nonce")
    )
    assert review.stable_worker_identity(before) != review.stable_worker_identity(
        after.replace("--role study", "--role exploration")
    )


def test_http_health_timeout_stays_short_while_publication_can_wait(monkeypatch):
    import io

    calls = []
    trusted = object()
    monkeypatch.setattr(review.ssl, "create_default_context", lambda **kwargs: trusted)

    def open_request(request, **kwargs):
        calls.append((request.full_url, kwargs))
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(review.urllib.request, "urlopen", open_request)
    config = {"ports": {"study": 19284}, "certificate": "local-only.pem"}
    assert review.http(config, "study", "/api/ready") == {"ok": True}
    assert review.http(config, "study", "/api/v1/admin/studies", body={}, timeout=600) == {
        "ok": True
    }
    assert [kwargs["timeout"] for _, kwargs in calls] == [5, 600]
    assert all(kwargs["context"] is trusted for _, kwargs in calls)


@pytest.mark.parametrize("expired,closed", [(True, False), (False, True), (False, False)])
def test_status_exposes_closure_before_showing_study_link(tmp_path, monkeypatch, expired, closed):
    raw = publication()
    if expired:
        raw["definition"]["closes_at"] = "2020-01-01T00:00:00+00:00"
    path = tmp_path / "publication.json"
    review.private_json(path, raw)
    config = {
        "root": str(tmp_path),
        "publication": str(path),
        "study_revision": "source-review",
        "ports": dict(postgres=19393, study=19394, exploration=19395),
        "database_password": "private-password-marker",
    }
    records = {role: {"pid": index + 2} for index, role in enumerate(review.ROLES)}
    monkeypatch.setattr(review, "processes", lambda _: records)
    monkeypatch.setattr(review, "owned", lambda *args: True)
    monkeypatch.setattr(review, "process_identity", lambda _: "owned")
    monkeypatch.setattr(review, "pg_ready", lambda *args: True)
    monkeypatch.setattr(review, "service_ready", lambda *args: True)
    preflights = []
    monkeypatch.setattr(review, "preflight", lambda config, **kw: preflights.append(kw))

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, args):
            assert args == ("source-review",)
            return self

        def fetchone(self):
            return (int(closed), json.dumps(raw["definition"]))

    monkeypatch.setattr(review, "database", lambda config: Connection())
    result = review.status(config)
    assert preflights == [{"allow_expired": True, "allow_closed": True, "check_ports": False}]
    assert result["closes_at"] == raw["definition"]["closes_at"]
    assert result["publication"]["closed"] is closed
    assert result["publication"]["expired"] is expired
    assert ("study_url" in result) is (not expired and not closed)
    assert "private-password-marker" not in json.dumps(result)
    if expired or closed:
        assert result["publication"]["next_action"] == "fresh-session --new-revision"
        with pytest.raises(review.ReviewError, match="fresh-session --new-revision"):
            review.require_open_publication(config)
    assert path.read_text() == json.dumps(raw, sort_keys=True, indent=2) + "\n"


def test_publication_status_does_not_inspect_an_unowned_database(tmp_path, monkeypatch):
    raw = publication()
    path = tmp_path / "publication.json"
    review.private_json(path, raw)
    config = {"root": str(tmp_path), "publication": str(path), "study_revision": "source-review"}
    monkeypatch.setattr(review, "processes", lambda _: {"postgres": {"pid": 5}})
    monkeypatch.setattr(review, "owned", lambda *args: False)
    monkeypatch.setattr(review, "database", lambda *args: pytest.fail("Foreign database queried"))
    state = review.publication_state(config)
    assert state["published"] is None and state["closed"] is None
    assert state["available_for_test_sessions"] is False
