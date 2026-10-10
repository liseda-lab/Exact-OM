"""Manage a durable, loopback-only development review environment.

The private root contains configuration, certificates, process ownership records,
PostgreSQL data, immutable review publications and fresh test invitation receipts.
This tool never adopts another server, resets a database or issues live invitations.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

SCHEMA = "exact-review-environment/1"
CLIENT_CONTRACT = "study-review/1"
DEFAULT_ROOT = Path.home() / "Library/Caches/exact-om-review/s1-20261010"
ROLES = ("postgres", "study", "exploration")


class ReviewError(RuntimeError):
    """A safe actionable failure that does not contain private configuration."""


def read_json(path):
    return json.loads(Path(path).read_text())


def private_json(path, value):
    """Replace one tool-owned record atomically, retaining restrictive permissions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tree_hash(path):
    digest = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode() + b"\0")
            digest.update(file_hash(item).encode() + b"\n")
    return digest.hexdigest()


def durable(path):
    """Reject temporary sole copies; the owner chooses another persistent location."""
    resolved = Path(path).expanduser().resolve()
    for temporary in (Path("/tmp").resolve(), Path("/private/var/folders")):
        if resolved == temporary or temporary in resolved.parents:
            raise ReviewError(
                "Review inputs and the environment root must be durable, outside temporary directories"
            )
    return resolved


def command(args, *, env=None, timeout=60):
    """Capture child output privately; command arguments never contain credentials."""
    try:
        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReviewError("A required local tool is missing or did not finish in time") from exc
    if result.returncode:
        raise ReviewError(f"Local tool {Path(args[0]).name} failed; existing data was preserved")
    return result.stdout.strip()


def runtime_versions(python):
    script = "import importlib.metadata,json,sys; print(json.dumps({'python':sys.version.split()[0],**{x:importlib.metadata.version(x) for x in ('fastapi','uvicorn','pydantic','psycopg','pyowl-core')}}))"
    return json.loads(command([str(python), "-c", script]))


def backend_identity(repo):
    """Hash service code/schema/config without importing native or matcher modules."""
    root = Path(repo).resolve()
    required = (
        "exact_inspect/service.py",
        "exact_inspect/study/api.py",
        "tools/review_environment.py",
        "pyproject.toml",
        "deploy/render/exact_study_requirements.txt",
    )
    if any(not (root / name).is_file() for name in required):
        raise ReviewError("Backend source files are missing; use the complete durable checkout")
    paths = {root / name for name in required}
    paths.update(
        path
        for path in (root / "exact_inspect").rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json", ".sql"}
        and "__pycache__" not in path.relative_to(root).parts
        and "static" not in path.relative_to(root).parts
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ReviewError("Backend source fingerprint cannot follow files outside its checkout")
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(file_hash(path).encode() + b"\n")
    try:
        commit = command(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], timeout=5)
    except ReviewError as exc:
        raise ReviewError(
            "Backend Git identity is unavailable; use a durable committed checkout"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ReviewError("Backend Git identity is invalid")
    return {
        "schema": "exact-review-backend-source/1",
        "git_commit": commit,
        "source_sha256": digest.hexdigest(),
        "file_count": len(paths),
    }


def verify_backend(config):
    recorded = config.get("backend")
    if not recorded or recorded.get("schema") != "exact-review-backend-source/1":
        raise ReviewError(
            "This legacy environment has no backend fingerprint; preserve it and run prepare "
            "with a separate durable root using this checkout"
        )
    current = backend_identity(config["repo"])
    if any(current[key] != recorded.get(key) for key in ("source_sha256", "file_count")):
        raise ReviewError(
            "Backend source bytes changed; preserve this environment and run prepare "
            "with a separate durable root"
        )
    # A later documentation-only commit does not change the verified runtime bytes.
    # Keep the original commit as provenance; never silently rewrite private config.
    return current


def validate_ports(ports):
    if set(ports) != set(ROLES) or len(set(ports.values())) != 3:
        raise ReviewError("Use three distinct explicit PostgreSQL, study and exploration ports")
    if any(type(port) is not int or not 1024 <= port <= 65535 for port in ports.values()):
        raise ReviewError("Review ports must be unprivileged TCP ports")


def port_available(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def validate_publication(raw, *, allow_expired=False):
    definition = raw["definition"]
    if (
        definition.get("purpose") != "development_review"
        or definition.get("synthetic") is not False
        or definition.get("correctness_metrics") != "disabled"
    ):
        raise ReviewError(
            "The bundle must explicitly declare development_review, real data and disabled correctness metrics"
        )
    protocol = definition.get("protocol_versions", {})
    expected = {
        "forms": "exact-study-forms/3",
        "setup": "setup/3",
        "consultation": "consultation/3",
        "methods": "study-methods/3",
        "export": "exact-study-analysis/4",
    }
    if definition.get("contract_version") != "exact-study/2.0" or any(
        protocol.get(k) != v for k, v in expected.items()
    ):
        raise ReviewError(
            "The review publication requires the study-review/1 negotiated forms and methods"
        )
    try:
        expires = datetime.fromisoformat(definition["closes_at"].replace("Z", "+00:00"))
    except (ValueError, KeyError) as exc:
        raise ReviewError("Publication expiry is invalid") from exc
    if expires.tzinfo is None:
        raise ReviewError("Publication expiry needs an explicit timezone")
    if not allow_expired and expires <= datetime.now(timezone.utc):
        raise ReviewError(
            "The publication expired; use fresh-session --new-revision to preserve it and create a fresh revision"
        )
    return definition


def verify_assets(raw, assets):
    """Verify supplied immutable assets and admission receipts without regenerating them."""
    root = Path(assets).resolve()
    for asset in raw["definition"]["assets"]:
        for key, hash_key in (
            ("path", "sha256"),
            ("admission_receipt_path", "admission_receipt_sha256"),
        ):
            if not asset.get(key):
                continue
            target = (root / asset[key]).resolve()
            if root not in target.parents or not target.is_file():
                raise ReviewError("A publication asset is absent or outside its admitted root")
            if file_hash(target) != asset[hash_key].removeprefix("sha256:"):
                raise ReviewError("A publication asset or admission receipt has changed")
            if key == "path" and target.stat().st_size != asset["size_bytes"]:
                raise ReviewError("A publication asset has a different byte length")


def verify_workspaces(raw, assets):
    """Verify all study workspace bindings even when exploration uses another bundle."""
    root = Path(assets).resolve()
    checked = {}
    for scope in raw["definition"].get("workspace_scopes", []):
        files = [(scope["index_path"], scope["index_sha256"])]
        for context in scope["contexts"]:
            files.extend(
                [
                    (context["path"] + "/manifest.json", context["manifest_sha256"]),
                    (context["path"] + "/context.sqlite", context["database_sha256"]),
                ]
            )
        for relative, expected in files:
            target = (root / relative).resolve()
            if root not in target.parents or not target.is_file():
                raise ReviewError(
                    "A study workspace resource is missing or outside its admitted root"
                )
            if target not in checked:
                checked[target] = file_hash(target)
            if checked[target] != expected.removeprefix("sha256:"):
                raise ReviewError("A bound study workspace index, manifest or database changed")


def validate_tls(cert, key):
    try:
        command(["openssl", "verify", "-CAfile", str(cert), str(cert)])
        command(["openssl", "x509", "-in", str(cert), "-checkend", "86400", "-noout"])
    except ReviewError as exc:
        raise ReviewError(
            "Local TLS is invalid, expired or expires within 24 hours; use renew-certificate"
        ) from exc
    names = command(["openssl", "x509", "-in", str(cert), "-noout", "-text"])
    alternatives = {value.strip() for line in names.splitlines() for value in line.split(",")}
    if not {"IP Address:127.0.0.1", "DNS:localhost"} <= alternatives:
        raise ReviewError("The local certificate must cover both localhost and 127.0.0.1")
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert), str(key))
    except (OSError, ssl.SSLError) as exc:
        raise ReviewError("The local TLS certificate and key do not match") from exc


def generate_tls(directory):
    """Create new TLS files without changing any previous certificate or key."""
    directory = Path(directory)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    cert, key = directory / "localhost.pem", directory / "localhost-key.pem"
    if cert.exists() or key.exists():
        raise ReviewError("Existing TLS files are preserved; select a new certificate directory")
    extensions = directory / "openssl.cnf"
    extensions.write_text(
        "[req]\ndistinguished_name=dn\nx509_extensions=local\nprompt=no\n[dn]\nCN=Exact local development review\n[local]\nsubjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1\nbasicConstraints=critical,CA:TRUE\nkeyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\n"
    )
    command(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "30",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-config",
            str(extensions),
        ]
    )
    os.chmod(key, 0o600)
    validate_tls(cert, key)
    return cert, key


def renew_certificate(config):
    """Retain DB/TLS history, stop owned HTTPS workers and install a fresh local cert."""
    preflight(config, allow_expired=True, allow_closed=True, check_tls=False)
    records = processes(config)
    for role in ("study", "exploration"):
        record = records.get(role)
        if (
            record
            and process_identity(record["pid"]) is not None
            and not owned(config, role, record)
        ):
            raise ReviewError(
                "Cannot renew TLS while an HTTPS worker has a different process identity"
            )
    cert, key = generate_tls(Path(config["root"]) / "tls" / ("renewal-" + uuid4().hex))
    for role in ("exploration", "study"):
        stop_role(config, role)
    config.update(certificate=str(cert), private_key=str(key))
    config["hashes"]["certificate"] = file_hash(cert)
    private_json(Path(config["root"]) / "environment.json", config)
    return {
        "certificate_renewed": True,
        "certificate": str(cert),
        "certificate_sha256": file_hash(cert),
        "database_preserved": True,
        "next": "Trust the new local certificate, then start; if the publication expired, run fresh-session --new-revision.",
    }


def new_publication(config, *, days=30):
    if not 1 <= days <= 90:
        raise ReviewError("Review expiry must be between 1 and 90 days")
    raw = read_json(config["source_publication"])
    validate_publication(raw, allow_expired=True)
    revision = (
        "review-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10]
    )
    raw["definition"]["study_revision"] = revision
    raw["definition"]["closes_at"] = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    target = Path(config["root"]) / "publications" / (revision + ".json")
    if target.exists():
        raise ReviewError("Fresh publication identity already exists")
    private_json(target, raw)
    config.update(publication=str(target), study_revision=revision)
    config["hashes"]["publication"] = file_hash(target)
    return raw


def load(root):
    config = read_json(Path(root) / "environment.json")
    if config.get("schema") != SCHEMA or Path(config["root"]).resolve() != Path(root).resolve():
        raise ReviewError("Environment configuration does not belong to this root")
    return config


def processes(config):
    path = Path(config["root"]) / "processes.json"
    return read_json(path) if path.exists() else {}


def process_identity(pid):
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "stat=", "-o", "lstart=", "-o", "command="],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode or not result.stdout.strip():
            return None
        state, identity = result.stdout.strip().split(None, 1)
        return None if state.startswith("Z") else identity
    except (OSError, subprocess.TimeoutExpired):
        return None


def stable_worker_identity(identity):
    """macOS venv launchers may exec the real Python after the first ps sample.

    Keep the process start time and complete managed-worker argument vector. Only
    the executable spelling can differ; PID, role, root and random nonce still bind.
    """
    parts = identity.split(None, 5)
    marker = " -m tools.review_environment "
    if len(parts) == 6 and marker in parts[5]:
        return " ".join(parts[:5]) + parts[5][parts[5].index(marker) :]
    return identity


def owned(config, role, record):
    if type(record.get("pid")) is not int or record["pid"] <= 1 or not record.get("nonce"):
        return False
    actual = process_identity(record["pid"])
    if actual is None:
        return False
    return (
        stable_worker_identity(record["identity"]) == stable_worker_identity(actual)
        and record["nonce"] in actual
        and (
            str(Path(config["root"]) / "pgdata") in actual
            if role == "postgres"
            else str(config["root"]) in actual
        )
    )


def publication_state(config, raw=None):
    """Read owner-safe closure metadata only from this environment's owned database."""
    raw = raw if raw is not None else read_json(config["publication"])
    definition = validate_publication(raw, allow_expired=True)
    if definition["study_revision"] != config["study_revision"]:
        raise ReviewError("The configured publication revision does not match its immutable file")
    result = {
        "study_revision": definition["study_revision"],
        "closes_at": definition["closes_at"],
        "expired": datetime.fromisoformat(definition["closes_at"].replace("Z", "+00:00"))
        <= datetime.now(timezone.utc),
        "published": None,
        "closed": None,
        "database_status": "not_checked_owned_database_unavailable",
    }
    record = processes(config).get("postgres")
    if record and owned(config, "postgres", record) and pg_ready(config, record):
        import psycopg

        try:
            with database(config) as db:
                row = db.execute(
                    "SELECT closed, payload FROM studies WHERE revision = %s",
                    (config["study_revision"],),
                ).fetchone()
        except (psycopg.errors.InvalidCatalogName, psycopg.errors.UndefinedTable):
            row = None  # First startup has not yet created/admitted this publication.
        except psycopg.Error as exc:
            raise ReviewError(
                "The owned review database could not report publication closure"
            ) from exc
        if not owned(config, "postgres", record):
            raise ReviewError("PostgreSQL ownership changed while checking publication closure")
        if row and datetime.fromisoformat(
            json.loads(row[1])["closes_at"].replace("Z", "+00:00")
        ) != datetime.fromisoformat(result["closes_at"].replace("Z", "+00:00")):
            raise ReviewError("The database publication expiry differs from the immutable input")
        result.update(
            published=row is not None,
            closed=bool(row[0]) if row else False,
            database_status="checked",
        )
    result["available_for_test_sessions"] = (
        result["published"] is True and result["closed"] is False and not result["expired"]
    )
    result["next_action"] = (
        "fresh-session --new-revision"
        if result["expired"] or result["closed"]
        else "fresh-session" if result["available_for_test_sessions"] else "start"
    )
    return result


def require_open_publication(config, raw=None, *, allow_expired=False, allow_closed=False):
    state = publication_state(config, raw)
    if (state["expired"] and not allow_expired) or (state["closed"] and not allow_closed):
        raise ReviewError(
            "The review publication is closed or expired; use fresh-session --new-revision "
            "to preserve it and create a fresh test revision"
        )
    return state


def preflight(config, *, allow_expired=False, allow_closed=False, check_ports=True, check_tls=True):
    """Fail closed on changed inputs, versions, TLS, ports and publication expiry."""
    verify_backend(config)
    validate_ports(config["ports"])
    for key in (
        "root",
        "repo",
        "assets",
        "publication",
        "source_publication",
        "package",
        "frontend",
        "python",
        "pgbin",
    ):
        durable(config[key])
    for key in ("publication", "source_publication", "package", "certificate"):
        if file_hash(config[key]) != config["hashes"][key]:
            raise ReviewError(
                f"Configured {key} bytes changed; preserve this environment and prepare a separate root"
            )
    if tree_hash(config["frontend"]) != config["hashes"]["frontend"]:
        raise ReviewError("The compiled frontend changed; prepare a separate environment root")
    if runtime_versions(config["python"]) != config["runtime_versions"]:
        raise ReviewError("The configured Python environment versions changed")
    postgres_version = command([str(Path(config["pgbin"]) / "postgres"), "--version"])
    if postgres_version != config["postgres_version"]:
        raise ReviewError("The configured PostgreSQL binary version changed")
    cluster = Path(config["root"]) / "pgdata"
    if not (cluster / "PG_VERSION").exists():
        raise ReviewError(
            "The preserved PostgreSQL cluster is incomplete; it will not be reinitialized"
        )
    if (cluster / "PG_VERSION").read_text().strip() != config["postgres_major"]:
        raise ReviewError("The preserved cluster and PostgreSQL binary major versions differ")
    if (
        read_json(Path(config["root"]) / "cluster-owner.json")["environment_id"]
        != config["environment_id"]
    ):
        raise ReviewError("PostgreSQL cluster ownership does not match this environment")
    if check_tls:
        validate_tls(config["certificate"], config["private_key"])
    raw = read_json(config["publication"])
    validate_publication(raw, allow_expired=allow_expired)
    verify_assets(raw, config["assets"])
    verify_workspaces(raw, config["assets"])
    from exact_inspect.study.models import Publish

    Publish.model_validate(raw)
    from exact_inspect.artifacts import validate_bundle

    validate_bundle(Path(config["package"]), verify_hashes=True)
    require_open_publication(config, raw, allow_expired=allow_expired, allow_closed=allow_closed)
    if check_ports:
        records = processes(config)
        for role, port in config["ports"].items():
            if not port_available(port) and not (
                role in records and owned(config, role, records[role])
            ):
                raise ReviewError(
                    f"Port {port} is occupied by a service this environment does not own"
                )
    return raw


def prepare(args):
    root = durable(args.root)
    config_path = root / "environment.json"
    if config_path.exists():
        config = load(root)
        for field in ("publication", "assets", "package", "frontend", "python", "pgbin"):
            supplied = getattr(args, field, None)
            previous = config["source_publication" if field == "publication" else field]
            if supplied is not None and supplied.expanduser().resolve() != Path(previous).resolve():
                raise ReviewError(
                    "Existing configuration differs; choose another root instead of replacing it"
                )
        for role in ROLES:
            supplied = getattr(args, role + "_port")
            if supplied is not None and supplied != config["ports"][role]:
                raise ReviewError(
                    "Existing configured ports differ; choose another root instead of replacing them"
                )
        raw = preflight(config)
        return {
            "prepared": True,
            "reused_existing_data": True,
            "root": str(root),
            "study_revision": config["study_revision"],
            "closes_at": raw["definition"]["closes_at"],
            "backend": config["backend"],
        }
    required = ("publication", "assets", "package", "frontend", "python", "pgbin")
    if any(getattr(args, key, None) is None for key in required):
        raise ReviewError(
            "First prepare requires publication, assets, package, frontend, python and pgbin paths"
        )
    values = {key: str(durable(getattr(args, key))) for key in required}
    # Keep the venv executable path, not its resolved system-Python symlink.
    values["python"] = str(args.python.expanduser().absolute())
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    if (root / "pgdata").exists():
        raise ReviewError(
            "An existing unowned PostgreSQL directory will never be initialized or replaced"
        )
    raw = read_json(values.pop("publication"))
    validate_publication(raw, allow_expired=True)
    for name in ("index.html", "participate/index.html"):
        if not (Path(values["frontend"]) / name).is_file():
            raise ReviewError("Build the complete frontend before preparing the review environment")
    verify_assets(raw, values["assets"])
    verify_workspaces(raw, values["assets"])
    from exact_inspect.study.models import Publish

    Publish.model_validate(raw)
    from exact_inspect.artifacts import validate_bundle

    validate_bundle(Path(values["package"]), verify_hashes=True)
    ports = dict(
        postgres=args.postgres_port or 19183,
        study=args.study_port or 19184,
        exploration=args.exploration_port or 19185,
    )
    validate_ports(ports)
    if any(not port_available(port) for port in ports.values()):
        raise ReviewError("A requested port is already occupied; no existing process was changed")
    config = {
        "schema": SCHEMA,
        "environment_id": uuid4().hex,
        "root": str(root),
        "repo": str(Path(__file__).resolve().parents[1]),
        **values,
        "ports": ports,
        "source_publication": str(durable(args.publication)),
        "client_contract": CLIENT_CONTRACT,
        "database_password": secrets.token_urlsafe(32),
        "signing_secret": secrets.token_urlsafe(32),
        "researcher_token": secrets.token_urlsafe(32),
        "database_name": "exact_review",
        "runtime_versions": runtime_versions(values["python"]),
        "backend": backend_identity(Path(__file__).resolve().parents[1]),
        "postgres_version": command([str(Path(values["pgbin"]) / "postgres"), "--version"]),
    }
    config["postgres_major"] = config["postgres_version"].split()[-1].split(".")[0]
    cert, key = generate_tls(root / "tls")
    config.update(certificate=str(cert), private_key=str(key))
    config["hashes"] = {
        key: file_hash(config[key]) for key in ("source_publication", "package", "certificate")
    }
    config["hashes"]["frontend"] = tree_hash(config["frontend"])
    new_publication(config, days=args.days)
    password_file = root / "initdb-password"
    fd = os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(config["database_password"] + "\n")
    try:
        command(
            [
                str(Path(config["pgbin"]) / "initdb"),
                "-D",
                str(root / "pgdata"),
                "-U",
                "review",
                "--pwfile=" + str(password_file),
                "--auth-host=scram-sha-256",
                "--auth-local=scram-sha-256",
                "--encoding=UTF8",
                "--no-locale",
            ],
            timeout=120,
        )
    finally:
        password_file.unlink(missing_ok=True)
    private_json(root / "cluster-owner.json", {"environment_id": config["environment_id"]})
    private_json(config_path, config)
    return {
        "prepared": True,
        "root": str(root),
        "study_revision": config["study_revision"],
        "closes_at": read_json(config["publication"])["definition"]["closes_at"],
        "certificate": str(cert),
        "client_contract": CLIENT_CONTRACT,
        "backend": config["backend"],
    }


def database(config, name=None):
    import psycopg

    return psycopg.connect(
        host="127.0.0.1",
        port=config["ports"]["postgres"],
        dbname=name or config["database_name"],
        user="review",
        password=config["database_password"],
        connect_timeout=2,
        autocommit=True,
        client_encoding="UTF8",
    )


def http(config, role, path, *, body=None, authenticated=False, timeout=5):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = "Bearer " + config["researcher_token"]
    request = urllib.request.Request(
        f"https://127.0.0.1:{config['ports'][role]}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
    )
    context = ssl.create_default_context(cafile=config["certificate"])
    with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
        return json.load(response)


def wait_for(predicate, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise ReviewError("A review service did not become ready; consult its private log")


def pg_ready(config, record):
    try:
        with database(config, "postgres") as db:
            row = db.execute(
                "SELECT current_setting('data_directory'), current_setting('cluster_name')"
            ).fetchone()
            return (
                Path(row[0]).resolve() == Path(config["root"]) / "pgdata"
                and row[1] == "exact-review-" + record["nonce"]
            )
    except Exception as exc:
        import psycopg

        if isinstance(exc, psycopg.Error):
            return False
        raise


def launch(config, role):
    root = Path(config["root"])
    nonce = uuid4().hex
    if role == "postgres":
        args = [
            str(Path(config["pgbin"]) / "postgres"),
            "-D",
            str(root / "pgdata"),
            "-h",
            "127.0.0.1",
            "-p",
            str(config["ports"][role]),
            "-k",
            "",
            "-c",
            "cluster_name=exact-review-" + nonce,
        ]
    else:
        args = [
            config["python"],
            "-m",
            "tools.review_environment",
            "_serve",
            "--root",
            str(root),
            "--role",
            role,
            "--nonce",
            nonce,
        ]
    logs = root / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(logs / (role + ".log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "ab") as log:
        child = subprocess.Popen(
            args,
            cwd=config["repo"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    identity = process_identity(child.pid)
    if identity is None or nonce not in identity:
        # A child still transitioning through exec is ours by its direct handle.
        for _ in range(20):
            time.sleep(0.05)
            identity = process_identity(child.pid)
            if identity and nonce in identity:
                break
        else:
            child.terminate()
            child.wait(timeout=5)
            raise ReviewError("Review child failed before its process identity could be recorded")
    record = {"pid": child.pid, "nonce": nonce, "identity": identity}
    records = processes(config)
    records[role] = record
    private_json(root / "processes.json", records)
    return record


def stop_role(config, role):
    records = processes(config)
    record = records.get(role)
    if record is None:
        return
    if process_identity(record["pid"]) is not None:
        if not owned(config, role, record):
            raise ReviewError(f"Refusing to stop {role}: PID ownership changed")
        os.kill(record["pid"], signal.SIGINT if role == "postgres" else signal.SIGTERM)
        wait_for(lambda: process_identity(record["pid"]) is None, timeout=30)
    records.pop(role)
    private_json(Path(config["root"]) / "processes.json", records)


def service_ready(config, role, record):
    if not record or not owned(config, role, record):
        return False
    if role == "postgres":
        ready = pg_ready(config, record)
    else:
        result = http(config, role, "/api/ready" if role == "study" else "/api/v1/health")
        ready = (
            result.get("ready") is True
            if role == "study"
            else result.get("status") == "available" and result.get("profile") == "local_app"
        )
    return bool(ready and owned(config, role, record))


def start(config):
    preflight(config)
    started = []
    try:
        records = processes(config)
        for role in ROLES:
            record = records.get(role)
            if not (record and owned(config, role, record)):
                if record and process_identity(record["pid"]) is not None:
                    raise ReviewError(f"Refusing to replace {role}: process ownership changed")
                record = launch(config, role)
                started.append(role)
            if role == "postgres":
                wait_for(lambda: service_ready(config, role, record))
                from psycopg import sql

                with database(config, "postgres") as db:
                    if not db.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s", (config["database_name"],)
                    ).fetchone():
                        db.execute(
                            sql.SQL("CREATE DATABASE {}").format(
                                sql.Identifier(config["database_name"])
                            )
                        )
                require_open_publication(config)
            else:
                # Study admission runs before its listener binds; large frozen ontology
                # bundles need the same bounded budget as an explicit publication POST.
                wait_for(
                    lambda: service_ready(config, role, record),
                    timeout=600 if role == "study" else 90,
                )
        result = status(config, preflight_checked=True)
        if not all(
            value["state"] == "owned" and value.get("ready")
            for value in result["services"].values()
        ):
            raise ReviewError("A service lost its owned identity or readiness during startup")
    except BaseException as original:
        failures = []
        for role in reversed(started):
            try:
                stop_role(config, role)
            except ReviewError:
                failures.append(role)
        if failures:
            raise ReviewError(
                "Startup failed; ownership checks prevented stopping: " + ", ".join(failures)
            ) from original
        raise
    return result


def status(config, *, preflight_checked=False):
    if not preflight_checked:
        preflight(config, allow_expired=True, allow_closed=True, check_ports=False)
    publication = publication_state(config)
    records = processes(config)
    report = {
        "root": config["root"],
        "study_revision": config["study_revision"],
        "client_contract": CLIENT_CONTRACT,
        "closes_at": publication["closes_at"],
        "publication": publication,
        "backend": config.get("backend"),
        "services": {},
    }
    for role in ROLES:
        record = records.get(role)
        state = (
            "stopped"
            if record is None or process_identity(record["pid"]) is None
            else "owned" if owned(config, role, record) else "ownership_mismatch"
        )
        report["services"][role] = {"state": state, "port": config["ports"][role]}
        if state == "owned":
            try:
                ready = service_ready(config, role, record)
            except (OSError, ValueError, urllib.error.URLError):
                ready = False
            report["services"][role]["ready"] = ready
    if publication["available_for_test_sessions"] and all(
        value["state"] == "owned" and value.get("ready") for value in report["services"].values()
    ):
        report["study_url"] = f"https://127.0.0.1:{config['ports']['study']}/participate/"
    exploration = report["services"]["exploration"]
    if exploration["state"] == "owned" and exploration.get("ready"):
        report["exploration_url"] = f"https://127.0.0.1:{config['ports']['exploration']}/"
    return report


def fresh_session(config, *, renew=False, days=30):
    preflight(config, allow_expired=renew, allow_closed=renew)
    if renew:
        new_publication(config, days=days)
        private_json(Path(config["root"]) / "environment.json", config)
    if not all(
        value["state"] == "owned" and value.get("ready")
        for value in status(config, preflight_checked=True)["services"].values()
    ):
        if not renew:
            raise ReviewError("Start this environment before creating a fresh test session")
        start(config)
    # Admission is idempotent. Retry a previously saved new revision after an interrupted
    # admission before issuing its invitation, even if existing workers stayed alive.
    http(
        config,
        "study",
        "/api/v1/admin/studies",
        body=read_json(config["publication"]),
        authenticated=True,
        timeout=600,
    )
    result = http(
        config,
        "study",
        "/api/v1/admin/studies/" + config["study_revision"] + "/invitations",
        body={"count": 1, "test": True},
        authenticated=True,
    )
    invitation = result["invitations"][0]
    target = Path(config["root"]) / "invitations" / (uuid4().hex + ".json")
    private_json(
        target,
        {
            "client_contract": CLIENT_CONTRACT,
            "study_revision": config["study_revision"],
            "invitation_url": urllib.parse.urljoin(
                f"https://127.0.0.1:{config['ports']['study']}", invitation["invitation"]
            ),
            **invitation,
        },
    )
    return {
        "created_test_session": True,
        "study_revision": config["study_revision"],
        "closes_at": read_json(config["publication"])["definition"]["closes_at"],
        "private_invitation_file": str(target),
        "client_contract": CLIENT_CONTRACT,
    }


def serve(config, role, nonce):
    if role not in {"study", "exploration"} or not nonce or len(nonce) != 32:
        raise ReviewError("Invalid managed review worker")
    verify_backend(config)
    if role == "study":
        from urllib.parse import quote

        from exact_inspect.study.api import create_study_app
        from exact_inspect.study.models import Publish

        raw = read_json(config["publication"])
        validate_publication(raw)
        url = f"postgresql://review:{quote(config['database_password'], safe='')}@127.0.0.1:{config['ports']['postgres']}/{config['database_name']}"
        app = create_study_app(
            url,
            config["signing_secret"],
            config["researcher_token"],
            f"https://127.0.0.1:{config['ports']['study']}",
            Path(config["assets"]),
            frontend_dir=Path(config["frontend"]),
        )
        app.state.study_store.publish(Publish.model_validate(raw))
    else:
        from exact_inspect.service import create_prepared_app

        app = create_prepared_app(
            Path(config["package"]), profile="local_app", frontend_dir=Path(config["frontend"])
        )
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=config["ports"][role],
        ssl_certfile=config["certificate"],
        ssl_keyfile=config["private_key"],
        log_level="warning",
        access_log=False,
    )


@contextlib.contextmanager
def environment_lock(root):
    import fcntl

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(root / ".environment.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReviewError(
                "Another command is currently changing this review environment"
            ) from exc
        yield
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "start",
            "status",
            "stop",
            "fresh-session",
            "renew-certificate",
            "_serve",
        ],
    )
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get("EXACT_REVIEW_ROOT", DEFAULT_ROOT))
    )
    for name in ("publication", "assets", "package", "frontend", "python", "pgbin"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--postgres-port", type=int)
    parser.add_argument("--study-port", type=int)
    parser.add_argument("--exploration-port", type=int)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--new-revision", action="store_true")
    parser.add_argument("--role", choices=["study", "exploration"], help=argparse.SUPPRESS)
    parser.add_argument("--nonce", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.action == "_serve":
            serve(load(args.root), args.role, args.nonce)
            return
        with environment_lock(durable(args.root)):
            if args.action == "prepare":
                result = prepare(args)
            else:
                config = load(args.root)
                if args.action == "start":
                    result = start(config)
                elif args.action == "status":
                    result = status(config)
                elif args.action == "renew-certificate":
                    result = renew_certificate(config)
                elif args.action == "stop":
                    for role in reversed(ROLES):
                        stop_role(config, role)
                    result = status(config, preflight_checked=True)
                else:
                    result = fresh_session(config, renew=args.new_revision, days=args.days)
        print(json.dumps(result, indent=2, sort_keys=True))
    except ReviewError as exc:
        parser.exit(1, f"Review environment: {exc}\n")
    except urllib.error.HTTPError as exc:
        parser.exit(
            1,
            f"Review environment: local service returned HTTP {exc.code}; no credentials or response body were printed. Existing data was preserved.\n",
        )
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        # Credential-bearing upstream errors are not echoed into terminal output.
        parser.exit(
            1,
            f"Review environment: {type(exc).__name__}; inputs or a local service are unavailable. Existing data was preserved.\n",
        )


if __name__ == "__main__":
    main()
