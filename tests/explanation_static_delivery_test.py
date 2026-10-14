"""Public build-asset caching and private profile boundaries, through actual applications."""

import hashlib

import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import publish_bundle
from exact_inspect.contracts import VisibilityPolicy
from exact_inspect.service import create_prepared_app
from exact_inspect.study.api import create_study_app

CACHE = "public, max-age=31536000, immutable"
CSS = b":root{--paper:#fff}body{display:grid}"
CSS_PATH = "/_next/static/css/" + hashlib.sha256(CSS).hexdigest()[:16] + ".css"


@pytest.fixture
def frontend(tmp_path):
    root = tmp_path / "frontend"
    asset = root / CSS_PATH.lstrip("/")
    asset.parent.mkdir(parents=True)
    asset.write_bytes(CSS)
    (asset.parent / "plain.css").write_bytes(CSS)
    for page in ("", "browse", "participate", "admin"):
        folder = root / page
        folder.mkdir(exist_ok=True)
        (folder / "index.html").write_text(
            f'<html><head><link rel="stylesheet" href="{CSS_PATH}">'
            "<script>self.boot=1</script></head><body>synthetic</body></html>"
        )
    return root


def application(profile, frontend, tmp_path):
    if profile == "study":
        return create_study_app(
            f"sqlite:///{tmp_path / 'study.sqlite'}",
            "synthetic-signing-secret-never-deploy-0000",
            "synthetic-admin-secret-never-deploy-000000",
            "https://testserver",
            tmp_path,
            allow_test_sqlite=True,
            frontend_dir=frontend,
        )
    package_root = tmp_path / "package"
    package_root.mkdir(exist_ok=True)
    package = publish_bundle(
        package_root, ontologies={}, policy=VisibilityPolicy(), audience="development_demo"
    )
    return create_prepared_app(package, profile=profile, frontend_dir=frontend)


@pytest.mark.parametrize("profile", ["public_demo", "study"])
def test_verified_css_keeps_immutable_cache_through_profile_protection(profile, frontend, tmp_path):
    with TestClient(
        application(profile, frontend, tmp_path), base_url="https://testserver"
    ) as client:
        response = client.get(CSS_PATH)
        assert response.status_code == 200
        assert response.content == CSS
        assert response.headers["cache-control"] == CACHE
        assert response.headers["content-type"] == "text/css; charset=utf-8"
        assert int(response.headers["content-length"]) == len(CSS)
        assert response.headers["x-content-type-options"] == "nosniff"
        head = client.head(CSS_PATH)
        assert head.status_code == 200 and head.content == b""
        assert int(head.headers["content-length"]) == len(CSS)
        assert head.headers["cache-control"] == CACHE


@pytest.mark.parametrize("profile", ["public_demo", "study"])
def test_unverified_assets_pages_private_reads_and_errors_remain_no_store(
    profile, frontend, tmp_path
):
    app = application(profile, frontend, tmp_path)
    with TestClient(app, base_url="https://testserver") as client:
        page = "/participate/" if profile == "study" else "/"
        html = client.get(page)
        assert html.status_code == 200 and "no-store" in html.headers["cache-control"]
        assert "script-src 'self' 'sha256-" in html.headers["content-security-policy"]
        assert "object-src 'none'" in html.headers["content-security-policy"]
        if profile == "study":
            assert "unsafe-inline" not in html.headers["content-security-policy"]
        paths = [
            "/_next/static/css/plain.css",
            "/_next/static/css/ffffffffffffffff.css",
            "/_next/%2e%2e/%2e%2e/private",
            "/api/v1/no-such-file.css",
            "/api/v1/study/resources/ffffffffffffffff.css",
        ]
        for path in paths:
            response = client.get(path, headers={"X-Exact-Verified-Asset": "true"})
            assert "no-store" in response.headers["cache-control"]
            assert "x-exact-read-id" not in response.headers
            if path != paths[0]:
                assert response.status_code in {401, 404, 422}
                assert "text/html" not in response.headers["content-type"]
        denied = client.get("/browse/" if profile == "study" else "/participate/")
        assert denied.status_code == 404 and "no-store" in denied.headers["cache-control"]
        invalid_range = client.get(CSS_PATH, headers={"Range": "bytes=999999999-"})
        assert invalid_range.status_code == 416
        assert "no-store" in invalid_range.headers["cache-control"]
        partial = client.get(CSS_PATH, headers={"Range": "bytes=2-8"})
        assert partial.status_code == 206 and partial.content == CSS[2:9]
        assert partial.headers["cache-control"] == CACHE


def test_export_mutation_cannot_keep_immutable_cache_and_new_files_are_not_admitted(
    frontend, tmp_path
):
    app = application("public_demo", frontend, tmp_path)
    with TestClient(app, base_url="https://testserver") as client:
        original = client.get(CSS_PATH)
        assert original.headers["etag"] == '"' + hashlib.sha256(CSS).hexdigest() + '"'
        (frontend / CSS_PATH.lstrip("/")).write_bytes(b"changed")
        changed = client.get(CSS_PATH)
        assert changed.status_code == 503 and changed.json()["code"] == "frontend_asset_changed"
        assert "no-store" in changed.headers["cache-control"]
        late = frontend / "_next/static/css/eeeeeeeeeeeeeeee.css"
        late.write_bytes(CSS)
        assert (
            "no-store" in client.get("/" + str(late.relative_to(frontend))).headers["cache-control"]
        )


def test_api_only_cannot_expose_or_cache_export_files(tmp_path):
    with TestClient(create_prepared_app(None)) as client:
        missing = client.get(CSS_PATH)
        assert missing.status_code == 404 and missing.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_static_diagnostics_correlate_completion_without_secrets(
    method, frontend, tmp_path, monkeypatch
):
    import json
    import re

    path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("EXACT_READ_DIAGNOSTICS", str(path))
    app = application("study", frontend, tmp_path)
    secret = "invitation-cookie-participant-secret"
    with TestClient(app, base_url="https://testserver") as client:
        response = client.request(
            method,
            CSS_PATH + "?invitation=" + secret,
            headers={"cookie": secret, "authorization": secret},
        )
    assert response.status_code == 200 and response.headers["cache-control"] == CACHE
    assert re.fullmatch(r"[a-f0-9]{32}", response.headers["x-exact-read-id"])
    assert response.headers["x-exact-asset-sha256"] == hashlib.sha256(CSS).hexdigest()
    text = path.read_text()
    assert secret not in text and CSS_PATH not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert len(rows) == 2 and rows[-1]["schema"] == "exact-static-transfer/1"
    done = rows[-1]
    assert done["read_id"] == response.headers["x-exact-read-id"] and done["complete"]
    assert done["sent_bytes"] == (len(CSS) if method == "GET" else 0)
    assert done["declared_length"] == len(CSS) and done["error_type"] is None
    assert done["length_matches"] is (True if method == "GET" else None)
    assert done["head_body_empty"] is (True if method == "HEAD" else None)


@pytest.mark.parametrize("ending", ["cancel", "send_error", "incomplete"])
def test_static_diagnostics_preserve_interruption_and_subsequent_read(ending, frontend):
    import asyncio
    import json

    from exact_inspect.public_assets import (
        PUBLIC_ASSET_SCOPE,
        build_public_asset_registry,
    )
    from exact_inspect.read_diagnostics import ReadDiagnostics

    asset = build_public_asset_registry(frontend)[CSS_PATH.lstrip("/")]
    records, messages = [], []
    secret = "invitation-cookie-participant-secret"

    async def receive():
        return {"type": "http.disconnect"}

    async def producer(scope, receive, send):
        scope[PUBLIC_ASSET_SCOPE] = asset
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"8"), (b"cache-control", CACHE.encode())],
            }
        )
        await send({"type": "http.response.body", "body": b"abc", "more_body": True})
        if ending == "cancel":
            assert (await receive())["type"] == "http.disconnect"
            raise asyncio.CancelledError(secret)
        if ending == "send_error":
            await send({"type": "http.response.body", "body": b"defgh", "more_body": False})

    async def send(message):
        if ending == "send_error" and message.get("body") == b"defgh":
            raise ConnectionResetError(secret)
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/" + secret,
        "query_string": secret.encode(),
        "headers": [(b"cookie", secret.encode())],
    }
    observed = ReadDiagnostics(producer, emit=records.append)
    if ending == "incomplete":
        asyncio.run(observed(scope, receive, send))
    else:
        with pytest.raises(asyncio.CancelledError if ending == "cancel" else ConnectionResetError):
            asyncio.run(observed(scope, receive, send))
    assert not records[-1]["complete"] and records[-1]["sent_bytes"] == 3
    assert records[-1]["length_matches"] is None
    assert secret not in json.dumps(records)
    assert messages[-1]["body"] == b"abc" and messages[-1]["more_body"]

    async def good(scope, receive, send):
        scope[PUBLIC_ASSET_SCOPE] = asset
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]}
        )
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    asyncio.run(ReadDiagnostics(good, emit=records.append)(dict(scope), receive, send))
    assert records[-1]["complete"] and records[-1]["length_matches"]
    assert records[-1]["read_id"] != records[-3]["read_id"]


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_offloaded_static_delivery_does_not_claim_observed_completion(method, frontend):
    import asyncio
    import json

    from exact_inspect.public_assets import (
        PUBLIC_ASSET_SCOPE,
        build_public_asset_registry,
    )
    from exact_inspect.read_diagnostics import ReadDiagnostics

    asset = build_public_asset_registry(frontend)[CSS_PATH.lstrip("/")]
    records, messages = [], []
    private_path = "/private/invitation-cookie-participant-secret.css"

    async def receive():
        return {"type": "http.disconnect"}

    async def producer(scope, receive, send):
        scope[PUBLIC_ASSET_SCOPE] = asset
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"8")]}
        )
        await send({"type": "http.response.pathsend", "path": private_path})

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": method, "path": private_path}
    asyncio.run(ReadDiagnostics(producer, emit=records.append)(scope, receive, send))
    assert messages[-1] == {"type": "http.response.pathsend", "path": private_path}
    done = records[-1]
    assert done["offloaded"] is True and done["complete"] is None
    assert done["sent_bytes"] is None and done["length_matches"] is None
    assert done["head_body_empty"] is None and done["error_type"] is None
    assert private_path not in json.dumps(records)


def test_diagnostic_output_is_bounded_and_new_rotations_stay_private(tmp_path):
    from exact_inspect.read_diagnostics import ReadDiagnostics

    log = tmp_path / "reads.jsonl"
    observer = ReadDiagnostics(None, path=log)
    for _ in range(3600):
        observer.emit({"synthetic_public_padding": "x" * 1000})
    files = list(tmp_path.glob("reads.jsonl*"))
    assert len(files) == 3
    assert sum(p.stat().st_size for p in files) <= 3 * 1048576
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in files)
