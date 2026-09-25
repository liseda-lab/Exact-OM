"""Profile-owned page allowlists and per-document CSP for the static frontend export."""

import base64
import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exact_inspect.frontend import content_security_policy, mount_frontend, pages_for

INLINE = "self.__boot=1"


@pytest.fixture
def export(tmp_path):
    root = tmp_path / "out"
    for page in ("", "browse", "library", "participate", "admin", "legacy"):
        directory = root / page if page else root
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text(
            f'<html><head><script>{INLINE}</script><script src="/_next/static/a.js"></script></head>'
            f"<body>{page or 'compare'}</body></html>"
        )
    (root / "_next" / "static").mkdir(parents=True)
    (root / "_next" / "static" / "a.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside the export")
    return root


def app_for(profile, export, **options):
    app = FastAPI()

    @app.get("/api/v1/health")
    def health():
        return {"status": "available"}

    mount_frontend(app, export, profile=profile, **options)
    return TestClient(app)


def test_each_profile_serves_only_its_own_pages(export):
    local = app_for("local_app", export)
    assert local.get("/").text.endswith("<body>compare</body></html>")
    assert local.get("/library/").status_code == 200
    for hidden in ("/participate/", "/admin/", "/legacy/"):
        assert local.get(hidden).status_code == 404
    demo = app_for("public_demo", export)
    assert demo.get("/browse/").status_code == 200
    assert demo.get("/library/").status_code == 404
    study = app_for("study", export, strict_styles=True, index_redirect="/participate/")
    assert study.get("/", follow_redirects=False).headers["location"] == "/participate/"
    assert study.get("/participate/").status_code == 200
    assert study.get("/admin/").status_code == 200
    for hidden in ("/browse/", "/library/", "/legacy/"):
        assert study.get(hidden).status_code == 404
    assert pages_for("study").isdisjoint(pages_for("local_app"))


def test_api_misses_assets_and_traversal_never_fall_through_to_pages(export):
    client = app_for("local_app", export)
    assert client.get("/api/v1/health").json() == {"status": "available"}
    missing = client.get("/api/v1/not-a-route")
    assert missing.status_code == 404 and missing.json()["code"] == "not_found"
    assert client.post("/api/v1/bundles/import", content=b"").status_code == 404
    assert client.get("/_next/static/a.js").text == "console.log(1)"
    assert client.get("/_next/static/../../secret.txt").status_code == 404
    assert client.get("/_next/%2e%2e/%2e%2e/secret.txt").status_code == 404
    assert client.get("/browse", follow_redirects=False).status_code == 308


def test_html_policy_allows_exact_inline_scripts_only(export):
    client = app_for("study", export, strict_styles=True)
    policy = client.get("/participate/").headers["content-security-policy"]
    digest = base64.b64encode(hashlib.sha256(INLINE.encode()).digest()).decode()
    assert f"'sha256-{digest}'" in policy
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert "frame-ancestors 'none'" in policy
    relaxed = content_security_policy(export / "index.html", strict_styles=False)
    assert "style-src 'self' 'unsafe-inline'" in relaxed
    assert "script-src 'self' 'unsafe-inline'" not in relaxed


def test_api_only_deployment_mounts_no_page_routes(export):
    app = FastAPI()
    mount_frontend(app, None, profile="local_app")
    assert TestClient(app).get("/").status_code == 404
