#!/usr/bin/env python3
"""Verify the installed local app, without a checkout, network or matcher run."""

import contextlib
import io
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import exact_inspect
import exact
from exact_inspect.app import create_app, resolve_frontend_dir
from exact_inspect.cli import main
from exact_inspect.settings import InspectSettings


def run():
    package = Path(exact_inspect.__file__).resolve().parent
    if "site-packages" not in package.parts:
        raise RuntimeError(f"Expected an installed package, got {package}")
    assert Path(exact.__file__).resolve().parent.parent == package.parent
    with tempfile.TemporaryDirectory(prefix="exact-inspect-installed-") as temporary:
        root = Path(temporary)
        for side in ("source", "target"):
            (root / (side + ".ofn")).write_text(
                f"Ontology(<urn:{side}> Declaration(Class(<urn:{side}:root>)))"
            )
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            assert (
                main(
                    [
                        "prepare",
                        "--source-ontology",
                        str(root / "source.ofn"),
                        "--target-ontology",
                        str(root / "target.ofn"),
                        "--output",
                        str(root / "prepared"),
                    ]
                )
                == 0
            )
        prepared = Path(json.loads(captured.getvalue())["package"])
        assert (
            main(["export", "--package", str(prepared), "--output", str(root / "bundle.zip")]) == 0
        )
        settings = InspectSettings(package=prepared, library_dir=root / "library")
        frontend = resolve_frontend_dir(settings)
        assert frontend == package / "static"
        with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
            for page in ("/", "/browse/", "/library/"):
                response = client.get(page)
                assert response.status_code == 200, (page, response.text)
                assert "Exact Explain" in response.text
                assert "script-src 'self' 'sha256-" in response.headers["Content-Security-Policy"]
            for page in ("/admin/", "/participate/", "/legacy/", "/api/v1/study/state"):
                assert client.get(page).status_code == 404
            for asset in frontend.rglob("*.js"):
                response = client.get("/" + asset.relative_to(frontend).as_posix())
                assert response.status_code == 200
                assert response.content == asset.read_bytes()
            assert client.get("/api/v1/health").json()["profile"] == "local_app"
            assert client.get("/api/v1/bundles").status_code == 200


if __name__ == "__main__":
    run()
    print(json.dumps({"status": "passed", "package": exact_inspect.__file__}))
