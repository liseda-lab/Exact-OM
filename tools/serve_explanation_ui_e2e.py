"""Serve a synthetic study over actual HTTPS and PostgreSQL for browser verification.

Build the small fixture with tools.build_explanation_ui_fixture first. This tool
never rewrites origins, calls providers, or creates non-test invitations. The
private configuration file lets Playwright authenticate without logging tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path

from tools.serve_explanation_ui_study import publication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--tls-cert", required=True, type=Path)
    parser.add_argument("--tls-key", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--frontend-dir", type=Path, default=Path("explanations_visualizer/out"))
    args = parser.parse_args()
    if not args.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        parser.error("This verification harness requires an isolated PostgreSQL database")
    frontend = args.frontend_dir.resolve()
    if not (frontend / "participate/index.html").is_file():
        parser.error("Build the frontend before starting the harness")
    assets = args.fixture.resolve() / "study-e2e"
    assets.mkdir(exist_ok=True)
    frozen = assets / "publication.json"
    if not frozen.exists():
        frozen.write_text(json.dumps(publication(args.fixture.resolve(), assets), indent=2))
    config_path = args.config.resolve()
    origin = f"https://127.0.0.1:{args.port}"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config["origin"] != origin:
            parser.error("Existing private configuration has a different origin")
    else:
        config = {
            "origin": origin,
            "researcher_token": secrets.token_urlsafe(32),
            "signing_secret": secrets.token_urlsafe(32),
            "publication": str(frozen),
            "study_revision": json.loads(frozen.read_text())["definition"]["study_revision"],
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(config, handle)

    from exact_inspect.study import create_study_app
    from exact_inspect.study.models import Publish

    app = create_study_app(
        args.database_url,
        config["signing_secret"],
        config["researcher_token"],
        origin,
        assets_dir=assets,
        frontend_dir=frontend,
    )
    app.state.study_store.publish(Publish.model_validate_json(frozen.read_text()))
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        ssl_certfile=str(args.tls_cert),
        ssl_keyfile=str(args.tls_key),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
