"""Serve the admitted synthetic v2 fixture over real loopback HTTPS/PostgreSQL."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path

from exact_inspect.study.api import create_study_app
from exact_inspect.study.models import Publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--tls-cert", type=Path, required=True)
    parser.add_argument("--tls-key", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18974)
    parser.add_argument("--frontend-dir", type=Path, default=Path("explanations_visualizer/out"))
    args = parser.parse_args()
    if not args.database_url.startswith(("postgresql://", "postgres://")):
        parser.error("Use a dedicated PostgreSQL test database")
    publication_path = args.fixture.resolve() / "publication.json"
    publication = Publish.model_validate_json(publication_path.read_text())
    if (
        not publication.definition.synthetic
        or publication.definition.contract_version != "exact-study/2.0"
    ):
        parser.error("Only explicit synthetic v2 fixtures are accepted")
    origin = f"https://127.0.0.1:{args.port}"
    if args.config.exists():
        config = json.loads(args.config.read_text())
        if (
            config["origin"] != origin
            or config["study_revision"] != publication.definition.study_revision
        ):
            parser.error("Private harness config does not match the fixture/origin")
    else:
        config = {
            "origin": origin,
            "researcher_token": secrets.token_urlsafe(32),
            "signing_secret": secrets.token_urlsafe(32),
            "publication": str(publication_path),
            "study_revision": publication.definition.study_revision,
        }
        with os.fdopen(
            os.open(args.config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        ) as handle:
            json.dump(config, handle)
    app = create_study_app(
        args.database_url,
        config["signing_secret"],
        config["researcher_token"],
        origin,
        args.fixture,
        frontend_dir=args.frontend_dir.resolve(),
    )
    app.state.study_store.publish(publication)
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        ssl_certfile=str(args.tls_cert),
        ssl_keyfile=str(args.tls_key),
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
