"""Serve the local explorer and its prepared-data API from one origin."""

from pathlib import Path
import sys

from .settings import InspectSettings


def resolve_frontend_dir(settings: InspectSettings) -> Path:
    if settings.frontend_dir is not None:
        paths = [settings.frontend_dir]
    else:
        package = Path(__file__).resolve().parent
        paths = [package / "static", package.parent / "explanations_visualizer" / "out"]
    for path in paths:
        if all(
            (path / page).is_file()
            for page in ("index.html", "browse/index.html", "library/index.html")
        ):
            return path.resolve()
    raise ValueError(
        "Frontend export not found. From the source checkout run "
        "`python scripts/build_frontend.py`, or pass --frontend-dir PATH to a built export. "
        "Release wheels include the frontend."
    )


def create_app(settings: InspectSettings):
    if sys.version_info < (3, 11):
        raise ValueError("Exact Explain requires Python 3.11 or 3.12")
    from fastapi.middleware.trustedhost import TrustedHostMiddleware

    from .service import create_prepared_app

    app = create_prepared_app(
        settings.package,
        profile=settings.profile,
        library_dir=settings.library_dir,
        frontend_dir=resolve_frontend_dir(settings),
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", settings.host]
    )
    return app
