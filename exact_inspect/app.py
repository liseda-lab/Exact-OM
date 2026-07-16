"""FastAPI application factory for Exact run inspection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .settings import InspectSettings


def _frontend_export(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    return resolved if (resolved / "index.html").is_file() else None


def resolve_frontend_dir(
    settings: InspectSettings,
    *,
    package_dir: Path | None = None,
    project_root: Path | None = None,
) -> Path | None:
    """Resolve the UI export without assuming a repository checkout."""

    if settings.frontend_dir is not None:
        explicit = _frontend_export(settings.frontend_dir)
        if explicit is not None:
            return explicit

    package_root = package_dir or Path(__file__).resolve().parent
    packaged = _frontend_export(package_root / "static")
    if packaged is not None:
        return packaged

    repo_root = project_root or package_root.parent
    return _frontend_export(repo_root / "explanations_visualizer" / "out")


def _fastapi_components() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - isolated package test
        raise RuntimeError(
            "exact-inspect requires FastAPI. Install the visualization extra with "
            '`pip install "exact-om[viz]"`.'
        ) from exc
    return FastAPI, HTTPException, Query, CORSMiddleware, StaticFiles


def create_app(settings: InspectSettings) -> Any:
    """Create an inspection app from explicit settings, with no global run path."""

    FastAPI, HTTPException, Query, CORSMiddleware, StaticFiles = _fastapi_components()
    try:
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover - guarded above
        raise RuntimeError("Install the visualization extra with `exact-om[viz]`.") from exc
    from .bundles import InspectionService

    if settings.run_dir is None:
        raise ValueError(
            "exact-inspect requires a run directory. Pass one to `exact-inspect open`, "
            "use `exact-inspect serve --run-dir`, or set EXACT_INSPECT_RUN_DIR."
        )
    if not settings.run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {settings.run_dir}")

    logger = logging.getLogger("exact_inspect")
    service = InspectionService(
        run_dir=settings.run_dir,
        analysis_dir=settings.analysis_dir,
        enable_ontology_info=settings.enable_ontology_info,
        source_ontology_path=settings.source_ontology_path,
        target_ontology_path=settings.target_ontology_path,
        logger=logger,
    )
    frontend_dir = resolve_frontend_dir(settings)
    app = FastAPI(title="Exact Inspect")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.inspection_service = service
    app.state.study_service = service
    app.state.frontend_dir = frontend_dir

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        return {
            **service.health_payload(),
            "frontend_mode": "static" if frontend_dir else "api-only",
            "frontend_dir": str(frontend_dir) if frontend_dir else None,
        }

    @app.get("/api/study/source")
    def api_study_source(
        source: str = Query(..., description="Exact source IRI/ID"),
    ) -> dict[str, Any]:
        try:
            return service.get_source_bundle(source)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown study source: {source}") from exc

    @app.get("/api/study/sources")
    def api_study_sources() -> list[dict[str, str]]:
        return service.source_options()

    @app.get("/api/study/node-info")
    def api_study_node_info(
        source: str = Query(...),
        target: str = Query(...),
        node_id: str = Query(...),
    ) -> dict[str, Any]:
        try:
            return service.get_node_info(source, target, node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown study source/target/node") from exc

    @app.get("/api/study/expand-node")
    def api_study_expand_node(
        source: str = Query(...),
        target: str = Query(...),
        node_id: str = Query(...),
    ) -> dict[str, Any]:
        try:
            return service.expand_node(source, target, node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown study source/target/node") from exc

    if frontend_dir is None:
        logger.warning(
            "No Exact Inspect frontend export was found; serving the API-only banner. "
            "Build explanations_visualizer or install a release wheel with the viz extra."
        )

        @app.get("/", response_class=HTMLResponse)
        def api_only_banner() -> str:
            return (
                "<!doctype html><html><head><title>Exact Inspect API</title></head>"
                "<body><h1>Exact Inspect is running in API-only mode</h1>"
                "<p>The frontend static export is not installed. The API is available under "
                "<code>/api/</code>.</p></body></html>"
            )

    else:
        logger.info("Serving Exact Inspect frontend from %s", frontend_dir)
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    return app


def create_study_visualizer_app(
    run_dir: Path,
    analysis_dir: Optional[Path] = None,
    enable_ontology_info: bool = True,
    logger: Optional[logging.Logger] = None,
    frontend_build_dir: Optional[Path] = None,
) -> Any:
    """Compatibility factory retained for callers of the old runtime package."""

    if logger is not None:
        logging.getLogger("exact_inspect").setLevel(logger.level)
    return create_app(
        InspectSettings(
            run_dir=run_dir,
            analysis_dir=analysis_dir,
            enable_ontology_info=enable_ontology_info,
            frontend_dir=frontend_build_dir,
        )
    )


__all__ = ["create_app", "create_study_visualizer_app", "resolve_frontend_dir"]
