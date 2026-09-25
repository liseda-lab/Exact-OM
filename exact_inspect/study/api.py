"""First-party study HTTP boundary; no exploration or provider routes are mounted."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .exports import csv_archive
from .models import (
    Consent,
    Consultation,
    EventAcknowledgement,
    EventBatch,
    Exchange,
    Invitations,
    Mutation,
    Publish,
    Questionnaire,
    Ranking,
    Resume,
    Setup,
    StudyCase,
    StudyState,
    TimingAcknowledgement,
    TimingSegment,
)
from .store import StudyError, StudyStore, canonical

COOKIE = "exact_study_session"


class RequestTooLarge(HTTPException):
    """Abort oversized JSON before model parsing or a transaction begins."""

    def __init__(self):
        super().__init__(413, "Request is too large")


class BodyLimitMiddleware:
    """Enforce streaming bounds even when Content-Length is absent or misleading."""

    def __init__(self, app, limit=1_000_000):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                raise RequestTooLarge()
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except RequestTooLarge:
            response = JSONResponse(
                {"detail": "Request is too large"},
                status_code=413,
                headers={
                    "Cache-Control": "private, no-store",
                    "Referrer-Policy": "no-referrer",
                },
            )
            await response(scope, receive, send)


class Authentication:
    """Sign pseudonymous cookie claims and keep invalid-exchange diagnostics opaque."""

    def __init__(self, signing_secret, researcher_token, origin):
        if len(signing_secret) < 32 or len(researcher_token) < 32:
            raise ValueError("Study signing and researcher secrets require at least 32 characters")
        if not origin.startswith("https://"):
            raise ValueError("Study origin must use HTTPS")
        self.key = signing_secret.encode()
        self.researcher_token = researcher_token
        self.origin = origin.rstrip("/")
        self.lock = threading.Lock()
        self.invalid_exchanges = []

    def cookie(self, sid, generation):
        payload = (
            base64.urlsafe_b64encode(canonical({"sid": sid, "generation": generation}).encode())
            .decode()
            .rstrip("=")
        )
        signature = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{signature}"

    def participant(self, request: Request):
        cookie = request.cookies.get(COOKIE, "")
        try:
            if len(cookie) > 512:
                raise ValueError()
            payload, signature = cookie.rsplit(".", 1)
            expected = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError()
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            return claims["sid"], int(claims["generation"])
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(401, "Session unavailable") from exc

    def researcher(self, request: Request):
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {self.researcher_token}"):
            raise HTTPException(401, "Researcher authentication required")

    def exchange_allowed(self):
        now = time.monotonic()
        with self.lock:
            self.invalid_exchanges = [t for t in self.invalid_exchanges if now - t < 60]
            if len(self.invalid_exchanges) >= 30:
                raise HTTPException(429, "Please retry later", headers={"Retry-After": "60"})

    def exchange_failed(self):
        with self.lock:
            self.invalid_exchanges.append(time.monotonic())


def create_study_router(store, *, signing_secret, researcher_token, origin):
    """Build isolated study and separately authenticated researcher routes."""
    auth = Authentication(signing_secret, researcher_token, origin)
    router = APIRouter(prefix="/api/v1")

    @router.post("/study/session", response_model=StudyState)
    def exchange(body: Exchange, response: Response):
        auth.exchange_allowed()
        try:
            sid, generation, state = store.exchange(body.secret)
        except StudyError:
            auth.exchange_failed()
            raise
        response.set_cookie(
            COOKIE,
            auth.cookie(sid, generation),
            secure=True,
            httponly=True,
            samesite="strict",
            max_age=30 * 24 * 3600,
            path="/api/v1/study",
        )
        return state

    @router.get("/study/state", response_model=StudyState)
    def state(identity=Depends(auth.participant)):
        return store.state(*identity)

    @router.put("/study/consent", response_model=StudyState)
    def consent(body: Consent, identity=Depends(auth.participant)):
        return store.mutate(*identity, "consent", body)

    @router.put("/study/setup", response_model=StudyState)
    def setup(body: Setup, identity=Depends(auth.participant)):
        return store.mutate(*identity, "setup", body)

    @router.put("/study/questionnaires/{form_id}", response_model=StudyState)
    def questionnaire(form_id: str, body: Questionnaire, identity=Depends(auth.participant)):
        return store.mutate(*identity, f"questionnaire:{form_id}", body)

    @router.get("/study/cases/current", response_model=StudyCase)
    def current(identity=Depends(auth.participant)):
        return store.current_case(*identity)

    @router.put("/study/cases/{case_id:path}/draft", response_model=StudyState)
    def draft(case_id: str, body: Ranking, identity=Depends(auth.participant)):
        return store.mutate(*identity, "draft", body, case_id)

    @router.post("/study/cases/{case_id:path}/submit", response_model=StudyState)
    def submit(case_id: str, body: Ranking, identity=Depends(auth.participant)):
        return store.mutate(*identity, "submit", body, case_id)

    @router.put("/study/cases/{case_id:path}/consultation", response_model=StudyState)
    def consultation(case_id: str, body: Consultation, identity=Depends(auth.participant)):
        return store.mutate(*identity, "consultation", body, case_id)

    @router.post("/study/events", response_model=EventAcknowledgement)
    def events(body: EventBatch, identity=Depends(auth.participant)):
        return store.events(*identity, body)

    @router.post("/study/timing", response_model=TimingAcknowledgement)
    def timing(body: TimingSegment, identity=Depends(auth.participant)):
        return store.timing(*identity, body)

    @router.post("/study/pause", response_model=StudyState)
    def pause(body: Mutation, identity=Depends(auth.participant)):
        return store.mutate(*identity, "pause", body)

    @router.post("/study/resume", response_model=StudyState)
    def resume(body: Resume, identity=Depends(auth.participant)):
        return store.mutate(*identity, "resume", body)

    @router.post("/study/complete", response_model=StudyState)
    def complete(body: Mutation, identity=Depends(auth.participant)):
        return store.mutate(*identity, "complete", body)

    @router.get("/study/resources/{asset_id:path}")
    def resource(asset_id: str, identity=Depends(auth.participant)):
        content, media_type = store.resource(*identity, asset_id)
        if isinstance(content, Path):
            return FileResponse(
                content, media_type=media_type, filename=asset_id.rsplit("/", 1)[-1] + ".ofn"
            )
        return Response(
            content, media_type=media_type, headers={"Content-Disposition": "attachment"}
        )

    @router.post("/admin/studies", dependencies=[Depends(auth.researcher)])
    def publish(body: Publish):
        return store.publish(body)

    @router.post(
        "/admin/studies/{revision:path}/invitations", dependencies=[Depends(auth.researcher)]
    )
    def issue(revision: str, body: Invitations):
        return store.issue(revision, body.count, test=body.test)

    @router.post("/admin/invitations/{session_id}/reissue", dependencies=[Depends(auth.researcher)])
    def reissue(session_id: str):
        return store.reissue(session_id)

    @router.post("/admin/invitations/{session_id}/revoke", dependencies=[Depends(auth.researcher)])
    def revoke(session_id: str):
        return store.reissue(session_id, revoke=True)

    @router.post("/admin/studies/{revision:path}/close", dependencies=[Depends(auth.researcher)])
    def close(revision: str):
        return store.close(revision)

    @router.get("/admin/studies/{revision:path}/progress", dependencies=[Depends(auth.researcher)])
    def progress(revision: str):
        return store.progress(revision)

    @router.post("/admin/studies/{revision:path}/exports", dependencies=[Depends(auth.researcher)])
    def export(
        revision: str,
        include_test: bool = False,
        include_keys: bool = False,
        format: Literal["json", "csv"] = "json",
    ):
        result = store.export(revision, include_test=include_test, include_keys=include_keys)
        if format == "csv":
            return Response(
                csv_archive(result),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{result["manifest"]["export_id"]}.zip"'
                },
            )
        return result

    @router.get("/admin/exports/{export_id}", dependencies=[Depends(auth.researcher)])
    def saved_export(export_id: str, format: Literal["json", "csv"] = "json"):
        result = store.saved_export(export_id)
        return (
            Response(csv_archive(result), media_type="application/zip")
            if format == "csv"
            else result
        )

    return router


def create_study_app(
    database_url,
    signing_secret,
    researcher_token,
    origin,
    assets_dir=None,
    *,
    allow_test_sqlite=False,
    frontend_dir=None,
):
    """Create the hosted study profile; only participant and researcher pages are served."""
    store = StudyStore(database_url, assets_dir, allow_test_sqlite=allow_test_sqlite)
    app = FastAPI(title="Exact ranking study", version="1.0", docs_url=None, redoc_url=None)
    app.state.study_store = store
    app.add_middleware(BodyLimitMiddleware)
    app.include_router(
        create_study_router(
            store,
            signing_secret=signing_secret,
            researcher_token=researcher_token,
            origin=origin,
        )
    )
    expected_origin = origin.rstrip("/")

    @app.middleware("http")
    async def protect(request: Request, call_next):
        # Mutations with participant cookies always require the deployment's origin.
        # The admin CLI authenticates with a separate bearer credential.
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and request.url.path.startswith("/api/v1/study")
            and request.headers.get("origin") != expected_origin
        ):
            response = JSONResponse({"detail": "Origin is not authorized"}, status_code=403)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Study pages carry a per-document policy (exact inline-script hashes, no
        # 'unsafe-inline'); everything else keeps the strict API default.
        if "content-security-policy" not in response.headers:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic's default "input" field can echo invitation secrets or free text.
        return JSONResponse(
            {
                "detail": [
                    {"loc": list(error["loc"]), "type": error["type"], "msg": error["msg"]}
                    for error in exc.errors()
                ]
            },
            status_code=422,
        )

    @app.exception_handler(StudyError)
    async def study_error(request, exc):
        return JSONResponse(
            {
                "detail": exc.message,
                **({"current_revision": exc.revision} if exc.revision is not None else {}),
            },
            status_code=exc.status,
        )

    @app.exception_handler(ValueError)
    async def input_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "profile": "study",
            "frontend": "static" if frontend_dir else "api-only",
        }

    @app.get("/api/ready")
    def ready():
        try:
            good = store.ready()
        except Exception:
            good = False
        return JSONResponse({"ready": good}, status_code=200 if good else 503)

    from ..frontend import mount_frontend

    mount_frontend(
        app, frontend_dir, profile="study", strict_styles=True, index_redirect="/participate/"
    )
    return app


def app_from_env():
    """Uvicorn factory; secrets remain outside command lines and checked-in config."""
    configured = os.environ.get("EXACT_STUDY_FRONTEND_DIR")
    bundled = Path(__file__).resolve().parents[1] / "static"
    frontend = Path(configured) if configured else bundled
    return create_study_app(
        database_url=os.environ["EXACT_STUDY_DATABASE_URL"],
        signing_secret=os.environ["EXACT_STUDY_SIGNING_SECRET"],
        researcher_token=os.environ["EXACT_STUDY_RESEARCHER_TOKEN"],
        origin=os.environ["EXACT_STUDY_ORIGIN"],
        assets_dir=Path(os.environ["EXACT_STUDY_ASSETS_DIR"]),
        frontend_dir=frontend if (frontend / "index.html").is_file() else None,
    )
