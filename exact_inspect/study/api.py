"""First-party study HTTP boundary; no exploration or provider routes are mounted."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import Headers, MutableHeaders

from .administration import RevisionPage, revisions
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
from .telemetry import (
    EventBatchV2,
    TimingAcknowledgementV2,
    TimingSegmentV2,
    save_events,
    save_timing,
)
from .v2_models import (
    ConsultationDraftV2,
    ConsultationV2,
    SetupV2,
    StudyCaseV2,
    StudyStateV2,
    TutorialAssessment,
    TutorialComplete,
    TutorialProgressMutation,
)

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


class StudyProtectionMiddleware:
    """Apply origin and security headers without reconstructing response streams.

    Forward the producer's body and completion messages unchanged. In particular,
    an interrupted producer must not acquire a synthetic final body: that would
    contradict Content-Length and replace the original cancellation/failure with
    an HTTP framing error.
    """

    def __init__(self, app, *, origin):
        self.app = app
        self.origin = origin.rstrip("/")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def protected_send(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": list(message.get("headers", []))}
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "private, no-store"
                headers["Referrer-Policy"] = "no-referrer"
                headers["X-Content-Type-Options"] = "nosniff"
                # Keep the document's exact script hashes and strict style policy.
                if "content-security-policy" not in headers:
                    headers["Content-Security-Policy"] = (
                        "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
                    )
            await send(message)

        # Cookie-authenticated mutations require the configured deployment origin;
        # the researcher CLI uses a separate bearer credential.
        if (
            scope["method"] not in {"GET", "HEAD", "OPTIONS"}
            and scope["path"].startswith("/api/v1/study")
            and Headers(scope=scope).get("origin") != self.origin
        ):
            response = JSONResponse({"detail": "Origin is not authorized"}, status_code=403)
            await response(scope, receive, protected_send)
        else:
            await self.app(scope, receive, protected_send)


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
        self.tutorial_requests = OrderedDict()

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

    def participant_write(
        self,
        request: Request,
        x_study_session: str | None = Header(
            default=None,
            alias="X-Study-Session",
            description="Expected session ID; a cookie switch returns 409 before mutation.",
        ),
    ):
        """Reject a stale tab's session binding before invoking any participant mutation."""
        identity = self.participant(request)
        if x_study_session is not None and x_study_session != identity[0]:
            raise HTTPException(
                409, "Study session changed; reopen the original invitation before saving."
            )
        return identity

    def researcher(self, request: Request):
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {self.researcher_token}"):
            raise HTTPException(401, "Researcher authentication required")

    def tutorial_write(
        self,
        request: Request,
        x_study_session: str | None = Header(default=None, alias="X-Study-Session"),
    ):
        """A short request throttle; pedagogical retries have no lifetime limit."""
        identity = self.participant_write(request, x_study_session)
        now = time.monotonic()
        with self.lock:
            timestamps = [
                value for value in self.tutorial_requests.pop(identity[0], []) if now - value < 60
            ]
            limited = len(timestamps) >= 120
            self.tutorial_requests[identity[0]] = timestamps if limited else [*timestamps, now]
            while len(self.tutorial_requests) > 2048:
                self.tutorial_requests.popitem(last=False)
        if limited:
            raise HTTPException(
                429,
                "Please retry shortly; your tutorial progress is saved",
                headers={"Retry-After": "60"},
            )
        return identity

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
    from .workspace import install_workspace_routes

    install_workspace_routes(router, store, auth.participant_write)

    @router.post(
        "/study/session",
        response_model=StudyState | StudyStateV2,
        response_model_exclude_unset=True,
    )
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

    @router.get(
        "/study/state", response_model=StudyState | StudyStateV2, response_model_exclude_unset=True
    )
    def state(identity=Depends(auth.participant)):
        return store.state(*identity)

    @router.put(
        "/study/consent",
        response_model=StudyState | StudyStateV2,
        response_model_exclude_unset=True,
    )
    def consent(body: Consent, identity=Depends(auth.participant_write)):
        return store.mutate(*identity, "consent", body)

    @router.put(
        "/study/setup", response_model=StudyState | StudyStateV2, response_model_exclude_unset=True
    )
    def setup(body: Setup | SetupV2, identity=Depends(auth.participant_write)):
        return store.mutate(*identity, "setup", body)

    @router.put(
        "/study/tutorial/progress", response_model=StudyStateV2, response_model_exclude_unset=True
    )
    def tutorial_progress(body: TutorialProgressMutation, identity=Depends(auth.tutorial_write)):
        return store.mutate(*identity, "tutorial_progress", body)

    @router.post(
        "/study/tutorial/assessment", response_model=StudyStateV2, response_model_exclude_unset=True
    )
    def tutorial_assessment(body: TutorialAssessment, identity=Depends(auth.tutorial_write)):
        return store.mutate(*identity, "tutorial_assessment", body)

    @router.post(
        "/study/tutorial/complete", response_model=StudyStateV2, response_model_exclude_unset=True
    )
    def tutorial_complete(body: TutorialComplete, identity=Depends(auth.tutorial_write)):
        return store.mutate(*identity, "tutorial_complete", body)

    @router.put(
        "/study/questionnaires/{form_id}",
        response_model=StudyState | StudyStateV2,
        response_model_exclude_unset=True,
    )
    def questionnaire(form_id: str, body: Questionnaire, identity=Depends(auth.participant_write)):
        return store.mutate(*identity, f"questionnaire:{form_id}", body)

    @router.get("/study/cases/current", response_model=StudyCase | StudyCaseV2)
    def current(identity=Depends(auth.participant_write)):
        return store.current_case(*identity)

    @router.put(
        "/study/cases/{case_id:path}/consultation/draft",
        response_model=StudyStateV2,
        response_model_exclude_unset=True,
    )
    def consultation_draft(
        case_id: str, body: ConsultationDraftV2, identity=Depends(auth.participant_write)
    ):
        return store.mutate(*identity, "consultation_draft", body, case_id)

    @router.put(
        "/study/cases/{case_id:path}/draft",
        response_model=StudyState | StudyStateV2,
        response_model_exclude_unset=True,
    )
    def draft(case_id: str, body: Ranking, identity=Depends(auth.participant_write)):
        return store.mutate(*identity, "draft", body, case_id)

    @router.post(
        "/study/cases/{case_id:path}/submit",
        response_model=StudyState | StudyStateV2,
        response_model_exclude_unset=True,
    )
    def submit(case_id: str, body: Ranking, identity=Depends(auth.participant_write)):
        return store.mutate(*identity, "submit", body, case_id)

    @router.put(
        "/study/cases/{case_id:path}/consultation",
        response_model=StudyState | StudyStateV2,
        response_model_exclude_unset=True,
    )
    def consultation(
        case_id: str, body: Consultation | ConsultationV2, identity=Depends(auth.participant_write)
    ):
        return store.mutate(*identity, "consultation", body, case_id)

    @router.post("/study/events", response_model=EventAcknowledgement)
    def events(body: EventBatch | EventBatchV2, identity=Depends(auth.participant_write)):
        return save_events(store, *identity, body)

    @router.post("/study/timing", response_model=TimingAcknowledgement | TimingAcknowledgementV2)
    def timing(body: TimingSegment | TimingSegmentV2, identity=Depends(auth.participant_write)):
        return save_timing(store, *identity, body)

    @router.post(
        "/study/pause", response_model=StudyState | StudyStateV2, response_model_exclude_unset=True
    )
    def pause(body: Mutation, identity=Depends(auth.participant_write)):
        return store.mutate(*identity, "pause", body)

    @router.post(
        "/study/resume", response_model=StudyState | StudyStateV2, response_model_exclude_unset=True
    )
    def resume(body: Resume, identity=Depends(auth.participant_write)):
        return store.mutate(*identity, "resume", body)

    @router.post(
        "/study/complete",
        response_model=StudyState | StudyStateV2,
        response_model_exclude_unset=True,
    )
    def complete(body: Mutation, identity=Depends(auth.participant_write)):
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

    @router.get(
        "/admin/studies", response_model=RevisionPage, dependencies=[Depends(auth.researcher)]
    )
    def list_revisions(
        limit: int = Query(50, ge=1, le=100),
        cursor: str | None = Query(None, max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$"),
    ):
        return revisions(store, limit=limit, cursor=cursor)

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
        analysis_schema: (
            Literal["exact-study-analysis/1", "exact-study-analysis/2", "exact-study-analysis/3"]
            | None
        ) = None,
    ):
        result = store.export(
            revision,
            include_test=include_test,
            include_keys=include_keys,
            analysis_schema=analysis_schema,
        )
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
    app.add_middleware(StudyProtectionMiddleware, origin=origin)
    app.include_router(
        create_study_router(
            store,
            signing_secret=signing_secret,
            researcher_token=researcher_token,
            origin=origin,
        )
    )

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
