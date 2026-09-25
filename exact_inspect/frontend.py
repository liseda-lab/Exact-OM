"""Serve the static frontend export with server-owned page allowlists per deployment profile.

One Next.js export contains every product page. Each profile exposes only its own pages
(exploration pages never appear in the study service and vice versa); ``_next`` assets are
inert shared bundles. HTML responses carry a Content-Security-Policy whose script hashes
are computed from the served file, so inline bootstrap scripts run without 'unsafe-inline'.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPLORATION_PAGES = frozenset({"", "browse"})
LOCAL_ONLY_PAGES = frozenset({"library"})
STUDY_PAGES = frozenset({"participate", "admin"})
_INLINE_SCRIPT = re.compile(rb"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)


def pages_for(profile: str) -> frozenset[str]:
    """Return the HTML routes a deployment profile may serve."""
    if profile == "local_app":
        return EXPLORATION_PAGES | LOCAL_ONLY_PAGES
    if profile == "public_demo":
        return EXPLORATION_PAGES
    if profile == "study":
        return STUDY_PAGES
    raise ValueError("Unknown deployment profile")


@lru_cache(maxsize=64)
def _script_hashes(path: str, mtime_ns: int, size: int) -> tuple[str, ...]:
    content = Path(path).read_bytes()
    return tuple(
        "'sha256-" + base64.b64encode(hashlib.sha256(body).digest()).decode() + "'"
        for body in _INLINE_SCRIPT.findall(content)
        if body.strip()
    )


def content_security_policy(html: Path, *, strict_styles: bool) -> str:
    """Build a per-document policy; only this document's exact inline scripts may run."""
    stat = html.stat()
    hashes = " ".join(_script_hashes(str(html), stat.st_mtime_ns, stat.st_size))
    return "; ".join(
        [
            "default-src 'self'",
            f"script-src 'self' {hashes}".strip(),
            "style-src 'self'" + ("" if strict_styles else " 'unsafe-inline'"),
            "img-src 'self' data: blob:",
            "font-src 'self'",
            "connect-src 'self'",
            "object-src 'none'",
            "base-uri 'none'",
            "frame-ancestors 'none'",
            "form-action 'self'",
        ]
    )


def _inside(root: Path, relative: str) -> Path | None:
    if not relative or "\\" in relative or "\x00" in relative:
        return None
    parts = [p for p in relative.split("/") if p]
    if any(p in {".", ".."} for p in parts):
        return None
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        return None
    return resolved


def mount_frontend(
    app: Any,
    frontend_dir: Path | None,
    *,
    profile: str,
    strict_styles: bool = False,
    index_redirect: str | None = None,
) -> None:
    """Register the catch-all static route last, after every API route."""
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

    allowed = pages_for(profile)
    if frontend_dir is None:
        # API-only deployment: no page routes exist, exactly as before a frontend was built.
        return
    root = frontend_dir.resolve()

    def not_found() -> Any:
        return JSONResponse(
            status_code=404,
            content={"code": "not_found", "message": "Resource unavailable", "retryable": False},
        )

    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )
    def unknown_api(path: str):
        # Registered after every real route: unknown API calls (e.g. import on a hosted
        # profile) are plain 404s, never page responses or method errors.
        return not_found()

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def frontend(path: str):
        if path.startswith("api/") or path == "api":
            return not_found()
        if path.startswith("_next/") or path in {"favicon.ico", "icon.svg"}:
            asset = _inside(root, path)
            if asset is None:
                return not_found()
            media_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            response = FileResponse(asset, media_type=media_type)
            if path.startswith("_next/static/"):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response
        page = path.strip("/")
        if page == "" and index_redirect:
            return RedirectResponse(index_redirect, status_code=307)
        if page not in allowed:
            return not_found()
        if path and not path.endswith("/"):
            return RedirectResponse("/" + page + "/", status_code=308)
        html = _inside(root, (page + "/" if page else "") + "index.html")
        if html is None:
            return not_found()
        response = FileResponse(html, media_type="text/html; charset=utf-8")
        response.headers["Content-Security-Policy"] = content_security_policy(
            html, strict_styles=strict_styles
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
