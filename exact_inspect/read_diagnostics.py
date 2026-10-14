"""Opt-in bounded HTTP-read diagnostics, without URLs, identifiers or user data.

Pure ASGI observation: never consume ahead, retry, swallow cancellation/errors, or
change bodies, content lengths, authentication or CSP. Each worker uses its own file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

from .public_assets import public_asset


class _PrivateRotatingFileHandler(RotatingFileHandler):
    def _open(self):
        return open(
            self.baseFilename,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            opener=lambda path, flags: os.open(path, flags, 0o600),
        )


class ReadDiagnostics:
    def __init__(self, app, *, path=None, emit=None):
        self.app = app
        self.emit = emit
        if emit is None:
            path = path or os.environ.get("EXACT_READ_DIAGNOSTICS")
            if path:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch(mode=0o600, exist_ok=True)
                handler = _PrivateRotatingFileHandler(target, maxBytes=1_048_576, backupCount=2)
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger = logging.getLogger("exact.reads." + uuid4().hex)
                logger.propagate = False
                logger.setLevel(logging.INFO)
                logger.addHandler(handler)
                self.emit = lambda row: logger.info(json.dumps(row, separators=(",", ":")))

    async def __call__(self, scope, receive, send):
        if (
            self.emit is None
            or scope["type"] != "http"
            or scope.get("method") not in {"GET", "HEAD"}
        ):
            return await self.app(scope, receive, send)
        started = time.monotonic()
        started_utc = datetime.now(timezone.utc).isoformat()
        identity = uuid4().hex
        api = scope.get("path", "").startswith("/api/v1/")
        active = False
        record = {
            "schema": "exact-read-diagnostic/1",
            "read_id": identity,
            "method": scope["method"],
            "status": None,
            "declared_length": None,
            "sent_bytes": 0,
            "complete": False,
            "disconnect_seen": False,
            "error_type": None,
        }

        def begin():
            nonlocal active
            asset = public_asset(scope)
            if not active and (api or asset is not None):
                if not api:
                    record.update(
                        schema="exact-static-transfer/1",
                        asset_sha256=asset.sha256,
                        asset_size=asset.stat.st_size,
                    )
                active = True
                self.emit(
                    {
                        "schema": record["schema"],
                        "read_id": identity,
                        "event": "started",
                        "time_utc": started_utc,
                        **({"asset_sha256": asset.sha256} if not api else {}),
                    }
                )

        if api:
            begin()

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                record["disconnect_seen"] = True
            return message

        async def observed_send(message):
            if message["type"] == "http.response.start":
                begin()
                if active:
                    record["status"] = message["status"]
                    headers = list(message.get("headers", []))
                    for name, value in headers:
                        if name.lower() == b"content-length" and value.isdigit():
                            record["declared_length"] = int(value)
                    extra = [(b"x-exact-read-id", identity.encode())]
                    if not api:
                        extra.append((b"x-exact-asset-sha256", record["asset_sha256"].encode()))
                    message = {**message, "headers": headers + extra}
            await send(message)
            if message["type"] == "http.response.body":
                record["sent_bytes"] += len(message.get("body", b""))
                record["complete"] = not message.get("more_body", False)
            elif message["type"] == "http.response.pathsend":
                # The extension hands delivery to the server. No terminal body or bytes
                # are visible here, so application handoff is not transfer completion.
                record.update(sent_bytes=None, complete=None, offloaded=True)

        try:
            await self.app(scope, observed_receive, observed_send)
        except BaseException as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            begin()  # A known asset can fail before its response headers were sent.
            if active:
                route = getattr(scope.get("route"), "path", None)
                record["route"] = (
                    route
                    if api and route and re.fullmatch(r"[/a-zA-Z0-9_{}:.\-]+", route)
                    else "unmatched" if api else "public_build_asset"
                )
                record["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
                record["event"] = "finished"
                if not api:
                    record["length_matches"] = (
                        record["sent_bytes"] == record["declared_length"]
                        if record["complete"]
                        and record["sent_bytes"] is not None
                        and record["declared_length"] is not None
                        and scope["method"] != "HEAD"
                        else None
                    )
                    record["head_body_empty"] = (
                        record["sent_bytes"] == 0
                        if scope["method"] == "HEAD" and record["sent_bytes"] is not None
                        else None
                    )
                self.emit(record)
