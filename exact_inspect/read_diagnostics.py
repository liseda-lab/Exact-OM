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
                handler = RotatingFileHandler(target, maxBytes=1_048_576, backupCount=2)
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
            or not scope.get("path", "").startswith("/api/v1/")
        ):
            return await self.app(scope, receive, send)
        started = time.monotonic()
        identity = uuid4().hex
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
        self.emit(
            {
                "schema": record["schema"],
                "read_id": identity,
                "event": "started",
                "time_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                record["disconnect_seen"] = True
            return message

        async def observed_send(message):
            if message["type"] == "http.response.start":
                record["status"] = message["status"]
                headers = list(message.get("headers", []))
                for name, value in headers:
                    if name.lower() == b"content-length" and value.isdigit():
                        record["declared_length"] = int(value)
                message = {
                    **message,
                    "headers": headers + [(b"x-exact-read-id", identity.encode())],
                }
            await send(message)
            if message["type"] == "http.response.body":
                record["sent_bytes"] += len(message.get("body", b""))
                record["complete"] = not message.get("more_body", False)

        try:
            await self.app(scope, observed_receive, observed_send)
        except BaseException as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            # Route templates are static application code; raw paths/queries never enter logs.
            route = getattr(scope.get("route"), "path", None)
            record["route"] = (
                route if route and re.fullmatch(r"[/a-zA-Z0-9_{}:.\-]+", route) else "unmatched"
            )
            record["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
            record["event"] = "finished"
            self.emit(record)
