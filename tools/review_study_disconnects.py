"""Loopback-only synthetic disconnect diagnostic, never a deployment entrypoint.

Run with the same required arguments as tools.serve_explanation_v2_e2e. It uses
that real HTTPS/PostgreSQL app and its unchanged document CSP, adding two clearly
marked diagnostic endpoints: an interrupted producer and a slow response for
client-abort/reload trials. No participant values or credentials are logged.
"""

from __future__ import annotations

import asyncio
import json

from starlette.datastructures import Headers
from starlette.responses import StreamingResponse
from starlette.routing import Route

from tools import serve_explanation_v2_e2e as harness


class CancelledProducer:
    async def __call__(self, scope, receive, send):
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"8")]}
        )
        await send({"type": "http.response.body", "body": b"part", "more_body": True})
        raise asyncio.CancelledError("injected diagnostic producer cancellation")


async def slow_response(request):
    async def chunks():
        try:
            for _ in range(512):
                yield b"x" * 16384
                await asyncio.sleep(0.01)
        finally:
            print(json.dumps({"diagnostic": "slow_producer_finalized"}), flush=True)

    return StreamingResponse(
        chunks(),
        media_type="application/octet-stream",
        headers={"content-length": str(512 * 16384)},
    )


class DiagnosticObserver:
    """Record original producer messages before forwarding, not network delivery.

    producer_bytes and producer_final_body describe attempted producer output
    before diagnostic chunk splitting. They do not prove that a disconnected
    browser received those bytes or the final body. The browser probe separately
    verifies its first received chunk and subsequent complete response hashes.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        delayed_resource = (
            scope["path"].startswith("/api/v1/study/resources/")
            and Headers(scope=scope).get("x-review-delay-body") == "1"
        )
        if not scope["path"].startswith("/_review/") and not delayed_resource:
            return await self.app(scope, receive, send)
        record = {
            "diagnostic": "authenticated_resource_delayed" if delayed_resource else scope["path"],
            "producer_bytes": 0,
            "producer_final_body": False,
            "disconnect_observed": False,
            "exception": None,
        }

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                record["disconnect_observed"] = True
            return message

        async def observed_send(message):
            if message["type"] == "http.response.body":
                record["producer_bytes"] += len(message.get("body", b""))
                record["producer_final_body"] = not message.get("more_body", False)
            if (
                delayed_resource
                and message["type"] == "http.response.body"
                and len(message.get("body", b"")) > 1024
            ):
                # Delay the admitted route's ORIGINAL body without changing bytes,
                # length, authentication, policy, or headers; only ASGI chunking differs.
                body = message["body"]
                await send({**message, "body": body[:1024], "more_body": True})
                await asyncio.sleep(1)
                await send({**message, "body": body[1024:]})
            else:
                await send(message)

        try:
            await self.app(scope, observed_receive, observed_send)
        except BaseException as exc:
            record["exception"] = type(exc).__name__
            raise
        finally:
            print(json.dumps(record, sort_keys=True), flush=True)


def main():
    create_app = harness.create_study_app

    def diagnostic_app(*args, **kwargs):
        app = create_app(*args, **kwargs)
        app.router.routes[:0] = [
            Route("/_review/cancel", CancelledProducer()),
            Route("/_review/slow", slow_response),
        ]
        app.add_middleware(DiagnosticObserver)
        return app

    harness.create_study_app = diagnostic_app
    harness.main()


if __name__ == "__main__":
    main()
